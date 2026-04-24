"""Backend model wrapper for chest X-ray inference and Grad-CAM."""

from __future__ import annotations

import base64
import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from torch import Tensor, nn
from torchvision import transforms

from data_loader import bone_suppression, lung_crop
from models import get_resnet50


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_SIZE = 448
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_resnet50.pt"
DEFAULT_SAMPLE_DIRS = (
    PROJECT_ROOT / "sample_images",
    PROJECT_ROOT / "assets" / "media",
)
ALLOWED_CHECKPOINT_EXTENSIONS = {".pt", ".pth"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_SAMPLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_GRADCAM_ALPHA = 0.45
PREDICTION_THRESHOLD = 0.5


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalizes a grayscale image to the ``uint8`` display range."""

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _sanitize_name(value: str) -> str:
    """Converts a filename or label into a safe slug."""

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return slug or "artifact"


def _pil_from_bytes(image_bytes: bytes) -> Image.Image:
    """Loads an image from bytes and converts it to single-channel grayscale."""

    return Image.open(io.BytesIO(image_bytes)).convert("L")


def _encode_gender(value: Optional[str]) -> float:
    """Encodes gender into the model's expected scalar representation."""

    if value is None:
        return 0.5

    normalized = value.strip().lower()
    if normalized in {"male", "m", "1"}:
        return 1.0
    if normalized in {"female", "f", "0"}:
        return 0.0
    return 0.5


def _encode_metadata(age: Optional[int], gender: Optional[str]) -> np.ndarray:
    """Builds the ``[age, gender, view]`` metadata vector used by the model."""

    age_value = float(np.clip((age or 0) / 100.0, 0.0, 1.2))
    gender_value = _encode_gender(gender)
    view_value = 0.25
    return np.array([age_value, gender_value, view_value], dtype=np.float32)


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    """Serializes a grayscale or RGB image array to PNG bytes."""

    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    elif image.ndim == 3 and image.shape[2] == 3:
        pil_image = Image.fromarray(image, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape for PNG export: {image.shape}")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_data_uri(png_bytes: bytes) -> str:
    """Converts PNG bytes into a base64 data URI."""

    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class LightweightGradCAM:
    """Minimal Grad-CAM implementation for ResNet-style classifiers."""

    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        """Stores activations for Grad-CAM."""

        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Optional[Tensor], ...],
    ) -> None:
        """Stores gradients for Grad-CAM."""

        if grad_output[0] is None:
            raise RuntimeError("Grad-CAM backward hook received no gradients.")
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: Tensor,
        class_index: int,
        metadata: Optional[Tensor] = None,
    ) -> np.ndarray:
        """Generates a normalized Grad-CAM heatmap."""

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor, metadata=metadata)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape (1, num_classes), got {tuple(logits.shape)}")

        target_score = logits[:, class_index].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)[0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        """Removes Grad-CAM hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()


@dataclass
class LoadedModelBundle:
    """Runtime metadata for the currently loaded checkpoint."""

    model: nn.Module
    class_names: List[str]
    expected_input_size: int
    metadata_enabled: bool
    checkpoint_path: Path


@dataclass
class PredictionResult:
    """Prediction payload stored in memory for API responses and exports."""

    checkpoint_path: str
    source_name: str
    input_size: int
    scores: Dict[str, float]
    predicted: str
    predicted_labels: List[str]
    gradcam_data_uri: str
    gradcam_png_bytes: bytes
    original_image: np.ndarray
    processed_image: np.ndarray
    bone_preview: np.ndarray
    crop_preview: np.ndarray
    metadata: Dict[str, object]
    checkpoint_loaded: bool = True

    def as_response(self) -> Dict[str, object]:
        """Converts the result into the API response schema."""

        return {
            "scores": self.scores,
            "predicted": self.predicted,
            "predicted_labels": self.predicted_labels,
            "gradcam": self.gradcam_data_uri,
            "input_size": self.input_size,
        }


class ChestXrayInferenceService:
    """High-level model service used by the FastAPI backend."""

    def __init__(
        self,
        default_checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
        sample_dirs: Sequence[Path] = DEFAULT_SAMPLE_DIRS,
        prediction_threshold: float = PREDICTION_THRESHOLD,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_checkpoint_path = Path(default_checkpoint_path)
        self.sample_dirs = [Path(path) for path in sample_dirs]
        self.prediction_threshold = prediction_threshold
        self.loaded_bundle: Optional[LoadedModelBundle] = None
        self._transform = self._build_transform(DEFAULT_INPUT_SIZE)

    @property
    def checkpoint_loaded(self) -> bool:
        """Returns whether a model checkpoint is currently loaded."""

        return self.loaded_bundle is not None

    @property
    def class_names(self) -> List[str]:
        """Returns the loaded class names, or an empty list if no model is active."""

        if self.loaded_bundle is None:
            return []
        return self.loaded_bundle.class_names

    @property
    def expected_input_size(self) -> int:
        """Returns the active input size."""

        if self.loaded_bundle is None:
            return DEFAULT_INPUT_SIZE
        return self.loaded_bundle.expected_input_size

    @property
    def metadata_enabled(self) -> bool:
        """Returns whether the current model expects metadata."""

        if self.loaded_bundle is None:
            return False
        return self.loaded_bundle.metadata_enabled

    def status(self) -> Dict[str, object]:
        """Returns an in-memory status snapshot for the API."""

        return {
            "checkpoint_loaded": self.checkpoint_loaded,
            "detected_classes": len(self.class_names),
            "expected_input_size": self.expected_input_size,
            "metadata_enabled": self.metadata_enabled,
        }

    def try_load_default_checkpoint(self) -> None:
        """Loads the default checkpoint when it exists."""

        if not self.default_checkpoint_path.exists():
            LOGGER.info("Default checkpoint not found at %s", self.default_checkpoint_path)
            return

        try:
            self.load_checkpoint(self.default_checkpoint_path)
            LOGGER.info("Loaded default checkpoint from %s", self.default_checkpoint_path)
        except Exception:
            LOGGER.exception("Failed to load default checkpoint from %s", self.default_checkpoint_path)

    def _build_transform(self, input_size: int) -> transforms.Compose:
        """Builds the torchvision preprocessing transform."""

        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size), antialias=True),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
            ]
        )

    def _resolve_checkpoint_config(
        self,
        checkpoint: Dict[str, object],
        state_dict: Dict[str, Tensor],
    ) -> Dict[str, object]:
        """Resolves model configuration from the checkpoint payload."""

        config = dict(checkpoint.get("config", {}))

        if "input_channels" not in config:
            conv_key = "backbone.conv1.weight"
            config["input_channels"] = int(state_dict[conv_key].shape[1]) if conv_key in state_dict else 1

        if "num_classes" not in config:
            if "classifier.1.weight" in state_dict:
                config["num_classes"] = int(state_dict["classifier.1.weight"].shape[0])
            elif "classifier.weight" in state_dict:
                config["num_classes"] = int(state_dict["classifier.weight"].shape[0])
            else:
                raise ValueError("Unable to infer num_classes from checkpoint state_dict.")

        if "input_size" not in config:
            config["input_size"] = DEFAULT_INPUT_SIZE

        config.setdefault("use_metadata", any(key.startswith("metadata_mlp.") for key in state_dict))
        config.setdefault("metadata_dim", 3)
        config.setdefault("metadata_hidden_dim", 64)
        config.setdefault("metadata_dropout", 0.1)
        config.setdefault("classifier_dropout", 0.0)

        return config

    def load_checkpoint(self, checkpoint_path: Path) -> LoadedModelBundle:
        """Loads a checkpoint from disk and activates it for inference."""

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        if checkpoint_path.suffix.lower() not in ALLOWED_CHECKPOINT_EXTENSIONS:
            raise ValueError(
                f"Unsupported checkpoint extension '{checkpoint_path.suffix}'. "
                f"Allowed extensions: {sorted(ALLOWED_CHECKPOINT_EXTENSIONS)}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            payload = checkpoint
        elif isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state_dict = checkpoint
            payload = {"state_dict": state_dict}
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected a dict with 'state_dict' or a raw state_dict payload."
            )

        config = self._resolve_checkpoint_config(payload, state_dict)
        class_names = list(payload.get("class_names", []))
        if not class_names:
            class_names = [f"class_{index}" for index in range(int(config["num_classes"]))]

        model = get_resnet50(
            input_channels=int(config["input_channels"]),
            pretrained=False,
            input_size=int(config["input_size"]),
            num_classes=int(config["num_classes"]),
            use_metadata=bool(config["use_metadata"]),
            metadata_dim=int(config["metadata_dim"]),
            metadata_hidden_dim=int(config["metadata_hidden_dim"]),
            metadata_dropout=float(config["metadata_dropout"]),
            classifier_dropout=float(config["classifier_dropout"]),
        )

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Checkpoint state_dict did not match the model architecture. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )

        model.to(self.device)
        model.eval()
        self._transform = self._build_transform(int(config["input_size"]))
        self.loaded_bundle = LoadedModelBundle(
            model=model,
            class_names=class_names,
            expected_input_size=int(config["input_size"]),
            metadata_enabled=bool(config["use_metadata"]),
            checkpoint_path=checkpoint_path,
        )
        return self.loaded_bundle

    def _build_metadata_tensor(self, age: Optional[int], gender: Optional[str]) -> Optional[Tensor]:
        """Builds the metadata tensor when the current model uses metadata."""

        if not self.metadata_enabled:
            return None

        metadata = _encode_metadata(age=age, gender=gender)
        return torch.from_numpy(metadata).unsqueeze(0).to(device=self.device)

    def _apply_preprocessing(
        self,
        image_array: np.ndarray,
        bone_flag: bool,
        crop_flag: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies optional preprocessing and returns preview images."""

        original_image = _normalize_to_uint8(image_array)
        bone_preview = _normalize_to_uint8(bone_suppression(original_image.copy()))
        crop_preview = _normalize_to_uint8(lung_crop(original_image.copy()))

        processed = original_image.copy()
        if bone_flag:
            processed = bone_suppression(processed)
        if crop_flag:
            processed = lung_crop(processed)
        processed = _normalize_to_uint8(processed)

        return original_image, processed, bone_preview, crop_preview

    def _tensor_from_image(self, image: np.ndarray) -> Tensor:
        """Converts a processed grayscale image into the model input tensor."""

        pil_image = Image.fromarray(image, mode="L")
        tensor = self._transform(pil_image).unsqueeze(0)
        return tensor.to(device=self.device, dtype=torch.float32)

    def _generate_gradcam(
        self,
        tensor: Tensor,
        base_image: np.ndarray,
        class_index: int,
        metadata_tensor: Optional[Tensor],
        alpha: float,
    ) -> Tuple[str, bytes]:
        """Generates a base64 Grad-CAM data URI for the selected class."""

        if self.loaded_bundle is None:
            raise RuntimeError("No checkpoint is currently loaded.")

        model = self.loaded_bundle.model
        if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4"):
            raise ValueError("The loaded model does not expose a ResNet-style layer4 block for Grad-CAM.")

        gradcam = LightweightGradCAM(model=model, target_module=model.backbone.layer4[-1].conv3)
        try:
            cam = gradcam.generate(tensor, class_index=class_index, metadata=metadata_tensor)
        finally:
            gradcam.close()

        cam_resized = cv2.resize(cam, (base_image.shape[1], base_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap_bgr = cv2.applyColorMap(np.uint8(cam_resized * 255.0), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        base_rgb = np.repeat(base_image[..., None], 3, axis=2).astype(np.float32) / 255.0
        overlay = np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0.0, 1.0)
        overlay_uint8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        gradcam_png = _image_to_png_bytes(overlay_uint8)
        return _png_data_uri(gradcam_png), gradcam_png

    def _predict_processed(
        self,
        source_name: str,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        bone_preview: np.ndarray,
        crop_preview: np.ndarray,
        age: Optional[int],
        gender: Optional[str],
        gradcam_alpha: float,
    ) -> PredictionResult:
        """Runs inference on a processed image and returns the prediction result."""

        if self.loaded_bundle is None:
            raise RuntimeError("No checkpoint is currently loaded. Upload a checkpoint or place one at the default path.")

        tensor = self._tensor_from_image(processed_image)
        metadata_tensor = self._build_metadata_tensor(age=age, gender=gender)

        with torch.inference_mode():
            logits = self.loaded_bundle.model(tensor, metadata=metadata_tensor)
            probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()

        scores = {
            class_name: float(score)
            for class_name, score in zip(self.loaded_bundle.class_names, probabilities.tolist())
        }
        predicted_labels = [name for name, score in scores.items() if score >= self.prediction_threshold]
        if not predicted_labels:
            predicted_labels = [max(scores, key=scores.get)]
        predicted = predicted_labels[0]
        target_index = self.loaded_bundle.class_names.index(predicted)
        gradcam_data_uri, gradcam_png_bytes = self._generate_gradcam(
            tensor=tensor,
            base_image=processed_image,
            class_index=target_index,
            metadata_tensor=metadata_tensor,
            alpha=gradcam_alpha,
        )

        metadata_payload = {
            "age": age,
            "gender": gender,
            "metadata_enabled": self.metadata_enabled,
        }
        return PredictionResult(
            checkpoint_path=str(self.loaded_bundle.checkpoint_path),
            source_name=source_name,
            input_size=self.loaded_bundle.expected_input_size,
            scores=scores,
            predicted=predicted,
            predicted_labels=predicted_labels,
            gradcam_data_uri=gradcam_data_uri,
            gradcam_png_bytes=gradcam_png_bytes,
            original_image=original_image,
            processed_image=processed_image,
            bone_preview=bone_preview,
            crop_preview=crop_preview,
            metadata=metadata_payload,
        )

    def predict_from_bytes(
        self,
        image_bytes: bytes,
        source_name: str,
        bone_flag: bool = False,
        crop_flag: bool = False,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        gradcam_alpha: float = DEFAULT_GRADCAM_ALPHA,
    ) -> PredictionResult:
        """Runs inference on an uploaded image payload."""

        image = _pil_from_bytes(image_bytes)
        image_array = np.asarray(image)
        original_image, processed_image, bone_preview, crop_preview = self._apply_preprocessing(
            image_array=image_array,
            bone_flag=bone_flag,
            crop_flag=crop_flag,
        )
        return self._predict_processed(
            source_name=source_name,
            original_image=original_image,
            processed_image=processed_image,
            bone_preview=bone_preview,
            crop_preview=crop_preview,
            age=age,
            gender=gender,
            gradcam_alpha=gradcam_alpha,
        )

    def _sample_registry(self) -> Dict[str, Path]:
        """Builds a whitelist of named sample images."""

        registry: Dict[str, Path] = {}
        for directory in self.sample_dirs:
            if not directory.exists():
                continue
            for path in directory.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in ALLOWED_SAMPLE_EXTENSIONS:
                    continue
                registry.setdefault(path.name, path.resolve())
        return registry

    def predict_from_sample(
        self,
        sample_name: str,
        bone_flag: bool = False,
        crop_flag: bool = False,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        gradcam_alpha: float = DEFAULT_GRADCAM_ALPHA,
    ) -> PredictionResult:
        """Runs inference on one of the whitelisted sample images."""

        registry = self._sample_registry()
        sample_path = registry.get(sample_name)
        if sample_path is None:
            raise FileNotFoundError(
                f"Unknown sample image '{sample_name}'. Available samples: {sorted(registry)}"
            )

        image = Image.open(sample_path).convert("L")
        image_array = np.asarray(image)
        original_image, processed_image, bone_preview, crop_preview = self._apply_preprocessing(
            image_array=image_array,
            bone_flag=bone_flag,
            crop_flag=crop_flag,
        )
        return self._predict_processed(
            source_name=sample_name,
            original_image=original_image,
            processed_image=processed_image,
            bone_preview=bone_preview,
            crop_preview=crop_preview,
            age=age,
            gender=gender,
            gradcam_alpha=gradcam_alpha,
        )

    def export_csv_bytes(self, result: PredictionResult) -> bytes:
        """Exports the prediction result as CSV bytes."""

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["source_name", "predicted", "class_name", "score", "age", "gender"])
        for class_name, score in result.scores.items():
            writer.writerow(
                [
                    result.source_name,
                    result.predicted,
                    class_name,
                    f"{score:.6f}",
                    result.metadata.get("age"),
                    result.metadata.get("gender"),
                ]
            )
        return buffer.getvalue().encode("utf-8")

    def export_pdf_bytes(self, result: PredictionResult) -> bytes:
        """Exports the prediction result as a simple PDF report."""

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        page_width, page_height = letter

        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(40, page_height - 40, "Chest X-ray Inference Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(40, page_height - 60, f"Source: {result.source_name}")
        pdf.drawString(40, page_height - 74, f"Predicted: {result.predicted}")
        pdf.drawString(40, page_height - 88, f"Checkpoint: {result.checkpoint_path}")
        pdf.drawString(
            40,
            page_height - 102,
            f"Metadata: age={result.metadata.get('age')}, gender={result.metadata.get('gender')}",
        )

        original_reader = ImageReader(io.BytesIO(_image_to_png_bytes(result.original_image)))
        gradcam_reader = ImageReader(io.BytesIO(result.gradcam_png_bytes))
        pdf.drawImage(original_reader, 40, page_height - 360, width=220, height=220, preserveAspectRatio=True)
        pdf.drawImage(gradcam_reader, 300, page_height - 360, width=220, height=220, preserveAspectRatio=True)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(40, page_height - 370, "Original")
        pdf.drawString(300, page_height - 370, "Grad-CAM")

        y_position = page_height - 400
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y_position, "Class Scores")
        pdf.setFont("Helvetica", 10)
        y_position -= 18
        for class_name, score in sorted(result.scores.items(), key=lambda item: item[1], reverse=True):
            pdf.drawString(50, y_position, f"{class_name}: {score:.4f}")
            y_position -= 14
            if y_position < 60:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y_position = page_height - 40

        pdf.save()
        return buffer.getvalue()
