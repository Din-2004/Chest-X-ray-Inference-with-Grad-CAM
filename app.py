"""Streamlit app for chest X-ray multilabel inference and Grad-CAM."""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pydicom
import streamlit as st
import torch
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor, nn

from data_loader import bone_suppression, lung_crop
from gradcam import GradCAMGenerator, build_model_from_checkpoint, build_overlay


WORKSPACE_ROOT = Path(__file__).resolve().parent
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_SUFFIXES = {".dcm", ".dicom"}
UPLOAD_SUFFIXES = tuple(sorted(suffix.lstrip(".") for suffix in RASTER_SUFFIXES | DICOM_SUFFIXES))
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def set_deterministic(seed: int = 42) -> None:
    """Configures deterministic behavior across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalizes an image into the ``uint8`` display range."""

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def load_dicom_from_bytes(dicom_bytes: bytes) -> np.ndarray:
    """Loads a DICOM image from bytes as a normalized grayscale array."""

    dicom = pydicom.dcmread(io.BytesIO(dicom_bytes), force=True)
    image = dicom.pixel_array.astype(np.float32)
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if str(getattr(dicom, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image

    return normalize_to_uint8(image)


def load_raster_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Loads a raster image from bytes as a normalized grayscale array."""

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return normalize_to_uint8(np.asarray(image))


def load_image_bytes(image_bytes: bytes, suffix: str) -> np.ndarray:
    """Loads DICOM or raster image bytes as a grayscale array."""

    suffix = suffix.lower()
    if suffix in DICOM_SUFFIXES:
        return load_dicom_from_bytes(image_bytes)
    if suffix in RASTER_SUFFIXES:
        return load_raster_from_bytes(image_bytes)
    raise ValueError(f"Unsupported image format: {suffix}")


def load_local_image(image_path: Path) -> np.ndarray:
    """Loads a local image path as a grayscale array."""

    return load_image_bytes(image_path.read_bytes(), image_path.suffix)


def encode_metadata(age: int, gender: str, view: str) -> np.ndarray:
    """Encodes UI metadata inputs into the model feature order ``[age, gender, view]``."""

    gender_map = {"Unknown": 0.5, "Female": 0.0, "Male": 1.0}
    view_map = {"Unknown": 0.25, "AP": 0.0, "PA": 0.5, "Lateral": 1.0}
    return np.array([age / 100.0, gender_map[gender], view_map[view]], dtype=np.float32)


def build_metadata_tensor(
    age: int,
    gender: str,
    view: str,
    device: torch.device,
) -> Tensor:
    """Builds a single-sample metadata tensor."""

    values = encode_metadata(age=age, gender=gender, view=view)
    return torch.from_numpy(values).unsqueeze(0).to(device=device)


def apply_preprocessing(
    image: np.ndarray,
    input_size: int,
    use_bone: bool,
    use_crop: bool,
) -> Dict[str, object]:
    """Applies preview and inference preprocessing variants."""

    original_image = normalize_to_uint8(image)
    bone_preview = normalize_to_uint8(bone_suppression(original_image.copy()))
    crop_preview = normalize_to_uint8(lung_crop(original_image.copy()))

    processed_image = original_image.copy()
    if use_bone:
        processed_image = bone_suppression(processed_image)
    if use_crop:
        processed_image = lung_crop(processed_image)
    processed_image = normalize_to_uint8(processed_image)

    resized = cv2.resize(processed_image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0

    return {
        "original": original_image,
        "bone": bone_preview,
        "crop": crop_preview,
        "processed": processed_image,
        "input_tensor": input_tensor,
    }


@st.cache_data(show_spinner=False)
def discover_local_checkpoints(workspace_root: str) -> List[str]:
    """Finds checkpoint files under the workspace."""

    root = Path(workspace_root)
    checkpoints = [
        str(path.resolve())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    ]
    return sorted(checkpoints)


@st.cache_data(show_spinner=False)
def discover_example_images(workspace_root: str) -> List[str]:
    """Finds example image files under common sample directories."""

    root = Path(workspace_root)
    candidates: List[Path] = []
    for folder in (root / "sample_images", root / "examples", root / "assets" / "media", root / "datasets"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES | DICOM_SUFFIXES:
                candidates.append(path.resolve())
    return [str(path) for path in sorted(dict.fromkeys(candidates))]


@st.cache_resource(show_spinner=False)
def load_model_resource(
    checkpoint_bytes: Optional[bytes],
    checkpoint_path: str,
) -> Tuple[nn.Module, List[str], Dict[str, object], torch.device]:
    """Loads a checkpoint and returns the restored model plus metadata."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_bytes is not None:
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'state_dict' entry.")

    model, class_names, config = build_model_from_checkpoint(checkpoint, device=device)
    return model, class_names, config, device


def resolve_checkpoint_selection(
    uploaded_checkpoint,
    selected_checkpoint_path: str,
    custom_checkpoint_path: str,
) -> Tuple[Optional[bytes], str, str]:
    """Resolves the active checkpoint source."""

    if uploaded_checkpoint is not None:
        return uploaded_checkpoint.getvalue(), "", uploaded_checkpoint.name

    if custom_checkpoint_path.strip():
        checkpoint_path = custom_checkpoint_path.strip()
        return None, checkpoint_path, Path(checkpoint_path).name

    if selected_checkpoint_path:
        return None, selected_checkpoint_path, Path(selected_checkpoint_path).name

    return None, "", ""


def checkpoint_signature(checkpoint_bytes: Optional[bytes], checkpoint_path: str) -> str:
    """Builds a stable signature for the currently selected checkpoint."""

    if checkpoint_bytes is not None:
        return hashlib.md5(checkpoint_bytes).hexdigest()
    return str(Path(checkpoint_path).resolve()) if checkpoint_path else ""


def run_inference(
    model: nn.Module,
    image_tensor: Tensor,
    device: torch.device,
    metadata_tensor: Optional[Tensor] = None,
) -> np.ndarray:
    """Runs model inference and returns sigmoid probabilities."""

    with torch.inference_mode():
        logits = model(image_tensor.to(device), metadata=metadata_tensor)
        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
    return probabilities.astype(np.float32)


def compute_gradcam_overlay(
    model: nn.Module,
    image_tensor: Tensor,
    processed_image: np.ndarray,
    class_index: int,
    alpha: float,
    device: torch.device,
    metadata_tensor: Optional[Tensor] = None,
) -> Tuple[np.ndarray, float]:
    """Computes a Grad-CAM overlay for a selected class."""

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4"):
        raise ValueError("Grad-CAM expects a ResNet-style backbone with layer4.")

    generator = GradCAMGenerator(model=model, target_module=model.backbone.layer4[-1].conv3)
    try:
        cam, _, probability = generator.generate(
            input_tensor=image_tensor.to(device),
            class_index=class_index,
            metadata=metadata_tensor,
        )
    finally:
        generator.close()

    _, overlay = build_overlay(original_image=processed_image, cam=cam, alpha=alpha)
    return overlay, probability


def build_prediction_table(class_names: Sequence[str], probabilities: np.ndarray) -> pd.DataFrame:
    """Builds a sorted prediction table."""

    frame = pd.DataFrame({"class_name": list(class_names), "score": probabilities.astype(float)})
    return frame.sort_values("score", ascending=False, ignore_index=True)


def build_export_dataframe(result: Dict[str, object]) -> pd.DataFrame:
    """Builds a CSV-friendly export table for one inference run."""

    class_names = result["class_names"]
    probabilities = result["probabilities"]
    rows = [
        {
            "source": result["source_label"],
            "class_name": class_name,
            "score": float(score),
            "bone_enabled": bool(result["settings"]["bone"]),
            "crop_enabled": bool(result["settings"]["crop"]),
            "age": int(result["metadata_inputs"]["age"]),
            "gender": str(result["metadata_inputs"]["gender"]),
            "view": str(result["metadata_inputs"]["view"]),
        }
        for class_name, score in zip(class_names, probabilities)
    ]
    return pd.DataFrame(rows)


def create_pdf_report(
    result: Dict[str, object],
    overlay_image: np.ndarray,
    selected_class_name: str,
    selected_probability: float,
) -> bytes:
    """Creates a simple PDF report for the current inference result."""

    buffer = io.BytesIO()
    prediction_table = build_prediction_table(result["class_names"], result["probabilities"]).head(8)

    with PdfPages(buffer) as pdf:
        overview = Figure(figsize=(12, 8))
        axes = overview.subplots(2, 2)
        axes = np.asarray(axes)
        axes[0, 0].imshow(result["original_image"], cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(result["bone_image"], cmap="gray")
        axes[0, 1].set_title("Bone-Suppressed Preview")
        axes[1, 0].imshow(result["crop_image"], cmap="gray")
        axes[1, 0].set_title("Lung-Cropped Preview")
        axes[1, 1].imshow(overlay_image)
        axes[1, 1].set_title(f"Grad-CAM: {selected_class_name} ({selected_probability:.3f})")
        for axis in axes.flat:
            axis.axis("off")
        overview.suptitle("Chest X-ray Inference Report")
        overview.tight_layout()
        pdf.savefig(overview)

        chart = Figure(figsize=(10, 6))
        axis = chart.subplots(1, 1)
        axis.barh(prediction_table["class_name"][::-1], prediction_table["score"][::-1], color="#3B82F6")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Confidence")
        axis.set_title("Top Predictions")
        chart.tight_layout()
        pdf.savefig(chart)

    buffer.seek(0)
    return buffer.getvalue()


def analyze_image(
    image_array: np.ndarray,
    source_label: str,
    model: nn.Module,
    class_names: Sequence[str],
    input_size: int,
    use_bone: bool,
    use_crop: bool,
    device: torch.device,
    metadata_tensor: Optional[Tensor],
    metadata_inputs: Dict[str, object],
) -> Dict[str, object]:
    """Preprocesses an image, runs inference, and returns a result dictionary."""

    processed = apply_preprocessing(
        image=image_array,
        input_size=input_size,
        use_bone=use_bone,
        use_crop=use_crop,
    )
    probabilities = run_inference(
        model=model,
        image_tensor=processed["input_tensor"],
        device=device,
        metadata_tensor=metadata_tensor,
    )
    top_class_index = int(np.argmax(probabilities))

    return {
        "source_label": source_label,
        "class_names": list(class_names),
        "probabilities": probabilities,
        "top_class_index": top_class_index,
        "original_image": processed["original"],
        "bone_image": processed["bone"],
        "crop_image": processed["crop"],
        "processed_image": processed["processed"],
        "input_tensor": processed["input_tensor"].cpu().numpy(),
        "settings": {"bone": use_bone, "crop": use_crop, "input_size": input_size},
        "metadata_inputs": metadata_inputs,
    }


def render_prediction_bars(prediction_table: pd.DataFrame, top_k: int = 6) -> None:
    """Renders confidence bars for the highest-scoring predictions."""

    st.markdown("#### Confidence Bars")
    for row in prediction_table.head(top_k).itertuples(index=False):
        left, right = st.columns([0.35, 0.65])
        left.write(f"{row.class_name}")
        right.progress(float(row.score), text=f"{row.score:.3f}")


def render_example_gallery(example_images: Sequence[str]) -> str:
    """Renders a thumbnail gallery and returns the selected example path."""

    if not example_images:
        st.info("No example images were found in the workspace.")
        return ""

    st.markdown("#### Example Gallery")
    preview_images = list(example_images[:6])
    selected_example = st.radio(
        "Choose a sample image",
        options=preview_images,
        horizontal=True,
        format_func=lambda value: Path(value).name,
    )

    columns = st.columns(min(3, len(preview_images)))
    for index, image_path in enumerate(preview_images):
        with columns[index % len(columns)]:
            try:
                preview = load_local_image(Path(image_path))
                st.image(preview, caption=Path(image_path).name, clamp=True)
            except Exception as exc:  # pragma: no cover - UI-only fallback
                st.warning(f"Preview unavailable for {Path(image_path).name}: {exc}")

    return selected_example


def main() -> None:
    """Renders the Streamlit inference app."""

    set_deterministic(42)
    st.set_page_config(page_title="Chest X-ray Inference", layout="wide")

    st.title("Chest X-ray Inference with Grad-CAM")
    st.write(
        "Upload a chest X-ray, choose optional preprocessing, inspect prediction scores, "
        "and export the result bundle as CSV or PDF."
    )

    local_checkpoints = discover_local_checkpoints(str(WORKSPACE_ROOT))
    example_images = discover_example_images(str(WORKSPACE_ROOT))

    with st.sidebar:
        st.header("Model")
        uploaded_checkpoint = st.file_uploader(
            "Upload checkpoint",
            type=[suffix.lstrip(".") for suffix in CHECKPOINT_SUFFIXES],
            accept_multiple_files=False,
        )
        selected_checkpoint_path = st.selectbox(
            "Local checkpoint",
            options=[""] + local_checkpoints,
            format_func=lambda value: "Choose local checkpoint" if not value else Path(value).name,
        )
        custom_checkpoint_path = st.text_input("Custom checkpoint path", value="")

        st.header("Preprocessing")
        use_bone = st.toggle("Enable bone suppression", value=False)
        use_crop = st.toggle("Enable lung field cropping", value=False)
        gradcam_alpha = st.slider("Grad-CAM intensity", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

        st.header("Patient Metadata")
        age_value = st.slider("Age", min_value=0, max_value=100, value=55)
        gender_value = st.selectbox("Gender", options=["Unknown", "Female", "Male"], index=0)
        view_value = st.selectbox("View", options=["Unknown", "AP", "PA", "Lateral"], index=0)

    checkpoint_bytes, checkpoint_path, checkpoint_label = resolve_checkpoint_selection(
        uploaded_checkpoint=uploaded_checkpoint,
        selected_checkpoint_path=selected_checkpoint_path,
        custom_checkpoint_path=custom_checkpoint_path,
    )
    active_checkpoint_signature = checkpoint_signature(checkpoint_bytes, checkpoint_path)
    checkpoint_ready = bool(checkpoint_bytes is not None or checkpoint_path)

    model: Optional[nn.Module] = None
    class_names: List[str] = []
    config: Dict[str, object] = {}
    device: Optional[torch.device] = None

    if checkpoint_ready:
        try:
            model, class_names, config, device = load_model_resource(checkpoint_bytes, checkpoint_path)
            st.sidebar.success(f"Loaded checkpoint: {checkpoint_label or Path(checkpoint_path).name}")
            st.sidebar.caption(f"Device: {device}")
        except Exception as exc:
            checkpoint_ready = False
            st.sidebar.error(f"Unable to load checkpoint: {exc}")
    else:
        st.sidebar.info("Upload or choose a checkpoint to enable inference.")

    input_size = int(config.get("input_size", 448)) if config else 448
    use_metadata = bool(config.get("use_metadata", False)) if config else False

    top_left, top_right = st.columns([1.3, 1.0])

    with top_left:
        st.subheader("Upload X-ray")
        uploaded_image = st.file_uploader(
            "Drag and drop a chest X-ray",
            type=UPLOAD_SUFFIXES,
            accept_multiple_files=False,
            key="uploaded_xray",
        )
        run_uploaded = st.button(
            "Run on uploaded X-ray",
            disabled=uploaded_image is None or not checkpoint_ready,
            use_container_width=True,
        )

        selected_example = render_example_gallery(example_images)
        run_sample = st.button(
            "Run on sample",
            disabled=(not checkpoint_ready) or (not selected_example),
            use_container_width=True,
        )

    with top_right:
        st.subheader("Session State")
        st.write(f"Checkpoint loaded: {'Yes' if checkpoint_ready else 'No'}")
        st.write(f"Detected classes: {len(class_names)}")
        st.write(f"Expected input size: {input_size}")
        st.write(f"Metadata branch enabled: {'Yes' if use_metadata else 'No'}")
        if local_checkpoints:
            st.caption("Discovered checkpoints")
            st.code("\n".join(local_checkpoints), language="text")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    metadata_inputs = {"age": age_value, "gender": gender_value, "view": view_value}
    metadata_tensor = None
    if device is not None and use_metadata:
        metadata_tensor = build_metadata_tensor(
            age=age_value,
            gender=gender_value,
            view=view_value,
            device=device,
        )

    if run_uploaded and uploaded_image is not None and model is not None and device is not None:
        try:
            image_array = load_image_bytes(uploaded_image.getvalue(), Path(uploaded_image.name).suffix)
            st.session_state.analysis_result = analyze_image(
                image_array=image_array,
                source_label=f"Uploaded file: {uploaded_image.name}",
                model=model,
                class_names=class_names,
                input_size=input_size,
                use_bone=use_bone,
                use_crop=use_crop,
                device=device,
                metadata_tensor=metadata_tensor,
                metadata_inputs=metadata_inputs,
            )
            st.session_state.analysis_result["checkpoint_signature"] = active_checkpoint_signature
        except Exception as exc:
            st.session_state.analysis_result = None
            st.error(f"Unable to process uploaded image: {exc}")

    if run_sample and selected_example and model is not None and device is not None:
        try:
            image_array = load_local_image(Path(selected_example))
            st.session_state.analysis_result = analyze_image(
                image_array=image_array,
                source_label=f"Sample image: {Path(selected_example).name}",
                model=model,
                class_names=class_names,
                input_size=input_size,
                use_bone=use_bone,
                use_crop=use_crop,
                device=device,
                metadata_tensor=metadata_tensor,
                metadata_inputs=metadata_inputs,
            )
            st.session_state.analysis_result["checkpoint_signature"] = active_checkpoint_signature
        except Exception as exc:
            st.session_state.analysis_result = None
            st.error(f"Unable to process sample image: {exc}")

    result = st.session_state.analysis_result
    if not result:
        st.info("Load a checkpoint, then run an uploaded image or a sample image to see predictions.")
        return

    if result.get("checkpoint_signature") != active_checkpoint_signature:
        st.warning("The checkpoint changed since the last inference. Run the image again to refresh the results.")
        return

    probabilities = np.asarray(result["probabilities"], dtype=np.float32)
    prediction_table = build_prediction_table(result["class_names"], probabilities)
    selected_class_index = st.selectbox(
        "Grad-CAM target class",
        options=list(range(len(result["class_names"]))),
        index=int(result["top_class_index"]),
        format_func=lambda idx: f"{result['class_names'][idx]} ({probabilities[idx]:.3f})",
    )

    if model is None or device is None:
        st.error("The model is not available in memory anymore. Reload the checkpoint.")
        return

    live_metadata_tensor = metadata_tensor if use_metadata else None
    input_tensor = torch.from_numpy(np.asarray(result["input_tensor"], dtype=np.float32))
    overlay_image, selected_probability = compute_gradcam_overlay(
        model=model,
        image_tensor=input_tensor,
        processed_image=np.asarray(result["processed_image"], dtype=np.uint8),
        class_index=selected_class_index,
        alpha=gradcam_alpha,
        device=device,
        metadata_tensor=live_metadata_tensor,
    )

    st.divider()
    st.subheader("Inference Results")
    st.caption(result["source_label"])
    st.caption(
        f"Preprocessing: bone={result['settings']['bone']}, crop={result['settings']['crop']} | "
        f"Metadata: age={result['metadata_inputs']['age']}, gender={result['metadata_inputs']['gender']}, "
        f"view={result['metadata_inputs']['view']}"
    )

    preview_columns = st.columns(4)
    with preview_columns[0]:
        st.image(result["original_image"], caption="Original", clamp=True)
    with preview_columns[1]:
        st.image(result["bone_image"], caption="Bone-Suppressed Preview", clamp=True)
    with preview_columns[2]:
        st.image(result["crop_image"], caption="Lung-Cropped Preview", clamp=True)
    with preview_columns[3]:
        st.image(
            overlay_image,
            caption=f"Grad-CAM: {result['class_names'][selected_class_index]} ({selected_probability:.3f})",
            clamp=True,
        )

    chart_col, table_col = st.columns([0.95, 1.05])
    with chart_col:
        render_prediction_bars(prediction_table)
        st.bar_chart(prediction_table.set_index("class_name")["score"], use_container_width=True)
    with table_col:
        st.markdown("#### Prediction Table")
        st.dataframe(prediction_table, hide_index=True, use_container_width=True)

    export_df = build_export_dataframe(result)
    pdf_bytes = create_pdf_report(
        result=result,
        overlay_image=overlay_image,
        selected_class_name=result["class_names"][selected_class_index],
        selected_probability=selected_probability,
    )

    export_left, export_right = st.columns(2)
    with export_left:
        st.download_button(
            label="Export results as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="xray_inference_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_right:
        st.download_button(
            label="Export results as PDF",
            data=pdf_bytes,
            file_name="xray_inference_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
