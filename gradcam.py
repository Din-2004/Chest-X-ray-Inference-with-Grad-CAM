"""Generate Grad-CAM overlays for chest X-ray checkpoints."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
from torch import Tensor, nn

from data_loader import bone_suppression, load_grayscale_image, lung_crop
from models import get_resnet50


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path saved by train.py.")
    parser.add_argument("--images", nargs="+", required=True, help="List of image paths to visualize.")
    parser.add_argument("--output_dir", type=Path, default=Path("gradcam_outputs"), help="Directory for PNG outputs.")
    parser.add_argument("--class_index", type=int, default=None, help="Optional class index to visualize.")
    parser.add_argument("--input_size", type=int, default=None, help="Optional input size override.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity for the heatmap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_bone", action="store_true", help="Force-enable the bone_suppression stub.")
    parser.add_argument("--use_crop", action="store_true", help="Force-enable the lung_crop stub.")
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
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
    """Normalizes an arbitrary grayscale image to the uint8 range."""

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def load_checkpoint_bundle(
    checkpoint_path: Path,
    map_location: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Loads a checkpoint saved by train.py."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, object],
    device: torch.device,
) -> Tuple[nn.Module, List[str], Dict[str, object]]:
    """Rebuilds a model from checkpoint config without re-downloading pretrained weights."""

    config = dict(checkpoint.get("config", {}))
    class_names = list(checkpoint.get("class_names", []))
    num_classes = int(config.get("num_classes", len(class_names)))

    if num_classes < 1:
        raise ValueError("Checkpoint config must specify a positive num_classes value.")

    if not class_names:
        class_names = [f"class_{index}" for index in range(num_classes)]

    model = get_resnet50(
        input_channels=int(config.get("input_channels", 1)),
        pretrained=False,
        input_size=int(config.get("input_size", 448)),
        num_classes=num_classes,
        use_metadata=bool(config.get("use_metadata", False)),
        metadata_dim=int(config.get("metadata_dim", 3)),
        metadata_hidden_dim=int(config.get("metadata_hidden_dim", 64)),
        metadata_dropout=float(config.get("metadata_dropout", 0.1)),
        classifier_dropout=float(config.get("classifier_dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, config


def preprocess_image(
    image_path: Path,
    input_size: int,
    apply_bone_suppression: bool,
    apply_lung_crop: bool,
) -> Tuple[np.ndarray, Tensor]:
    """Loads an image and prepares both display and model input views."""

    image = load_grayscale_image(image_path)
    if apply_bone_suppression:
        image = bone_suppression(image)
    if apply_lung_crop:
        image = lung_crop(image)

    image = normalize_to_uint8(image)
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    return image, input_tensor


def safe_slug(value: str) -> str:
    """Converts a label into a filesystem-friendly slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return slug or "class"


class GradCAMGenerator:
    """Small Grad-CAM helper for ResNet-based image classifiers."""

    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.target_module = target_module
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        """Stores activations from the target convolutional block."""

        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[Tensor], ...],
        grad_output: Tuple[Optional[Tensor], ...],
    ) -> None:
        """Stores gradients from the target convolutional block."""

        if grad_output[0] is None:
            raise RuntimeError("Grad-CAM backward hook did not receive output gradients.")
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: Tensor,
        class_index: Optional[int] = None,
        metadata: Optional[Tensor] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generates a normalized Grad-CAM heatmap for one image."""

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor, metadata=metadata)
        probabilities = torch.sigmoid(logits)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape (1, num_classes), got {tuple(logits.shape)}")

        if class_index is None:
            class_index = int(torch.argmax(logits[0]).item())

        target_score = logits[:, class_index].sum()
        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)

        return cam.cpu().numpy(), class_index, float(probabilities[0, class_index].item())

    def close(self) -> None:
        """Removes registered hooks."""

        self.forward_handle.remove()
        self.backward_handle.remove()


def build_overlay(original_image: np.ndarray, cam: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Builds heatmap and overlay RGB images."""

    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(np.uint8(cam_resized * 255.0), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    base_rgb = np.repeat(original_image[..., None], 3, axis=2).astype(np.float32) / 255.0
    overlay = np.clip((1.0 - alpha) * base_rgb + alpha * heatmap_rgb, 0.0, 1.0)
    return heatmap_rgb, overlay


def save_gradcam_figure(
    original_image: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    class_name: str,
    probability: float,
    output_path: Path,
) -> None:
    """Saves a side-by-side Grad-CAM visualization."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Overlay\n{class_name} ({probability:.3f})")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Loads a checkpoint and writes Grad-CAM PNGs for the requested images."""

    args = parse_args()
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint_bundle(args.checkpoint, map_location=device)
    model, class_names, config = build_model_from_checkpoint(checkpoint, device=device)

    saved_args = dict(checkpoint.get("args", {}))
    input_size = int(args.input_size if args.input_size is not None else config.get("input_size", 448))
    use_bone = bool(saved_args.get("use_bone", False)) or args.use_bone
    use_crop = bool(saved_args.get("use_crop", False)) or args.use_crop

    if not hasattr(model, "backbone") or not hasattr(model.backbone, "layer4"):
        raise ValueError("Grad-CAM expects a ResNet-style backbone with layer4.")

    target_module = model.backbone.layer4[-1].conv3
    gradcam = GradCAMGenerator(model=model, target_module=target_module)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"output_dir={args.output_dir}")

    try:
        for index, image_arg in enumerate(args.images, start=1):
            image_path = Path(image_arg)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            print(f"gradcam image {index}/{len(args.images)}: {image_path}")
            original_image, input_tensor = preprocess_image(
                image_path=image_path,
                input_size=input_size,
                apply_bone_suppression=use_bone,
                apply_lung_crop=use_crop,
            )
            input_tensor = input_tensor.to(device)

            cam, class_index, probability = gradcam.generate(input_tensor=input_tensor, class_index=args.class_index)
            if class_index < 0 or class_index >= len(class_names):
                raise IndexError(f"class_index {class_index} is out of range for {len(class_names)} classes.")

            class_name = class_names[class_index]
            heatmap_rgb, overlay_rgb = build_overlay(original_image=original_image, cam=cam, alpha=args.alpha)

            output_name = f"{image_path.stem}_gradcam_{safe_slug(class_name)}.png"
            output_path = args.output_dir / output_name
            save_gradcam_figure(
                original_image=original_image,
                heatmap_rgb=heatmap_rgb,
                overlay_rgb=overlay_rgb,
                class_name=class_name,
                probability=probability,
                output_path=output_path,
            )
            print(f"saved={output_path}")
    finally:
        gradcam.close()


if __name__ == "__main__":
    main()
