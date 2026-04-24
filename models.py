"""Model definitions for chest X-ray classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


PathLike = Union[str, Path]


def _build_first_conv(original_conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    """Creates a ResNet stem convolution that supports non-RGB inputs."""

    if input_channels < 1:
        raise ValueError("input_channels must be >= 1.")

    new_conv = nn.Conv2d(
        in_channels=input_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        dilation=original_conv.dilation,
        groups=original_conv.groups,
        bias=original_conv.bias is not None,
        padding_mode=original_conv.padding_mode,
    )

    with torch.no_grad():
        if input_channels == original_conv.in_channels:
            new_conv.weight.copy_(original_conv.weight)
        else:
            mean_weight = original_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, input_channels, 1, 1))

        if original_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(original_conv.bias)

    return new_conv


class ResNet50WithMetadata(nn.Module):
    """ResNet50 image classifier with optional metadata fusion.

    The optional metadata tensor should have shape ``(batch_size, 3)`` and
    follow the feature order ``[age, gender, view]``. Typical encodings are:

    - age: normalized scalar such as ``age / 100``
    - gender: binary or one-hot reduced to a scalar
    - view: scalar code for AP/PA/Lateral or a normalized ordinal encoding
    """

    def __init__(
        self,
        input_channels: int = 1,
        pretrained: bool = True,
        input_size: int = 448,
        num_classes: int = 14,
        use_metadata: bool = False,
        metadata_dim: int = 3,
        metadata_hidden_dim: int = 64,
        metadata_dropout: float = 0.1,
        classifier_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.conv1 = _build_first_conv(backbone.conv1, input_channels)

        image_feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        self.metadata_dim = metadata_dim
        self.metadata_hidden_dim = metadata_hidden_dim
        self.metadata_dropout = metadata_dropout
        self.classifier_dropout = classifier_dropout
        self.image_feature_dim = image_feature_dim
        self.pretrained = pretrained

        if use_metadata:
            self.metadata_mlp = nn.Sequential(
                nn.Linear(metadata_dim, metadata_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=metadata_dropout),
                nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
                nn.ReLU(inplace=True),
            )
            classifier_input_dim = image_feature_dim + metadata_hidden_dim
        else:
            self.metadata_mlp = None
            classifier_input_dim = image_feature_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(classifier_input_dim, num_classes),
        )

    def _prepare_metadata(self, metadata: Optional[Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Returns a metadata tensor aligned with the image batch."""

        if metadata is None:
            metadata = torch.zeros(batch_size, self.metadata_dim, device=device)
        else:
            metadata = metadata.to(device=device, dtype=torch.float32)

        if metadata.ndim != 2 or metadata.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Expected metadata shape (batch_size, {self.metadata_dim}), got {tuple(metadata.shape)}"
            )

        if metadata.shape[0] != batch_size:
            raise ValueError(
                f"Metadata batch size {metadata.shape[0]} does not match image batch size {batch_size}."
            )

        return metadata

    def forward_features(self, images: Tensor, metadata: Optional[Tensor] = None) -> Tensor:
        """Extracts image features and optionally concatenates metadata features."""

        image_features = self.backbone(images)

        if not self.use_metadata:
            return image_features

        metadata_features = self.metadata_mlp(
            self._prepare_metadata(metadata, batch_size=images.shape[0], device=images.device)
        )
        return torch.cat([image_features, metadata_features], dim=1)

    def forward(self, images: Tensor, metadata: Optional[Tensor] = None) -> Tensor:
        """Runs the model and returns classification logits."""

        features = self.forward_features(images, metadata=metadata)
        return self.classifier(features)

    def get_config(self) -> Dict[str, Any]:
        """Returns the constructor arguments needed to recreate this model."""

        return {
            "input_channels": self.input_channels,
            "pretrained": self.pretrained,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "use_metadata": self.use_metadata,
            "metadata_dim": self.metadata_dim,
            "metadata_hidden_dim": self.metadata_hidden_dim,
            "metadata_dropout": self.metadata_dropout,
            "classifier_dropout": self.classifier_dropout,
        }

    def save_model(self, path: PathLike) -> None:
        """Saves model weights together with the model configuration."""

        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_model(
        cls,
        path: PathLike,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "ResNet50WithMetadata":
        """Loads a saved model checkpoint."""

        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            model = cls(**config)
            model.load_state_dict(checkpoint["state_dict"])
            return model

        raise ValueError("Checkpoint does not contain a saved model config and state_dict.")


def get_resnet50(
    input_channels: int = 1,
    pretrained: bool = True,
    input_size: int = 448,
    num_classes: int = 14,
    use_metadata: bool = False,
    metadata_dim: int = 3,
    metadata_hidden_dim: int = 64,
    metadata_dropout: float = 0.1,
    classifier_dropout: float = 0.0,
) -> ResNet50WithMetadata:
    """Builds a ResNet50 model for chest X-ray classification.

    Args:
        input_channels: Number of input image channels. Defaults to ``1``.
        pretrained: Whether to initialize the backbone with ImageNet weights.
        input_size: Expected input image size. Stored for downstream use.
        num_classes: Number of output classes.
        use_metadata: Whether to fuse ``age/gender/view`` metadata features.
        metadata_dim: Number of metadata input features. Defaults to ``3``.
        metadata_hidden_dim: Hidden size of the metadata MLP branch.
        metadata_dropout: Dropout used inside the metadata MLP.
        classifier_dropout: Dropout used before the final classifier.

    Returns:
        A ``ResNet50WithMetadata`` instance.
    """

    return ResNet50WithMetadata(
        input_channels=input_channels,
        pretrained=pretrained,
        input_size=input_size,
        num_classes=num_classes,
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
        metadata_hidden_dim=metadata_hidden_dim,
        metadata_dropout=metadata_dropout,
        classifier_dropout=classifier_dropout,
    )


def save_model(model: ResNet50WithMetadata, path: PathLike) -> None:
    """Helper wrapper for ``model.save_model(path)``."""

    model.save_model(path)


def load_model(
    path: PathLike,
    map_location: Optional[Union[str, torch.device]] = None,
) -> ResNet50WithMetadata:
    """Loads a saved ``ResNet50WithMetadata`` model from disk."""

    return ResNet50WithMetadata.load_model(path, map_location=map_location)
