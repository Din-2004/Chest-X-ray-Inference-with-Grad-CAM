"""Data loading utilities for chest X-ray deep learning projects.

This module provides a flexible PyTorch dataset and dataloader pipeline that:

- reads ChestX-ray14 and/or OpenI image folders with CSV label files,
- supports DICOM and common image formats such as PNG/JPG,
- converts every sample into a single-channel tensor,
- exposes optional preprocessing hooks for bone suppression and lung cropping,
- applies albumentations-based training augmentations, and
- creates a deterministic validation split.

The CSV parsing logic is intentionally tolerant of common schema variations.
It can read label information either from a text label column such as
``Finding Labels`` or from one-hot encoded label columns.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


SUPPORTED_IMAGE_SUFFIXES = {
    ".dcm",
    ".dicom",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

PATH_COLUMN_CANDIDATES = (
    "path",
    "image_path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "image",
    "image_id",
    "dicom_id",
    "study_id",
    "uid",
    "Image Index",
)

TEXT_LABEL_COLUMN_CANDIDATES = (
    "Finding Labels",
    "finding_labels",
    "labels",
    "label",
    "findings",
    "finding",
    "disease",
    "diseases",
    "target",
    "targets",
)

AGE_COLUMN_CANDIDATES = (
    "Age",
    "age",
    "patient_age",
    "Patient Age",
)

GENDER_COLUMN_CANDIDATES = (
    "Gender",
    "gender",
    "sex",
    "Sex",
    "patient_gender",
    "Patient Gender",
    "Patient Sex",
)

VIEW_COLUMN_CANDIDATES = (
    "View Position",
    "view_position",
    "view",
    "View",
    "projection",
    "Projection",
    "viewposition",
)

METADATA_COLUMNS = {
    "patientid",
    "patient_id",
    "patient",
    "subject_id",
    "study_id",
    "series_id",
    "uid",
    "image_id",
    "image index",
    "path",
    "image_path",
    "filepath",
    "file_path",
    "filename",
    "file_name",
    "view",
    "viewposition",
    "projection",
    "split",
    "age",
    "patient age",
    "patient_age",
    "gender",
    "patient gender",
    "patient_gender",
    "sex",
    "patient sex",
    "patient_sex",
}


@dataclass(frozen=True)
class SampleRecord:
    """Represents one image sample before it is converted into tensors."""

    image_path: Path
    labels: Tuple[str, ...]
    source: str
    sample_id: str
    metadata: Optional[Tuple[float, float, float]] = None


def bone_suppression(image: np.ndarray) -> np.ndarray:
    """Placeholder for a bone suppression algorithm.

    Replace this stub with a domain-specific model or image-processing pipeline
    if you want to suppress rib and clavicle structures before training.

    Args:
        image: Grayscale image array with shape ``(H, W)``.

    Returns:
        The processed grayscale image. The default implementation returns the
        input unchanged.
    """

    return image


def lung_crop(image: np.ndarray) -> np.ndarray:
    """Placeholder for a lung field cropping algorithm.

    Replace this stub with a segmentation-based crop or a heuristic detector if
    you want to focus the model on the lung region.

    Args:
        image: Grayscale image array with shape ``(H, W)``.

    Returns:
        The cropped grayscale image. The default implementation returns the
        input unchanged.
    """

    return image


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalizes an arbitrary grayscale image to the ``uint8`` range."""

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_value) / (max_value - min_value)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _load_dicom(path: Path) -> np.ndarray:
    """Loads a DICOM image and returns a normalized grayscale array."""

    dicom = pydicom.dcmread(str(path), force=True)
    image = dicom.pixel_array.astype(np.float32)
    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if str(getattr(dicom, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() - image

    return _normalize_to_uint8(image)


def _load_raster(path: Path) -> np.ndarray:
    """Loads a PNG/JPG-style image and returns a normalized grayscale array."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return _normalize_to_uint8(image)


def load_grayscale_image(path: Path) -> np.ndarray:
    """Loads a DICOM or raster image as a single grayscale array."""

    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        return _load_dicom(path)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return _load_raster(path)
    raise ValueError(f"Unsupported image format: {path}")


def _seed_worker(worker_id: int) -> None:
    """Seeds DataLoader workers for reproducible augmentations."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _default_train_transform(image_size: int) -> A.Compose:
    """Builds the training-time albumentations pipeline."""

    return A.Compose(
        [
            A.Rotate(limit=7, border_mode=cv2.BORDER_REPLICATE, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
        ]
    )


def _default_eval_transform(image_size: int) -> A.Compose:
    """Builds the validation-time preprocessing pipeline."""

    return A.Compose([A.Resize(height=image_size, width=image_size)])


def _tokenize_labels(value: object) -> List[str]:
    """Splits a text label cell into normalized class names."""

    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    labels = [token.strip() for token in re.split(r"[|;,]", text) if token.strip()]
    return labels


def _infer_binary_label_columns(frame: pd.DataFrame) -> List[str]:
    """Infers one-hot encoded label columns from a CSV file."""

    binary_columns: List[str] = []
    for column in frame.columns:
        normalized = column.strip().lower()
        if normalized in METADATA_COLUMNS:
            continue

        series = frame[column].dropna()
        if series.empty:
            continue

        if pd.api.types.is_bool_dtype(series):
            binary_columns.append(column)
            continue

        if not pd.api.types.is_numeric_dtype(series):
            continue

        unique_values = {float(value) for value in pd.unique(series)}
        if unique_values.issubset({0.0, 1.0}):
            binary_columns.append(column)

    return binary_columns


def _find_first_present_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    """Returns the first candidate column present in a DataFrame."""

    normalized_map = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        match = normalized_map.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def _build_file_index(root: Path) -> Dict[str, Path]:
    """Builds a lookup table for image paths under ``root``."""

    index: Dict[str, Path] = {}
    if not root.exists():
        return index

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        index.setdefault(path.name.lower(), path)
        index.setdefault(path.stem.lower(), path)

    return index


def _resolve_image_path(
    root: Path,
    value: object,
    file_index: Optional[Dict[str, Path]] = None,
) -> Tuple[Optional[Path], Optional[Dict[str, Path]]]:
    """Resolves an image path from a CSV cell value."""

    if pd.isna(value):
        return None, file_index

    raw_value = str(value).strip()
    if not raw_value:
        return None, file_index

    candidate = Path(raw_value)
    direct_candidates = []

    if candidate.is_absolute():
        direct_candidates.append(candidate)
    else:
        direct_candidates.append(root / candidate)
        direct_candidates.append(root / candidate.name)

    for path in direct_candidates:
        if path.exists() and path.is_file():
            return path.resolve(), file_index

    if file_index is None:
        file_index = _build_file_index(root)

    lookup_keys = [candidate.name.lower(), candidate.stem.lower(), raw_value.lower()]
    for key in lookup_keys:
        resolved = file_index.get(key)
        if resolved is not None:
            return resolved.resolve(), file_index

    return None, file_index


def _extract_labels(
    row: pd.Series,
    text_label_column: Optional[str],
    binary_label_columns: Sequence[str],
) -> List[str]:
    """Extracts label names from either text or one-hot encoded columns."""

    if text_label_column is not None:
        labels = _tokenize_labels(row[text_label_column])
        if labels:
            return labels

    labels: List[str] = []
    for column in binary_label_columns:
        value = row[column]
        if pd.isna(value):
            continue
        if bool(float(value)):
            labels.append(column)
    return labels


def _parse_age(value: object) -> float:
    """Normalizes patient age into a 0-1 float using ``age / 100``."""

    if pd.isna(value):
        return 0.0

    text = str(value).strip()
    if not text:
        return 0.0

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return 0.0

    age = float(match.group(1))
    return float(np.clip(age / 100.0, 0.0, 1.2))


def _parse_gender(value: object) -> float:
    """Encodes gender into a single float feature."""

    if pd.isna(value):
        return 0.5

    text = str(value).strip().lower()
    if text in {"m", "male", "1"}:
        return 1.0
    if text in {"f", "female", "0"}:
        return 0.0
    return 0.5


def _parse_view(value: object) -> float:
    """Encodes radiographic view into a single float feature."""

    if pd.isna(value):
        return 0.25

    text = str(value).strip().lower()
    if text == "ap":
        return 0.0
    if text == "pa":
        return 0.5
    if text in {"lateral", "lat", "ll", "rl"}:
        return 1.0
    return 0.25


def _extract_metadata(
    row: pd.Series,
    age_column: Optional[str],
    gender_column: Optional[str],
    view_column: Optional[str],
) -> Optional[Tuple[float, float, float]]:
    """Extracts optional metadata features as ``(age, gender, view)``."""

    if age_column is None and gender_column is None and view_column is None:
        return None

    age_value = _parse_age(row[age_column]) if age_column is not None else 0.0
    gender_value = _parse_gender(row[gender_column]) if gender_column is not None else 0.5
    view_value = _parse_view(row[view_column]) if view_column is not None else 0.25
    return (age_value, gender_value, view_value)


def _build_records_from_csv(root: Path, csv_path: Path, source_name: str) -> List[SampleRecord]:
    """Builds sample records from a dataset root folder and a CSV label file."""

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV label file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    path_column = _find_first_present_column(frame.columns, PATH_COLUMN_CANDIDATES)
    if path_column is None:
        raise ValueError(
            f"Could not infer an image path column in {csv_path}. "
            f"Tried: {', '.join(PATH_COLUMN_CANDIDATES)}"
        )

    text_label_column = _find_first_present_column(frame.columns, TEXT_LABEL_COLUMN_CANDIDATES)
    age_column = _find_first_present_column(frame.columns, AGE_COLUMN_CANDIDATES)
    gender_column = _find_first_present_column(frame.columns, GENDER_COLUMN_CANDIDATES)
    view_column = _find_first_present_column(frame.columns, VIEW_COLUMN_CANDIDATES)
    binary_label_columns = _infer_binary_label_columns(frame)
    if text_label_column is not None and text_label_column in binary_label_columns:
        binary_label_columns = [column for column in binary_label_columns if column != text_label_column]

    if text_label_column is None and not binary_label_columns:
        raise ValueError(
            f"Could not infer labels from {csv_path}. "
            "Expected a text label column or one-hot encoded label columns."
        )

    records: List[SampleRecord] = []
    file_index: Optional[Dict[str, Path]] = None

    for _, row in frame.iterrows():
        image_path, file_index = _resolve_image_path(root, row[path_column], file_index)
        if image_path is None:
            continue

        labels = tuple(_extract_labels(row, text_label_column, binary_label_columns))
        metadata = _extract_metadata(row, age_column, gender_column, view_column)
        sample_id = image_path.stem
        records.append(
            SampleRecord(
                image_path=image_path,
                labels=labels,
                source=source_name,
                sample_id=sample_id,
                metadata=metadata,
            )
        )

    if not records:
        raise ValueError(
            f"No images from {csv_path} could be resolved under dataset root {root}."
        )

    return records


class ChestXrayDataset(Dataset):
    """PyTorch dataset for combined ChestX-ray14/OpenI style data.

    Args:
        records: Sample metadata entries.
        class_names: Ordered label vocabulary used to build multi-hot targets.
        transform: Albumentations pipeline applied after optional preprocessing.
        apply_bone_suppression: Whether to call :func:`bone_suppression`.
        apply_lung_crop: Whether to call :func:`lung_crop`.

    Returns:
        A mapping with keys ``image``, ``label``, ``path``, ``source``, and
        ``sample_id``. When age/gender/view columns are available, the sample
        also includes a ``metadata`` tensor ordered as ``[age, gender, view]``.
        The ``image`` tensor has shape ``(1, H, W)`` and dtype
        ``torch.float32``. The ``label`` tensor is a multi-hot vector ordered
        by ``class_names``.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        class_names: Sequence[str],
        transform: Optional[A.Compose] = None,
        apply_bone_suppression: bool = False,
        apply_lung_crop: bool = False,
    ) -> None:
        self.records = list(records)
        self.class_names = list(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.transform = transform
        self.apply_bone_suppression = apply_bone_suppression
        self.apply_lung_crop = apply_lung_crop
        self.has_metadata = any(record.metadata is not None for record in self.records)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""

        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Loads one sample and converts it to tensors."""

        record = self.records[index]
        image = load_grayscale_image(record.image_path)

        if self.apply_bone_suppression:
            image = bone_suppression(image)
        if self.apply_lung_crop:
            image = lung_crop(image)
        image = _normalize_to_uint8(image)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError(
                f"Expected a single-channel image after preprocessing, got shape {image.shape}"
            )

        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        label_tensor = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in record.labels:
            class_index = self.class_to_index.get(label)
            if class_index is not None:
                label_tensor[class_index] = 1.0

        sample = {
            "image": image_tensor,
            "label": label_tensor,
            "path": str(record.image_path),
            "source": record.source,
            "sample_id": record.sample_id,
        }
        if self.has_metadata:
            metadata_values = record.metadata if record.metadata is not None else (0.0, 0.5, 0.25)
            sample["metadata"] = torch.tensor(metadata_values, dtype=torch.float32)

        return sample


def build_datasets(
    chestxray14_root: Optional[Path] = None,
    chestxray14_csv: Optional[Path] = None,
    openi_root: Optional[Path] = None,
    openi_csv: Optional[Path] = None,
    image_size: int = 224,
    val_fraction: float = 0.2,
    seed: int = 42,
    apply_bone_suppression: bool = False,
    apply_lung_crop: bool = False,
    train_transform: Optional[A.Compose] = None,
    val_transform: Optional[A.Compose] = None,
) -> Tuple[ChestXrayDataset, ChestXrayDataset, List[str]]:
    """Builds train/validation datasets and a shared class vocabulary.

    Args:
        chestxray14_root: Root folder containing ChestX-ray14 images.
        chestxray14_csv: CSV file containing ChestX-ray14 labels.
        openi_root: Root folder containing OpenI images.
        openi_csv: CSV file containing OpenI labels.
        image_size: Output image size used by the default transforms.
        val_fraction: Fraction of samples reserved for validation.
        seed: Random seed used for the deterministic split.
        apply_bone_suppression: Whether to run the stub hook before transforms.
        apply_lung_crop: Whether to run the stub hook before transforms.
        train_transform: Optional custom training transform.
        val_transform: Optional custom validation transform.

    Returns:
        A tuple ``(train_dataset, val_dataset, class_names)``.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    if (chestxray14_root is None) != (chestxray14_csv is None):
        raise ValueError("Provide both chestxray14_root and chestxray14_csv together.")
    if (openi_root is None) != (openi_csv is None):
        raise ValueError("Provide both openi_root and openi_csv together.")

    sources: List[Tuple[Path, Path, str]] = []
    if chestxray14_root is not None and chestxray14_csv is not None:
        sources.append((Path(chestxray14_root), Path(chestxray14_csv), "ChestX-ray14"))
    if openi_root is not None and openi_csv is not None:
        sources.append((Path(openi_root), Path(openi_csv), "OpenI"))

    if not sources:
        raise ValueError("Provide at least one dataset root and CSV pair.")

    records: List[SampleRecord] = []
    for root, csv_path, source_name in sources:
        records.extend(_build_records_from_csv(root, csv_path, source_name))

    if len(records) < 2:
        raise ValueError("At least two resolved samples are required to create a validation split.")

    class_names = sorted({label for record in records for label in record.labels})
    if not class_names:
        raise ValueError("No labels were found after parsing the provided CSV files.")

    indices = np.arange(len(records))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_count = int(len(records) * val_fraction)
    val_count = min(max(val_count, 1), len(records) - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_records = [records[index] for index in train_indices]
    val_records = [records[index] for index in val_indices]

    train_dataset = ChestXrayDataset(
        records=train_records,
        class_names=class_names,
        transform=train_transform or _default_train_transform(image_size),
        apply_bone_suppression=apply_bone_suppression,
        apply_lung_crop=apply_lung_crop,
    )
    val_dataset = ChestXrayDataset(
        records=val_records,
        class_names=class_names,
        transform=val_transform or _default_eval_transform(image_size),
        apply_bone_suppression=apply_bone_suppression,
        apply_lung_crop=apply_lung_crop,
    )

    return train_dataset, val_dataset, class_names


def build_dataloaders(
    chestxray14_root: Optional[Path] = None,
    chestxray14_csv: Optional[Path] = None,
    openi_root: Optional[Path] = None,
    openi_csv: Optional[Path] = None,
    image_size: int = 224,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    apply_bone_suppression: bool = False,
    apply_lung_crop: bool = False,
    train_transform: Optional[A.Compose] = None,
    val_transform: Optional[A.Compose] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Builds deterministic train/validation dataloaders.

    Returns:
        A tuple ``(train_loader, val_loader, class_names)``.
    """

    train_dataset, val_dataset, class_names = build_datasets(
        chestxray14_root=chestxray14_root,
        chestxray14_csv=chestxray14_csv,
        openi_root=openi_root,
        openi_csv=openi_csv,
        image_size=image_size,
        val_fraction=val_fraction,
        seed=seed,
        apply_bone_suppression=apply_bone_suppression,
        apply_lung_crop=apply_lung_crop,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader, class_names


def _build_arg_parser() -> argparse.ArgumentParser:
    """Creates the CLI parser used by the smoke-test entry point."""

    parser = argparse.ArgumentParser(
        description="Smoke-test the Chest X-ray dataset and dataloader pipeline."
    )
    parser.add_argument("--chestxray14-root", type=Path, help="ChestX-ray14 image root folder.")
    parser.add_argument("--chestxray14-csv", type=Path, help="ChestX-ray14 label CSV path.")
    parser.add_argument("--openi-root", type=Path, help="OpenI image root folder.")
    parser.add_argument("--openi-csv", type=Path, help="OpenI label CSV path.")
    parser.add_argument("--image-size", type=int, default=224, help="Output image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the loader.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--bone-suppression",
        action="store_true",
        help="Run the bone_suppression stub before augmentation.",
    )
    parser.add_argument(
        "--lung-crop",
        action="store_true",
        help="Run the lung_crop stub before augmentation.",
    )
    return parser


def _print_dataset_summary(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: Sequence[str],
) -> None:
    """Prints a short summary for CLI smoke testing."""

    print(f"Classes ({len(class_names)}): {list(class_names)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    batch = next(iter(train_loader))
    image_batch: Tensor = batch["image"]
    label_batch: Tensor = batch["label"]
    print(f"Train batch image shape: {tuple(image_batch.shape)}")
    print(f"Train batch label shape: {tuple(label_batch.shape)}")
    print(f"First sample source: {batch['source'][0]}")
    print(f"First sample path: {batch['path'][0]}")


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not any(
        [
            args.chestxray14_root,
            args.chestxray14_csv,
            args.openi_root,
            args.openi_csv,
        ]
    ):
        parser.print_help()
        raise SystemExit(0)

    train_loader, val_loader, class_names = build_dataloaders(
        chestxray14_root=args.chestxray14_root,
        chestxray14_csv=args.chestxray14_csv,
        openi_root=args.openi_root,
        openi_csv=args.openi_csv,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        apply_bone_suppression=args.bone_suppression,
        apply_lung_crop=args.lung_crop,
    )
    _print_dataset_summary(train_loader, val_loader, class_names)
