"""Training script for chest X-ray multi-label classification."""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_loader import SUPPORTED_IMAGE_SUFFIXES, build_dataloaders
from models import get_resnet50


DATASET_HINTS: Dict[str, Dict[str, Sequence[str]]] = {
    "chestxray14": {
        "dir_names": (
            "ChestX-ray14",
            "chestxray14",
            "chest_xray14",
            "nih",
            "nih_chestxray14",
        ),
        "csv_names": (
            "Data_Entry_2017.csv",
            "chestxray14.csv",
            "chestxray14_labels.csv",
            "chest_xray14.csv",
            "labels.csv",
        ),
        "tokens": ("chest", "xray14", "nih", "dataentry", "dataentry2017"),
    },
    "openi": {
        "dir_names": (
            "OpenI",
            "openi",
            "open_i",
            "indiana",
            "nlmcxr",
        ),
        "csv_names": (
            "openi.csv",
            "openi_labels.csv",
            "openi_metadata.csv",
            "indiana.csv",
            "labels.csv",
        ),
        "tokens": ("openi", "indiana", "nlmcxr", "iu"),
    },
}


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Train a ResNet50 model on chest X-ray data.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root folder for the dataset(s).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam.")
    parser.add_argument("--use_bone", action="store_true", help="Enable the bone_suppression stub.")
    parser.add_argument("--use_crop", action="store_true", help="Enable the lung_crop stub.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dry_run", action="store_true", help="Run one short sanity-check epoch.")
    parser.add_argument("--input_size", type=int, default=448, help="Input image size.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--use_metadata", action="store_true", help="Fuse age/gender/view metadata when available.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints") / "best_resnet50.pt",
        help="Path for the best model checkpoint.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for the ResNet50 backbone.",
    )
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


def _normalize_name(value: str) -> str:
    """Normalizes a path or file name for fuzzy matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _contains_supported_images(path: Path) -> bool:
    """Returns whether a directory contains at least one supported image file."""

    if not path.exists() or not path.is_dir():
        return False

    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            return True
    return False


def _iter_candidate_directories(base_dir: Path) -> Iterable[Path]:
    """Yields likely dataset directories from the provided base directory."""

    yield base_dir

    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            yield child

    for child in sorted(base_dir.glob("*/*")):
        if child.is_dir():
            yield child


def _find_dataset_root(data_dir: Path, dataset_key: str) -> Optional[Path]:
    """Finds a dataset root directory using common naming conventions."""

    hints = DATASET_HINTS[dataset_key]
    alias_names = {_normalize_name(name) for name in hints["dir_names"]}
    token_names = tuple(_normalize_name(token) for token in hints["tokens"])

    candidates: List[Tuple[int, int, Path]] = []
    for candidate in _iter_candidate_directories(data_dir):
        normalized_name = _normalize_name(candidate.name)
        score = 0

        if normalized_name in alias_names:
            score += 100
        if any(token in normalized_name for token in token_names):
            score += 30
        if score == 0:
            continue
        if _contains_supported_images(candidate):
            score += 10

        candidates.append((score, -len(candidate.parts), candidate))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][2]


def _score_csv_candidate(csv_path: Path, dataset_key: str, dataset_root: Optional[Path]) -> int:
    """Scores a CSV file as a likely label file for a dataset."""

    hints = DATASET_HINTS[dataset_key]
    normalized_name = _normalize_name(csv_path.name)
    normalized_stem = _normalize_name(csv_path.stem)
    preferred_names = {_normalize_name(name) for name in hints["csv_names"]}
    tokens = tuple(_normalize_name(token) for token in hints["tokens"])

    score = 0
    if normalized_name in preferred_names:
        score += 100
    if any(token in normalized_name or token in normalized_stem for token in tokens):
        score += 35
    if any(token in normalized_name for token in ("label", "labels", "finding", "annotation", "entry", "meta")):
        score += 15
    if dataset_root is not None and dataset_root in csv_path.parents:
        score += 10
    return score


def _find_label_csv(data_dir: Path, dataset_key: str, dataset_root: Optional[Path]) -> Optional[Path]:
    """Finds a likely label CSV for the requested dataset."""

    search_roots: List[Path] = []
    if dataset_root is not None:
        search_roots.append(dataset_root)
    if data_dir not in search_roots:
        search_roots.append(data_dir)

    candidates: List[Tuple[int, int, Path]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue

        for csv_path in search_root.rglob("*.csv"):
            resolved = csv_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            score = _score_csv_candidate(csv_path, dataset_key, dataset_root)
            if score > 0:
                candidates.append((score, -len(csv_path.parts), resolved))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]

    if dataset_root is not None:
        fallback_csvs = sorted(dataset_root.rglob("*.csv"))
        if len(fallback_csvs) == 1:
            return fallback_csvs[0].resolve()

    return None


def discover_data_sources(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Discovers dataset roots and CSV files from a single data directory."""

    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    chest_root = _find_dataset_root(data_dir, "chestxray14")
    openi_root = _find_dataset_root(data_dir, "openi")

    chest_csv = _find_label_csv(data_dir, "chestxray14", chest_root)
    openi_csv = _find_label_csv(data_dir, "openi", openi_root)

    sources = {
        "chestxray14_root": chest_root,
        "chestxray14_csv": chest_csv,
        "openi_root": openi_root,
        "openi_csv": openi_csv,
    }

    for dataset_key in ("chestxray14", "openi"):
        root_key = f"{dataset_key}_root"
        csv_key = f"{dataset_key}_csv"
        if (sources[root_key] is None) != (sources[csv_key] is None):
            sources[root_key] = None
            sources[csv_key] = None

    found_pairs = [
        key
        for key in ("chestxray14", "openi")
        if sources[f"{key}_root"] is not None and sources[f"{key}_csv"] is not None
    ]
    if not found_pairs:
        raise FileNotFoundError(
            "Could not find a valid ChestX-ray14 or OpenI root/CSV pair under "
            f"{data_dir}. Expected folders such as 'ChestX-ray14' or 'OpenI' and a matching CSV labels file."
        )

    return sources


def compute_macro_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Computes macro AUROC while skipping classes without both positive and negative examples."""

    if targets.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Expected 2D arrays for targets and probabilities.")
    if targets.shape != probabilities.shape:
        raise ValueError(
            f"targets and probabilities must have the same shape, got {targets.shape} and {probabilities.shape}."
        )

    per_class_scores: List[float] = []
    for class_index in range(targets.shape[1]):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]

        if np.unique(class_targets).size < 2:
            continue

        per_class_scores.append(float(roc_auc_score(class_targets, class_probabilities)))

    if not per_class_scores:
        return float("nan")

    return float(np.mean(per_class_scores))


def compute_per_class_aurocs(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, float]:
    """Computes AUROC for each class independently."""

    if targets.shape != probabilities.shape:
        raise ValueError(
            f"targets and probabilities must have the same shape, got {targets.shape} and {probabilities.shape}."
        )
    if targets.ndim != 2:
        raise ValueError("Expected 2D targets and probabilities arrays.")
    if len(class_names) != targets.shape[1]:
        raise ValueError(
            f"class_names length {len(class_names)} does not match number of classes {targets.shape[1]}."
        )

    scores: Dict[str, float] = {}
    for class_index, class_name in enumerate(class_names):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]
        if np.unique(class_targets).size < 2:
            scores[class_name] = float("nan")
            continue
        scores[class_name] = float(roc_auc_score(class_targets, class_probabilities))
    return scores


def _extract_metadata_tensor(batch: Dict[str, object], device: torch.device) -> Optional[Tensor]:
    """Returns optional metadata if the dataloader provides it."""

    metadata = batch.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("Expected batch['metadata'] to be a torch.Tensor when present.")
    return metadata.to(device=device, dtype=torch.float32, non_blocking=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
    phase_name: str = "train",
) -> Dict[str, float]:
    """Runs one training or validation epoch."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_samples = 0
    all_targets: List[Tensor] = []
    all_probabilities: List[Tensor] = []

    total_steps = len(loader)
    if max_batches is not None:
        total_steps = min(total_steps, max_batches)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        metadata = _extract_metadata_tensor(batch, device=device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images, metadata=metadata)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.append(targets.detach().cpu())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu())

        if step == 1 or step == total_steps or step % max(1, total_steps // 5) == 0:
            print(f"{phase_name} step {step}/{total_steps} loss={loss.item():.4f}")

        if max_batches is not None and step >= max_batches:
            break

    mean_loss = total_loss / max(total_samples, 1)
    targets_np = torch.cat(all_targets, dim=0).numpy()
    probabilities_np = torch.cat(all_probabilities, dim=0).numpy()
    auroc = compute_macro_auroc(targets_np, probabilities_np)
    per_class_aurocs = compute_per_class_aurocs(targets_np, probabilities_np, class_names=class_names)

    return {
        "loss": mean_loss,
        "auroc": auroc,
        "per_class_aurocs": per_class_aurocs,
    }


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    val_auroc: float,
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> None:
    """Saves a best-model checkpoint compatible with models.load_model()."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": model.get_config(),
        "epoch": epoch,
        "val_auroc": val_auroc,
        "class_names": list(class_names),
        "args": vars(args),
    }
    torch.save(payload, checkpoint_path)


def _is_better_auroc(current: float, best: float) -> bool:
    """Returns whether the current AUROC should replace the best score."""

    if math.isnan(current):
        return math.isnan(best)
    if math.isnan(best):
        return True
    return current > best


def build_training_components(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, List[str], nn.Module]:
    """Builds dataloaders and model based on the command-line configuration."""

    sources = discover_data_sources(args.data_dir)
    train_loader, val_loader, class_names = build_dataloaders(
        chestxray14_root=sources["chestxray14_root"],
        chestxray14_csv=sources["chestxray14_csv"],
        openi_root=sources["openi_root"],
        openi_csv=sources["openi_csv"],
        image_size=args.input_size,
        batch_size=args.batch,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.workers,
        apply_bone_suppression=args.use_bone,
        apply_lung_crop=args.use_crop,
    )

    model = get_resnet50(
        input_channels=1,
        pretrained=not args.no_pretrained,
        input_size=args.input_size,
        num_classes=len(class_names),
        use_metadata=args.use_metadata,
    )
    return train_loader, val_loader, class_names, model


def main() -> None:
    """Runs end-to-end training."""

    args = parse_args()
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names, model = build_training_components(args)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    max_batches = 1 if args.dry_run else None
    total_epochs = 1 if args.dry_run else args.epochs
    best_val_auroc = float("nan")

    print(f"device={device}")
    print(f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    print(f"num_classes={len(class_names)}")
    if args.dry_run:
        print("dry_run enabled: training will stop after one batch per phase.")

    for epoch in range(1, total_epochs + 1):
        print(f"\nEpoch {epoch}/{total_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            optimizer=optimizer,
            max_batches=max_batches,
            phase_name="train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            optimizer=None,
            max_batches=max_batches,
            phase_name="val",
        )

        scheduler.step(val_metrics["loss"])

        train_auroc_text = "nan" if math.isnan(train_metrics["auroc"]) else f"{train_metrics['auroc']:.4f}"
        val_auroc_text = "nan" if math.isnan(val_metrics["auroc"]) else f"{val_metrics['auroc']:.4f}"
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "epoch_summary "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_auroc={train_auroc_text} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auroc={val_auroc_text} "
            f"lr={current_lr:.6g}"
        )
        print(
            "train_per_class_auroc "
            + " ".join(
                f"{class_name}={'nan' if math.isnan(score) else f'{score:.4f}'}"
                for class_name, score in train_metrics["per_class_aurocs"].items()
            )
        )
        print(
            "val_per_class_auroc "
            + " ".join(
                f"{class_name}={'nan' if math.isnan(score) else f'{score:.4f}'}"
                for class_name, score in val_metrics["per_class_aurocs"].items()
            )
        )

        if _is_better_auroc(val_metrics["auroc"], best_val_auroc):
            best_val_auroc = val_metrics["auroc"]
            save_checkpoint(
                model=model,
                checkpoint_path=args.checkpoint,
                epoch=epoch,
                val_auroc=val_metrics["auroc"],
                class_names=class_names,
                args=args,
            )
            print(f"saved_best_checkpoint={args.checkpoint}")

    print("training_complete")


if __name__ == "__main__":
    main()
