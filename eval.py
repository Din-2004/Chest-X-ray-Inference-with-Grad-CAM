"""Evaluation script for chest X-ray multi-label classification."""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn

from data_loader import SUPPORTED_IMAGE_SUFFIXES, build_dataloaders
from models import get_resnet50

import matplotlib.pyplot as plt  # noqa: E402


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

    parser = argparse.ArgumentParser(description="Evaluate a saved chest X-ray checkpoint.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root folder for the dataset(s).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path saved by train.py.")
    parser.add_argument("--output_dir", type=Path, default=Path("eval_results"), help="Directory for CSVs and plots.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size override.")
    parser.add_argument("--seed", type=int, default=None, help="Seed override for the validation split.")
    parser.add_argument("--input_size", type=int, default=None, help="Input size override.")
    parser.add_argument("--val_fraction", type=float, default=None, help="Validation split override.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for precision/recall.")
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


def _extract_metadata_tensor(batch: Dict[str, object], device: torch.device) -> Optional[Tensor]:
    """Returns optional metadata if the dataloader provides it."""

    metadata = batch.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("Expected batch['metadata'] to be a torch.Tensor when present.")
    return metadata.to(device=device, dtype=torch.float32, non_blocking=True)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Runs model inference over the evaluation loader."""

    all_probabilities: List[Tensor] = []
    all_targets: List[Tensor] = []
    all_paths: List[str] = []
    all_sources: List[str] = []
    all_sample_ids: List[str] = []

    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
            metadata = _extract_metadata_tensor(batch, device=device)

            logits = model(images, metadata=metadata)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu())
            all_targets.append(targets.cpu())
            all_paths.extend(batch["path"])
            all_sources.extend(batch["source"])
            all_sample_ids.extend(batch["sample_id"])

            if step == 1 or step == len(loader) or step % max(1, len(loader) // 5) == 0:
                print(f"eval step {step}/{len(loader)}")

    return {
        "probabilities": torch.cat(all_probabilities, dim=0).numpy(),
        "targets": torch.cat(all_targets, dim=0).numpy(),
        "paths": all_paths,
        "sources": all_sources,
        "sample_ids": all_sample_ids,
    }


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Computes per-class metrics, summary metrics, and confusion matrices."""

    predictions = (probabilities >= threshold).astype(np.int64)
    metric_rows: List[Dict[str, object]] = []
    confusion_rows: List[Dict[str, object]] = []

    for class_index, class_name in enumerate(class_names):
        class_targets = targets[:, class_index].astype(np.int64)
        class_probabilities = probabilities[:, class_index]
        class_predictions = predictions[:, class_index]

        auroc = float("nan")
        if np.unique(class_targets).size >= 2:
            auroc = float(roc_auc_score(class_targets, class_probabilities))

        precision = float(precision_score(class_targets, class_predictions, zero_division=0))
        recall = float(recall_score(class_targets, class_predictions, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(class_targets, class_predictions, labels=[0, 1]).ravel()

        metric_rows.append(
            {
                "class_name": class_name,
                "auroc": auroc,
                "precision": precision,
                "recall": recall,
                "support_positive": int(class_targets.sum()),
                "support_negative": int((1 - class_targets).sum()),
            }
        )
        confusion_rows.append(
            {
                "class_name": class_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    macro_auroc = float(metrics_df["auroc"].dropna().mean()) if metrics_df["auroc"].notna().any() else float("nan")
    macro_precision = float(metrics_df["precision"].mean()) if not metrics_df.empty else float("nan")
    macro_recall = float(metrics_df["recall"].mean()) if not metrics_df.empty else float("nan")
    micro_precision = float(precision_score(targets.ravel(), predictions.ravel(), zero_division=0))
    micro_recall = float(recall_score(targets.ravel(), predictions.ravel(), zero_division=0))

    summary_df = pd.DataFrame(
        [
            {
                "macro_auroc": macro_auroc,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "threshold": threshold,
                "num_samples": int(targets.shape[0]),
                "num_classes": int(targets.shape[1]),
            }
        ]
    )

    return metrics_df, confusion_df, summary_df


def build_predictions_dataframe(
    evaluation_outputs: Dict[str, object],
    class_names: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    """Creates a per-sample predictions table."""

    probabilities = np.asarray(evaluation_outputs["probabilities"])
    targets = np.asarray(evaluation_outputs["targets"]).astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)

    rows: List[Dict[str, object]] = []
    for row_index in range(probabilities.shape[0]):
        row: Dict[str, object] = {
            "sample_id": evaluation_outputs["sample_ids"][row_index],
            "path": evaluation_outputs["paths"][row_index],
            "source": evaluation_outputs["sources"][row_index],
        }
        for class_index, class_name in enumerate(class_names):
            safe_name = class_name.replace(" ", "_")
            row[f"target_{safe_name}"] = int(targets[row_index, class_index])
            row[f"prob_{safe_name}"] = float(probabilities[row_index, class_index])
            row[f"pred_{safe_name}"] = int(predictions[row_index, class_index])
        rows.append(row)

    return pd.DataFrame(rows)


def save_auroc_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Saves a bar plot for per-class AUROC."""

    plt.figure(figsize=(max(10, len(metrics_df) * 0.8), 6))
    sns.barplot(data=metrics_df, x="class_name", y="auroc", color="#4C78A8")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.title("AUROC Per Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_precision_recall_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Saves a grouped bar plot for per-class precision and recall."""

    plot_df = metrics_df.melt(
        id_vars="class_name",
        value_vars=["precision", "recall"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(max(10, len(metrics_df) * 0.8), 6))
    sns.barplot(data=plot_df, x="class_name", y="score", hue="metric")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.title("Precision and Recall Per Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_plots(confusion_df: pd.DataFrame, output_path: Path) -> None:
    """Saves a grid of per-class confusion matrices."""

    num_classes = len(confusion_df)
    ncols = max(1, math.ceil(math.sqrt(num_classes)))
    nrows = max(1, math.ceil(num_classes / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis, (_, row) in zip(axes_array.flat, confusion_df.iterrows()):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        axis.set_title(str(row["class_name"]))

    for axis in axes_array.flatten()[num_classes:]:
        axis.axis("off")

    fig.suptitle("Confusion Matrices Per Class", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Runs evaluation, metric export, and plot generation."""

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint_bundle(args.checkpoint, map_location=device)
    model, checkpoint_class_names, config = build_model_from_checkpoint(checkpoint, device=device)

    saved_args = dict(checkpoint.get("args", {}))
    seed = int(args.seed if args.seed is not None else saved_args.get("seed", 42))
    input_size = int(args.input_size if args.input_size is not None else config.get("input_size", 448))
    batch_size = int(args.batch if args.batch is not None else saved_args.get("batch", 8))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else saved_args.get("val_fraction", 0.2))
    use_bone = bool(saved_args.get("use_bone", False)) or args.use_bone
    use_crop = bool(saved_args.get("use_crop", False)) or args.use_crop

    set_deterministic(seed)
    sources = discover_data_sources(args.data_dir)
    _, val_loader, data_class_names = build_dataloaders(
        chestxray14_root=sources["chestxray14_root"],
        chestxray14_csv=sources["chestxray14_csv"],
        openi_root=sources["openi_root"],
        openi_csv=sources["openi_csv"],
        image_size=input_size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=args.workers,
        apply_bone_suppression=use_bone,
        apply_lung_crop=use_crop,
    )

    if checkpoint_class_names != data_class_names:
        raise ValueError(
            "Class names from the checkpoint do not match the current dataset split. "
            f"Checkpoint classes: {checkpoint_class_names}; dataset classes: {data_class_names}"
        )

    print(f"device={device}")
    print(f"eval_samples={len(val_loader.dataset)}")
    print(f"num_classes={len(checkpoint_class_names)}")

    evaluation_outputs = evaluate(model=model, loader=val_loader, device=device)
    metrics_df, confusion_df, summary_df = compute_metrics(
        targets=np.asarray(evaluation_outputs["targets"]),
        probabilities=np.asarray(evaluation_outputs["probabilities"]),
        class_names=checkpoint_class_names,
        threshold=args.threshold,
    )
    predictions_df = build_predictions_dataframe(
        evaluation_outputs=evaluation_outputs,
        class_names=checkpoint_class_names,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "per_class_metrics.csv"
    confusion_csv = args.output_dir / "confusion_matrices.csv"
    summary_csv = args.output_dir / "summary.csv"
    predictions_csv = args.output_dir / "prediction_scores.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    confusion_df.to_csv(confusion_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    save_auroc_plot(metrics_df, args.output_dir / "auroc_per_class.png")
    save_precision_recall_plot(metrics_df, args.output_dir / "precision_recall_per_class.png")
    save_confusion_plots(confusion_df, args.output_dir / "confusion_matrices.png")

    summary = summary_df.iloc[0]
    macro_auroc_text = "nan" if pd.isna(summary["macro_auroc"]) else f"{summary['macro_auroc']:.4f}"
    print(f"macro_auroc={macro_auroc_text}")
    print(f"macro_precision={summary['macro_precision']:.4f}")
    print(f"macro_recall={summary['macro_recall']:.4f}")
    print(f"saved_metrics={metrics_csv}")
    print(f"saved_plots={args.output_dir}")


if __name__ == "__main__":
    main()
