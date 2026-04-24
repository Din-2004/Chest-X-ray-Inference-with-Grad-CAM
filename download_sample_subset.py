"""Create a tiny synthetic ChestX-ray14-style sample subset for smoke testing.

This is a download stub. It does not fetch real medical data. Instead, it
creates a deterministic synthetic dataset with a ChestX-ray14-like CSV layout
so the training, evaluation, and Streamlit flows can be exercised locally.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


LABELS = ("Atelectasis", "Cardiomegaly", "Effusion", "Infiltration")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Create a synthetic ChestX-ray14-style sample subset.")
    parser.add_argument("--output_dir", type=Path, default=Path("sample_data"), help="Root directory for sample data.")
    parser.add_argument(
        "--sample_images_dir",
        type=Path,
        default=Path("sample_images"),
        help="Directory to populate with a few example images for the app.",
    )
    parser.add_argument("--num_samples", type=int, default=24, help="Number of synthetic images to generate.")
    parser.add_argument("--image_size", type=int, default=320, help="Square image size in pixels.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist.")
    return parser.parse_args()


def _ellipse_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    """Return an ellipse mask over the provided coordinate grid."""

    return (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0


def _label_set(index: int) -> List[str]:
    """Return a deterministic label set for one synthetic sample."""

    labels = [LABELS[index % len(LABELS)]]
    if index % 6 in {0, 3}:
        labels.append(LABELS[(index + 1) % len(LABELS)])
    return sorted(set(labels))


def _build_synthetic_image(index: int, image_size: int, seed: int) -> np.ndarray:
    """Generate one synthetic grayscale image with class-linked patterns."""

    rng = np.random.default_rng(seed + index)
    labels = _label_set(index)

    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    image = np.full((image_size, image_size), 208.0, dtype=np.float32)

    image -= 22.0 * (yy / image_size)
    image += 6.0 * np.sin(xx / image_size * np.pi * 3.0)

    left_lung = _ellipse_mask(xx, yy, image_size * 0.34, image_size * 0.48, image_size * 0.15, image_size * 0.27)
    right_lung = _ellipse_mask(xx, yy, image_size * 0.66, image_size * 0.48, image_size * 0.15, image_size * 0.27)
    lungs = left_lung | right_lung
    image[lungs] -= 60.0

    spine = np.abs(xx - image_size * 0.5) < image_size * 0.015
    image[spine] += 14.0

    if "Cardiomegaly" in labels:
        heart = _ellipse_mask(xx, yy, image_size * 0.50, image_size * 0.63, image_size * 0.16, image_size * 0.12)
        image[heart] += 40.0

    if "Atelectasis" in labels:
        atelectasis_patch = _ellipse_mask(xx, yy, image_size * 0.29, image_size * 0.72, image_size * 0.12, image_size * 0.08)
        image[atelectasis_patch] += 34.0

    if "Effusion" in labels:
        effusion_band = (yy > image_size * 0.76) & (lungs | _ellipse_mask(xx, yy, image_size * 0.50, image_size * 0.78, image_size * 0.32, image_size * 0.08))
        image[effusion_band] += 28.0

    if "Infiltration" in labels:
        for _ in range(5):
            cx = rng.uniform(image_size * 0.25, image_size * 0.75)
            cy = rng.uniform(image_size * 0.25, image_size * 0.70)
            rx = rng.uniform(image_size * 0.03, image_size * 0.08)
            ry = rng.uniform(image_size * 0.03, image_size * 0.08)
            patch = _ellipse_mask(xx, yy, cx, cy, rx, ry)
            image[patch] += rng.uniform(12.0, 26.0)

    image += rng.normal(loc=0.0, scale=6.0, size=image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return image


def _write_csv(csv_path: Path, filenames: Iterable[str]) -> None:
    """Write a ChestX-ray14-style label CSV."""

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Image Index", "Finding Labels"])
        writer.writeheader()
        for index, filename in enumerate(filenames):
            writer.writerow(
                {
                    "Image Index": filename,
                    "Finding Labels": "|".join(_label_set(index)),
                }
            )


def create_sample_subset(
    output_dir: Path,
    sample_images_dir: Path,
    num_samples: int,
    image_size: int,
    seed: int,
    force: bool,
) -> None:
    """Create the synthetic dataset and a few reusable sample images."""

    dataset_root = output_dir / "ChestX-ray14"
    images_dir = dataset_root / "images"
    csv_path = dataset_root / "Data_Entry_2017.csv"

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    if force and sample_images_dir.exists():
        shutil.rmtree(sample_images_dir)

    if csv_path.exists() and any(images_dir.glob("*.png")) and not force:
        print(f"Sample subset already exists at {dataset_root}")
    else:
        images_dir.mkdir(parents=True, exist_ok=True)
        filenames: List[str] = []
        for index in range(num_samples):
            filename = f"sample_{index:03d}.png"
            image = _build_synthetic_image(index=index, image_size=image_size, seed=seed)
            Image.fromarray(image, mode="L").save(images_dir / filename)
            filenames.append(filename)

        _write_csv(csv_path, filenames)
        print(f"Created synthetic dataset at {dataset_root}")

    sample_images_dir.mkdir(parents=True, exist_ok=True)
    preview_indices = [0, min(5, num_samples - 1), min(11, num_samples - 1)]
    for preview_index in preview_indices:
        source = images_dir / f"sample_{preview_index:03d}.png"
        if source.exists():
            shutil.copy2(source, sample_images_dir / source.name)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "This folder contains a synthetic ChestX-ray14-style sample subset created by "
        "download_sample_subset.py for smoke testing only.\n",
        encoding="utf-8",
    )
    print(f"Prepared sample images at {sample_images_dir}")


def main() -> None:
    """Entry point."""

    args = parse_args()
    create_sample_subset(
        output_dir=args.output_dir,
        sample_images_dir=args.sample_images_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
