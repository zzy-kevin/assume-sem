"""Micrograph dataset ingestion, scanning, manifest generation, and preprocessing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from assume.config import (
    ACTIVE_CLASSES,
    CLASS_TO_ID,
    CROP_RATIO,
    DATASET_DIR,
    DEFAULT_DATASET_DIR,
    EXCLUDED_CLASSES,
    EXPECTED_TOTAL_MICROGRAPHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_IMAGE_SIZE,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)

logger = logging.getLogger(__name__)


def preprocess_micrograph(
    image_or_path: Union[Image.Image, str, Path],
    crop_ratio: float = CROP_RATIO,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
) -> Image.Image:
    """Preprocess a SEM micrograph by removing the instrument databar and resizing.

    Applies relative proportional height cropping to cleanly excise the databar
    across both standard (1024x768) and non-standard (3072x2304, 2048x1536, etc.)
    resolutions, followed by bilinear interpolation to the target model dimensions.

    Args:
        image_or_path: PIL Image object or file path.
        crop_ratio: Fraction of image height to retain from top (default 0.75).
        target_size: Output (width, height) tuple (default (512, 288)).

    Returns:
        PIL.Image in RGB mode resized to target_size.
    """
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened_img:
            img = opened_img.copy()
    else:
        img = image_or_path

    already_preprocessed = (
        getattr(img, "_is_preprocessed", False)
        or bool(getattr(img, "info", {}).get("preprocessed", False))
        or img.size == target_size
    )

    # Ensure 3-channel RGB representation
    if img.mode != "RGB":
        img = img.convert("RGB")

    if already_preprocessed:
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        img.info["preprocessed"] = True
        setattr(img, "_is_preprocessed", True)
        return img

    raw_w, raw_h = img.size
    crop_h = max(1, int(round(raw_h * crop_ratio)))

    # Box: (left, upper, right, lower)
    cropped = img.crop((0, 0, raw_w, crop_h))
    resized = cropped.resize(target_size, Image.Resampling.BILINEAR)
    resized.info["preprocessed"] = True
    setattr(resized, "_is_preprocessed", True)
    return resized


def scan_dataset(
    dataset_dir: Union[str, Path] = DATASET_DIR,
    excluded_classes: Sequence[str] = EXCLUDED_CLASSES,
    active_classes: Sequence[str] = ACTIVE_CLASSES,
) -> List[Dict[str, Any]]:
    """Scan dataset directory across active classes, excluding specified subdirectories.

    Guards against NTFS case-insensitivity file duplication by case-folding suffixes
    and sorting files deterministically.

    Args:
        dataset_dir: Root dataset directory containing class subdirectories.
        excluded_classes: Sequence of directory names to exclude.
        active_classes: Ordered sequence of valid active class names.

    Returns:
        List of dict records for every discovered micrograph.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    excluded_set = set(excluded_classes)
    records: List[Dict[str, Any]] = []

    # Iterate deterministically over sorted class directories
    for class_dir in sorted(dataset_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name in excluded_set:
            logger.info("Excluding directory: %s", class_name)
            continue
        if class_name not in active_classes:
            logger.warning("Directory %s not in active classes list; skipping", class_name)
            continue

        class_id = CLASS_TO_ID[class_name]

        # Scan files, case-insensitively filtering for .jpg/.jpeg
        # Use set of resolved paths to avoid duplicate hits on Windows filesystems
        seen_stems = set()
        class_files = []
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg"}:
                if file_path.name in seen_stems:
                    continue
                seen_stems.add(file_path.name)
                class_files.append(file_path)

        class_files.sort(key=lambda p: p.name)

        for file_path in class_files:
            # Fast header inspection for raw dimensions
            try:
                with Image.open(file_path) as im:
                    raw_w, raw_h = im.size
            except Exception as e:
                logger.error("Failed to read header of %s: %s", file_path, e)
                raw_w, raw_h = 1024, 768

            records.append({
                "image_id": file_path.stem,
                "class_label": class_name,
                "class_id": class_id,
                "rel_path": file_path.relative_to(dataset_path).as_posix(),
                "full_path": str(file_path.resolve()),
                "raw_w": raw_w,
                "raw_h": raw_h,
            })

    # Sort records deterministically by (class_id, image_id)
    records.sort(key=lambda r: (r["class_id"], r["image_id"]))

    # Assign sequential emb_index [0..N-1]
    for idx, rec in enumerate(records):
        rec["emb_index"] = idx

    return records


def build_manifest(
    dataset_dir: Union[str, Path] = DATASET_DIR,
    excluded_classes: Sequence[str] = EXCLUDED_CLASSES,
) -> pd.DataFrame:
    """Build and validate a complete pandas DataFrame manifest of the SEM dataset.

    Args:
        dataset_dir: Dataset directory path.
        excluded_classes: Excluded class folder names.

    Returns:
        DataFrame with columns: image_id, class_label, class_id, rel_path, full_path,
        emb_index, raw_w, raw_h.
    """
    records = scan_dataset(dataset_dir=dataset_dir, excluded_classes=excluded_classes)
    df = pd.DataFrame(records)

    # Column ordering contract
    columns = [
        "image_id",
        "class_label",
        "class_id",
        "rel_path",
        "full_path",
        "emb_index",
        "raw_w",
        "raw_h",
    ]
    df = df[columns]

    logger.info("Manifest created with %d micrographs across %d classes.", len(df), df["class_label"].nunique())
    return df


class MicrographDataset(Dataset):
    """PyTorch Dataset for preprocessing and serving micrographs for DINOv3."""

    def __init__(
        self,
        image_paths: Sequence[Union[str, Path]],
        crop_ratio: float = CROP_RATIO,
        target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.crop_ratio = crop_ratio
        self.target_size = target_size
        self.mean = list(mean)
        self.std = list(std)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        preprocessed = preprocess_micrograph(
            path,
            crop_ratio=self.crop_ratio,
            target_size=self.target_size,
        )
        # Convert PIL Image [0, 255] (H, W, 3) to PyTorch tensor [0.0, 1.0] (3, H, W)
        tensor = TF.to_tensor(preprocessed)
        # Apply standard ImageNet normalization
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor, idx
