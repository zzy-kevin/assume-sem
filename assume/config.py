"""Global configuration constants for ASSUME pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

# Project filesystem layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# Default dataset location (configurable via environment variable)
DEFAULT_DATASET_DIR: Path = Path(r"D:\UOFT\Grad1_local\SEM\online\t2v4p-2x965")
DATASET_DIR: Path = Path(os.getenv("SEM_DATASET_PATH", str(DEFAULT_DATASET_DIR)))

# Downsampled copy of the 1,543 evaluation micrographs shipped with the repo (CC BY 4.0)
MICROGRAPH_PACK_DIR: Path = DATA_DIR / "micrographs"


def resolve_micrograph(path) -> Path:
    """Return `path` if it exists, else the bundled copy at data/micrographs/<class>/<file>.

    Manifests store absolute Windows paths from the original machine; this makes them
    portable without rewriting the manifests.
    """
    p = Path(path)
    if p.exists():
        return p
    parts = str(path).replace("\\", "/").split("/")
    alt = MICROGRAPH_PACK_DIR / parts[-2] / parts[-1]
    return alt if alt.exists() else p

# Dataset classes and exclusion specification
EXCLUDED_CLASSES: Tuple[str, ...] = ("Powder_highres",)
ACTIVE_CLASSES: Tuple[str, ...] = (
    "Biological",
    "Fibres",
    "Films_Coated_Surface",
    "Nanowires",
    "Particles",
    "Patterned_surface",
    "Porous_Sponge",
    "Powder",
)

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(ACTIVE_CLASSES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(ACTIVE_CLASSES)}

EXPECTED_TOTAL_MICROGRAPHS: int = 14620

# Micrograph relative proportional preprocessing
CROP_RATIO: float = 0.75  # Retain top 75% height to cleanly excise instrument databar
TARGET_WIDTH: int = 512
TARGET_HEIGHT: int = 288
TARGET_IMAGE_SIZE: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)  # (width, height)

# DINOv3 ViT-S/16+ teacher configuration
DINOV3_MODEL_ID: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
EMBEDDING_DIM: int = 384
PATCH_SIZE: int = 16
GRID_WIDTH: int = TARGET_WIDTH // PATCH_SIZE   # 512 // 16 = 32
GRID_HEIGHT: int = TARGET_HEIGHT // PATCH_SIZE # 288 // 16 = 18
NUM_PATCH_TOKENS: int = GRID_WIDTH * GRID_HEIGHT  # 32 * 18 = 576
NUM_SPECIAL_TOKENS: int = 5  # 1 [CLS] + 4 register tokens
TOTAL_TOKENS: int = NUM_SPECIAL_TOKENS + NUM_PATCH_TOKENS  # 581
PATCH_SLICE: slice = slice(NUM_SPECIAL_TOKENS, TOTAL_TOKENS)

# Image normalization (standard ImageNet parameters)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Output artifact paths
MANIFEST_PATH: Path = DATA_DIR / "manifest.parquet"
EMBEDDING_PATH: Path = DATA_DIR / "dinov3_pooled.npy"
ROOT_MANIFEST_PATH: Path = PROJECT_ROOT / "manifest.parquet"
ROOT_EMBEDDING_PATH: Path = PROJECT_ROOT / "dinov3_pooled.npy"

# Load .env if present (allowing local .env override)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

DEFAULT_VERTEX_PROJECT_ID: str = ""  # set VERTEX_PROJECT_ID in .env
DEFAULT_VERTEX_API_KEY: str = ""  # set VERTEX_API_KEY in .env; never commit keys
DEFAULT_VERTEX_MODEL: str = "gemini-2.5-flash"
VERTEX_PROJECT_ID: str = os.getenv("VERTEX_PROJECT_ID", DEFAULT_VERTEX_PROJECT_ID)
VERTEX_API_KEY: str = os.getenv("VERTEX_API_KEY", DEFAULT_VERTEX_API_KEY)
VERTEX_MODEL: str = os.getenv("VERTEX_MODEL", DEFAULT_VERTEX_MODEL)

