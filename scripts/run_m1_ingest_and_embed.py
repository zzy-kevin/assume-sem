"""Milestone 1 Pipeline Runner: Ingest Micrographs & Compute DINOv3 Teacher Embeddings.

This script scans all in-scope micrographs (14,620 across 8 classes, excluding Powder_highres),
generates and saves data/manifest.parquet, precomputes L2-normalized 384-dimensional DINOv3
spatial patch mean-pooled embeddings, and saves data/dinov3_pooled.npy.
Copies/links are also maintained at project root.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import pandas as pd
import torch

from assume.config import (
    DATA_DIR,
    DATASET_DIR,
    DINOV3_MODEL_ID,
    EMBEDDING_DIM,
    EMBEDDING_PATH,
    EXCLUDED_CLASSES,
    EXPECTED_TOTAL_MICROGRAPHS,
    MANIFEST_PATH,
    ROOT_EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
)
from assume.data.dinov3_embedder import DINOv3Embedder
from assume.data.ingestion import build_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_m1_ingest_and_embed")


def link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink from dst to src; fallback to copy2 if symlinks fail."""
    if dst.resolve() == src.resolve():
        return
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception as e:
            logger.warning("Could not remove existing file %s: %s", dst, e)

    try:
        dst.symlink_to(src)
        logger.info("Symlinked %s -> %s", dst, src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)


def run_pipeline(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = DATA_DIR,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | None = None,
    force: bool = False,
) -> None:
    """Execute the ingestion and embedding generation pipeline."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_target = output_dir / "manifest.parquet"
    embedding_target = output_dir / "dinov3_pooled.npy"

    logger.info("============================================================")
    logger.info("Starting ASSUME Milestone 1 Ingestion & Pre-Embedding")
    logger.info("Dataset Directory: %s", dataset_dir)
    logger.info("Output Directory:  %s", output_dir)
    logger.info("Batch Size:        %d", batch_size)
    logger.info("Num Workers:       %d", num_workers)
    logger.info("============================================================")

    # Step 1: Scan dataset and generate manifest
    if manifest_target.exists() and not force:
        logger.info("Loading existing manifest from %s", manifest_target)
        manifest_df = pd.read_parquet(manifest_target, engine="pyarrow")
    else:
        logger.info("Scanning dataset and building manifest...")
        manifest_df = build_manifest(
            dataset_dir=dataset_dir,
            excluded_classes=EXCLUDED_CLASSES,
        )
        logger.info("Writing manifest table to %s (engine=pyarrow)...", manifest_target)
        manifest_df.to_parquet(manifest_target, engine="pyarrow", index=False)

    logger.info("Manifest contains %d rows across %d classes.", len(manifest_df), manifest_df["class_label"].nunique())

    if len(manifest_df) != EXPECTED_TOTAL_MICROGRAPHS:
        logger.warning(
            "Found %d micrographs, but expected %d. Check excluded classes or dataset path.",
            len(manifest_df),
            EXPECTED_TOTAL_MICROGRAPHS,
        )

    # Step 2: DINOv3 Feature Pre-Embedding
    if embedding_target.exists() and not force:
        logger.info("Loading existing embeddings from %s", embedding_target)
        embeddings = np.load(embedding_target)
        if embeddings.shape != (len(manifest_df), EMBEDDING_DIM):
            logger.warning(
                "Existing embeddings shape %s does not match expected (%d, %d); recomputing.",
                embeddings.shape,
                len(manifest_df),
                EMBEDDING_DIM,
            )
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        logger.info("Initializing DINOv3 Teacher Embedder (%s)...", DINOV3_MODEL_ID)
        embedder = DINOv3Embedder(device=device)

        logger.info("Extracting embeddings for %d micrographs...", len(manifest_df))
        t_embed_start = time.time()
        try:
            embeddings = embedder.embed_manifest(
                manifest_df=manifest_df,
                batch_size=batch_size,
                num_workers=num_workers,
                show_progress=True,
            )
        except RuntimeError as re:
            if "DataLoader worker" in str(re) and num_workers > 0:
                logger.warning("DataLoader worker failed on this platform; retrying with num_workers=0...")
                embeddings = embedder.embed_manifest(
                    manifest_df=manifest_df,
                    batch_size=batch_size,
                    num_workers=0,
                    show_progress=True,
                )
            else:
                raise

        t_embed_elapsed = time.time() - t_embed_start
        fps = len(manifest_df) / max(1e-3, t_embed_elapsed)
        logger.info(
            "Completed feature extraction in %.2f s (%.1f micrographs/sec)",
            t_embed_elapsed,
            fps,
        )

        logger.info("Saving embeddings matrix to %s...", embedding_target)
        np.save(embedding_target, embeddings)

    # Step 3: Link/copy artifacts to project root
    logger.info("Publishing artifacts to root directory...")
    link_or_copy(manifest_target, ROOT_MANIFEST_PATH)
    link_or_copy(embedding_target, ROOT_EMBEDDING_PATH)

    # Step 4: Verification of deliverables
    logger.info("Verifying output deliverables...")
    assert manifest_target.exists(), f"Missing {manifest_target}"
    assert embedding_target.exists(), f"Missing {embedding_target}"
    assert ROOT_MANIFEST_PATH.exists(), f"Missing {ROOT_MANIFEST_PATH}"
    assert ROOT_EMBEDDING_PATH.exists(), f"Missing {ROOT_EMBEDDING_PATH}"

    loaded_manifest = pd.read_parquet(manifest_target, engine="pyarrow")
    loaded_embs = np.load(embedding_target)

    assert len(loaded_manifest) == EXPECTED_TOTAL_MICROGRAPHS, (
        f"Expected {EXPECTED_TOTAL_MICROGRAPHS} rows, got {len(loaded_manifest)}"
    )
    assert loaded_embs.shape == (EXPECTED_TOTAL_MICROGRAPHS, EMBEDDING_DIM), (
        f"Expected shape ({EXPECTED_TOTAL_MICROGRAPHS}, {EMBEDDING_DIM}), got {loaded_embs.shape}"
    )
    assert loaded_embs.dtype == np.float32, f"Expected float32, got {loaded_embs.dtype}"

    norms = np.linalg.norm(loaded_embs, axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0)))
    logger.info("Verification summary:")
    logger.info("  Manifest rows:      %d", len(loaded_manifest))
    logger.info("  Embeddings shape:   %s", loaded_embs.shape)
    logger.info("  Embeddings dtype:   %s", loaded_embs.dtype)
    logger.info("  Max L2 norm error:  %g", max_norm_err)
    assert max_norm_err < 1e-4, f"L2 normalization verification failed: {max_norm_err}"

    total_time = time.time() - start_time
    logger.info("Milestone 1 successfully completed in %.2f seconds!", total_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASSUME Milestone 1 Ingestion & Pre-Embedding")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to SEM dataset")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing files")
    args = parser.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        force=args.force,
    )


if __name__ == "__main__":
    main()
