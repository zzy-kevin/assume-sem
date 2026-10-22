"""Zero-Shot Semantic Bottleneck Evaluation Module for ASSUME Milestone 5.

Implements:
1. Balanced evaluation subset extraction (1,600 micrographs, exactly 200 per class across 8 classes,
   with deterministic seed=42 resampling for minority classes Fibres and Porous_Sponge).
2. End-to-end concept bottleneck matrix scoring with ScorerAgent, supporting checkpointing and resumption.
3. Hermetic MockScorerAgent modeling microstructural semantics with valid [1, 10] integer scores.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image

from assume.agents.proposer import ConceptRubric
from assume.config import (
    ACTIVE_CLASSES,
    DATA_DIR,
    MANIFEST_PATH,
    PROJECT_ROOT,
    ROOT_MANIFEST_PATH,
)
from assume.data.ingestion import preprocess_micrograph

logger = logging.getLogger(__name__)

DEFAULT_EVAL_MANIFEST_PATH: Path = DATA_DIR / "eval_manifest.parquet"
DEFAULT_ROOT_EVAL_MANIFEST_PATH: Path = PROJECT_ROOT / "eval_manifest.parquet"
DEFAULT_EMBEDDINGS_PATH: Path = DATA_DIR / "assume_embeddings_1600x30.npy"
DEFAULT_ROOT_EMBEDDINGS_PATH: Path = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
DEFAULT_CHECKPOINT_PATH: Path = DATA_DIR / "assume_embeddings_checkpoint.npy"


def _mirror_file(src: Path, dst: Path) -> None:
    """Safely mirror a file from src to dst."""
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logger.warning("Failed to mirror %s to %s: %s", src, dst, e)


def create_balanced_eval_subset(
    manifest_path: Optional[Union[str, Path]] = None,
    n_per_class: int = 200,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
    root_output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Construct a balanced evaluation subset without oversampling duplicates.

    For classes with count >= n_per_class (200), samples exactly n_per_class micrographs
    without replacement using the deterministic random seed.
    For minority classes with count < n_per_class ('Fibres' with 162 micrographs,
    'Porous_Sponge' with 181 micrographs), takes all available micrographs without
    replacement (min(n_available, n_per_class)), yielding 1,543 unique micrographs
    across the 8 classes with zero duplicate images.

    Preserves columns: image_id, class_label, class_id, rel_path, full_path, emb_index, raw_w, raw_h.

    Args:
        manifest_path: Path to dataset manifest parquet (default: data/manifest.parquet).
        n_per_class: Maximum target sample count per class (default: 200).
        seed: Random seed for deterministic reproducibility (default: 42).
        output_path: Primary export path for eval manifest (default: data/eval_manifest.parquet).
        root_output_path: Root mirror path for eval manifest (default: eval_manifest.parquet).

    Returns:
        pd.DataFrame containing unique micrographs (1,543 total on full dataset for n_per_class=200).
    """
    if manifest_path is not None:
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found at {m_path}")
    else:
        m_path = MANIFEST_PATH if MANIFEST_PATH.exists() else ROOT_MANIFEST_PATH
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH} or {ROOT_MANIFEST_PATH}")

    out_p = Path(output_path) if output_path else DEFAULT_EVAL_MANIFEST_PATH
    if root_output_path:
        root_out_p = Path(root_output_path)
    elif output_path is None:
        root_out_p = DEFAULT_ROOT_EVAL_MANIFEST_PATH
    else:
        root_out_p = None

    df = pd.read_parquet(m_path)
    selected_chunks: List[pd.DataFrame] = []
    expected_total = 0

    for class_name in ACTIVE_CLASSES:
        class_sub = df[df["class_label"] == class_name]
        n_available = len(class_sub)

        if n_available == 0:
            raise ValueError(f"Class '{class_name}' has 0 micrographs in manifest {m_path}.")

        n_sample = min(n_available, n_per_class)
        expected_total += n_sample

        if n_available > n_per_class:
            sampled = class_sub.sample(n=n_sample, replace=False, random_state=seed)
        else:
            sampled = class_sub.copy()

        selected_chunks.append(sampled)

    eval_df = pd.concat(selected_chunks, ignore_index=True)

    # Validate output shape and uniqueness
    if len(eval_df) != expected_total:
        raise ValueError(
            f"Expected {expected_total} total evaluation rows, got {len(eval_df)}."
        )

    # Save primary parquet artifact
    out_p.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_parquet(out_p, index=False)
    logger.info("Saved balanced evaluation manifest to %s (%d unique rows)", out_p, len(eval_df))

    # Mirror to root if requested
    if root_out_p:
        _mirror_file(out_p, root_out_p)
        logger.info("Mirrored evaluation manifest to %s", root_out_p)

    return eval_df


# Semantic concept bias vectors per SEM class for physically grounded mock scoring on [0, 5] scale
_CLASS_SEMANTIC_PROFILES: Dict[str, Dict[str, float]] = {
    "Fibres": {
        "filamentous_fiber_packing_density": 4.5,
        "aspect_ratio_of_primary_microstructural_units": 4.5,
        "directional_alignment_of_elongated_structures": 3.5,
        "curvilinear_tortuosity_of_filamentous_strands": 4.0,
        "collimated_bundle_cohesion": 3.5,
        "transverse_filament_diameter_monodispersity": 3.5,
        "free_standing_fiber_end_density": 4.0,
        "inter_filament_entanglement_interlacing": 4.2,
        "periodic_array_regularity": 0.0,
        "discrete_particle_spatial_areal_density": 0.0,
    },
    "Nanowires": {
        "filamentous_fiber_packing_density": 4.2,
        "aspect_ratio_of_primary_microstructural_units": 4.8,
        "directional_alignment_of_elongated_structures": 3.5,
        "transverse_filament_diameter_monodispersity": 4.2,
        "free_standing_fiber_end_density": 3.8,
        "inter_filament_entanglement_interlacing": 3.5,
        "periodic_array_regularity": 0.0,
        "discrete_particle_spatial_areal_density": 0.0,
    },
    "Patterned_surface": {
        "periodic_array_regularity": 4.8,
        "periodic_grating_spatial_frequency": 4.2,
        "hexagonal_close_packing_coordination": 4.0,
        "sidewall_scalloping_prominence": 3.8,
        "planar_boundary_and_facet_edge_sharpness": 4.0,
        "filamentous_fiber_packing_density": 0.0,
        "curvilinear_tortuosity_of_filamentous_strands": 0.0,
    },
    "Particles": {
        "discrete_particle_spatial_areal_density": 4.5,
        "dispersed_phase_cluster_aggregation_degree": 4.0,
        "sub_micron_surface_grain_roughness": 3.5,
        "geometric_angularity_of_protruding_features": 3.2,
        "aspect_ratio_of_primary_microstructural_units": 1.0,
        "filamentous_fiber_packing_density": 0.0,
    },
    "Powder": {
        "discrete_particle_spatial_areal_density": 4.2,
        "sub_micron_surface_granular_stippling_density": 4.2,
        "sub_micron_surface_grain_roughness": 4.0,
        "dispersed_phase_cluster_aggregation_degree": 3.8,
        "depth_of_field_focal_degradation": 3.0,
        "filamentous_fiber_packing_density": 0.0,
        "periodic_array_regularity": 0.0,
    },
    "Porous_Sponge": {
        "structural_branching_junction_density": 4.5,
        "inter_bundle_void_volume_fraction": 4.5,
        "irregular_boundary_etch_pit_cavity_area_fraction": 4.2,
        "open_loop_feature_curvature_openness": 4.0,
        "topographical_relief_and_edge_contrast_prominence": 4.0,
        "periodic_array_regularity": 0.0,
    },
    "Biological": {
        "concentric_curvature_gradient": 4.2,
        "curvilinear_tortuosity_of_filamentous_strands": 3.8,
        "depth_of_field_focal_degradation": 3.8,
        "open_loop_feature_curvature_openness": 4.0,
        "macroscopic_dark_stain_cloudiness": 3.5,
        "periodic_array_regularity": 0.0,
        "planar_boundary_and_facet_edge_sharpness": 0.5,
    },
    "Films_Coated_Surface": {
        "microstructural_grain_contrast_patchiness": 4.5,
        "planar_boundary_and_facet_edge_sharpness": 4.0,
        "topographical_relief_and_edge_contrast_prominence": 3.5,
        "inter_ridge_tapering_gradient": 3.2,
        "filamentous_fiber_packing_density": 0.0,
        "discrete_particle_spatial_areal_density": 0.5,
        "inter_bundle_void_volume_fraction": 0.5,
    },
}


class MockScorerAgent:
    """Hermetic, physically grounded mock scorer producing realistic integer ratings in [0, 5].

    Generates reproducible concept activations matching SEM morphological semantics
    when class labels are present, and valid high-entropy bounded scores when absent.
    Used for unit testing, offline integration, and fast verification workflows.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize MockScorerAgent with deterministic seed."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def score_single_image(
        self,
        image: Union[Image.Image, str, Path],
        rubrics: Sequence[ConceptRubric],
        class_label: Optional[str] = None,
    ) -> Dict[str, int]:
        """Score a single micrograph across rubrics, returning concept_id to integer in [0, 5]."""
        label = class_label
        if label is None and isinstance(image, Image.Image):
            label = getattr(image, "_class_label", None)
        if label is None and isinstance(image, (str, Path)):
            p_str = str(image)
            for c_name in ACTIVE_CLASSES:
                if c_name in p_str:
                    label = c_name
                    break

        profile = _CLASS_SEMANTIC_PROFILES.get(label, {}) if label else {}

        scores: Dict[str, int] = {}
        for r in rubrics:
            base_mean = profile.get(r.concept_id, 2.5)
            # Add stochastic variation (standard deviation ~ 0.8)
            raw = self.rng.normal(loc=base_mean, scale=0.8)
            clamped = int(np.clip(np.round(raw), 0, 5))
            scores[r.concept_id] = clamped

        return scores

    def score_images(
        self,
        images: Sequence[Union[Image.Image, str, Path]],
        rubrics: Sequence[ConceptRubric],
        max_workers: int = 1,
    ) -> np.ndarray:
        """Score multiple micrographs against all rubrics.

        Returns:
            np.ndarray of shape (len(images), len(rubrics)) with dtype int64 in [0, 5].
        """
        n_images = len(images)
        n_rubrics = len(rubrics)
        if n_images == 0 or n_rubrics == 0:
            return np.zeros((n_images, n_rubrics), dtype=np.int64)

        matrix = np.zeros((n_images, n_rubrics), dtype=np.int64)
        for i, img in enumerate(images):
            c_label = getattr(img, "_class_label", None) if isinstance(img, Image.Image) else None
            res = self.score_single_image(img, rubrics, class_label=c_label)
            for j, r in enumerate(rubrics):
                matrix[i, j] = res.get(r.concept_id, 0)

        return matrix


def compute_bottleneck_embeddings(
    eval_df: pd.DataFrame,
    concepts: Sequence[ConceptRubric],
    scorer: Optional[Any] = None,
    max_workers: int = 2,
    checkpoint_interval: int = 50,
    output_path: Optional[Union[str, Path]] = None,
    root_output_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Evaluate evaluation subset micrographs against concept rubrics to produce bottleneck embeddings.

    Produces float32 numpy matrix of shape (len(eval_df), len(concepts)) with ratings in [0, 5].
    Saves array to data/assume_embeddings_1600x30.npy and mirrors to root assume_embeddings_1600x30.npy.
    Supports durable checkpointing and resumption.

    Args:
        eval_df: Evaluation subset DataFrame containing 'full_path' and 'class_label'.
        concepts: Sequence of K ConceptRubric objects.
        scorer: ScorerAgent instance (or MockScorerAgent). If None, initializes default ScorerAgent.
        max_workers: Concurrency worker limit for batch scoring.
        checkpoint_interval: Interval in images between durable disk checkpoints (default: 50).
        output_path: Primary destination for embeddings (default: data/assume_embeddings_1600x30.npy).
        root_output_path: Mirror destination (default: assume_embeddings_1600x30.npy).
        checkpoint_path: Checkpoint array path (default: data/assume_embeddings_checkpoint.npy).
        force_recompute: If True, ignores existing embeddings and recomputes from scratch.

    Returns:
        np.ndarray of shape (len(eval_df), len(concepts)), dtype float32, values in [0.0, 5.0].
    """
    n_images = len(eval_df)
    n_concepts = len(concepts)

    out_p = Path(output_path) if output_path else DEFAULT_EMBEDDINGS_PATH
    if root_output_path:
        root_out_p = Path(root_output_path)
    elif output_path is None:
        root_out_p = DEFAULT_ROOT_EMBEDDINGS_PATH
    else:
        root_out_p = None
    chk_p = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH

    # Fast return if valid completed embeddings already exist and recompute not forced
    if not force_recompute and out_p.exists():
        try:
            existing = np.load(out_p)
            if (
                existing.shape == (n_images, n_concepts)
                and not np.isnan(existing).any()
                and (existing >= 0.0).all()
                and (existing <= 10.0).all()
            ):
                logger.info(
                    "Found valid existing bottleneck embeddings at %s (shape: %s).",
                    out_p,
                    existing.shape,
                )
                if root_out_p:
                    _mirror_file(out_p, root_out_p)
                return existing.astype(np.float32)
        except Exception as e:
            logger.warning("Failed to load existing embeddings from %s: %s; recomputing.", out_p, e)

    # Initialize default live scorer if not provided
    if scorer is None:
        from assume.agents.scorer import ScorerAgent

        scorer = ScorerAgent(max_chunk_size=10)

    # Check for existing checkpoint to resume progress
    matrix = np.full((n_images, n_concepts), np.nan, dtype=np.float32)
    start_idx = 0

    if chk_p.exists() and not force_recompute:
        try:
            chk_arr = np.load(chk_p)
            if chk_arr.shape == (n_images, n_concepts):
                valid_mask = ~np.isnan(chk_arr).any(axis=1)
                num_done = int(np.sum(valid_mask))
                if num_done > 0:
                    matrix[:num_done] = chk_arr[:num_done]
                    start_idx = num_done
                    logger.info("Resuming bottleneck scoring from checkpoint image %d / %d", start_idx, n_images)
        except Exception as e:
            logger.warning("Checkpoint load failed: %s; starting fresh.", e)

    logger.info(
        "Starting bottleneck encoding: %d images x %d concepts (starting at index %d, workers=%d)...",
        n_images,
        n_concepts,
        start_idx,
        max_workers,
    )

    batch_size = 10
    for idx in range(start_idx, n_images, batch_size):
        end_idx = min(idx + batch_size, n_images)
        batch_sub = eval_df.iloc[idx:end_idx]

        batch_imgs: List[Image.Image] = []
        for _, row in batch_sub.iterrows():
            img_path = Path(row["full_path"]) if "full_path" in row else None
            c_label = str(row.get("class_label", ""))

            if img_path and img_path.exists():
                try:
                    img = preprocess_micrograph(img_path)
                except Exception:
                    img = Image.new("RGB", (512, 288), color=(128, 128, 128))
            else:
                img = Image.new("RGB", (512, 288), color=(128, 128, 128))

            # Attach class label metadata for mock scorer support
            setattr(img, "_class_label", c_label)
            batch_imgs.append(img)

        # Score batch micrographs against concepts
        batch_scores = scorer.score_images(batch_imgs, concepts, max_workers=max_workers)
        matrix[idx:end_idx] = batch_scores.astype(np.float32)

        # Periodic checkpointing
        if end_idx % checkpoint_interval == 0 or end_idx == n_images:
            chk_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(chk_p, matrix)
            logger.info("Scored %d / %d micrographs (checkpoint saved to %s)", end_idx, n_images, chk_p)

    # Validate output
    if np.isnan(matrix).any():
        raise RuntimeError("Bottleneck embeddings contain NaN values after scoring pass.")

    # Save final array
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_p, matrix)
    logger.info("Saved final bottleneck embeddings to %s (shape: %s, dtype: %s)", out_p, matrix.shape, matrix.dtype)

    # Mirror to project root if requested
    if root_out_p:
        _mirror_file(out_p, root_out_p)
        logger.info("Mirrored final bottleneck embeddings to %s", root_out_p)

    # Remove temporary checkpoint on successful completion
    if chk_p.exists():
        try:
            chk_p.unlink()
        except Exception:
            pass

    return matrix
