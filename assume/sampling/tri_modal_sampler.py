"""Tri-Modal Active Batch Sampler for ASSUME.

This module implements Milestone 2 (R2): an active discovery batch sampler
that combines three complementary sampling policies in equal 33%/33%/33% proportions:
1. 33% Uniform Random: Unbiased exploration ensuring broad coverage of rare morphologies.
2. 33% Teacher Farthest-Point Sampling (FPS): Vectorized cosine distance maximization
   in frozen DINOv3 feature space.
3. 33% Student Discrepancy / Collision Sampling: Pairs of micrographs that produce highly
   similar student concept representations (sim > 0.85) despite being distant in teacher
   space (DINOv3 cosine dist > 0.40).

At step 0 or when no valid colliding pairs exist (cold start), the sampler falls back
to 100% Random + FPS (e.g., 3 random + 3 FPS for B=6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from assume.config import EMBEDDING_PATH, MANIFEST_PATH, ROOT_EMBEDDING_PATH, ROOT_MANIFEST_PATH

logger = logging.getLogger(__name__)

# Default active learning discrepancy thresholds
DEFAULT_SIM_THRESHOLD: float = 0.85
DEFAULT_TEACHER_DIST_THRESHOLD: float = 0.40
DEFAULT_BATCH_SIZE: int = 6


class TriModalBatchSampler:
    """Active batch sampler implementing the 33/33/33 tri-modal policy for ASSUME."""

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path, pd.DataFrame]] = None,
        dinov3_features_path: Optional[Union[str, Path, np.ndarray]] = None,
        seed: Optional[int] = None,
        track_history: bool = True,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
        teacher_dist_threshold: float = DEFAULT_TEACHER_DIST_THRESHOLD,
    ) -> None:
        """Initialize TriModalBatchSampler.

        Args:
            manifest_path: Path to manifest.parquet, or pre-loaded pd.DataFrame.
                           If None, checks default and root manifest paths.
            dinov3_features_path: Path to dinov3_pooled.npy, or pre-loaded np.ndarray.
                                 If None, checks default and root embedding paths.
            seed: Optional integer seed for reproducible pseudo-random sampling.
            track_history: If True, remembers previously sampled indices to prevent
                           re-sampling in the random and FPS streams across steps.
            sim_threshold: Minimum student concept cosine similarity to qualify as a collision.
            teacher_dist_threshold: Minimum teacher DINOv3 cosine distance for collision.
        """
        self.sim_threshold: float = float(sim_threshold)
        self.teacher_dist_threshold: float = float(teacher_dist_threshold)
        self.track_history: bool = bool(track_history)
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # 1. Load teacher features
        self.features: np.ndarray = self._load_features(dinov3_features_path)
        self.n_samples: int = int(len(self.features))

        # 2. Load or construct manifest metadata
        self.manifest: pd.DataFrame = self._load_manifest(manifest_path, self.n_samples)

        # 3. State tracking
        self.sampled_indices: Set[int] = set()
        self.sampled_collision_pairs: Set[Tuple[int, int]] = set()
        self._min_dists_to_sampled: np.ndarray = np.full(self.n_samples, np.inf, dtype=np.float32)
        self.last_batch_info: Dict[str, Any] = {}

    def _load_features(
        self, features_source: Optional[Union[str, Path, np.ndarray]]
    ) -> np.ndarray:
        """Load and unit-normalize teacher features."""
        if isinstance(features_source, np.ndarray):
            features = np.asarray(features_source, dtype=np.float32)
        elif features_source is not None:
            p = Path(features_source)
            if not p.exists():
                raise FileNotFoundError(f"DINOv3 features file not found: {p}")
            features = np.load(p).astype(np.float32)
        else:
            # Fallback discovery
            if EMBEDDING_PATH.exists():
                features = np.load(EMBEDDING_PATH).astype(np.float32)
            elif ROOT_EMBEDDING_PATH.exists():
                features = np.load(ROOT_EMBEDDING_PATH).astype(np.float32)
            else:
                raise FileNotFoundError(
                    f"DINOv3 features not found at {EMBEDDING_PATH} or {ROOT_EMBEDDING_PATH}"
                )

        if features.ndim != 2:
            raise ValueError(f"Features must be a 2D array, got shape {features.shape}")

        # Ensure L2 normalization: ||v||_2 = 1.0
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        zero_norms = (norms == 0.0).squeeze()
        norms[norms == 0.0] = 1.0
        normalized = features / norms
        if np.any(zero_norms):
            normalized[zero_norms] = 0.0

        return normalized.astype(np.float32)

    def _load_manifest(
        self, manifest_source: Optional[Union[str, Path, pd.DataFrame]], n_samples: int
    ) -> pd.DataFrame:
        """Load or build DataFrame manifest."""
        if isinstance(manifest_source, pd.DataFrame):
            return manifest_source.copy()
        elif manifest_source is not None:
            p = Path(manifest_source)
            if p.exists():
                return pd.read_parquet(p)

        # Fallback discovery
        if MANIFEST_PATH.exists():
            return pd.read_parquet(MANIFEST_PATH)
        elif ROOT_MANIFEST_PATH.exists():
            return pd.read_parquet(ROOT_MANIFEST_PATH)

        # Minimal fallback synthetic dataframe
        return pd.DataFrame({"emb_index": np.arange(n_samples)})

    def reset(self) -> None:
        """Reset sampler history and tracked distances."""
        self.sampled_indices.clear()
        self.sampled_collision_pairs.clear()
        self._min_dists_to_sampled.fill(np.inf)
        self.last_batch_info.clear()

    def mark_sampled(self, indices: Iterable[int]) -> None:
        """Mark a set of image indices as sampled and update FPS reference distances."""
        idx_list = [int(i) for i in indices]
        if not idx_list:
            return

        self.sampled_indices.update(idx_list)

        # Incrementally update minimum distances to all sampled points
        if len(idx_list) == 1:
            dist = 1.0 - np.dot(self.features, self.features[idx_list[0]])
            self._min_dists_to_sampled = np.minimum(self._min_dists_to_sampled, dist)
        else:
            sub_feats = self.features[idx_list]  # (K, 384)
            sims = np.dot(self.features, sub_feats.T)  # (N, K)
            dists = 1.0 - np.max(sims, axis=1)  # (N,)
            self._min_dists_to_sampled = np.minimum(self._min_dists_to_sampled, dists)

    def get_colliding_pairs(
        self,
        concept_scores: np.ndarray,
        scored_indices: Optional[List[int]] = None,
        sim_threshold: Optional[float] = None,
        teacher_dist_threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """Identify all image pairs satisfying the student collision discrepancy conditions.

        Condition 1 (Student Collision): applicability-masked, mean-centered cosine similarity > sim_threshold
        Condition 2 (Teacher Discrepancy): d_DINO(v_i, v_j) > teacher_dist_threshold

        Only compares each pair (A, B) across active concepts where max(s_A, s_B) > 0 so shared
        0 ("absent / not applicable") scores do not inflate cosine similarity, and subtracts
        column means across active concepts before computing similarity. Also filters out
        previously sampled collision pairs when track_history is enabled.

        Args:
            concept_scores: Array of shape (M, K) of concept scores in [0, 5].
            scored_indices: List of length M mapping each row of concept_scores to
                            its global image index in [0, n_samples - 1].
                            If None, defaults to list(range(M)).
            sim_threshold: Minimum student cosine similarity (defaults to instance threshold).
            teacher_dist_threshold: Minimum teacher cosine distance (defaults to instance threshold).

        Returns:
            List of (idx_a, idx_b) tuples of global image indices meeting both criteria,
            ranked by discrepancy score (concept_sim * teacher_dist) descending.
        """
        if concept_scores is None:
            return []

        scores_arr = np.asarray(concept_scores, dtype=np.float32)
        if scores_arr.ndim != 2 or scores_arr.shape[0] < 2 or scores_arr.shape[1] == 0:
            return []

        if not np.isfinite(scores_arr).all():
            return []

        m_samples = scores_arr.shape[0]
        if scored_indices is None:
            indices = list(range(m_samples))
        else:
            indices = [int(x) for x in scored_indices]
            if len(indices) != m_samples:
                raise ValueError(
                    f"scored_indices length ({len(indices)}) does not match concept_scores rows ({m_samples})"
                )

        thresh_sim = self.sim_threshold if sim_threshold is None else float(sim_threshold)
        thresh_dist = self.teacher_dist_threshold if teacher_dist_threshold is None else float(teacher_dist_threshold)

        # 1. Applicability masking: for pair (i, j), concept k is active iff max(s_ik, s_jk) > 0
        pos_mask = (scores_arr > 0.0).astype(np.float32)  # (M, K)
        zero_mask = 1.0 - pos_mask  # (M, K), 1 where score <= 0

        # Column mean-centering across samples when m_samples >= 4 and column has non-zero variance
        centered = scores_arr.copy()
        if m_samples >= 4:
            col_means = np.mean(scores_arr, axis=0, keepdims=True)
            col_stds = np.std(scores_arr, axis=0, keepdims=True)
            var_cols = (col_stds > 1e-6).squeeze(axis=0)
            if np.any(var_cols):
                centered[:, var_cols] = scores_arr[:, var_cols] - col_means[:, var_cols]

        # Vectorized active-only dot product:
        # Active indicator for pair (i, j) on concept k is (1 - zero_mask[i, k] * zero_mask[j, k])
        cz = centered * zero_mask
        num_centered = np.dot(centered, centered.T) - np.dot(cz, cz.T)  # (M, M)

        c_sq = centered * centered
        c_sq_z = c_sq * zero_mask
        total_sq = np.sum(c_sq, axis=1, keepdims=True)  # (M, 1)
        norm_sq_i = np.maximum(total_sq - np.dot(c_sq_z, zero_mask.T), 0.0)  # (M, M)
        denom_centered = np.sqrt(norm_sq_i * norm_sq_i.T)

        # Fallback uncentered active-only cosine similarity for pairs where centered norm vanishes
        sz = scores_arr * zero_mask
        num_raw = np.dot(scores_arr, scores_arr.T) - np.dot(sz, sz.T)
        s_sq = scores_arr * scores_arr
        s_sq_z = s_sq * zero_mask
        total_raw_sq = np.sum(s_sq, axis=1, keepdims=True)
        norm_raw_sq_i = np.maximum(total_raw_sq - np.dot(s_sq_z, zero_mask.T), 0.0)
        denom_raw = np.sqrt(norm_raw_sq_i * norm_raw_sq_i.T)

        # Require at least one mutually or singly active concept (denom_raw > 1e-12)
        concept_sim = np.zeros((m_samples, m_samples), dtype=np.float32)
        valid_centered = (denom_centered > 1e-6) & (denom_raw > 1e-12)
        valid_fallback = (~valid_centered) & (denom_raw > 1e-12)
        if np.any(valid_centered):
            concept_sim[valid_centered] = num_centered[valid_centered] / denom_centered[valid_centered]
        if np.any(valid_fallback):
            concept_sim[valid_fallback] = num_raw[valid_fallback] / denom_raw[valid_fallback]

        # 2. Compute pairwise teacher cosine distance: (M, M)
        teacher_feats = self.features[indices]  # (M, 384)
        teacher_sim = np.dot(teacher_feats, teacher_feats.T)
        teacher_dist = 1.0 - np.clip(teacher_sim, -1.0, 1.0)  # (M, M)

        # 3. Detect colliding pairs in the upper triangle (i < j)
        collision_mask = (concept_sim > thresh_sim) & (teacher_dist > thresh_dist)
        upper_mask = np.triu(collision_mask, k=1)
        row_idx, col_idx = np.where(upper_mask)

        if len(row_idx) == 0:
            return []

        # 4. Rank candidate pairs by discrepancy strength
        discrepancies = concept_sim[row_idx, col_idx] * teacher_dist[row_idx, col_idx]
        sorted_order = np.argsort(-discrepancies)

        results: List[Tuple[int, int]] = []
        for rank_idx in sorted_order:
            idx_a = int(indices[row_idx[rank_idx]])
            idx_b = int(indices[col_idx[rank_idx]])
            if idx_a == idx_b:
                continue
            canon_pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            if self.track_history and canon_pair in self.sampled_collision_pairs:
                continue
            results.append((idx_a, idx_b))

        return results

    def _sample_random(
        self,
        n_points: int,
        exclude_indices: Set[int],
    ) -> List[int]:
        """Sample n_points uniformly at random without replacement from available pool."""
        if n_points <= 0:
            return []

        all_indices = np.arange(self.n_samples)
        if exclude_indices:
            mask = np.ones(self.n_samples, dtype=bool)
            # Filter valid excluded indices within bounds
            valid_excluded = [i for i in exclude_indices if 0 <= i < self.n_samples]
            if valid_excluded:
                mask[valid_excluded] = False
            available = all_indices[mask]
        else:
            available = all_indices

        if len(available) >= n_points:
            chosen = self.rng.choice(available, size=n_points, replace=False)
            return [int(x) for x in chosen]

        # In extreme starvation, sample as many as available then fill from any non-excluded
        chosen = [int(x) for x in available]
        needed = n_points - len(chosen)
        remaining = [i for i in range(self.n_samples) if i not in set(chosen)]
        if remaining:
            fill = self.rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
            chosen.extend([int(x) for x in fill])

        return chosen

    def _sample_fps(
        self,
        n_points: int,
        reference_indices: Sequence[int],
        exclude_indices: Set[int],
    ) -> List[int]:
        """Select n_points maximizing minimum cosine distance to reference and selected points.

        Args:
            n_points: Number of points to select.
            reference_indices: Reference points to maximize distance from.
            exclude_indices: Indices forbidden from selection.

        Returns:
            List of selected integer indices of length n_points.
        """
        if n_points <= 0:
            return []

        # Initialize distance vector
        ref_list = [int(x) for x in reference_indices if 0 <= x < self.n_samples]
        if ref_list:
            if len(ref_list) == 1:
                ref_feat = self.features[ref_list[0]]
                dists = 1.0 - np.dot(self.features, ref_feat)
            else:
                ref_feats = self.features[ref_list]  # (R, 384)
                sims = np.dot(self.features, ref_feats.T)  # (N, R)
                dists = 1.0 - np.max(sims, axis=1)  # (N,)
        else:
            # Seed with distance from a random available seed point
            available = [i for i in range(self.n_samples) if i not in exclude_indices]
            seed_idx = int(self.rng.choice(available)) if available else 0
            ref_feat = self.features[seed_idx]
            dists = 1.0 - np.dot(self.features, ref_feat)

        selected: List[int] = []
        forbidden = set(exclude_indices)

        for _ in range(n_points):
            current_dists = dists.copy()
            # Mask out forbidden points
            valid_forbidden = [i for i in forbidden if 0 <= i < self.n_samples]
            if valid_forbidden:
                current_dists[valid_forbidden] = -np.inf

            best_idx = int(np.argmax(current_dists))
            if current_dists[best_idx] == -np.inf:
                # Pool exhausted: fallback to any unchosen index
                fallback_choices = [i for i in range(self.n_samples) if i not in forbidden]
                if fallback_choices:
                    best_idx = int(self.rng.choice(fallback_choices))
                else:
                    break

            selected.append(best_idx)
            forbidden.add(best_idx)

            # Update minimum distances with the newly selected point
            new_dist = 1.0 - np.dot(self.features, self.features[best_idx])
            dists = np.minimum(dists, new_dist)

        return selected

    def sample_batch(
        self,
        step: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        concept_scores: Optional[np.ndarray] = None,
        scored_indices: Optional[List[int]] = None,
        update_history: bool = True,
    ) -> List[int]:
        """Sample an active batch of images cycling through a single mode per step.

        Single-Mode Batch Cycling Policy:
        - Step 3t (step % 3 == 0): 100% Teacher Farthest-Point Sampling (FPS) (6 images).
        - Step 3t + 1 (step % 3 == 1): 100% Uniform Random Sampling (6 images).
        - Step 3t + 2 (step % 3 == 2): 100% Student Collision Sampling (3 colliding pairs = 6 images).
          If no valid colliding pairs exist yet or concept_scores is None, falls back to FPS.

        Args:
            step: Iteration index (0-indexed).
            batch_size: Desired batch size (default 6).
            concept_scores: Optional (M, K) array of student concept scores.
            scored_indices: Optional list mapping rows of concept_scores to global dataset indices.
            update_history: If True, marks selected batch images and collision pairs as sampled.

        Returns:
            List of exactly batch_size integer dataset indices with zero overlap.
        """
        if batch_size <= 0:
            return []

        # Resolve scored_indices if concept_scores provided without explicit index mapping
        if concept_scores is not None and scored_indices is None:
            m_scores = len(concept_scores)
            if m_scores == self.n_samples:
                scored_indices = list(range(self.n_samples))
            elif len(self.sampled_indices) == m_scores and len(self.sampled_indices) > 0:
                scored_indices = list(self.sampled_indices)
            else:
                scored_indices = list(range(m_scores))

        # Base pool exclusion (respect sampled history if enough unsampled points remain)
        exclude_pool = set(self.sampled_indices) if (
            self.track_history and (len(self.sampled_indices) + batch_size <= self.n_samples)
        ) else set()

        batch: List[int] = []
        batch_set: Set[int] = set()
        rand_pts: List[int] = []
        fps_pts: List[int] = []
        col_pts: List[int] = []
        selected_collision_pairs: List[Tuple[int, int]] = []

        mode_idx = step % 3
        if mode_idx == 0:
            strategy = "fps"
            ref_pts = list(self.sampled_indices)
            fps_pts = self._sample_fps(
                batch_size,
                reference_indices=ref_pts,
                exclude_indices=exclude_pool,
            )
            batch.extend(fps_pts)
            batch_set.update(fps_pts)

        elif mode_idx == 1:
            strategy = "random"
            rand_pts = self._sample_random(
                batch_size,
                exclude_indices=exclude_pool,
            )
            batch.extend(rand_pts)
            batch_set.update(rand_pts)

        else:
            # Step 3t + 2: Collision mode (3 pairs = 6 images for batch_size=6)
            colliding_pairs: List[Tuple[int, int]] = []
            if concept_scores is not None:
                colliding_pairs = self.get_colliding_pairs(
                    concept_scores=concept_scores,
                    scored_indices=scored_indices,
                    sim_threshold=self.sim_threshold,
                    teacher_dist_threshold=self.teacher_dist_threshold,
                )

            target_pairs = max(1, batch_size // 2)
            for pair_a, pair_b in colliding_pairs:
                if pair_a not in batch_set and pair_b not in batch_set and pair_a != pair_b:
                    col_pts.extend([pair_a, pair_b])
                    batch_set.update([pair_a, pair_b])
                    selected_collision_pairs.append((int(pair_a), int(pair_b)))
                    if update_history and self.track_history:
                        canon_pair = (min(int(pair_a), int(pair_b)), max(int(pair_a), int(pair_b)))
                        self.sampled_collision_pairs.add(canon_pair)
                    if len(selected_collision_pairs) >= target_pairs:
                        break

            batch.extend(col_pts)

            if len(col_pts) == 0:
                strategy = "fps"
            else:
                strategy = "collision"

            # Fill any remaining slots (or full batch on fallback) via Teacher FPS
            deficit = batch_size - len(batch)
            if deficit > 0:
                ref_pts = list(self.sampled_indices) + col_pts
                fps_pts = self._sample_fps(
                    deficit,
                    reference_indices=ref_pts,
                    exclude_indices=exclude_pool.union(batch_set),
                )
                batch.extend(fps_pts)
                batch_set.update(fps_pts)

        # Safety: Guarantee exact batch size and zero overlap within batch
        if len(batch_set) < batch_size:
            missing = batch_size - len(batch_set)
            leftover = [i for i in range(self.n_samples) if i not in batch_set]
            if leftover:
                fill = self.rng.choice(leftover, size=min(missing, len(leftover)), replace=False)
                for f in fill:
                    batch.append(int(f))
                    batch_set.add(int(f))

        final_batch = [int(x) for x in batch[:batch_size]]

        # Record metadata for inspection and auditing
        self.last_batch_info = {
            "step": step,
            "strategy": strategy,
            "batch_mode": strategy,
            "batch_size": len(final_batch),
            "random_indices": [int(x) for x in rand_pts],
            "fps_indices": [int(x) for x in fps_pts],
            "collision_indices": [int(x) for x in col_pts],
            "collision_pairs": selected_collision_pairs,
            "batch": final_batch,
        }

        # Update tracked history if requested
        if update_history and self.track_history:
            self.mark_sampled(final_batch)

        return final_batch

    def get_image_metadata(self, index: int) -> Dict[str, Any]:
        """Retrieve metadata for a specific dataset image index."""
        if 0 <= index < len(self.manifest):
            row = self.manifest.iloc[index]
            return row.to_dict()
        return {"emb_index": index}

    def __len__(self) -> int:
        """Return total number of dataset samples."""
        return self.n_samples
