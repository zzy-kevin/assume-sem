"""Statistical Pruner for ASSUME Iterative Discovery Loop.

Identifies and removes uninformative ("dead") concept dimensions (std < 0.20 or >90% identical),
computes resilient Pearson correlation matrices with zero-variance protection, and detects
collinear concept pairs (|r| > 0.80) to route to the Visual Merger Agent.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from assume.agents.proposer import ConceptRubric

logger = logging.getLogger(__name__)

DEFAULT_MIN_STD: float = 0.20
DEFAULT_COLLINEAR_THRESHOLD: float = 0.80
DEFAULT_MAX_IDENTICAL_FRACTION: float = 0.90


class StatisticalPruner:
    """Statistical pruner and correlation analyzer for candidate concept dimensions."""

    def __init__(
        self,
        min_std: float = DEFAULT_MIN_STD,
        collinear_threshold: float = DEFAULT_COLLINEAR_THRESHOLD,
        max_identical_fraction: float = DEFAULT_MAX_IDENTICAL_FRACTION,
    ) -> None:
        """Initialize StatisticalPruner.

        Args:
            min_std: Minimum standard deviation across evaluation images to avoid pruning (default 0.20).
            collinear_threshold: Absolute Pearson correlation threshold (|r| > threshold) for collinearity (default 0.80).
            max_identical_fraction: Maximum fraction of identical scores allowed before considering dead (default 0.90).
        """
        self.min_std: float = float(min_std)
        self.collinear_threshold: float = float(collinear_threshold)
        self.max_identical_fraction: float = float(max_identical_fraction)

    def identify_dead_dimensions(
        self,
        scores: np.ndarray,
        min_std: Optional[float] = None,
        max_identical_fraction: Optional[float] = None,
    ) -> List[int]:
        """Identify column indices of dead concept dimensions.

        A concept dimension is considered dead if:
        1. Standard deviation across samples is strictly less than min_std (e.g. < 0.20), OR
        2. Greater than max_identical_fraction of scores are identical (e.g. > 90% identical).

        Args:
            scores: 2D numpy array of shape (N, K) containing concept scores.
            min_std: Optional override for minimum standard deviation.
            max_identical_fraction: Optional override for maximum identical fraction.

        Returns:
            List of column indices (0-indexed) identified as dead dimensions.
        """
        if scores is None or scores.size == 0 or scores.ndim != 2:
            return []

        n_samples, n_concepts = scores.shape
        if n_samples < 2 or n_concepts == 0:
            return []

        std_thresh = self.min_std if min_std is None else float(min_std)
        ident_thresh = (
            self.max_identical_fraction
            if max_identical_fraction is None
            else float(max_identical_fraction)
        )

        dead_indices: List[int] = []
        for j in range(n_concepts):
            col = scores[:, j]
            col_std = float(np.std(col))

            # Check identical fraction
            _, counts = np.unique(col, return_counts=True)
            max_count = int(np.max(counts)) if len(counts) > 0 else 0
            identical_frac = float(max_count) / float(n_samples) if n_samples > 0 else 0.0

            if (col_std < std_thresh) or (identical_frac > ident_thresh):
                dead_indices.append(j)

        return dead_indices

    def prune_dead_dimensions(
        self,
        scores: np.ndarray,
        rubrics: Sequence[ConceptRubric],
        min_std: Optional[float] = None,
        max_identical_fraction: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[ConceptRubric], List[str]]:
        """Prune zero-variance or uninformative concept dimensions.

        Args:
            scores: Concept score matrix of shape (N, K).
            rubrics: Sequence of K ConceptRubric objects corresponding to columns.
            min_std: Minimum acceptable standard deviation (defaults to instance setting, e.g. 0.20).
            max_identical_fraction: Max identical fraction allowed (defaults to instance setting, e.g. 0.90).

        Returns:
            Tuple of:
            - filtered_scores: Array of shape (N, K_active) with dead columns removed.
            - retained_rubrics: List of retained ConceptRubric objects.
            - pruned_concept_names: List of strings containing names of pruned concepts.
        """
        rubric_list = list(rubrics)
        n_rubrics = len(rubric_list)

        if n_rubrics == 0:
            return (
                np.zeros((len(scores) if scores is not None else 0, 0), dtype=np.float32),
                [],
                [],
            )

        if scores is None or scores.size == 0:
            return (
                np.zeros((0, n_rubrics), dtype=np.float32),
                rubric_list,
                [],
            )

        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.ndim != 2 or scores_arr.shape[1] != n_rubrics:
            raise ValueError(
                f"scores shape {scores_arr.shape} does not match rubric count {n_rubrics}"
            )

        n_samples = scores_arr.shape[0]
        if n_samples < 2:
            # Insufficient samples to reliably compute variance; retain all
            return scores_arr, rubric_list, []

        dead_indices_set = set(
            self.identify_dead_dimensions(
                scores_arr,
                min_std=min_std,
                max_identical_fraction=max_identical_fraction,
            )
        )

        keep_indices: List[int] = [i for i in range(n_rubrics) if i not in dead_indices_set]
        pruned_indices: List[int] = [i for i in range(n_rubrics) if i in dead_indices_set]

        filtered_scores = scores_arr[:, keep_indices] if keep_indices else np.zeros((n_samples, 0), dtype=np.float32)
        retained_rubrics = [rubric_list[i] for i in keep_indices]
        pruned_concept_names = [rubric_list[i].name for i in pruned_indices]

        if pruned_concept_names:
            logger.info(
                "Pruned %d dead concept dimension(s) (std < %.2f or > %.0f%% identical): %s",
                len(pruned_concept_names),
                self.min_std if min_std is None else min_std,
                (self.max_identical_fraction if max_identical_fraction is None else max_identical_fraction) * 100,
                pruned_concept_names,
            )

        return filtered_scores, retained_rubrics, pruned_concept_names

    def compute_correlation_matrix(
        self,
        scores: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute active-only Pearson correlation matrix R in R^(K x K) across concept score columns.

        For each pair of concepts (i, j), correlation is computed across the micrographs where
        at least one of the two concepts is active (max(s_i, s_j) > 0), preventing shared 0
        ("absent / not applicable") scores from inducing false collinearity.

        Handles constant / zero-variance dimensions cleanly:
        - Non-diagonal correlations involving zero-variance dimensions are set to 0.0.
        - Diagonal entries are strictly 1.0.
        - Numerical clipping guarantees output in [-1.0, 1.0] with exact symmetry.

        Args:
            scores: Concept score matrix of shape (N, K).
            eps: Epsilon threshold for zero-variance detection.

        Returns:
            Pearson correlation matrix of shape (K, K) with dtype float32.
        """
        if scores is None or scores.size == 0 or scores.ndim != 2:
            return np.zeros((0, 0), dtype=np.float32)

        n_samples, n_concepts = scores.shape
        if n_concepts == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if n_samples < 2:
            # Single sample or empty: identity matrix
            return np.eye(n_concepts, dtype=np.float32)

        scores_arr = np.asarray(scores, dtype=np.float64)
        corr = np.eye(n_concepts, dtype=np.float32)

        # If scores are non-negative (0..5 scale where 0 = absent), filter per-pair to max(s_i, s_j) > 0
        has_negatives = bool(np.any(scores_arr < 0.0))

        for i in range(n_concepts - 1):
            col_i = scores_arr[:, i : i + 1]  # (N, 1)
            cols_j = scores_arr[:, i + 1 :]  # (N, K - i - 1)

            if has_negatives:
                mask = np.ones_like(cols_j, dtype=np.float64)
            else:
                mask = ((col_i > 0.0) | (cols_j > 0.0)).astype(np.float64)

            counts = np.sum(mask, axis=0)  # (K - i - 1,)
            valid_count = counts >= 2.0
            safe_counts = np.where(valid_count, counts, 1.0)

            xi = col_i * mask
            xj = cols_j * mask

            mean_i = np.sum(xi, axis=0, keepdims=True) / safe_counts
            mean_j = np.sum(xj, axis=0, keepdims=True) / safe_counts

            ci = (col_i - mean_i) * mask
            cj = (cols_j - mean_j) * mask

            cov = np.sum(ci * cj, axis=0) / safe_counts
            std_i = np.sqrt(np.sum(ci * ci, axis=0) / safe_counts)
            std_j = np.sqrt(np.sum(cj * cj, axis=0) / safe_counts)

            denom = std_i * std_j
            valid = valid_count & (denom > eps)
            safe_denom = np.where(valid, denom, 1.0)

            r_vals = np.where(valid, cov / safe_denom, 0.0).astype(np.float32)
            corr[i, i + 1 :] = r_vals
            corr[i + 1 :, i] = r_vals

        # Symmetric enforcement and range clipping
        corr = (corr + corr.T) / 2.0
        corr = np.clip(corr, -1.0, 1.0)

        # Zero out any non-finite entries (NaN / Inf safety)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        return corr.astype(np.float32)

    def find_collinear_pairs(
        self,
        corr_matrix: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """Identify pairs of concepts with absolute Pearson correlation exceeding threshold.

        Searches upper triangle (i < j) for |r_ij| > threshold.
        Results are sorted descending by absolute correlation magnitude.

        Args:
            corr_matrix: Correlation matrix of shape (K, K).
            threshold: Correlation threshold (defaults to instance collinear_threshold, e.g. 0.80).

        Returns:
            List of (i, j, r_ij) tuples where i < j and |r_ij| > threshold,
            sorted by |r_ij| in descending order.
        """
        if corr_matrix is None or corr_matrix.ndim != 2:
            return []

        k = corr_matrix.shape[0]
        if k < 2 or corr_matrix.shape[1] != k:
            return []

        thresh = self.collinear_threshold if threshold is None else float(threshold)

        pairs: List[Tuple[int, int, float]] = []
        for i in range(k):
            for j in range(i + 1, k):
                r_val = float(corr_matrix[i, j])
                if abs(r_val) > thresh:
                    pairs.append((i, j, r_val))

        # Sort descending by absolute correlation magnitude
        pairs.sort(key=lambda item: abs(item[2]), reverse=True)
        return pairs
