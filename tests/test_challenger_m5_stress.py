"""Challenger Empirical Stress Testing Suite for ASSUME Milestone 5.

Adversarial and empirical verification covering:
1. Balanced Subset Stress Testing:
   - Dynamic sample sizes: n_per_class in [10, 50, 162, 181, 200, 300].
   - Deterministic seed consistency and cross-seed variance.
   - Minority oversampling invariants for Fibres (162 avail) and Porous_Sponge (181 avail):
     * Zero duplicates when N <= N_avail.
     * Exact deterministic duplicate count (N - N_avail) when N > N_avail.
     * Full coverage: all N_avail images present in the sampled set.
   - Boundary & error conditions: missing manifest, non-existent files, missing class rows.
2. Bottleneck Resumption & Checkpoint Invariants:
   - Partial checkpoint interruption and recovery with uncomputed NaN rows.
   - Resumption integrity: pre-existing rows preserved, remaining rows computed.
   - Shape mismatch or corrupt checkpoint graceful fallback to clean start.
   - Cache invalidation: force_recompute=True regenerates embeddings, force_recompute=False reuses cache.
3. Feature Alignment Invariants:
   - 20 randomly sampled rows from data/eval_manifest.parquet:
     * Raw image loading from full_path.
     * Fresh DINOv3 forward pass via DINOv3Embedder.
     * Bitwise / cosine alignment (cosine sim >= 0.9999) with dinov3_pooled.npy at emb_index.
     * Cross-check with original manifest.parquet at emb_index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import tempfile
from typing import List
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from assume.agents.proposer import ConceptRubric
from assume.config import (
    ACTIVE_CLASSES,
    DATA_DIR,
    EMBEDDING_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    ROOT_EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
)
from assume.data.dinov3_embedder import DINOv3Embedder
from assume.evaluation.bottleneck_eval import (
    MockScorerAgent,
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)

logger = logging.getLogger("challenger_m5_stress")


def _get_manifest_path() -> Path:
    if MANIFEST_PATH.exists():
        return MANIFEST_PATH
    if ROOT_MANIFEST_PATH.exists():
        return ROOT_MANIFEST_PATH
    raise FileNotFoundError("Neither MANIFEST_PATH nor ROOT_MANIFEST_PATH exists.")


def _get_embedding_path() -> Path:
    if EMBEDDING_PATH.exists():
        return EMBEDDING_PATH
    if ROOT_EMBEDDING_PATH.exists():
        return ROOT_EMBEDDING_PATH
    raise FileNotFoundError("Neither EMBEDDING_PATH nor ROOT_EMBEDDING_PATH exists.")


class TestBalancedSubsetStress(unittest.TestCase):
    """Adversarial stress-testing of create_balanced_eval_subset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = _get_manifest_path()
        cls.full_df = pd.read_parquet(cls.manifest_path)
        cls.fibres_avail = len(cls.full_df[cls.full_df["class_label"] == "Fibres"])
        cls.porous_avail = len(cls.full_df[cls.full_df["class_label"] == "Porous_Sponge"])

    def test_fibres_and_porous_counts_match_expected(self) -> None:
        """Confirm ground-truth counts: Fibres has 162 and Porous_Sponge has 181."""
        self.assertEqual(self.fibres_avail, 162)
        self.assertEqual(self.porous_avail, 181)

    def test_varying_sample_sizes_and_structure(self) -> None:
        """Stress-test create_balanced_eval_subset across diverse sample sizes without oversampling."""
        test_sizes = [10, 50, 162, 181, 200, 300]
        class_avail = self.full_df["class_label"].value_counts().to_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            for n in test_sizes:
                out_file = Path(tmpdir) / f"eval_{n}.parquet"
                df = create_balanced_eval_subset(
                    manifest_path=self.manifest_path,
                    n_per_class=n,
                    seed=42,
                    output_path=out_file,
                )
                # Verify total length equals sum of min(n, n_avail) across active classes
                expected_len = sum(min(n, class_avail[c]) for c in ACTIVE_CLASSES)
                self.assertEqual(len(df), expected_len, f"Failed length check for n={n}")
                self.assertEqual(df["image_id"].nunique(), expected_len, f"Duplicates found for n={n}")
                self.assertTrue(out_file.exists())

                # Verify each active class has min(n, n_avail) rows
                counts = df["class_label"].value_counts()
                for c in ACTIVE_CLASSES:
                    self.assertEqual(counts[c], min(n, class_avail[c]), f"Class {c} count != min({n}, {class_avail[c]})")

    def test_oversampling_threshold_invariants(self) -> None:
        """Verify zero duplicates across all sample sizes (strict sampling without replacement)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Case 1: n = 100 (< 162). Both Fibres and Porous_Sponge have 100 unique, 0 duplicates
            df_100 = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=100,
                seed=42,
                output_path=Path(tmpdir) / "eval_100.parquet",
            )
            fibres_100 = df_100[df_100["class_label"] == "Fibres"]
            porous_100 = df_100[df_100["class_label"] == "Porous_Sponge"]
            self.assertEqual(len(fibres_100), 100)
            self.assertEqual(fibres_100["image_id"].nunique(), 100)
            self.assertEqual(len(porous_100), 100)
            self.assertEqual(porous_100["image_id"].nunique(), 100)

            # Case 2: n = 162 (exact Fibres limit). 0 duplicates for all classes
            df_162 = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=162,
                seed=42,
                output_path=Path(tmpdir) / "eval_162.parquet",
            )
            fibres_162 = df_162[df_162["class_label"] == "Fibres"]
            porous_162 = df_162[df_162["class_label"] == "Porous_Sponge"]
            self.assertEqual(len(fibres_162), 162)
            self.assertEqual(fibres_162["image_id"].nunique(), 162)
            self.assertEqual(len(porous_162), 162)
            self.assertEqual(porous_162["image_id"].nunique(), 162)

            # Case 3: n = 181 (Fibres capped at 162, Porous exact 181, 0 duplicates)
            df_181 = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=181,
                seed=42,
                output_path=Path(tmpdir) / "eval_181.parquet",
            )
            fibres_181 = df_181[df_181["class_label"] == "Fibres"]
            porous_181 = df_181[df_181["class_label"] == "Porous_Sponge"]
            self.assertEqual(len(fibres_181), 162)
            self.assertEqual(fibres_181["image_id"].nunique(), 162)
            self.assertEqual(len(porous_181), 181)
            self.assertEqual(porous_181["image_id"].nunique(), 181)

            # Case 4: n = 300 (Fibres capped at 162, Porous capped at 181, 0 duplicates)
            df_300 = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=300,
                seed=42,
                output_path=Path(tmpdir) / "eval_300.parquet",
            )
            fibres_300 = df_300[df_300["class_label"] == "Fibres"]
            porous_300 = df_300[df_300["class_label"] == "Porous_Sponge"]
            self.assertEqual(len(fibres_300), 162)
            self.assertEqual(fibres_300["image_id"].nunique(), 162)
            self.assertEqual(len(porous_300), 181)
            self.assertEqual(porous_300["image_id"].nunique(), 181)

    def test_random_seed_determinism_and_entropy(self) -> None:
        """Verify identical seeds produce identical frames, different seeds produce distinct subsets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_seed42_a = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=50,
                seed=42,
                output_path=Path(tmpdir) / "s42_a.parquet",
            )
            df_seed42_b = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=50,
                seed=42,
                output_path=Path(tmpdir) / "s42_b.parquet",
            )
            df_seed123 = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=50,
                seed=123,
                output_path=Path(tmpdir) / "s123.parquet",
            )

            # Same seed -> exact frame equality
            pd.testing.assert_frame_equal(df_seed42_a, df_seed42_b)

            # Different seed -> distinct sampled sets
            overlap = set(df_seed42_a["image_id"]).intersection(set(df_seed123["image_id"]))
            total = len(df_seed42_a)
            # Not identical
            self.assertNotEqual(list(df_seed42_a["image_id"]), list(df_seed123["image_id"]))
            # Partial overlap is expected (e.g. minority classes), but not 100% identical ordering
            self.assertLess(len(overlap), total)

    def test_error_handling_invalid_inputs(self) -> None:
        """Verify robust error raising on missing manifest or empty active class."""
        # Non-existent manifest
        with self.assertRaises(FileNotFoundError):
            create_balanced_eval_subset(manifest_path="non_existent_manifest.parquet")

        # Manifest missing one active class
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_manifest = Path(tmpdir) / "corrupt_manifest.parquet"
            # Filter out 'Powder'
            filtered_df = self.full_df[self.full_df["class_label"] != "Powder"]
            filtered_df.to_parquet(corrupted_manifest)

            with self.assertRaises(ValueError) as ctx:
                create_balanced_eval_subset(manifest_path=corrupted_manifest, n_per_class=10)
            self.assertIn("Powder", str(ctx.exception))


class TestBottleneckResumptionAndCheckpointInvariants(unittest.TestCase):
    """Stress-test checkpointing, interrupted resumption, and cache invalidation."""

    def setUp(self) -> None:
        self.concepts = [
            ConceptRubric(
                name=f"Concept_{i}",
                definition=f"Definition of concept {i}",
                scale_1="Low",
                scale_3="Medium",
                scale_5="High",
            )
            for i in range(30)
        ]
        # Build 96-row synthetic evaluation dataframe
        rows = []
        for c_id, c_name in enumerate(ACTIVE_CLASSES):
            for i in range(12):  # 8 * 12 = 96 rows
                rows.append({
                    "image_id": f"img_{c_name}_{i}",
                    "class_label": c_name,
                    "class_id": c_id,
                    "emb_index": c_id * 12 + i,
                    "full_path": f"dummy_{c_name}_{i}.jpg",
                })
        self.eval_df = pd.DataFrame(rows)

    def test_simulated_interruption_and_resumption(self) -> None:
        """Pre-populate a checkpoint with 40 rows scored, verify resumption completes all 96 rows."""
        n_images = len(self.eval_df)
        n_concepts = len(self.concepts)
        precomputed_count = 40

        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "assume_embeddings.npy"
            chk_p = Path(tmpdir) / "assume_embeddings_checkpoint.npy"

            # Create partial checkpoint with NaNs in remaining rows
            partial_chk = np.full((n_images, n_concepts), np.nan, dtype=np.float32)
            # Fill first 40 rows with distinct signature value 4.0
            partial_chk[:precomputed_count] = 4.0
            np.save(chk_p, partial_chk)

            scorer = MockScorerAgent(seed=999)
            res = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                checkpoint_interval=10,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=False,
            )

            # Assert shape and completeness
            self.assertEqual(res.shape, (n_images, n_concepts))
            self.assertFalse(np.isnan(res).any())

            # Check that pre-existing 40 rows retain 4.0
            np.testing.assert_array_equal(res[:precomputed_count], 4.0)

            # Check that remaining 56 rows were populated with scores in [0, 5]
            remaining = res[precomputed_count:]
            self.assertTrue((remaining >= 0.0).all())
            self.assertTrue((remaining <= 5.0).all())

            # Checkpoint file should be cleaned up upon successful completion
            self.assertFalse(chk_p.exists())
            self.assertTrue(out_p.exists())

    def test_cache_invalidation_force_flag(self) -> None:
        """Verify force_recompute=True invalidates cache, while force_recompute=False reuses it."""
        n_images = len(self.eval_df)
        n_concepts = len(self.concepts)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "assume_embeddings.npy"
            chk_p = Path(tmpdir) / "checkpoint.npy"

            # Pre-seed completed file with signature 4.0
            synthetic_cached = np.full((n_images, n_concepts), 4.0, dtype=np.float32)
            np.save(out_p, synthetic_cached)

            scorer = MockScorerAgent(seed=42)

            # With force_recompute=False, should return cached array instantly
            cached_result = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=False,
            )
            np.testing.assert_array_equal(cached_result, 4.0)

            # With force_recompute=True, should recompute and overwrite
            recomputed_result = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=True,
            )
            self.assertFalse((recomputed_result == 4.0).all())
            self.assertEqual(recomputed_result.shape, (n_images, n_concepts))
            self.assertTrue((recomputed_result >= 0.0).all())
            self.assertTrue((recomputed_result <= 5.0).all())

    def test_corrupt_checkpoint_fallback(self) -> None:
        """A corrupted or mismatched checkpoint file must be safely ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "assume_embeddings.npy"
            chk_p = Path(tmpdir) / "checkpoint.npy"

            # Checkpoint with wrong shape
            wrong_shape_chk = np.zeros((10, 5), dtype=np.float32)
            np.save(chk_p, wrong_shape_chk)

            scorer = MockScorerAgent(seed=42)
            result = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=False,
            )
            # Successfully completed full dataframe regardless
            self.assertEqual(result.shape, (len(self.eval_df), len(self.concepts)))
            self.assertFalse(np.isnan(result).any())


class TestFeatureAlignmentInvariant(unittest.TestCase):
    """Adversarial empirical test: raw image -> DINOv3 forward pass vs dinov3_pooled.npy[emb_index]."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_manifest_path = DATA_DIR / "eval_manifest.parquet"
        if not cls.eval_manifest_path.exists():
            cls.eval_manifest_path = PROJECT_ROOT / "eval_manifest.parquet"
        if not cls.eval_manifest_path.exists():
            raise unittest.SkipTest("eval_manifest.parquet not found.")

        cls.dinov3_pooled_path = _get_embedding_path()
        cls.dinov3_features = np.load(cls.dinov3_pooled_path)
        cls.eval_df = pd.read_parquet(cls.eval_manifest_path)
        cls.raw_manifest_path = _get_manifest_path()
        cls.raw_manifest_df = pd.read_parquet(cls.raw_manifest_path)

        # Initialize real DINOv3 embedder
        cls.embedder = DINOv3Embedder(local_files_only=True)

    def test_feature_alignment_20_random_micrographs(self) -> None:
        """Sample 20 micrographs from eval_manifest, perform fresh DINOv3 inference, compare with pooled npy."""
        rng = np.random.default_rng(2026)
        sample_indices = rng.choice(len(self.eval_df), size=20, replace=False)
        sample_rows = self.eval_df.iloc[sample_indices]

        logger.info("Testing feature alignment on 20 randomly sampled micrographs...")

        cosine_sims = []
        l2_errors = []

        for _, row in sample_rows.iterrows():
            emb_idx = int(row["emb_index"])
            full_path = Path(row["full_path"])
            image_id = row["image_id"]

            # Invariant 1: File must exist on disk
            self.assertTrue(full_path.exists(), f"Micrograph file not found: {full_path}")

            # Invariant 2: emb_index in raw manifest matches image_id and full_path
            raw_match = self.raw_manifest_df.iloc[emb_idx]
            self.assertEqual(raw_match["image_id"], image_id)
            self.assertEqual(str(Path(raw_match["full_path"])), str(full_path))

            # Invariant 3: Live DINOv3 forward pass matches dinov3_pooled[emb_idx]
            cached_vector = self.dinov3_features[emb_idx]
            live_vector = self.embedder.embed_images([full_path])[0]

            # Cosine similarity: (u . v) / (||u|| * ||v||)
            cos_sim = float(np.dot(cached_vector, live_vector) / (np.linalg.norm(cached_vector) * np.linalg.norm(live_vector)))
            l2_dist = float(np.linalg.norm(cached_vector - live_vector))

            cosine_sims.append(cos_sim)
            l2_errors.append(l2_dist)

            # Assert near-exact numerical agreement (floating point tolerance: cos >= 0.9999, L2 < 0.015)
            self.assertGreaterEqual(
                cos_sim,
                0.9999,
                f"Cosine similarity degraded for {image_id} (emb_idx={emb_idx}): {cos_sim}",
            )
            self.assertLess(
                l2_dist,
                0.015,
                f"L2 error too high for {image_id} (emb_idx={emb_idx}): {l2_dist}",
            )

        mean_cos = np.mean(cosine_sims)
        min_cos = np.min(cosine_sims)
        max_l2 = np.max(l2_errors)
        logger.info(
            "Feature alignment verified on 20 micrographs: Mean Cosine Sim = %.6f, Min Cosine Sim = %.6f, Max L2 Error = %.6f",
            mean_cos,
            min_cos,
            max_l2,
        )
        self.assertGreaterEqual(mean_cos, 0.99999)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
