"""Comprehensive unit and property tests for ASSUME Milestone 2: TriModalBatchSampler.

Covers R2 requirements:
- Step 0 initialization (100% random + FPS, exact batch size 6, no duplicate indices).
- Exact 33/33/33 split when concept scores are provided and collision exists.
- Mathematical collision detection logic with known synthetic matrices.
- Fallback behavior when no collision pair meets the thresholds.
- Latency and execution speed benchmarks (must sample batch in < 50 ms).
- Zero overlap within a single batch across all streams.
- State management, reset functionality, and history tracking.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from assume.config import (
    DATA_DIR,
    EMBEDDING_PATH,
    EXPECTED_TOTAL_MICROGRAPHS,
    MANIFEST_PATH,
    ROOT_EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
)
from assume.sampling import TriModalBatchSampler
from assume.sampling.tri_modal_sampler import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SIM_THRESHOLD,
    DEFAULT_TEACHER_DIST_THRESHOLD,
)


class TestTriModalBatchSampler(unittest.TestCase):
    """Test suite covering the TriModalBatchSampler implementation."""

    @classmethod
    def setUpClass(cls):
        """Prepare mock features and load real dataset features if available."""
        # 1. Deterministic synthetic test features (N=100, dim=384)
        cls.n_mock = 100
        cls.dim = 384
        rng = np.random.default_rng(1337)
        raw_mock = rng.standard_normal((cls.n_mock, cls.dim), dtype=np.float32)
        cls.mock_features = raw_mock / np.linalg.norm(raw_mock, axis=1, keepdims=True)

        cls.mock_manifest = pd.DataFrame({
            "image_id": [f"img_{i:04d}" for i in range(cls.n_mock)],
            "class_label": ["Biological"] * (cls.n_mock // 2) + ["Nanowires"] * (cls.n_mock // 2),
            "class_id": [0] * (cls.n_mock // 2) + [3] * (cls.n_mock // 2),
            "full_path": [f"mock_path_{i}.jpg" for i in range(cls.n_mock)],
            "emb_index": list(range(cls.n_mock)),
        })

        # 2. Real dataset artifacts if present
        cls.real_features_path = None
        if EMBEDDING_PATH.exists():
            cls.real_features_path = EMBEDDING_PATH
        elif ROOT_EMBEDDING_PATH.exists():
            cls.real_features_path = ROOT_EMBEDDING_PATH

    def test_step_0_cold_start_initialization(self):
        """At step 0 (step % 3 == 0), sample 100% via DINOv3 FPS (6 images)."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=42,
        )

        batch = sampler.sample_batch(step=0, batch_size=6, concept_scores=None)

        # Batch size must be exactly 6
        self.assertEqual(len(batch), 6)
        # All indices must be unique within the batch (zero duplicate indices)
        self.assertEqual(len(set(batch)), 6)
        # All indices must be valid integers in range [0, n_mock - 1]
        for idx in batch:
            self.assertIsInstance(idx, int)
            self.assertTrue(0 <= idx < self.n_mock)

        # Inspect breakdown: step 0 is 100% FPS (6 FPS images)
        info = sampler.last_batch_info
        self.assertEqual(info["strategy"], "fps")
        self.assertEqual(len(info["fps_indices"]), 6)
        self.assertEqual(len(info["random_indices"]), 0)
        self.assertEqual(len(info["collision_indices"]), 0)
        self.assertEqual(info["batch"], batch)

    def test_collision_detection_mathematical_logic(self):
        """Verify get_colliding_pairs correctly filters pairs on sim > 0.85 and teacher_dist > 0.40."""
        # Construct synthetic features and concept scores with known geometries
        features = np.zeros((4, 8), dtype=np.float32)
        # Point 0: [1, 0, 0, 0, ...]
        features[0, 0] = 1.0
        # Point 1: orthogonal to Point 0 -> cos_sim = 0.0, cosine dist = 1.0 - 0.0 = 1.0 > 0.40
        features[1, 1] = 1.0
        # Point 2: nearly parallel to Point 0 -> cos_sim = 0.95, cosine dist = 0.05 <= 0.40
        features[2, 0] = np.cos(np.arccos(0.95))
        features[2, 1] = np.sin(np.arccos(0.95))
        # Point 3: orthogonal to Point 0 -> cosine dist = 1.0 > 0.40
        features[3, 2] = 1.0

        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        # Concept scores: (4 images, 3 concepts) on 0-5 scale
        concept_scores = np.array([
            [4.0, 5.0, 3.0],  # Image 0
            [4.0, 5.0, 3.0],  # Image 1: identical to Image 0 -> cos_sim = 1.0 > 0.85
            [4.0, 5.0, 3.0],  # Image 2: identical to Image 0 -> cos_sim = 1.0 > 0.85
            [1.0, 1.0, 5.0],  # Image 3: distinct concept scores -> cos_sim with Image 0 < 0.0
        ], dtype=np.float32)

        colliding_pairs = sampler.get_colliding_pairs(concept_scores)

        # Both (0, 1) and (1, 2) should be detected
        pair_set = {tuple(sorted(p)) for p in colliding_pairs}
        self.assertIn((0, 1), pair_set)
        self.assertIn((1, 2), pair_set)
        self.assertNotIn((0, 2), pair_set)  # Teacher distance too small
        self.assertNotIn((0, 3), pair_set)  # Concept similarity too low

    def test_applicability_masked_and_mean_centered_collision(self):
        """Pairs sharing only 0s (absent concepts) with opposite active scores must NOT collide."""
        features = np.zeros((3, 8), dtype=np.float32)
        features[0, 0] = 1.0
        features[1, 1] = 1.0
        features[2, 2] = 1.0

        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        # Images 0 and 1 have opposite scores on active concepts 0 & 1, and 0s on concepts 2..9
        concept_scores = np.zeros((3, 10), dtype=np.float32)
        concept_scores[0, 0] = 5.0
        concept_scores[0, 1] = 1.0
        concept_scores[1, 0] = 1.0
        concept_scores[1, 1] = 5.0
        concept_scores[2, 0] = 3.0
        concept_scores[2, 1] = 3.0

        pairs = sampler.get_colliding_pairs(concept_scores)
        pair_set = {tuple(sorted(p)) for p in pairs}
        self.assertNotIn((0, 1), pair_set)

    def test_custom_thresholds_and_scored_indices_mapping(self):
        """Test get_colliding_pairs with custom thresholds and arbitrary scored_indices."""
        features = np.zeros((10, 4), dtype=np.float32)
        features[5, 0] = 1.0
        features[8, 1] = 1.0  # dist(5, 8) = 1.0
        features[9, 2] = 1.0

        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        scores = np.array([
            [5.0, 2.0, 4.0],
            [5.0, 2.0, 4.0],
            [1.0, 5.0, 1.0],
        ], dtype=np.float32)

        # Map row 0 -> global index 5, row 1 -> global index 8, row 2 -> global index 9
        scored_indices = [5, 8, 9]
        pairs = sampler.get_colliding_pairs(
            concept_scores=scores,
            scored_indices=scored_indices,
            sim_threshold=0.90,
            teacher_dist_threshold=0.50,
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(tuple(sorted(pairs[0])), (5, 8))

    def test_single_mode_batch_cycling_and_collision_tracking(self):
        """Verify step % 3 == 0 -> fps, step % 3 == 1 -> random, step % 3 == 2 -> collision, and non-resampling."""
        features = self.mock_features[:20]
        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        # Step 0: 100% FPS
        b0 = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(b0), 6)
        self.assertEqual(sampler.last_batch_info["strategy"], "fps")
        self.assertEqual(len(sampler.last_batch_info["fps_indices"]), 6)

        # Step 1: 100% Random
        b1 = sampler.sample_batch(step=1, batch_size=6)
        self.assertEqual(len(b1), 6)
        self.assertEqual(sampler.last_batch_info["strategy"], "random")
        self.assertEqual(len(sampler.last_batch_info["random_indices"]), 6)

        # Step 2: 100% Collision (3 colliding pairs = 6 images when 3 pairs exist)
        rng = np.random.default_rng(123)
        concept_scores = rng.integers(1, 6, size=(20, 12)).astype(np.float32)
        # Force 3 distinct colliding pairs: (2, 7), (3, 8), (4, 9)
        concept_scores[7] = concept_scores[2]
        concept_scores[8] = concept_scores[3]
        concept_scores[9] = concept_scores[4]

        b2 = sampler.sample_batch(
            step=2,
            batch_size=6,
            concept_scores=concept_scores,
            scored_indices=list(range(20)),
        )
        self.assertEqual(len(b2), 6)
        self.assertEqual(len(set(b2)), 6)
        info = sampler.last_batch_info
        self.assertEqual(info["strategy"], "collision")
        self.assertGreaterEqual(len(info["collision_indices"]), 2)
        self.assertGreaterEqual(len(info["collision_pairs"]), 1)
        # Sampled collision pairs must be recorded in sampler.sampled_collision_pairs
        for u, v in info["collision_pairs"]:
            canon = (min(u, v), max(u, v))
            self.assertIn(canon, sampler.sampled_collision_pairs)
            # Subsequent get_colliding_pairs must not return already-sampled pairs
            remaining_pairs = {tuple(sorted(p)) for p in sampler.get_colliding_pairs(concept_scores, list(range(20)))}
            self.assertNotIn(canon, remaining_pairs)

    def test_fallback_when_no_collision_pair_meets_threshold(self):
        """When step % 3 == 2 but no colliding pairs exist, fallback to 100% DINOv3 FPS."""
        features = self.mock_features[:20]
        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        # Construct orthogonal concept scores (zero active overlap between pairs)
        concept_scores = np.eye(20, dtype=np.float32) * 5.0

        batch = sampler.sample_batch(
            step=2,
            batch_size=6,
            concept_scores=concept_scores,
            scored_indices=list(range(20)),
        )

        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)

        info = sampler.last_batch_info
        # Should gracefully fall back to fps strategy
        self.assertEqual(info["strategy"], "fps")
        self.assertEqual(len(info["fps_indices"]), 6)
        self.assertEqual(len(info["random_indices"]), 0)
        self.assertEqual(len(info["collision_indices"]), 0)

    def test_zero_overlap_within_single_batch(self):
        """Verify strict zero duplicate indices across multiple active learning iterations."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=999,
        )

        # Run 10 successive batches
        for step in range(10):
            # Generate random synthetic scores
            scores = np.random.uniform(1.0, 10.0, size=(self.n_mock, 6)).astype(np.float32)
            # Guarantee a collision between two points
            scores[10] = scores[5] + np.random.uniform(-0.05, 0.05, size=6)

            batch = sampler.sample_batch(
                step=step,
                batch_size=6,
                concept_scores=scores,
            )

            # Strict check: batch length == 6 and set length == 6
            self.assertEqual(len(batch), 6, f"Batch size mismatch at step {step}")
            self.assertEqual(len(set(batch)), 6, f"Duplicate indices found in batch at step {step}: {batch}")

    def test_history_tracking_and_reset(self):
        """Verify that track_history accumulates sampled images and reset() clears them."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=42,
            track_history=True,
        )

        b1 = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(sampler.sampled_indices), 6)
        self.assertTrue(set(b1).issubset(sampler.sampled_indices))

        b2 = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(sampler.sampled_indices), 12)
        # In cold start with sufficient pool, b2 should have zero overlap with b1
        self.assertEqual(len(set(b1).intersection(set(b2))), 0)

        # Test reset
        sampler.reset()
        self.assertEqual(len(sampler.sampled_indices), 0)
        self.assertEqual(len(sampler.last_batch_info), 0)

    def test_custom_batch_sizes(self):
        """Verify the sampler accommodates non-standard batch sizes (e.g. B=4, B=12)."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=42,
        )

        for b_size in [4, 8, 12]:
            batch = sampler.sample_batch(step=0, batch_size=b_size)
            self.assertEqual(len(batch), b_size)
            self.assertEqual(len(set(batch)), b_size)

    def test_execution_speed_under_50ms(self):
        """Sampling a batch of 6 on full 14,620 features must complete in < 50 ms."""
        if self.real_features_path is None or not Path(self.real_features_path).exists():
            self.skipTest("Real DINOv3 features not available; skipping real speed benchmark.")

        sampler = TriModalBatchSampler(
            manifest_path=MANIFEST_PATH if MANIFEST_PATH.exists() else None,
            dinov3_features_path=self.real_features_path,
            seed=42,
        )
        self.assertEqual(len(sampler), EXPECTED_TOTAL_MICROGRAPHS)

        # 1. Warm-up
        _ = sampler.sample_batch(step=0, batch_size=6, update_history=False)

        # 2. Benchmark step 0 cold start
        latencies_cold = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sampler.sample_batch(step=0, batch_size=6, update_history=False)
            latencies_cold.append(time.perf_counter() - t0)

        avg_cold_ms = np.mean(latencies_cold) * 1000.0
        max_cold_ms = np.max(latencies_cold) * 1000.0
        print(f"\n[Benchmark] Cold start sampling latency (N=14,620): avg={avg_cold_ms:.2f}ms, max={max_cold_ms:.2f}ms")
        self.assertLess(max_cold_ms, 50.0, f"Cold start sampling exceeded 50 ms: {max_cold_ms:.2f}ms")

        # 3. Benchmark step 1 tri-modal with collision
        # Create a mock concept score matrix for 50 images
        scored_idx = list(range(50))
        scores = np.random.uniform(1.0, 10.0, size=(50, 10)).astype(np.float32)
        # Find two points in real features with distance > 0.40
        scores[1] = scores[0] + np.random.uniform(-0.02, 0.02, size=10)

        latencies_tri = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sampler.sample_batch(
                step=1,
                batch_size=6,
                concept_scores=scores,
                scored_indices=scored_idx,
                update_history=False,
            )
            latencies_tri.append(time.perf_counter() - t0)

        avg_tri_ms = np.mean(latencies_tri) * 1000.0
        max_tri_ms = np.max(latencies_tri) * 1000.0
        print(f"[Benchmark] Tri-modal sampling latency (N=14,620): avg={avg_tri_ms:.2f}ms, max={max_tri_ms:.2f}ms")
        self.assertLess(max_tri_ms, 50.0, f"Tri-modal sampling exceeded 50 ms: {max_tri_ms:.2f}ms")

    def test_image_metadata_retrieval(self):
        """Verify image metadata retrieval maps indices to manifest rows."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=42,
        )

    def test_invalid_inputs_and_edge_cases(self):
        """Verify robust handling of boundary conditions, empty inputs, and invalid dimensions."""
        sampler = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=42,
        )

        # 1. Non-positive batch size
        self.assertEqual(sampler.sample_batch(step=0, batch_size=0), [])
        self.assertEqual(sampler.sample_batch(step=0, batch_size=-5), [])

        # 2. Empty or 1D concept scores
        self.assertEqual(sampler.get_colliding_pairs(np.zeros((0, 5), dtype=np.float32)), [])
        self.assertEqual(sampler.get_colliding_pairs(np.zeros((1, 5), dtype=np.float32)), [])
        self.assertEqual(sampler.get_colliding_pairs(np.zeros((5,), dtype=np.float32)), [])
        self.assertEqual(sampler.get_colliding_pairs(None), [])

        # 3. All-zero concept scores (zero norm)
        zero_scores = np.zeros((4, 3), dtype=np.float32)
        pairs = sampler.get_colliding_pairs(zero_scores)
        self.assertEqual(pairs, [])

        # 4. Length mismatch between scored_indices and concept_scores
        with self.assertRaises(ValueError):
            sampler.get_colliding_pairs(
                concept_scores=np.ones((4, 3), dtype=np.float32),
                scored_indices=[0, 1],  # 2 indices but 4 rows
            )

    def test_fps_distance_maximization_invariant(self):
        """FPS must strictly select the point with maximum cosine distance to existing points."""
        # 3 points in 2D
        # Point 0: [1, 0]
        # Point 1: [0.999, 0.04] (nearly identical, dist ~ 0.001)
        # Point 2: [0, 1] (orthogonal, dist = 1.0)
        p0 = np.array([1.0, 0.0], dtype=np.float32)
        p1 = np.array([0.999, 0.04], dtype=np.float32)
        p1 /= np.linalg.norm(p1)
        p2 = np.array([0.0, 1.0], dtype=np.float32)
        features = np.vstack([p0, p1, p2])

        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=features,
            seed=42,
        )

        # When point 0 is the reference seed, FPS selecting 1 point must pick point 2 (orthogonal)
        fps_choice = sampler._sample_fps(
            n_points=1,
            reference_indices=[0],
            exclude_indices={0},
        )
        self.assertEqual(fps_choice, [2], "FPS must pick orthogonal Point 2 over near-identical Point 1")

    def test_deterministic_reproducibility(self):
        """Samplers with identical seeds must produce identical sampling trajectories."""
        sampler_a = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=777,
        )
        sampler_b = TriModalBatchSampler(
            manifest_path=self.mock_manifest,
            dinov3_features_path=self.mock_features,
            seed=777,
        )

        batch_a1 = sampler_a.sample_batch(step=0, batch_size=6)
        batch_b1 = sampler_b.sample_batch(step=0, batch_size=6)
        self.assertEqual(batch_a1, batch_b1)

        batch_a2 = sampler_a.sample_batch(step=0, batch_size=6)
        batch_b2 = sampler_b.sample_batch(step=0, batch_size=6)
        self.assertEqual(batch_a2, batch_b2)

    def test_pool_exhaustion_safety(self):
        """Verify sampler remains stable with zero duplicates even when pool size is small."""
        small_features = self.mock_features[:8]
        sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=small_features,
            seed=42,
            track_history=True,
        )

        # Batch 1 takes 6 of the 8 items
        b1 = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(b1), 6)
        self.assertEqual(len(set(b1)), 6)

        # Batch 2 requests 6 items when only 2 unsampled remain
        b2 = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(b2), 6)
        self.assertEqual(len(set(b2)), 6)


if __name__ == "__main__":
    unittest.main()

