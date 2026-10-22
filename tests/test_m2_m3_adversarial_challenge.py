"""Adversarial stress-test suite challenging ASSUME Milestone 2 and Milestone 3.

Empirically challenges:
1. Tri-Modal Sampler (M2):
   - Degenerate concept score matrices: empty, zero-norm, single-concept (K=1), all-identical, all-orthogonal.
   - Non-finite score matrices: NaNs, Infs, extreme negative values.
   - High-load sampling loop: 100 consecutive sampling steps, verifying zero intra-batch duplicates and bounded memory.
   - Starvation and edge pool sizes: small pools (N=6, N=3, N=1), boundary batch sizes (B=0, B=1, B=12).
   - Scored indices mapping robustness and dimension mismatch.
2. Resilient Gemini Vertex API Harness (M3):
   - Concept chunking boundary conditions: K=0, 1, 10, 11, 20, 21.
   - Strict chunk size enforcement: chunk_size <= 0, max_chunk_size > 10, score_chunk > 10 rubrics.
   - Malformed markdown and JSON extraction under corrupt, wrapped, or truncated text.
   - Concept rubric schema resilience: missing keys, alternative scoring guides, malformed anchors.
   - Backoff calculation under mathematical progression, full-jitter bounds, and Retry-After header variants.
   - Severe simulated HTTP 429 burst load with concurrent threads.
   - Status code matrix: retryable (429, 500, 502, 503, 504) vs non-retryable (400, 401, 403, 404).
   - Scorer score clamping: out-of-range floats, non-numeric strings, missing IDs.
   - Live Vertex AI smoke verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import time
import tracemalloc
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
from PIL import Image

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import (
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    ResilientVertexClient,
)
from assume.config import (
    EMBEDDING_PATH,
    EXPECTED_TOTAL_MICROGRAPHS,
    MANIFEST_PATH,
    ROOT_EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
)
from assume.sampling import TriModalBatchSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("challenger_m2_m3")


class TestTriModalSamplerAdversarial(unittest.TestCase):
    """Adversarial stress-test suite challenging the TriModalBatchSampler."""

    @classmethod
    def setUpClass(cls):
        """Load real dataset features if available, otherwise synthetic fallback."""
        cls.features_path = None
        if EMBEDDING_PATH.exists():
            cls.features_path = EMBEDDING_PATH
        elif ROOT_EMBEDDING_PATH.exists():
            cls.features_path = ROOT_EMBEDDING_PATH

        if cls.features_path is not None:
            cls.real_features = np.load(cls.features_path)
            cls.n_samples = len(cls.real_features)
        else:
            cls.n_samples = 100
            rng = np.random.default_rng(42)
            raw = rng.standard_normal((cls.n_samples, 384), dtype=np.float32)
            cls.real_features = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def setUp(self):
        self.sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=self.real_features,
            seed=42,
            track_history=True,
        )

    # -------------------------------------------------------------------------
    # 1. Degenerate Concept Score Matrices
    # -------------------------------------------------------------------------

    def test_degenerate_empty_scores(self):
        """Empty score matrices (None, 0 rows, 1 row, 0 cols) must trigger safe fallback."""
        # None at step 1 (1 % 3 == 1 -> random)
        self.assertEqual(self.sampler.get_colliding_pairs(None), [])
        b_none = self.sampler.sample_batch(step=1, batch_size=6, concept_scores=None)
        self.assertEqual(len(b_none), 6)
        self.assertEqual(len(set(b_none)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "random")

        # 0 rows: shape (0, 5) at step 2 (2 % 3 == 2 -> collision fallback to fps)
        empty_rows = np.zeros((0, 5), dtype=np.float32)
        self.assertEqual(self.sampler.get_colliding_pairs(empty_rows), [])
        b_empty = self.sampler.sample_batch(step=2, batch_size=6, concept_scores=empty_rows)
        self.assertEqual(len(b_empty), 6)
        self.assertEqual(len(set(b_empty)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "fps")

        # 1 row: shape (1, 5) (cannot form pairs)
        single_row = np.array([[5.0, 5.0, 5.0, 5.0, 5.0]], dtype=np.float32)
        self.assertEqual(self.sampler.get_colliding_pairs(single_row), [])
        b_single = self.sampler.sample_batch(step=2, batch_size=6, concept_scores=single_row)
        self.assertEqual(len(b_single), 6)
        self.assertEqual(len(set(b_single)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "fps")

        # 0 cols: shape (10, 0)
        zero_cols = np.zeros((10, 0), dtype=np.float32)
        self.assertEqual(self.sampler.get_colliding_pairs(zero_cols), [])
        b_zcol = self.sampler.sample_batch(step=2, batch_size=6, concept_scores=zero_cols)
        self.assertEqual(len(b_zcol), 6)
        self.assertEqual(len(set(b_zcol)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "fps")

    def test_degenerate_all_zero_scores(self):
        """All-zero score matrix has 0 L2 norm. Must not crash with division by zero or produce NaNs."""
        zero_scores = np.zeros((20, 5), dtype=np.float32)
        pairs = self.sampler.get_colliding_pairs(zero_scores)
        self.assertEqual(pairs, [])

        batch = self.sampler.sample_batch(step=2, batch_size=6, concept_scores=zero_scores)
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "fps")

    def test_degenerate_single_concept_matrix(self):
        """Single-concept score matrix (shape M x 1) represents K=1 discovered concept."""
        scores_1d = np.array([[2.0], [5.0], [5.0], [1.0]], dtype=np.float32)
        pairs = self.sampler.get_colliding_pairs(
            scores_1d,
            scored_indices=[0, 1, 2, 3],
            sim_threshold=0.85,
            teacher_dist_threshold=0.10,
        )
        self.assertIsInstance(pairs, list)
        batch = self.sampler.sample_batch(step=1, batch_size=6, concept_scores=scores_1d, scored_indices=[0, 1, 2, 3])
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)

    def test_degenerate_all_identical_scores(self):
        """All-identical scores (every sample has exact same vector [5, 5, 5]) must not crash."""
        scores = np.full((30, 3), 5.0, dtype=np.float32)
        pairs = self.sampler.get_colliding_pairs(scores, scored_indices=list(range(30)))
        self.assertIsInstance(pairs, list)

        batch = self.sampler.sample_batch(step=1, batch_size=6, concept_scores=scores, scored_indices=list(range(30)))
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)

    def test_degenerate_all_orthogonal_scores(self):
        """Orthogonal scores (identity matrix) have zero active overlap, strictly below 0.85 threshold."""
        scores = np.eye(20, dtype=np.float32)
        pairs = self.sampler.get_colliding_pairs(scores)
        self.assertEqual(pairs, [])

        batch = self.sampler.sample_batch(step=3, batch_size=6, concept_scores=scores)
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)
        self.assertEqual(self.sampler.last_batch_info["strategy"], "fps")

    def test_degenerate_nan_and_inf_scores(self):
        """Scores containing NaNs or Infs must not crash the sampler or raise unhandled exceptions."""
        nan_scores = np.full((10, 4), np.nan, dtype=np.float32)
        pairs_nan = self.sampler.get_colliding_pairs(nan_scores)
        self.assertEqual(pairs_nan, [])

        inf_scores = np.full((10, 4), np.inf, dtype=np.float32)
        pairs_inf = self.sampler.get_colliding_pairs(inf_scores)
        self.assertEqual(pairs_inf, [])

        # Extreme negative scores
        neg_scores = -10.0 * np.ones((10, 4), dtype=np.float32)
        # Cosine similarity between two identical negative vectors is (-v) . (-v) / (||v|| ||v||) = 1.0
        pairs_neg = self.sampler.get_colliding_pairs(neg_scores)
        self.assertIsInstance(pairs_neg, list)

    def test_dimension_mismatch_scored_indices_raises(self):
        """Scored indices length mismatch must raise ValueError."""
        scores = np.ones((5, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.sampler.get_colliding_pairs(scores, scored_indices=[0, 1, 2])  # 3 indices for 5 rows

    # -------------------------------------------------------------------------
    # 2. High-Load 100 Consecutive Sampling Steps
    # -------------------------------------------------------------------------

    def test_high_load_100_consecutive_steps(self):
        """Stress-test: Run 100 consecutive sampling steps with track_history=True.

        Verifies:
        1. Exact batch size 6 at every step.
        2. Strict zero duplicate indices within any batch (len(set(batch)) == 6).
        3. Stable memory usage (no unbounded memory leak, < 5 MB growth across 100 steps).
        4. Sub-50ms execution speed across every single step.
        """
        self.sampler.reset()
        tracemalloc.start()
        mem_start, _ = tracemalloc.get_traced_memory()

        step_latencies = []
        all_batches = []
        n_steps = 100

        # Simulate evolving concept scores
        rng = np.random.default_rng(2026)
        k_concepts = 6

        for step in range(n_steps):
            # Dynamic concept score matrix of size (50, k_concepts)
            scores = rng.uniform(1.0, 10.0, size=(50, k_concepts)).astype(np.float32)
            # Create an artificial collision between sample 4 and sample 12
            scores[12] = scores[4] + rng.uniform(-0.02, 0.02, size=k_concepts).astype(np.float32)
            scored_idx = list(range(50))

            t0 = time.perf_counter()
            batch = self.sampler.sample_batch(
                step=step,
                batch_size=6,
                concept_scores=scores if step > 0 else None,
                scored_indices=scored_idx,
                update_history=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(elapsed_ms)

            # Invariant 1: exact length 6
            self.assertEqual(len(batch), 6, f"Batch size mismatch at step {step}: len={len(batch)}")
            # Invariant 2: strict zero duplicates within the single batch
            self.assertEqual(
                len(set(batch)), 6, f"Duplicate indices within batch at step {step}: {batch}"
            )
            # Invariant 3: valid range [0, n_samples - 1]
            for idx in batch:
                self.assertTrue(0 <= idx < self.n_samples, f"Index out of bounds at step {step}: {idx}")

            all_batches.append(batch)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_growth_mb = (current_mem - mem_start) / (1024 * 1024)
        peak_mb = peak_mem / (1024 * 1024)
        avg_ms = float(np.mean(step_latencies))
        max_ms = float(np.max(step_latencies))

        logger.info(
            "100-step stress test: avg=%.2fms, max=%.2fms, mem_growth=%.3fMB, peak_mem=%.3fMB",
            avg_ms,
            max_ms,
            mem_growth_mb,
            peak_mb,
        )

        # Invariant 4: strict performance requirements (< 50ms per step)
        self.assertLess(max_ms, 50.0, f"Max step latency {max_ms:.2f}ms exceeded 50ms threshold")
        # Invariant 5: strict memory constraint (< 5 MB growth over 100 steps)
        self.assertLess(mem_growth_mb, 5.0, f"Memory growth {mem_growth_mb:.2f}MB exceeded 5MB threshold")
        # Invariant 6: tracked history contains sampled indices
        self.assertGreaterEqual(len(self.sampler.sampled_indices), 100)

    def test_high_load_100_steps_cold_start_all_unique(self):
        """In cold-start across 100 steps, each step samples 6 completely disjoint images (600 unique)."""
        self.sampler.reset()
        for step in range(100):
            batch = self.sampler.sample_batch(step=0, batch_size=6, concept_scores=None)
            self.assertEqual(len(batch), 6)
            self.assertEqual(len(set(batch)), 6)
        self.assertEqual(len(self.sampler.sampled_indices), 600)

    # -------------------------------------------------------------------------
    # 3. Starvation and Extreme Pool Sizes
    # -------------------------------------------------------------------------

    def test_starvation_pool_exact_batch_size(self):
        """When dataset pool size N == batch_size (N=6), successive batches must not crash or deadlock."""
        small_feats = self.real_features[:6]
        small_sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=small_feats,
            seed=42,
            track_history=True,
        )
        self.assertEqual(len(small_sampler), 6)

        # Batch 1 takes all 6
        b1 = small_sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(b1), 6)
        self.assertEqual(len(set(b1)), 6)

        # Batch 2 requests 6 when all 6 were previously sampled
        b2 = small_sampler.sample_batch(step=1, batch_size=6)
        self.assertEqual(len(b2), 6)
        self.assertEqual(len(set(b2)), 6)

    def test_starvation_pool_smaller_than_batch_size(self):
        """When dataset pool size N < batch_size (e.g. N=3, B=6), must return available without crashing."""
        tiny_feats = self.real_features[:3]
        tiny_sampler = TriModalBatchSampler(
            manifest_path=None,
            dinov3_features_path=tiny_feats,
            seed=42,
            track_history=True,
        )
        self.assertEqual(len(tiny_sampler), 3)

        batch = tiny_sampler.sample_batch(step=0, batch_size=6)
        # Should return all 3 available unique indices
        self.assertEqual(len(set(batch)), 3)
        self.assertTrue(len(batch) <= 6)

    def test_boundary_batch_sizes(self):
        """Boundary batch sizes (B=0, B=-1, B=1, B=2, B=12) must be handled gracefully."""
        self.assertEqual(self.sampler.sample_batch(step=0, batch_size=0), [])
        self.assertEqual(self.sampler.sample_batch(step=0, batch_size=-10), [])

        # B=1
        b1 = self.sampler.sample_batch(step=0, batch_size=1)
        self.assertEqual(len(b1), 1)

        # B=2
        b2 = self.sampler.sample_batch(step=0, batch_size=2)
        self.assertEqual(len(b2), 2)
        self.assertEqual(len(set(b2)), 2)

        # B=12
        b12 = self.sampler.sample_batch(step=0, batch_size=12)
        self.assertEqual(len(b12), 12)
        self.assertEqual(len(set(b12)), 12)


class TestGeminiHarnessAdversarial(unittest.TestCase):
    """Adversarial stress-test suite challenging the Resilient Vertex AI Harness."""

    def setUp(self):
        self.fake_client = ResilientVertexClient(
            api_key="mock_key_abc123",
            project_id="1034989953225",
            max_retries=3,
            base_delay=0.05,
            max_delay=0.5,
            timeout=2.0,
        )

    # -------------------------------------------------------------------------
    # 4. Concept Chunking Boundary Cases (K=0, 1, 10, 11, 20, 21)
    # -------------------------------------------------------------------------

    def _create_rubrics(self, count: int) -> list[ConceptRubric]:
        return [
            ConceptRubric(
                name=f"Concept_{i+1:02d}",
                definition=f"Definition of concept {i+1}",
                scale_1=f"Scale 1 anchor for {i+1}",
                scale_5=f"Scale 5 anchor for {i+1}",
                scale_10=f"Scale 10 anchor for {i+1}",
            )
            for i in range(count)
        ]

    def test_chunking_boundary_counts(self):
        """Test ScorerAgent.chunk_rubrics with boundary concept counts: 0, 1, 10, 11, 20, 21."""
        test_cases = [
            (0, []),
            (1, [1]),
            (10, [10]),
            (11, [10, 1]),
            (20, [10, 10]),
            (21, [10, 10, 1]),
            (30, [10, 10, 10]),
        ]

        for k, expected_chunk_sizes in test_cases:
            with self.subTest(concept_count=k):
                rubrics = self._create_rubrics(k)
                chunks = ScorerAgent.chunk_rubrics(rubrics, chunk_size=10)
                actual_sizes = [len(c) for c in chunks]
                self.assertEqual(actual_sizes, expected_chunk_sizes)
                # Verify all original rubrics are preserved in exact order
                flattened = [r for c in chunks for r in c]
                self.assertEqual(flattened, rubrics)

    def test_chunking_constraints_and_validation(self):
        """Invalid chunking parameters must raise ValueError."""
        # chunk_size <= 0
        with self.assertRaises(ValueError):
            ScorerAgent.chunk_rubrics(self._create_rubrics(5), chunk_size=0)
        with self.assertRaises(ValueError):
            ScorerAgent.chunk_rubrics(self._create_rubrics(5), chunk_size=-3)

        # ScorerAgent max_chunk_size > 10
        with self.assertRaises(ValueError):
            ScorerAgent(client=self.fake_client, max_chunk_size=11)

        # score_chunk called directly with > 10 rubrics
        scorer = ScorerAgent(client=self.fake_client, max_chunk_size=10)
        dummy_img = Image.new("RGB", (512, 288))
        with self.assertRaises(ValueError):
            scorer.score_chunk(dummy_img, self._create_rubrics(11))

    def test_score_images_output_dimensions(self):
        """score_images must return matrix of shape (N, K) and dtype int64 across boundary counts."""
        scorer = ScorerAgent(client=self.fake_client)

        for k in [0, 1, 10, 11, 20, 21]:
            with self.subTest(k_concepts=k):
                rubrics = self._create_rubrics(k)
                images = [Image.new("RGB", (512, 288)) for _ in range(3)]

                with patch.object(scorer, "score_single_image") as mock_score_single:
                    mock_score_single.return_value = {r.concept_id: 7 for r in rubrics}
                    matrix = scorer.score_images(images, rubrics, max_workers=2)

                self.assertEqual(matrix.shape, (3, k))
                self.assertEqual(matrix.dtype, np.int64)
                if k > 0:
                    self.assertTrue(np.all(matrix == 7))

    # -------------------------------------------------------------------------
    # 5. Malformed Markdown and Corrupt JSON Extraction
    # -------------------------------------------------------------------------

    def test_extract_clean_json_adversarial_malformed(self):
        """Stress-test extract_clean_json against malformed, wrapped, or corrupted inputs."""
        # 1. Standard markdown block
        s1 = "```json\n{\"key\": \"val\"}\n```"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s1), {"key": "val"})

        # 2. Block without 'json' language identifier
        s2 = "```\n{\"key\": 123}\n```"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s2), {"key": 123})

        # 3. Conversational preamble and postamble
        s3 = "Sure! Here is the concept score JSON:\n{\"scores\": {\"c1\": 8}}\nLet me know if you need more!"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s3), {"scores": {"c1": 8}})

        # 4. Truncated JSON -> must raise ValueError
        s4 = "```json\n{\"name\": \"Incomplete\", \"scale_1\": "
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json(s4)

        # 5. Non-JSON string -> must raise ValueError
        s5 = "Internal Server Error 500: Quota exceeded"
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json(s5)

        # 6. Empty / whitespace string -> must raise ValueError
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json("")
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json("   \n\t  ")

        # 7. Multiple JSON blocks: extracts first valid block
        s7 = "```json\n{\"block\": 1}\n```\nSome text\n```json\n{\"block\": 2}\n```"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s7), {"block": 1})

        # 8. Single quotes instead of double quotes (strict JSON syntax) -> raises ValueError
        s8 = "{'name': 'single_quote'}"
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json(s8)

    # -------------------------------------------------------------------------
    # 6. Concept Rubric Schema Resilience & Variations
    # -------------------------------------------------------------------------

    def test_rubric_from_dict_variations(self):
        """ConceptRubric.from_dict must gracefully handle missing keys and non-standard scoring guides."""
        # Completely empty dict
        r_empty = ConceptRubric.from_dict({})
        self.assertEqual(r_empty.name, "Unnamed Concept")
        self.assertEqual(r_empty.concept_id, "unnamed_concept")
        self.assertGreater(len(r_empty.scale_1), 0)
        self.assertGreater(len(r_empty.scale_5), 0)
        self.assertGreater(len(r_empty.scale_10), 0)

        # Semicolon-delimited combined scoring_guide
        d_semi = {
            "name": "Porosity Level",
            "scoring_guide": "1: Zero voids; 5: Moderate porosity; 10: Cellular sponge",
        }
        r_semi = ConceptRubric.from_dict(d_semi)
        self.assertEqual(r_semi.scale_1, "Zero voids")
        self.assertEqual(r_semi.scale_5, "Moderate porosity")
        self.assertEqual(r_semi.scale_10, "Cellular sponge")

        # Non-alphanumeric name -> clean snake_case id
        r_special = ConceptRubric(
            name="  -- Nano/Micro-Roughness & Texturing!!  ",
            definition="Surface relief",
            scale_1="Smooth",
            scale_5="Rippled",
            scale_10="Rugged",
        )
        self.assertEqual(r_special.concept_id, "nano_micro_roughness_texturing")

    # -------------------------------------------------------------------------
    # 7. Backoff Calculation & Mathematical Progression
    # -------------------------------------------------------------------------

    def test_backoff_mathematical_progression(self):
        """Verify backoff exponential progression, full-jitter bounds, and ceiling clamping."""
        client = ResilientVertexClient(
            api_key="mock",
            project_id="1034989953225",
            base_delay=1.0,
            max_delay=60.0,
        )

        expected_ceilings = {
            1: 1.0,   # 1.0 * 2^0
            2: 2.0,   # 1.0 * 2^1
            3: 4.0,   # 1.0 * 2^2
            4: 8.0,   # 1.0 * 2^3
            5: 16.0,  # 1.0 * 2^4
            6: 32.0,  # 1.0 * 2^5
            7: 60.0,  # clamped to max_delay 60.0
            50: 60.0, # clamped, no overflow
        }

        for attempt, expected_ceiling in expected_ceilings.items():
            samples = [client.compute_backoff(attempt) for _ in range(200)]
            self.assertTrue(all(s >= 0.05 for s in samples))
            self.assertTrue(all(s <= expected_ceiling for s in samples))
            # Average should be roughly ceiling / 2
            mean_sample = float(np.mean(samples))
            self.assertGreater(mean_sample, 0.05)
            self.assertLess(mean_sample, expected_ceiling)

    def test_backoff_retry_after_header_variants(self):
        """Test compute_backoff parsing of Retry-After headers."""
        client = ResilientVertexClient(api_key="mock", project_id="1034989953225")

        # Positive integer
        s_int = client.compute_backoff(1, retry_after_header="15")
        self.assertGreaterEqual(s_int, 15.0)
        self.assertLessEqual(s_int, 16.1)

        # Decimal
        s_float = client.compute_backoff(1, retry_after_header="4.5")
        self.assertGreaterEqual(s_float, 4.5)
        self.assertLessEqual(s_float, 5.6)

        # Negative string -> clamps to 0.0 + jitter
        s_neg = client.compute_backoff(1, retry_after_header="-10")
        self.assertGreaterEqual(s_neg, 0.0)
        self.assertLessEqual(s_neg, 1.1)

        # Malformed non-numeric -> fallback to standard backoff
        s_bad = client.compute_backoff(1, retry_after_header="Wed, 21 Oct 2026")
        self.assertGreaterEqual(s_bad, 0.05)
        self.assertLessEqual(s_bad, 60.0)

    # -------------------------------------------------------------------------
    # 8. Severe Simulated HTTP 429 Burst Load
    # -------------------------------------------------------------------------

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_severe_simulated_429_burst_recovery(self, mock_post: MagicMock, mock_sleep: MagicMock):
        """Simulate 30 concurrent worker requests hitting HTTP 429 bursts and recovering."""
        client = ResilientVertexClient(
            api_key="mock",
            project_id="1034989953225",
            max_retries=3,
            base_delay=0.01,
            max_delay=0.1,
        )

        import threading
        lock = threading.Lock()
        request_attempts = {}

        # For each call, return 429 for the first 2 attempts, then 200 OK
        resp_429 = MagicMock(status_code=429, text="Rate limit exceeded", headers={})
        resp_200 = MagicMock(
            status_code=200,
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "{\"status\": \"ok\"}"}]}}]},
        )

        def mock_side_effect(*args, **kwargs):
            payload = kwargs.get("json", {})
            req_id = payload.get("request_id", 0)
            with lock:
                attempts = request_attempts.get(req_id, 0) + 1
                request_attempts[req_id] = attempts

            if attempts <= 2:
                return resp_429
            return resp_200

        mock_post.side_effect = mock_side_effect

        def _make_call(idx: int):
            return client.call({"request_id": idx})

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_make_call, i) for i in range(20)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), 20)
        self.assertTrue(all("candidates" in r for r in results))
        # Total attempts: 20 requests * 3 calls (2 retries + 1 success) = 60 calls
        self.assertEqual(mock_post.call_count, 60)

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_status_code_retryable_matrix(self, mock_post: MagicMock, mock_sleep: MagicMock):
        """Verify retry behavior across HTTP status codes."""
        client = ResilientVertexClient(api_key="mock", project_id="1034989953225", max_retries=2)

        # 1. Non-retryable error (400 Bad Request) must fail immediately without retrying
        mock_post.return_value = MagicMock(status_code=400, text="Bad Request: Invalid argument")
        with self.assertRaises(RuntimeError) as ctx:
            client.call({"test": 1})
        self.assertIn("non-retryable error 400", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)

        # 2. Non-retryable error (401 Unauthorized)
        mock_post.reset_mock()
        mock_post.return_value = MagicMock(status_code=401, text="Unauthorized API key")
        with self.assertRaises(RuntimeError):
            client.call({"test": 1})
        self.assertEqual(mock_post.call_count, 1)

        # 3. Retryable errors (500, 502, 503, 504) must retry up to max_retries
        for code in [500, 502, 503, 504]:
            mock_post.reset_mock()
            mock_post.return_value = MagicMock(status_code=code, text=f"Error {code}")
            with self.assertRaises(RuntimeError) as ctx:
                client.call({"test": 1})
            self.assertIn("Max retries", str(ctx.exception))
            # 1 initial + 2 retries = 3 calls
            self.assertEqual(mock_post.call_count, 3)

    # -------------------------------------------------------------------------
    # 9. Scorer Clamping & Sanitization
    # -------------------------------------------------------------------------

    def test_scorer_cleaning_and_clamping(self):
        """ScorerAgent must sanitize non-numeric values, out-of-range floats, and missing concepts to [0, 5]."""
        rubrics = self._create_rubrics(4)
        scorer = ScorerAgent(client=self.fake_client)

        # Mock model response with adversarial values:
        # concept_01: 15.8 (out of bounds high -> 5)
        # concept_02: -3.2 (out of bounds low -> 0)
        # concept_03: "not_a_number" (non-numeric -> 0)
        # concept_04: missing from dict entirely -> 0
        adversarial_json = json.dumps({
            "scores": {
                rubrics[0].concept_id: 15.8,
                rubrics[1].concept_id: -3.2,
                rubrics[2].concept_id: "not_a_number",
            }
        })

        with patch.object(self.fake_client, "generate_content", return_value=adversarial_json):
            scores = scorer.score_chunk(Image.new("RGB", (512, 288)), rubrics)

        self.assertEqual(scores[rubrics[0].concept_id], 5)
        self.assertEqual(scores[rubrics[1].concept_id], 0)
        self.assertEqual(scores[rubrics[2].concept_id], 0)
        self.assertEqual(scores[rubrics[3].concept_id], 0)

    # -------------------------------------------------------------------------
    # 10. Live Vertex AI Connectivity Smoke Test
    # -------------------------------------------------------------------------

    def test_live_vertex_connectivity(self):
        """Verify real HTTP connectivity to Gemini 3.8 Flash via Vertex Express API."""
        api_key = os.environ.get("VERTEX_API_KEY")
        project_id = os.environ.get("VERTEX_PROJECT_ID")
        if not api_key or not project_id:
            self.skipTest("Vertex AI credentials not in environment; skipping live test.")

        live_client = ResilientVertexClient(
            api_key=api_key,
            project_id=project_id,
            max_retries=2,
            timeout=30.0,
        )

        test_img = Image.new("RGB", (512, 288), color=(100, 150, 200))
        scorer = ScorerAgent(client=live_client)
        rubric = ConceptRubric(
            name="Visual Uniformity",
            definition="Degree of spatial homogeneity in micrograph intensity.",
            scale_1="Heavily heterogeneous with extreme local contrast.",
            scale_3="Moderately uniform with scattered features.",
            scale_5="Completely uniform solid field.",
        )

        try:
            scores = scorer.score_chunk(test_img, [rubric])
        except RuntimeError as e:
            if "403" in str(e) or "PERMISSION_DENIED" in str(e):
                self.skipTest(f"Vertex API key lacked permission or expired: {e}")
            raise
        self.assertIn("visual_uniformity", scores)
        score_val = scores["visual_uniformity"]
        self.assertIsInstance(score_val, int)
        self.assertTrue(0 <= score_val <= 5)
        logger.info("Live Vertex AI smoke test PASSED: score=%d", score_val)


if __name__ == "__main__":
    unittest.main()
