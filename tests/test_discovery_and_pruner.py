"""Comprehensive Unit and Integration Tests for ASSUME Milestone 4.

Tests:
1. StatisticalPruner:
   - Dead dimension identification and pruning (std < 0.20 or >90% identical).
   - Zero-variance handling and Pearson correlation matrix properties.
   - Collinear pair detection (|r| > 0.80) and sorting.
2. ConceptDiscoveryLoop:
   - Cold start step 0 initialization.
   - Active tri-modal step iteration.
   - Diversity evaluation trigger every 3 batches.
   - Integration with merger for collinear pairs and pruner for dead dimensions.
   - Full hermetic end-to-end loop simulation from 0 to K=30 concepts.
3. Concept Dictionary Export:
   - Verification of strictly atomic single-property rubrics.
   - Verification of 1, 5, 10 monotonic scoring guides and dual-path saving.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence
import unittest

import numpy as np
from PIL import Image

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.loop.discovery_loop import ConceptDiscoveryLoop
from assume.loop.pruner import StatisticalPruner
from assume.sampling.tri_modal_sampler import TriModalBatchSampler


def _create_sample_rubric(idx: int, name: Optional[str] = None) -> ConceptRubric:
    """Create a test concept rubric."""
    concept_name = name or f"Concept_{idx}"
    return ConceptRubric(
        name=concept_name,
        definition=f"Physical morphological definition for {concept_name}.",
        scale_1=f"Minimal / absent expression of {concept_name}.",
        scale_5=f"Intermediate expression of {concept_name}.",
        scale_10=f"Maximal expression of {concept_name}.",
    )


class TestStatisticalPruner(unittest.TestCase):
    """Unit tests for StatisticalPruner."""

    def setUp(self) -> None:
        self.pruner = StatisticalPruner(
            min_std=0.20,
            collinear_threshold=0.80,
            max_identical_fraction=0.90,
        )

    def test_prune_dead_dimensions_zero_variance(self) -> None:
        """Constant columns (std = 0.0) must be pruned."""
        N = 20
        scores = np.column_stack([
            np.full(N, 5.0),                          # dead (constant 5.0)
            np.linspace(1.0, 10.0, N),                # active (wide variance)
            np.full(N, 8.0),                          # dead (constant 8.0)
        ])
        rubrics = [_create_sample_rubric(i) for i in range(3)]

        filtered, retained, pruned = self.pruner.prune_dead_dimensions(scores, rubrics)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].name, "Concept_1")
        self.assertEqual(filtered.shape, (N, 1))
        self.assertListEqual(pruned, ["Concept_0", "Concept_2"])

    def test_prune_dead_dimensions_low_variance(self) -> None:
        """Columns with std < 0.20 must be pruned."""
        N = 20
        # std for [5, 5, 5, ..., 5.2] is ~0.04 < 0.20
        low_var_col = np.array([5.0] * 19 + [5.2])
        high_var_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        scores = np.column_stack([low_var_col, high_var_col])
        rubrics = [_create_sample_rubric(0, "LowVar"), _create_sample_rubric(1, "HighVar")]

        filtered, retained, pruned = self.pruner.prune_dead_dimensions(scores, rubrics)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].name, "HighVar")
        self.assertIn("LowVar", pruned)

    def test_prune_dead_dimensions_identical_fraction(self) -> None:
        """Columns with >90% identical values must be pruned even if std >= 0.20."""
        N = 20
        # 19 out of 20 identical (95% identical)
        mostly_identical = np.array([5.0] * 19 + [10.0])  # std ~1.09 > 0.20, but 95% identical
        varied = np.tile([1.0, 3.0, 5.0, 8.0], 5)
        scores = np.column_stack([mostly_identical, varied])
        rubrics = [_create_sample_rubric(0, "MostlyIdentical"), _create_sample_rubric(1, "Varied")]

        filtered, retained, pruned = self.pruner.prune_dead_dimensions(scores, rubrics)

        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].name, "Varied")
        self.assertIn("MostlyIdentical", pruned)

    def test_prune_dead_dimensions_edge_cases(self) -> None:
        """Test empty and edge-case score arrays."""
        # Empty rubrics
        f_scores, ret, pr = self.pruner.prune_dead_dimensions(np.zeros((10, 0)), [])
        self.assertEqual(len(ret), 0)
        self.assertEqual(len(pr), 0)

        # Empty scores
        rubrics = [_create_sample_rubric(0)]
        f_scores, ret, pr = self.pruner.prune_dead_dimensions(np.zeros((0, 1)), rubrics)
        self.assertEqual(len(ret), 1)

        # Single sample (cannot compute variance)
        scores_1 = np.array([[5.0, 6.0]])
        rubrics_2 = [_create_sample_rubric(0), _create_sample_rubric(1)]
        f_scores, ret, pr = self.pruner.prune_dead_dimensions(scores_1, rubrics_2)
        self.assertEqual(len(ret), 2)

    def test_compute_correlation_matrix_math(self) -> None:
        """Verify Pearson correlation calculation and numerical stability."""
        N = 30
        x = np.linspace(1.0, 10.0, N)
        y = 2.0 * x + 1.0       # Perfect positive correlation (r = 1.0)
        z = -3.0 * x + 5.0      # Perfect negative correlation (r = -1.0)
        w = np.sin(x)           # Non-linear / low correlation

        scores = np.column_stack([x, y, z, w])
        R = self.pruner.compute_correlation_matrix(scores)

        self.assertEqual(R.shape, (4, 4))
        # Diagonal must be strictly 1.0
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-5)
        # R must be symmetric
        np.testing.assert_allclose(R, R.T, atol=1e-5)
        # Perfect correlation pairs
        self.assertAlmostEqual(R[0, 1], 1.0, places=4)
        self.assertAlmostEqual(R[0, 2], -1.0, places=4)
        self.assertTrue(-1.0 <= R[0, 3] <= 1.0)

    def test_compute_correlation_matrix_zero_variance_dimensions(self) -> None:
        """Verify that zero-variance dimensions yield 0.0 correlation without NaN."""
        N = 25
        x = np.linspace(1.0, 10.0, N)
        const = np.full(N, 7.0)  # zero variance
        scores = np.column_stack([x, const])

        R = self.pruner.compute_correlation_matrix(scores)

        self.assertFalse(np.isnan(R).any())
        self.assertEqual(R[0, 0], 1.0)
        self.assertEqual(R[1, 1], 1.0)
        self.assertEqual(R[0, 1], 0.0)
        self.assertEqual(R[1, 0], 0.0)

    def test_find_collinear_pairs(self) -> None:
        """Verify collinear pair detection with |r| > 0.80 and descending sort."""
        # Synthetic 4x4 correlation matrix
        R = np.array([
            [1.0,  0.85, 0.40, -0.92],
            [0.85, 1.0,  0.79,  0.10],
            [0.40, 0.79, 1.0,  -0.50],
            [-0.92, 0.10, -0.50, 1.0],
        ], dtype=np.float32)

        pairs = self.pruner.find_collinear_pairs(R, threshold=0.80)

        # Expected collinear pairs: (0, 3) with r = -0.92, and (0, 1) with r = 0.85
        # (1, 2) has r=0.79 <= 0.80, so excluded
        self.assertEqual(len(pairs), 2)
        # Sorted descending by |r|
        self.assertEqual(pairs[0][0], 0)
        self.assertEqual(pairs[0][1], 3)
        self.assertAlmostEqual(pairs[0][2], -0.92, places=2)

        self.assertEqual(pairs[1][0], 0)
        self.assertEqual(pairs[1][1], 1)
        self.assertAlmostEqual(pairs[1][2], 0.85, places=2)


class MockProposerAgent:
    """Hermetic mock proposer that yields synthetic atomic concept rubrics."""

    def __init__(self) -> None:
        self.counter = 0

    def propose(
        self,
        images: Sequence[Any],
        existing_concepts: Optional[Sequence[Any]] = None,
        n_proposals: int = 4,
    ) -> List[ConceptRubric]:
        proposals: List[ConceptRubric] = []
        for _ in range(n_proposals):
            self.counter += 1
            name = f"Mock Concept {self.counter}"
            proposals.append(ConceptRubric(
                name=name,
                definition=f"Morphological measure for {name}.",
                scale_1=f"Absence of {name}",
                scale_5=f"Moderate {name}",
                scale_10=f"Extreme {name}",
            ))
        return proposals


class MockScorerAgent:
    """Hermetic mock scorer producing valid integer scores in [1, 10]."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def score_images(
        self,
        images: Sequence[Any],
        rubrics: Sequence[ConceptRubric],
        max_workers: int = 1,
    ) -> np.ndarray:
        n_img = len(images)
        n_rub = len(rubrics)
        if n_img == 0 or n_rub == 0:
            return np.zeros((n_img, n_rub), dtype=np.int64)

        # Generate scores with natural variance in [1, 10]
        base_scores = self.rng.integers(1, 11, size=(n_img, n_rub), dtype=np.int64)
        return base_scores


class MockVisualMergerAgent:
    """Hermetic mock merger evaluating colliding pairs."""

    def __init__(self, action: str = "KEEP_BOTH") -> None:
        self.action = action
        self.call_count = 0

    def evaluate_collision(
        self,
        rubric_a: ConceptRubric,
        rubric_b: ConceptRubric,
        images: Sequence[Any],
        correlation_r: Optional[float] = None,
    ) -> MergeDecision:
        self.call_count += 1
        if self.action == "MERGE":
            unified = ConceptRubric(
                name=f"{rubric_a.name} & {rubric_b.name} Unified",
                definition=f"Unified morphological feature synthesizing {rubric_a.name} and {rubric_b.name}.",
                scale_1=rubric_a.scale_1,
                scale_5=rubric_a.scale_5,
                scale_10=rubric_a.scale_10,
            )
            return MergeDecision(
                action="MERGE",
                rationale="Concepts capture redundant visual manifestation in SEM micrographs.",
                merged_rubric=unified,
            )
        return MergeDecision(
            action="KEEP_BOTH",
            rationale="Concepts represent independent physical axes.",
            merged_rubric=None,
        )


class MockSampler:
    """Hermetic mock sampler simulating TriModalBatchSampler."""

    def __init__(self, n_samples: int = 100, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.sampled_indices = set()
        self.manifest = None  # uses synthetic fallback images

    def __len__(self) -> int:
        return self.n_samples

    def _sample_fps(
        self,
        n_points: int,
        reference_indices: Sequence[int],
        exclude_indices: set,
    ) -> List[int]:
        available = [i for i in range(self.n_samples) if i not in exclude_indices]
        return [int(x) for x in self.rng.choice(available, size=min(n_points, len(available)), replace=False)]

    def sample_batch(
        self,
        step: int,
        batch_size: int = 6,
        concept_scores: Optional[np.ndarray] = None,
        scored_indices: Optional[List[int]] = None,
    ) -> List[int]:
        chosen = [int(x) for x in self.rng.choice(self.n_samples, size=batch_size, replace=False)]
        self.sampled_indices.update(chosen)
        return chosen


class TestConceptDiscoveryLoop(unittest.TestCase):
    """Integration and behavioral tests for ConceptDiscoveryLoop."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_path = Path(self.temp_dir.name) / "data" / "concept_dictionary.json"
        self.root_path = Path(self.temp_dir.name) / "concept_dictionary.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loop_cold_start_step_0(self) -> None:
        """Verify cold start step 0 initialization and concept proposal (checkpoint-only scoring)."""
        sampler = MockSampler(n_samples=50)
        proposer = MockProposerAgent()
        scorer = MockScorerAgent()
        merger = MockVisualMergerAgent()
        pruner = StatisticalPruner()

        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            proposer=proposer,
            scorer=scorer,
            merger=merger,
            pruner=pruner,
            target_k=10,
            batch_size=6,
            proposals_per_batch=4,
            output_path=self.out_path,
            root_output_path=self.root_path,
            step_delay=0.0,
        )

        step_info = loop.run_step(0)

        self.assertEqual(step_info["step"], 0)
        self.assertEqual(len(loop.concepts), 4)
        # Checkpoint-only scoring: step 0 does not score the 6 discovery images or pad with fake 5s
        self.assertEqual(loop.eval_scores.size, 0)
        self.assertIsNone(step_info["eval_summary"])  # Not an eval interval step

    def test_loop_diversity_eval_trigger(self) -> None:
        """Verify that diversity evaluation triggers on periodic checkpoint steps."""
        sampler = MockSampler(n_samples=50)
        proposer = MockProposerAgent()
        scorer = MockScorerAgent()
        merger = MockVisualMergerAgent()
        pruner = StatisticalPruner()

        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            proposer=proposer,
            scorer=scorer,
            merger=merger,
            pruner=pruner,
            target_k=15,
            batch_size=6,
            eval_interval=3,
            eval_batch_size=12,
            output_path=self.out_path,
            root_output_path=self.root_path,
            step_delay=0.0,
        )

        # Step 0: batch 1 -> no eval
        info_0 = loop.run_step(0)
        self.assertIsNone(info_0["eval_summary"])

        # Step 1: batch 2 -> no eval
        info_1 = loop.run_step(1)
        self.assertIsNone(info_1["eval_summary"])

        # Step 2: batch 3 -> eval triggers!
        info_2 = loop.run_step(2)
        self.assertIsNotNone(info_2["eval_summary"])
        self.assertIn("active_k", info_2["eval_summary"])

    def test_distinguishing_micrographs_selection_and_sign_awareness(self) -> None:
        """Verify selection of distinguishing micrographs for merger, including negative correlation."""
        sampler = MockSampler(n_samples=30)
        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            output_path=self.out_path,
            root_output_path=self.root_path,
        )

        images = [Image.new("RGB", (100, 100), color=(i * 10, i * 10, i * 10)) for i in range(6)]
        scores_i = np.array([5, 1, 3, 0, 4, 2])
        scores_j = np.array([1, 5, 3, 0, 2, 4])

        distinguishing_pos = loop._select_distinguishing_micrographs(
            scores_i, scores_j, images, n_images=3, correlation_r=0.85
        )
        distinguishing_neg = loop._select_distinguishing_micrographs(
            scores_i, scores_j, images, n_images=3, correlation_r=-0.85
        )

        self.assertEqual(len(distinguishing_pos), 3)
        self.assertEqual(len(distinguishing_neg), 3)
        # When r < 0, image 0 (scores_i=5, scores_j=1 -> eff_sj=5) has highest joint sum (10)
        self.assertEqual(distinguishing_neg[0], images[0])

    def test_active_only_correlation_ignores_shared_zeros(self) -> None:
        """Verify compute_correlation_matrix computes correlation only where max(s_i, s_j) > 0."""
        pruner = StatisticalPruner()
        # Concepts 0 and 1 are negatively correlated (-1.0) on the 4 active images, and 0 on 20 images
        s0 = np.array([5, 1, 5, 1] + [0] * 20, dtype=np.float32)
        s1 = np.array([1, 5, 1, 5] + [0] * 20, dtype=np.float32)
        R = pruner.compute_correlation_matrix(np.column_stack([s0, s1]))
        self.assertAlmostEqual(R[0, 1], -1.0, places=4)

    def test_keep_both_whitelist_prevents_repeated_merger_calls(self) -> None:
        """Verify pairs adjudicated as KEEP_BOTH are whitelisted and not re-sent to Merger."""
        sampler = MockSampler(n_samples=50)
        proposer = MockProposerAgent()
        merger = MockVisualMergerAgent(action="KEEP_BOTH")
        pruner = StatisticalPruner(collinear_threshold=0.80)

        class CollinearScorer:
            def score_images(self, images, rubrics, max_workers=1):
                N = len(images)
                K = len(rubrics)
                base = np.tile([1, 2, 3, 4, 5, 1], (N // 6 + 1))[:N]
                return np.column_stack([base for _ in range(K)])

        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            proposer=proposer,
            scorer=CollinearScorer(),
            merger=merger,
            pruner=pruner,
            target_k=10,
            batch_size=6,
             proposals_per_batch=2,
            eval_interval=1,
            eval_batch_size=12,
            output_path=self.out_path,
            root_output_path=self.root_path,
            step_delay=0.0,
        )

        # First eval: 2 concepts -> 1 collinear pair -> 1 merger call -> added to verified_orthogonal_pairs
        loop.concepts = [_create_sample_rubric(0, "C0"), _create_sample_rubric(1, "C1")]
        loop.evaluate_and_prune()
        self.assertEqual(merger.call_count, 1)
        self.assertEqual(len(loop.verified_orthogonal_pairs), 1)

        # Second eval on same 2 concepts: pair is whitelisted -> 0 additional merger calls
        loop.evaluate_and_prune()
        self.assertEqual(merger.call_count, 1)

    def test_loop_collinear_merging_execution(self) -> None:
        """Verify that collinear pairs detected in eval trigger VisualMerger MERGE action."""
        sampler = MockSampler(n_samples=50)
        proposer = MockProposerAgent()
        merger = MockVisualMergerAgent(action="MERGE")
        pruner = StatisticalPruner(collinear_threshold=0.80)

        # Custom scorer that forces concept 0 and concept 1 to be collinear
        class CollinearMockScorer:
            def score_images(self, images, rubrics, max_workers=1):
                N = len(images)
                K = len(rubrics)
                mat = np.random.randint(1, 6, size=(N, K))
                if K >= 2:
                    mat[:, 1] = mat[:, 0]  # Perfect collinearity between concept 0 and 1
                return mat

        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            proposer=proposer,
            scorer=CollinearMockScorer(),
            merger=merger,
            pruner=pruner,
            target_k=10,
            batch_size=6,
            eval_interval=1,  # Eval every step for this test
            eval_batch_size=12,
            output_path=self.out_path,
            root_output_path=self.root_path,
            step_delay=0.0,
        )

        # Run step 0: proposes 4 concepts, CollinearMockScorer makes 0 and 1 identical
        step_info = loop.run_step(0)
        self.assertIsNotNone(step_info["eval_summary"])
        self.assertGreaterEqual(merger.call_count, 1)
        self.assertGreaterEqual(step_info["eval_summary"]["merged"], 1)

    def test_loop_full_hermetic_convergence_to_target_k(self) -> None:
        """Hermetic end-to-end loop simulation from 0 to K=30 concepts."""
        sampler = MockSampler(n_samples=100, seed=123)
        proposer = MockProposerAgent()
        scorer = MockScorerAgent(seed=123)
        merger = MockVisualMergerAgent(action="KEEP_BOTH")
        pruner = StatisticalPruner(min_std=0.10)  # low threshold for random scores

        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            proposer=proposer,
            scorer=scorer,
            merger=merger,
            pruner=pruner,
            target_k=30,
            batch_size=6,
            eval_batch_size=15,
            eval_interval=3,
            proposals_per_batch=4,
            max_iterations=15,
            output_path=self.out_path,
            root_output_path=self.root_path,
            seed=123,
            step_delay=0.0,
        )

        final_concepts = loop.run()

        # Must reach exactly target_k=30
        self.assertEqual(len(final_concepts), 30)
        self.assertTrue(self.out_path.exists())
        self.assertTrue(self.root_path.exists())

        # Validate saved JSON structure
        with open(self.out_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 30)

        for item in data:
            self.assertIn("concept_id", item)
            self.assertIn("name", item)
            self.assertIn("definition", item)
            self.assertIn("scale_0", item)
            self.assertIn("scale_1", item)
            self.assertIn("scale_3", item)
            self.assertIn("scale_5", item)
            self.assertIn("scoring_guide", item)
            self.assertTrue(len(item["name"]) > 0)
            self.assertTrue(len(item["definition"]) > 0)
            self.assertTrue(
                "0:" in item["scoring_guide"]
                and "1:" in item["scoring_guide"]
                and "3:" in item["scoring_guide"]
                and "5:" in item["scoring_guide"]
            )

    def test_live_concept_dictionary_json_artifact(self) -> None:
        """Validate the production 60-concept concept_dictionary.json."""
        from assume.config import DATA_DIR, PROJECT_ROOT

        data_dict_path = DATA_DIR / "concept_dictionary.json"
        root_dict_path = PROJECT_ROOT / "concept_dictionary.json"

        self.assertTrue(data_dict_path.exists(), f"Missing {data_dict_path}")
        self.assertTrue(root_dict_path.exists(), f"Missing {root_dict_path}")

        for p in (data_dict_path, root_dict_path):
            with open(p, "r", encoding="utf-8") as f:
                concepts_data = json.load(f)

            self.assertIsInstance(concepts_data, list)
            self.assertEqual(len(concepts_data), 60, f"Expected 60 concepts, found {len(concepts_data)}")

            seen_ids = set()
            seen_names = set()

            for idx, c in enumerate(concepts_data):
                self.assertIn("concept_id", c)
                self.assertIn("name", c)
                self.assertIn("definition", c)
                self.assertIn("scale_0", c)
                self.assertIn("scale_1", c)
                self.assertIn("scale_3", c)
                self.assertIn("scale_5", c)
                self.assertIn("scoring_guide", c)

                # Ensure uniqueness
                self.assertNotIn(c["concept_id"], seen_ids, f"Duplicate concept_id: {c['concept_id']}")
                self.assertNotIn(c["name"].lower(), seen_names, f"Duplicate name: {c['name']}")
                seen_ids.add(c["concept_id"])
                seen_names.add(c["name"].lower())

                # Ensure non-empty and well-formed anchors
                self.assertGreater(len(c["name"]), 3)
                self.assertGreater(len(c["definition"]), 10)
                self.assertGreater(len(c["scale_1"]), 5)
                self.assertGreater(len(c["scale_3"]), 5)
                self.assertGreater(len(c["scale_5"]), 5)

                # Round-trip check with ConceptRubric
                rubric = ConceptRubric.from_dict(c)
                self.assertEqual(rubric.concept_id, c["concept_id"])
                self.assertEqual(rubric.name, c["name"])

    def test_concept_discovery_loop_accepts_model_argument(self) -> None:
        """ConceptDiscoveryLoop must accept model keyword argument and forward to default agents."""
        sampler = MockSampler(n_samples=20)
        loop = ConceptDiscoveryLoop(
            sampler=sampler,
            model="gemini-2.5-flash",
            output_path=self.out_path,
            root_output_path=self.root_path,
        )
        self.assertEqual(loop.model, "gemini-2.5-flash")
        self.assertEqual(loop.proposer.client.model, "gemini-2.5-flash")
        self.assertEqual(loop.scorer.client.model, "gemini-2.5-flash")
        self.assertEqual(loop.merger.client.model, "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
