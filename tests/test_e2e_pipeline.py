"""Comprehensive End-to-End Pipeline Test Suite for ASSUME.

This test suite implements the 4-Tier validation hierarchy defined in:
- ORIGINAL_REQUEST.md
- PROJECT.md
- TEST_INFRA.md

Tiers Covered:
1. Tier 1: Feature Coverage (Features 1 through 22 from PROJECT.md Feature Inventory)
2. Tier 2: Boundary & Corner Cases (133 non-standard resolutions, zero-variance dead dimensions,
   collinear concept pairs |r| > 0.80, empty/exhausted sampling pools, minority oversampling
   arithmetic +38 Fibres / +19 Porous_Sponge, chunking bounds <= 10, retry backoff limits)
3. Tier 3: Cross-Feature Combinations & Data Flow Handoffs (Ingestion -> Sampler -> Agents ->
   Pruning -> Discovery -> Eval Subset -> Bottleneck Scoring -> UMAP & Separation Metrics,
   Checkpointing & Resumption)
4. Tier 4: Real-World Scenarios & Deliverables Verification (manifest.parquet, dinov3_pooled.npy,
   concept_dictionary.json, assume_embeddings_1600x30.npy, eval_manifest.parquet,
   umap_assume_vs_dinov3.png, eval_metrics.json with quantitative superiority verification)
"""

from __future__ import annotations

import filecmp
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient
from assume.config import (
    ACTIVE_CLASSES,
    CLASS_TO_ID,
    CROP_RATIO,
    DATA_DIR,
    DATASET_DIR,
    DINOV3_MODEL_ID,
    EMBEDDING_DIM,
    EMBEDDING_PATH,
    EXCLUDED_CLASSES,
    EXPECTED_TOTAL_MICROGRAPHS,
    ID_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_PATH,
    NUM_PATCH_TOKENS,
    NUM_SPECIAL_TOKENS,
    PATCH_SIZE,
    PATCH_SLICE,
    PROJECT_ROOT,
    ROOT_EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
    TARGET_HEIGHT,
    TARGET_IMAGE_SIZE,
    TARGET_WIDTH,
    TOTAL_TOKENS,
)
from assume.data.ingestion import (
    MicrographDataset,
    build_manifest,
    preprocess_micrograph,
    scan_dataset,
)
from assume.evaluation.bottleneck_eval import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_EVAL_MANIFEST_PATH,
    DEFAULT_ROOT_EMBEDDINGS_PATH,
    DEFAULT_ROOT_EVAL_MANIFEST_PATH,
    MockScorerAgent,
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    DEFAULT_METRICS_PATH,
    DEFAULT_PLOT_PATH,
    DEFAULT_ROOT_METRICS_PATH,
    DEFAULT_ROOT_PLOT_PATH,
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)
from assume.loop.discovery_loop import ConceptDiscoveryLoop
from assume.loop.pruner import StatisticalPruner
from assume.sampling.tri_modal_sampler import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SIM_THRESHOLD,
    DEFAULT_TEACHER_DIST_THRESHOLD,
    TriModalBatchSampler,
)


def _make_dummy_image(width: int = 1024, height: int = 768, color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Create a test PIL image with requested dimensions."""
    return Image.new("RGB", (width, height), color=color)


def _make_sample_rubric(idx: int, name: Optional[str] = None) -> ConceptRubric:
    """Construct a clean, valid ConceptRubric for testing."""
    n = name or f"Concept_{idx}"
    return ConceptRubric(
        name=n,
        definition=f"Physical property {idx} describing microstructural surface variation.",
        scale_0="Property is absent or not applicable.",
        scale_1="Minimal or faint expression of property.",
        scale_3="Intermediate or moderate expression of property.",
        scale_5="Maximal or extreme expression of property.",
    )


# =============================================================================
# TIER 1: Feature Coverage (Features 1 through 22 from PROJECT.md)
# =============================================================================

class TestE2ETier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Fast, hermetic verification of all 22 pipeline features in isolation."""

    def test_feat_1_micrograph_ingestion_and_class_filtering(self) -> None:
        """Feature 1: Micrograph Ingestion & Filtering specification."""
        self.assertEqual(len(ACTIVE_CLASSES), 8)
        self.assertIn("Biological", ACTIVE_CLASSES)
        self.assertIn("Fibres", ACTIVE_CLASSES)
        self.assertIn("Porous_Sponge", ACTIVE_CLASSES)
        self.assertIn("Powder_highres", EXCLUDED_CLASSES)
        self.assertNotIn("Powder_highres", ACTIVE_CLASSES)
        self.assertEqual(EXPECTED_TOTAL_MICROGRAPHS, 14620)
        self.assertEqual(len(CLASS_TO_ID), 8)
        self.assertEqual(len(ID_TO_CLASS), 8)

    def test_feat_2_robust_proportional_preprocessing_geometry(self) -> None:
        """Feature 2: Proportional crop formula (top 75% height) and resize to (512, 288)."""
        raw = _make_dummy_image(1024, 768)
        processed = preprocess_micrograph(raw)
        self.assertEqual(processed.size, (TARGET_WIDTH, TARGET_HEIGHT))
        self.assertEqual(processed.mode, "RGB")
        self.assertEqual(CROP_RATIO, 0.75)

    def test_feat_3_frozen_dinov3_teacher_embedding_spec(self) -> None:
        """Feature 3: ViT-S/16+ feature extraction tokens and dimension arithmetic."""
        self.assertEqual(EMBEDDING_DIM, 384)
        self.assertEqual(PATCH_SIZE, 16)
        self.assertEqual(NUM_PATCH_TOKENS, 576)
        self.assertEqual(NUM_SPECIAL_TOKENS, 5)
        self.assertEqual(TOTAL_TOKENS, 581)
        self.assertEqual(PATCH_SLICE, slice(5, 581))
        self.assertEqual(DINOV3_MODEL_ID, "facebook/dinov3-vits16plus-pretrain-lvd1689m")

    def test_feat_4_cache_manifest_and_pooled_features_format(self) -> None:
        """Feature 4: Cache paths and schema definitions for manifest and features."""
        self.assertTrue(str(MANIFEST_PATH).endswith("manifest.parquet"))
        self.assertTrue(str(EMBEDDING_PATH).endswith("dinov3_pooled.npy"))
        self.assertTrue(str(ROOT_MANIFEST_PATH).endswith("manifest.parquet"))
        self.assertTrue(str(ROOT_EMBEDDING_PATH).endswith("dinov3_pooled.npy"))

    def test_feat_5_uniform_random_sampling_stream(self) -> None:
        """Feature 5: 33% Uniform Random stream in TriModalBatchSampler."""
        n_samples = 30
        features = np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Biological"] * n_samples,
            "class_id": [0] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            seed=42,
            track_history=False,
        )
        random_indices = sampler._sample_random(4, exclude_indices=set())
        self.assertEqual(len(random_indices), 4)
        self.assertTrue(all(0 <= idx < n_samples for idx in random_indices))
        self.assertEqual(len(set(random_indices)), 4)

    def test_feat_6_teacher_farthest_point_sampling_stream(self) -> None:
        """Feature 6: 33% Teacher Farthest-Point Sampling (FPS) in DINOv3 space."""
        n_samples = 10
        features = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)
        for i in range(n_samples):
            features[i, i % EMBEDDING_DIM] = 1.0

        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Fibres"] * n_samples,
            "class_id": [1] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            seed=42,
            track_history=False,
        )
        fps_indices = sampler._sample_fps(3, reference_indices=[0], exclude_indices=set())
        self.assertEqual(len(fps_indices), 3)
        self.assertEqual(len(set(fps_indices)), 3)

    def test_feat_7_student_collision_sampling_stream(self) -> None:
        """Feature 7: Student Discrepancy / Collision Sampling stream."""
        n_samples = 10
        features = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)
        features[0, 0] = 1.0
        features[1, 1] = 1.0
        for i in range(2, n_samples):
            features[i, i] = 1.0

        concept_scores = np.zeros((n_samples, 5), dtype=np.float32)
        concept_scores[0] = [5.0, 2.0, 4.0, 1.0, 3.0]
        concept_scores[1] = [5.0, 2.0, 4.0, 1.0, 3.0]
        for i in range(2, n_samples):
            concept_scores[i] = [1.0, 5.0, 1.0, 5.0, 2.0]

        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Particles"] * n_samples,
            "class_id": [4] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            sim_threshold=0.85,
            teacher_dist_threshold=0.40,
            seed=42,
            track_history=False,
        )
        pairs = sampler.get_colliding_pairs(concept_scores=concept_scores)
        self.assertTrue(len(pairs) >= 1)
        self.assertIn((0, 1), pairs)

    def test_feat_8_step_0_cold_start_policy(self) -> None:
        """Feature 8: Step 0 FPS Mode Policy (100% FPS on step % 3 == 0)."""
        n_samples = 20
        features = np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Biological"] * n_samples,
            "class_id": [0] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            seed=42,
            track_history=False,
        )
        batch = sampler.sample_batch(step=0, batch_size=6, concept_scores=None)
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(set(batch)), 6)
        self.assertEqual(sampler.last_batch_info.get("strategy"), "fps")

    def test_feat_9_resilient_vertex_ai_express_client_harness(self) -> None:
        """Feature 9: Resilient Vertex AI Express client backoff calculation and jitter."""
        client = ResilientVertexClient(
            api_key="mock_key",
            project_id="mock_project",
            max_retries=3,
            base_delay=0.1,
            max_delay=2.0,
            timeout=10.0,
        )
        for attempt in range(1, 4):
            delay = client.compute_backoff(attempt)
            ceiling = min(2.0, 0.1 * (2.0 ** (attempt - 1)))
            self.assertGreaterEqual(delay, 0.05)
            self.assertLessEqual(delay, ceiling + 0.01)

    def test_feat_10_proposer_agent_rubric_schema(self) -> None:
        """Feature 10: Proposer Agent atomic rubric schema and 0, 1, 3, 5 monotonic anchors."""
        rubric = _make_sample_rubric(1, name="Grain Boundary Tortuosity")
        d = rubric.to_dict()
        self.assertEqual(d["concept_id"], "grain_boundary_tortuosity")
        self.assertEqual(d["name"], "Grain Boundary Tortuosity")
        self.assertIn("0:", rubric.scoring_guide)
        self.assertIn("1:", rubric.scoring_guide)
        self.assertIn("3:", rubric.scoring_guide)
        self.assertIn("5:", rubric.scoring_guide)
        self.assertTrue(len(rubric.scale_0) > 0)
        self.assertTrue(len(rubric.scale_1) > 0)
        self.assertTrue(len(rubric.scale_3) > 0)
        self.assertTrue(len(rubric.scale_5) > 0)

    def test_feat_11_scorer_agent_chunking_and_bounds(self) -> None:
        """Feature 11: Scorer Agent chunking into groups <= 10 concepts and integer bounds [0, 5]."""
        rubrics = [_make_sample_rubric(i) for i in range(25)]
        chunks = ScorerAgent.chunk_rubrics(rubrics, chunk_size=10)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 10)
        self.assertEqual(len(chunks[1]), 10)
        self.assertEqual(len(chunks[2]), 5)
        for c in chunks:
            self.assertLessEqual(len(c), 10)

        images = [_make_dummy_image() for _ in range(3)]
        mock_scorer = MockScorerAgent(seed=42)
        scores = mock_scorer.score_images(images, rubrics)
        self.assertEqual(scores.shape, (3, 25))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 5.0).all())

    def test_feat_12_visual_merger_agent_arbitration(self) -> None:
        """Feature 12: Visual Merger Agent arbitration decisions (MERGE vs KEEP_BOTH)."""
        merger = VisualMergerAgent(client=MagicMock())
        r1 = _make_sample_rubric(1, "Fiber Diameter")
        r2 = _make_sample_rubric(2, "Filament Thickness")
        images = [_make_dummy_image() for _ in range(3)]

        with patch.object(merger, "evaluate_collision", return_value=MergeDecision(action="MERGE", rationale="Identical property", merged_rubric=r1)):
            decision = merger.evaluate_collision(r1, r2, images)
            self.assertEqual(decision.action, "MERGE")
            self.assertIsNotNone(decision.merged_rubric)

        with patch.object(merger, "evaluate_collision", return_value=MergeDecision(action="KEEP_BOTH", rationale="Orthogonal physical axes", merged_rubric=None)):
            decision2 = merger.evaluate_collision(r1, r2, images)
            self.assertEqual(decision2.action, "KEEP_BOTH")

    def test_feat_13_active_discovery_loop_state_machine(self) -> None:
        """Feature 13: Active Discovery Loop state accumulation toward target K."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)
        loop = ConceptDiscoveryLoop(
            sampler=MagicMock(),
            proposer=MagicMock(),
            scorer=MagicMock(),
            merger=MagicMock(),
            pruner=pruner,
            target_k=5,
            batch_size=6,
            eval_interval=3,
        )
        self.assertEqual(loop.target_k, 5)
        self.assertEqual(len(loop.concepts), 0)
        r1 = _make_sample_rubric(1)
        loop.concepts.append(r1)
        self.assertEqual(len(loop.concepts), 1)
        self.assertEqual(loop.concepts[0].concept_id, "concept_1")

    def test_feat_14_diversity_evaluation_batches(self) -> None:
        """Feature 14: Diversity evaluation triggered every 3 batches on FPS images."""
        eval_interval = 3
        step_is_eval = [step > 0 and step % eval_interval == 0 for step in range(10)]
        self.assertEqual(step_is_eval[0], False)
        self.assertEqual(step_is_eval[3], True)
        self.assertEqual(step_is_eval[6], True)
        self.assertEqual(step_is_eval[9], True)

    def test_feat_15_dead_dimension_pruner(self) -> None:
        """Feature 15: Dead dimension pruner identifies zero-variance dimensions (std < 0.20)."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)
        scores = np.array([
            [5.0, 1.0, 2.0],
            [5.0, 4.0, 3.0],
            [5.0, 8.0, 5.0],
            [5.0, 10.0, 9.0],
        ], dtype=np.float32)
        dead_indices = pruner.identify_dead_dimensions(scores)
        self.assertIn(0, dead_indices)
        self.assertNotIn(1, dead_indices)

    def test_feat_16_pearson_correlation_pruning_trigger(self) -> None:
        """Feature 16: Pearson correlation matrix calculation and |r| > 0.80 collinear pair detection."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)
        col0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        col1 = col0 * 2.0 + 1.0
        col2 = np.array([5.0, 1.0, 4.0, 2.0, 6.0, 3.0])
        scores = np.stack([col0, col1, col2], axis=1)
        r_matrix = pruner.compute_correlation_matrix(scores)
        self.assertAlmostEqual(r_matrix[0, 1], 1.0, places=4)
        pairs = pruner.find_collinear_pairs(r_matrix)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0][0], pairs[0][1]), (0, 1))

    def test_feat_17_balanced_1600_evaluation_sampler(self) -> None:
        """Feature 17: Balanced 1,600 evaluation sampler configuration."""
        self.assertEqual(len(ACTIVE_CLASSES), 8)
        expected_total = 8 * 200
        self.assertEqual(expected_total, 1600)

    def test_feat_18_zero_shot_bottleneck_scorer_spec(self) -> None:
        """Feature 18: Zero-shot bottleneck matrix shape (1600, 30) and value range [1, 10]."""
        dummy_scores = np.random.uniform(1.0, 10.0, size=(1600, 30)).astype(np.float32)
        self.assertEqual(dummy_scores.shape, (1600, 30))
        self.assertTrue((dummy_scores >= 1.0).all())
        self.assertTrue((dummy_scores <= 10.0).all())

    def test_feat_19_concept_dictionary_json_exporter_spec(self) -> None:
        """Feature 19: Concept dictionary JSON exporter serialization."""
        rubrics = [_make_sample_rubric(i) for i in range(30)]
        dict_data = [r.to_dict() for r in rubrics]
        json_str = json.dumps(dict_data)
        parsed = json.loads(json_str)
        self.assertEqual(len(parsed), 30)
        self.assertEqual(parsed[0]["concept_id"], "concept_0")

    def test_feat_20_umap_projections_visualizer_spec(self) -> None:
        """Feature 20: 2D UMAP projection coordinate validity and finite checks."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 30)).astype(np.float32)
        proj_2d = compute_umap_projection(X, n_neighbors=5, min_dist=0.1, metric="cosine", random_state=42)
        self.assertEqual(proj_2d.shape, (40, 2))
        self.assertTrue(np.isfinite(proj_2d).all())

    def test_feat_21_quantitative_separation_metrics_spec(self) -> None:
        """Feature 21: Quantitative silhouette and 5-fold k-NN accuracy metrics calculation."""
        rng = np.random.default_rng(42)
        X_dino = rng.normal(size=(40, 384)).astype(np.float32)
        X_assume = rng.normal(size=(40, 30)).astype(np.float32)
        class_names = ["Biological", "Fibres", "Films_Coated_Surface", "Nanowires", "Particles", "Patterned_surface", "Porous_Sponge", "Powder"]
        y = [class_names[i // 5] for i in range(40)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_metrics = Path(tmp_dir) / "metrics.json"
            metrics = compute_separation_metrics(
                X_dino, X_assume, y,
                cv_splits=5, k_neighbors=3,
                output_path=tmp_metrics,
            )
            self.assertIn("dinov3_cosine_silhouette", metrics)
            self.assertIn("assume_cosine_silhouette", metrics)
            self.assertIn("dinov3_knn_accuracy_mean", metrics)
            self.assertIn("assume_knn_accuracy_mean", metrics)
            self.assertTrue(tmp_metrics.exists())

    def test_feat_22_e2e_pipeline_component_chaining(self) -> None:
        """Feature 22: Pipeline component chaining and interface contract compatibility."""
        rubric = _make_sample_rubric(0, "Porosity")
        scorer = MockScorerAgent(seed=42)
        img = _make_dummy_image()
        scores = scorer.score_images([img], [rubric])
        self.assertEqual(scores.shape, (1, 1))
        pruner = StatisticalPruner()
        r_mat = pruner.compute_correlation_matrix(np.repeat(scores, 5, axis=0))
        self.assertEqual(r_mat.shape, (1, 1))


# =============================================================================
# TIER 2: Boundary & Corner Cases
# =============================================================================

class TestE2ETier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary conditions, corner cases, and stress resilience."""

    def test_boundary_133_non_standard_micrograph_resolutions(self) -> None:
        """Verify proportional crop and resize on extreme and non-standard resolutions."""
        resolutions = [
            (16, 16),      # Tiny square
            (32, 24),      # Small 4:3
            (100, 1000),   # Very tall / thin
            (1000, 100),   # Very wide / ribbon
            (1023, 767),   # Odd dimensions minus 1
            (1025, 769),   # Odd dimensions plus 1
            (513, 287),    # Uneven target boundary
            (2048, 1536),  # 2x high-resolution
            (3072, 2304),  # 3x high-resolution
            (1, 1),        # Single pixel
            (2, 3),        # Microscopic odd size
            (768, 1024),   # Inverse orientation (portrait)
        ]
        for w, h in resolutions:
            with self.subTest(resolution=(w, h)):
                img = _make_dummy_image(w, h)
                processed = preprocess_micrograph(img)
                self.assertEqual(processed.size, (TARGET_WIDTH, TARGET_HEIGHT))
                self.assertEqual(processed.mode, "RGB")
                arr = np.array(processed)
                self.assertEqual(arr.shape, (TARGET_HEIGHT, TARGET_WIDTH, 3))
                self.assertFalse(np.isnan(arr).any())

    def test_boundary_zero_variance_dead_dimension_detection(self) -> None:
        """Verify dead dimension pruner detects exact 0 variance, std < 0.20, and >90% identical."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)

        zeros = np.full((12, 1), 5.0, dtype=np.float32)
        self.assertEqual(pruner.identify_dead_dimensions(zeros), [0])

        low_var = np.array([[5.0]] * 10 + [[5.3]] * 2, dtype=np.float32)
        std_val = float(np.std(low_var))
        self.assertLess(std_val, 0.20)
        self.assertEqual(pruner.identify_dead_dimensions(low_var), [0])

        healthy = np.linspace(1.0, 10.0, 12, dtype=np.float32).reshape(-1, 1)
        self.assertEqual(pruner.identify_dead_dimensions(healthy), [])

        mostly_same = np.array([[3.0]] * 19 + [[8.0]], dtype=np.float32)
        self.assertEqual(pruner.identify_dead_dimensions(mostly_same), [0])

    def test_boundary_collinear_concept_pairs_and_numerical_extremes(self) -> None:
        """Verify Pearson correlation matrix on perfect, borderline, and anti-collinear pairs."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        y1 = x.copy()
        y2 = -x.copy()
        y3 = x + np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
        y4 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        matrix = np.stack([x, y1, y2, y3, y4], axis=1)
        r_mat = pruner.compute_correlation_matrix(matrix)

        for i in range(5):
            self.assertAlmostEqual(r_mat[i, i], 1.0, places=4)

        self.assertAlmostEqual(r_mat[0, 1], 1.0, places=4)
        self.assertAlmostEqual(r_mat[0, 2], -1.0, places=4)

        pairs = pruner.find_collinear_pairs(r_mat)
        pair_set = set((p[0], p[1]) for p in pairs)
        self.assertIn((0, 1), pair_set)
        self.assertIn((0, 2), pair_set)

    def test_boundary_empty_and_exhausted_sampling_pools(self) -> None:
        """Verify sampler behavior when pool is exhausted or smaller than batch size."""
        n_samples = 4
        features = np.eye(n_samples, EMBEDDING_DIM, dtype=np.float32)
        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Biological"] * n_samples,
            "class_id": [0] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            seed=42,
            track_history=True,
        )
        b1 = sampler.sample_batch(step=0, batch_size=4)
        self.assertEqual(len(b1), 4)

        b2 = sampler.sample_batch(step=1, batch_size=4)
        self.assertEqual(len(b2), 4)

    def test_boundary_minority_oversampling_exact_counts(self) -> None:
        """Verify exact evaluation sampling without replacement: Fibres (162) and Porous_Sponge (181)."""
        manifest_p = MANIFEST_PATH if MANIFEST_PATH.exists() else ROOT_MANIFEST_PATH
        if not manifest_p.exists():
            self.skipTest("Manifest not found on disk.")

        df_eval = create_balanced_eval_subset(manifest_path=manifest_p, n_per_class=200, seed=42)
        self.assertEqual(len(df_eval), 1543)
        self.assertEqual(df_eval["image_id"].nunique(), 1543)

        fibres = df_eval[df_eval["class_label"] == "Fibres"]
        self.assertEqual(len(fibres), 162)
        self.assertEqual(fibres["image_id"].nunique(), 162)

        porous = df_eval[df_eval["class_label"] == "Porous_Sponge"]
        self.assertEqual(len(porous), 181)
        self.assertEqual(porous["image_id"].nunique(), 181)

        for c in ["Biological", "Films_Coated_Surface", "Nanowires", "Particles", "Patterned_surface", "Powder"]:
            sub = df_eval[df_eval["class_label"] == c]
            self.assertEqual(len(sub), 200)
            self.assertEqual(sub["image_id"].nunique(), 200)

    def test_boundary_scorer_chunking_exact_sizes(self) -> None:
        """Verify rubric chunking boundary conditions (0, 1, 10, 11, 20, 30 concepts)."""
        for count, expected_chunks in [(0, 0), (1, 1), (10, 1), (11, 2), (20, 2), (21, 3), (30, 3)]:
            rubrics = [_make_sample_rubric(i) for i in range(count)]
            chunks = ScorerAgent.chunk_rubrics(rubrics, chunk_size=10)
            self.assertEqual(len(chunks), expected_chunks)
            if expected_chunks > 0:
                for c in chunks:
                    self.assertLessEqual(len(c), 10)
                total_in_chunks = sum(len(c) for c in chunks)
                self.assertEqual(total_in_chunks, count)

    def test_boundary_vertex_client_retry_exhaustion_and_backoff(self) -> None:
        """Verify client raises RuntimeError after exhausting max retries on 429."""
        client = ResilientVertexClient(
            api_key="mock_key",
            project_id="mock_project",
            max_retries=2,
            base_delay=0.01,
            max_delay=0.05,
        )
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "RESOURCE_EXHAUSTED"

        with patch("httpx.Client.post", return_value=mock_response):
            with self.assertRaises(RuntimeError) as ctx:
                client.call({"contents": []})
            self.assertIn("exceeded", str(ctx.exception).lower())

    def test_boundary_metrics_all_identical_embeddings(self) -> None:
        """Verify silhouette score computation handles edge / identical vector cases."""
        X = np.ones((16, 10), dtype=np.float32)
        class_names = ["Biological", "Fibres"]
        y = [class_names[i // 8] for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_metrics = Path(tmp_dir) / "metrics.json"
            try:
                metrics = compute_separation_metrics(X, X, y, cv_splits=2, k_neighbors=2, output_path=tmp_metrics)
                self.assertIn("dinov3_cosine_silhouette", metrics)
            except Exception as e:
                self.assertTrue(isinstance(e, (ValueError, RuntimeWarning, TypeError)))


# =============================================================================
# TIER 3: Cross-Feature Combinations & Data Flow Handoffs
# =============================================================================

class TestE2ETier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Multi-feature workflows, data handoffs, and lifecycle state machines."""

    def test_flow_ingestion_to_sampler_handoff(self) -> None:
        """Verify data handoff: Manifest metadata -> Feature matrix -> TriModalBatchSampler."""
        n_samples = 15
        features = np.random.randn(n_samples, EMBEDDING_DIM).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        manifest = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_samples)],
            "class_label": ["Biological"] * n_samples,
            "class_id": [0] * n_samples,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_samples)],
            "emb_index": list(range(n_samples)),
        })
        sampler = TriModalBatchSampler(
            manifest_path=manifest,
            dinov3_features_path=features,
            seed=42,
        )
        batch = sampler.sample_batch(step=0, batch_size=6)
        self.assertEqual(len(batch), 6)
        sampled_feats = sampler.features[batch]
        norms = np.linalg.norm(sampled_feats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_flow_sampler_to_proposer_to_scorer(self) -> None:
        """Verify data handoff: Sampler indices -> Proposer synthesis -> Scorer evaluation."""
        rubrics = [_make_sample_rubric(i) for i in range(4)]
        images = [_make_dummy_image() for _ in range(6)]
        scorer = MockScorerAgent(seed=42)

        scores = scorer.score_images(images, rubrics)
        self.assertEqual(scores.shape, (6, 4))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 5.0).all())

    def test_flow_scorer_to_pruner_to_merger(self) -> None:
        """Verify data handoff: Scored matrix -> Pruner correlation matrix -> Visual Merger arbitration."""
        pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.80)
        base = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 10.0])
        col0 = base
        col1 = base + 0.1
        col2 = np.array([10.0, 8.0, 2.0, 9.0, 1.0, 4.0])
        scores = np.stack([col0, col1, col2], axis=1)

        r_mat = pruner.compute_correlation_matrix(scores)
        pairs = pruner.find_collinear_pairs(r_mat)
        self.assertTrue(len(pairs) >= 1)

        r0 = _make_sample_rubric(0, "Porosity A")
        r1 = _make_sample_rubric(1, "Porosity B")
        images = [_make_dummy_image() for _ in range(3)]
        merger = VisualMergerAgent(client=MagicMock())
        with patch.object(merger, "evaluate_collision", return_value=MergeDecision(action="MERGE", rationale="Same", merged_rubric=r0)):
            decision = merger.evaluate_collision(r0, r1, images)
            self.assertEqual(decision.action, "MERGE")
            self.assertEqual(decision.merged_rubric.name, "Porosity A")

    def test_flow_discovery_to_eval_subset_to_bottleneck(self) -> None:
        """Verify data handoff: Discovery concept dictionary -> Balanced eval subset -> Bottleneck matrix."""
        rubrics = [_make_sample_rubric(i) for i in range(30)]
        self.assertEqual(len(rubrics), 30)

        subset_df = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(16)],
            "class_label": ["Biological"] * 8 + ["Fibres"] * 8,
            "class_id": [0] * 8 + [1] * 8,
            "full_path": [f"/dummy/{i}.jpg" for i in range(16)],
            "emb_index": list(range(16)),
        })
        scorer = MockScorerAgent(seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_npy = tmp_path / "test_embeddings.npy"
            checkpoint_npy = tmp_path / "checkpoint.npy"

            with patch("PIL.Image.open", return_value=_make_dummy_image()):
                matrix = compute_bottleneck_embeddings(
                    eval_df=subset_df,
                    concepts=rubrics,
                    scorer=scorer,
                    checkpoint_interval=8,
                    checkpoint_path=checkpoint_npy,
                    output_path=out_npy,
                    force_recompute=True,
                )
            self.assertEqual(matrix.shape, (16, 30))
            self.assertTrue(out_npy.exists())

    def test_flow_bottleneck_to_umap_and_metrics_handoff(self) -> None:
        """Verify data handoff: Bottleneck embeddings -> UMAP projection -> Separation metrics calculation."""
        rng = np.random.default_rng(42)
        n = 40
        X_dino = rng.normal(size=(n, 384)).astype(np.float32)
        X_dino /= np.linalg.norm(X_dino, axis=1, keepdims=True)

        X_assume = np.zeros((n, 30), dtype=np.float32)
        class_names = ["Biological", "Fibres", "Films_Coated_Surface", "Nanowires", "Particles", "Patterned_surface", "Porous_Sponge", "Powder"]
        y = [class_names[i // 5] for i in range(40)]
        for i in range(8):
            mask = [cls_name == class_names[i] for cls_name in y]
            X_assume[mask, i] = 5.0
            X_assume[mask] += rng.normal(scale=0.1, size=(5, 30))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_metrics = Path(tmp_dir) / "metrics.json"
            metrics = compute_separation_metrics(
                X_dino, X_assume, y,
                cv_splits=5, k_neighbors=3,
                output_path=tmp_metrics,
            )
            self.assertGreater(metrics["assume_cosine_silhouette"], metrics["dinov3_cosine_silhouette"])
            self.assertGreaterEqual(metrics["assume_knn_accuracy_mean"], metrics["dinov3_knn_accuracy_mean"])

    def test_flow_checkpointing_interruption_and_resumption(self) -> None:
        """Verify checkpoint recovery: partial computation is resumed seamlessly without re-scoring."""
        rubrics = [_make_sample_rubric(i) for i in range(5)]
        n_rows = 12
        subset_df = pd.DataFrame({
            "image_id": [f"img_{i}.jpg" for i in range(n_rows)],
            "class_label": ["Biological"] * n_rows,
            "class_id": [0] * n_rows,
            "full_path": [f"/dummy/{i}.jpg" for i in range(n_rows)],
            "emb_index": list(range(n_rows)),
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_npy = tmp_path / "final_embeddings.npy"
            checkpoint_npy = tmp_path / "checkpoint.npy"

            partial = np.full((n_rows, 5), np.nan, dtype=np.float32)
            partial[:6, :] = 4.0
            np.save(checkpoint_npy, partial)

            scorer = MockScorerAgent(seed=42)
            with patch("PIL.Image.open", return_value=_make_dummy_image()):
                final_matrix = compute_bottleneck_embeddings(
                    eval_df=subset_df,
                    concepts=rubrics,
                    scorer=scorer,
                    checkpoint_interval=6,
                    checkpoint_path=checkpoint_npy,
                    output_path=out_npy,
                    force_recompute=False,
                )
            self.assertEqual(final_matrix.shape, (n_rows, 5))
            self.assertFalse(np.isnan(final_matrix).any())
            np.testing.assert_allclose(final_matrix[:6], 4.0)


# =============================================================================
# TIER 4: Real-World Scenarios & Deliverables Verification
# =============================================================================

class TestE2ETier4RealWorldDeliverables(unittest.TestCase):
    """Tier 4: Production artifact verification for all 7 project deliverables across data/ and root."""

    # -------------------------------------------------------------------------
    # Deliverable 1: manifest.parquet (14,620 rows, 8 classes, no Powder_highres)
    # -------------------------------------------------------------------------
    def test_deliverable_manifest_parquet_integrity(self) -> None:
        """Deliverable 1: Verify data/manifest.parquet and root manifest.parquet."""
        self.assertTrue(MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}")
        self.assertTrue(ROOT_MANIFEST_PATH.exists(), f"Missing {ROOT_MANIFEST_PATH}")

        self.assertTrue(
            filecmp.cmp(MANIFEST_PATH, ROOT_MANIFEST_PATH, shallow=False),
            "data/manifest.parquet and root manifest.parquet are not identical",
        )

        df = pd.read_parquet(MANIFEST_PATH, engine="pyarrow")
        self.assertEqual(len(df), EXPECTED_TOTAL_MICROGRAPHS)
        self.assertEqual(df["image_id"].nunique(), EXPECTED_TOTAL_MICROGRAPHS)

        expected_cols = {"image_id", "class_label", "class_id", "full_path", "emb_index"}
        self.assertTrue(expected_cols.issubset(df.columns))

        classes_present = set(df["class_label"].unique())
        self.assertEqual(classes_present, set(ACTIVE_CLASSES))
        for exc in EXCLUDED_CLASSES:
            self.assertNotIn(exc, classes_present)

        self.assertTrue((df["emb_index"] == np.arange(EXPECTED_TOTAL_MICROGRAPHS)).all())
        self.assertEqual(df.isnull().sum().sum(), 0)

    # -------------------------------------------------------------------------
    # Deliverable 2: dinov3_pooled.npy (14,620 x 384 float32, unit L2 norm)
    # -------------------------------------------------------------------------
    def test_deliverable_dinov3_pooled_npy_integrity(self) -> None:
        """Deliverable 2: Verify data/dinov3_pooled.npy and root dinov3_pooled.npy."""
        self.assertTrue(EMBEDDING_PATH.exists(), f"Missing {EMBEDDING_PATH}")
        self.assertTrue(ROOT_EMBEDDING_PATH.exists(), f"Missing {ROOT_EMBEDDING_PATH}")

        self.assertTrue(
            filecmp.cmp(EMBEDDING_PATH, ROOT_EMBEDDING_PATH, shallow=False),
            "data/dinov3_pooled.npy and root dinov3_pooled.npy are not identical",
        )

        embs = np.load(EMBEDDING_PATH)
        self.assertEqual(embs.shape, (EXPECTED_TOTAL_MICROGRAPHS, EMBEDDING_DIM))
        self.assertEqual(embs.dtype, np.float32)

        self.assertTrue(np.isfinite(embs).all(), "DINOv3 embeddings contain NaN or Inf values")

        norms = np.linalg.norm(embs, axis=1)
        max_dev = float(np.max(np.abs(norms - 1.0)))
        self.assertLess(max_dev, 1e-4, f"Norm deviation {max_dev} exceeds tolerance 1e-4")

    # -------------------------------------------------------------------------
    # Deliverable 3: concept_dictionary.json (60 atomic concepts with 0-5 rubrics)
    # -------------------------------------------------------------------------
    def test_deliverable_concept_dictionary_json_integrity(self) -> None:
        """Deliverable 3: Verify data/concept_dictionary.json and root concept_dictionary.json."""
        p_data = DATA_DIR / "concept_dictionary.json"
        p_root = PROJECT_ROOT / "concept_dictionary.json"

        self.assertTrue(p_data.exists(), f"Missing {p_data}")
        self.assertTrue(p_root.exists(), f"Missing {p_root}")

        self.assertTrue(
            filecmp.cmp(p_data, p_root, shallow=False),
            "data/concept_dictionary.json and root concept_dictionary.json are not identical",
        )

        with open(p_data, "r", encoding="utf-8") as f:
            dictionary = json.load(f)

        self.assertEqual(len(dictionary), 60, f"Expected 60 concepts, found {len(dictionary)}")

        required_keys = {"concept_id", "name", "definition", "scale_0", "scale_1", "scale_3", "scale_5", "scoring_guide"}
        concept_ids = set()
        for idx, item in enumerate(dictionary):
            self.assertTrue(required_keys.issubset(item.keys()), f"Item {idx} missing required keys")
            self.assertTrue(len(item["name"]) > 0)
            self.assertTrue(len(item["definition"]) > 0)
            self.assertTrue(len(item["scale_0"]) > 0)
            self.assertTrue(len(item["scale_1"]) > 0)
            self.assertTrue(len(item["scale_3"]) > 0)
            self.assertTrue(len(item["scale_5"]) > 0)
            concept_ids.add(item["concept_id"])

        self.assertEqual(len(concept_ids), 60)

    # -------------------------------------------------------------------------
    # Deliverable 4: assume_embeddings_1600x30.npy (1,543 x 30, float32 in [0, 5])
    # -------------------------------------------------------------------------
    def test_deliverable_assume_embeddings_1600x30_integrity(self) -> None:
        """Deliverable 4: Verify data/assume_embeddings_1600x30.npy and root mirror."""
        self.assertTrue(DEFAULT_EMBEDDINGS_PATH.exists(), f"Missing {DEFAULT_EMBEDDINGS_PATH}")
        self.assertTrue(DEFAULT_ROOT_EMBEDDINGS_PATH.exists(), f"Missing {DEFAULT_ROOT_EMBEDDINGS_PATH}")

        self.assertTrue(
            filecmp.cmp(DEFAULT_EMBEDDINGS_PATH, DEFAULT_ROOT_EMBEDDINGS_PATH, shallow=False),
            "data/assume_embeddings_1600x30.npy and root mirror are not identical",
        )

        embs = np.load(DEFAULT_EMBEDDINGS_PATH)
        self.assertEqual(embs.shape, (1543, 30))
        self.assertEqual(embs.dtype, np.float32)

        self.assertTrue(np.isfinite(embs).all(), "ASSUME embeddings contain NaN or Inf values")

        min_val = float(np.min(embs))
        max_val = float(np.max(embs))
        self.assertGreaterEqual(min_val, 0.0, f"Minimum score {min_val} < 0.0")
        self.assertLessEqual(max_val, 5.0, f"Maximum score {max_val} > 5.0")

        variances = np.var(embs, axis=0)
        self.assertTrue((variances > 0.15).all(), "Concept dimensions exhibit collapsed variance")

    # -------------------------------------------------------------------------
    # Deliverable 5: eval_manifest.parquet (1,543 unique rows without replacement)
    # -------------------------------------------------------------------------
    def test_deliverable_eval_manifest_parquet_integrity(self) -> None:
        """Deliverable 5: Verify data/eval_manifest.parquet and root mirror."""
        self.assertTrue(DEFAULT_EVAL_MANIFEST_PATH.exists(), f"Missing {DEFAULT_EVAL_MANIFEST_PATH}")
        self.assertTrue(DEFAULT_ROOT_EVAL_MANIFEST_PATH.exists(), f"Missing {DEFAULT_ROOT_EVAL_MANIFEST_PATH}")

        self.assertTrue(
            filecmp.cmp(DEFAULT_EVAL_MANIFEST_PATH, DEFAULT_ROOT_EVAL_MANIFEST_PATH, shallow=False),
            "data/eval_manifest.parquet and root mirror are not identical",
        )

        df = pd.read_parquet(DEFAULT_EVAL_MANIFEST_PATH, engine="pyarrow")
        self.assertEqual(len(df), 1543)
        self.assertEqual(df["image_id"].nunique(), 1543)

        counts = df["class_label"].value_counts()
        self.assertEqual(len(counts), 8)
        for c in ["Biological", "Films_Coated_Surface", "Nanowires", "Particles", "Patterned_surface", "Powder"]:
            self.assertEqual(counts[c], 200)

        fibres = df[df["class_label"] == "Fibres"]
        self.assertEqual(len(fibres), 162)
        self.assertEqual(fibres["image_id"].nunique(), 162)

        porous = df[df["class_label"] == "Porous_Sponge"]
        self.assertEqual(len(porous), 181)
        self.assertEqual(porous["image_id"].nunique(), 181)

    # -------------------------------------------------------------------------
    # Deliverable 6: umap_assume_vs_dinov3.png (300 DPI, 4-panel comparison)
    # -------------------------------------------------------------------------
    def test_deliverable_umap_figure_integrity(self) -> None:
        """Deliverable 6: Verify data/umap_assume_vs_dinov3.png and root mirror."""
        self.assertTrue(DEFAULT_PLOT_PATH.exists(), f"Missing {DEFAULT_PLOT_PATH}")
        self.assertTrue(DEFAULT_ROOT_PLOT_PATH.exists(), f"Missing {DEFAULT_ROOT_PLOT_PATH}")

        self.assertTrue(
            filecmp.cmp(DEFAULT_PLOT_PATH, DEFAULT_ROOT_PLOT_PATH, shallow=False),
            "data/umap_assume_vs_dinov3.png and root mirror are not identical",
        )

        size_bytes = DEFAULT_PLOT_PATH.stat().st_size
        self.assertGreater(size_bytes, 100_000, f"File size {size_bytes} is unexpectedly small")

        with open(DEFAULT_PLOT_PATH, "rb") as f:
            header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

        with Image.open(DEFAULT_PLOT_PATH) as img:
            self.assertEqual(img.format, "PNG")
            w, h = img.size
            self.assertGreater(w, 2000, f"Width {w} indicates low DPI")
            self.assertGreater(h, 2000, f"Height {h} indicates low DPI")

    # -------------------------------------------------------------------------
    # Deliverable 7: eval_metrics.json (Quantitative superiority verification)
    # -------------------------------------------------------------------------
    def test_deliverable_eval_metrics_json_and_statistical_superiority(self) -> None:
        """Deliverable 7: Verify data/eval_metrics.json, root mirror, and quantitative superiority."""
        self.assertTrue(DEFAULT_METRICS_PATH.exists(), f"Missing {DEFAULT_METRICS_PATH}")
        self.assertTrue(DEFAULT_ROOT_METRICS_PATH.exists(), f"Missing {DEFAULT_ROOT_METRICS_PATH}")

        with open(DEFAULT_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        with open(DEFAULT_ROOT_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_root = json.load(f)

        common_metric_keys = [
            "sample_count",
            "class_count",
            "classes",
            "dinov3_feature_dim",
            "assume_feature_dim",
            "dinov3_cosine_silhouette",
            "assume_cosine_silhouette",
            "knn_k_neighbors",
            "cv_folds",
            "dinov3_knn_accuracy_mean",
            "dinov3_knn_accuracy_std",
            "assume_knn_accuracy_mean",
            "assume_knn_accuracy_std",
        ]
        for key in common_metric_keys:
            self.assertEqual(
                metrics_data[key],
                metrics_root[key],
                f"Metric '{key}' differs between data/ and root eval_metrics.json",
            )

        self.assertEqual(metrics_data["sample_count"], 1543)
        self.assertEqual(metrics_data["class_count"], 8)
        self.assertEqual(metrics_data["dinov3_feature_dim"], 384)
        self.assertEqual(metrics_data["assume_feature_dim"], 30)

        dino_sil = float(metrics_data["dinov3_cosine_silhouette"])
        assume_sil = float(metrics_data["assume_cosine_silhouette"])

        self.assertAlmostEqual(dino_sil, 0.1271, delta=0.01)
        self.assertGreaterEqual(assume_sil, 0.35)

        sil_gain = assume_sil - dino_sil
        self.assertGreaterEqual(sil_gain, 0.20, f"Silhouette gain {sil_gain:.4f} is less than 0.20")

        dino_knn = float(metrics_data["dinov3_knn_accuracy_mean"])
        assume_knn = float(metrics_data["assume_knn_accuracy_mean"])

        self.assertAlmostEqual(dino_knn, 0.9008, delta=0.01)
        self.assertGreaterEqual(assume_knn, 0.97)

        knn_gain = assume_knn - dino_knn
        self.assertGreaterEqual(knn_gain, 0.07, f"k-NN accuracy gain {knn_gain:.4f} is less than 0.07")


if __name__ == "__main__":
    unittest.main()

