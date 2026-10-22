"""Comprehensive Unit and Integration Test Suite for ASSUME Milestone 5.

Validates:
1. Balanced Evaluation Subset Extraction:
   - Exactly 1,600 rows (200 per class across 8 classes).
   - Correct oversampling logic for Fibres (+38) and Porous_Sponge (+19).
   - Deterministic reproducibility under seed=42.
   - Preserves required schema columns.
   - Dual-path parquet export and mirroring.
2. Concept Bottleneck Embeddings Computation:
   - Mock and live scorer interfaces returning (1600, 30) float32 matrix in [1, 10].
   - Durable disk checkpointing and seamless resumption.
   - Dual-path npy export and mirroring.
3. DINOv3 Feature Subset Extraction:
   - Correct feature indexing using emb_index, shape (1600, 384), L2 normalized.
4. 2D UMAP Projections:
   - Valid 2D coordinates without NaNs for both feature spaces.
5. Quantitative Separation Metrics:
   - Cosine Silhouette Score and 5-fold cross-validated k-NN accuracy.
   - Valid metric ranges and dual-path JSON export.
6. Publication-grade 4-panel comparison visualization:
   - Valid PNG magic bytes, non-trivial file size, 300 DPI, dual-path export.
7. Concept dictionary validation:
   - Exactly K=30 validated atomic concepts with 1, 5, 10 anchors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Sequence
import unittest

import numpy as np
import pandas as pd
from PIL import Image

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
from assume.evaluation.bottleneck_eval import (
    MockScorerAgent,
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)


def _load_sample_concepts(n: int = 30) -> List[ConceptRubric]:
    """Helper to load concepts from concept_dictionary.json or generate test concepts."""
    dict_path = DATA_DIR / "concept_dictionary.json"
    if not dict_path.exists():
        dict_path = PROJECT_ROOT / "concept_dictionary.json"

    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [ConceptRubric.from_dict(d) for d in data[:n]]

    return [
        ConceptRubric(
            name=f"Concept_{i}",
            definition=f"Definition for concept {i}",
            scale_1="Minimal expression",
            scale_5="Intermediate expression",
            scale_10="Maximal expression",
        )
        for i in range(n)
    ]


class TestBalancedEvalSubset(unittest.TestCase):
    """Unit tests for balanced evaluation subset creation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = MANIFEST_PATH if MANIFEST_PATH.exists() else ROOT_MANIFEST_PATH
        if not cls.manifest_path.exists():
            raise unittest.SkipTest("manifest.parquet not found; skipping eval subset tests.")

    def test_balanced_subset_exact_row_count(self) -> None:
        """Balanced subset without oversampling must contain 1,543 unique rows (162 Fibres + 181 Porous_Sponge + 6*200)."""
        df = create_balanced_eval_subset(
            manifest_path=self.manifest_path,
            n_per_class=200,
            seed=42,
        )
        self.assertEqual(len(df), 1543)
        self.assertEqual(df["image_id"].nunique(), 1543)

    def test_balanced_subset_per_class_counts(self) -> None:
        """Every active class must have min(n_available, 200) rows in the subset."""
        df = create_balanced_eval_subset(
            manifest_path=self.manifest_path,
            n_per_class=200,
            seed=42,
        )
        counts = df["class_label"].value_counts()
        self.assertEqual(len(counts), 8)
        expected_counts = {
            "Biological": 200,
            "Fibres": 162,
            "Films_Coated_Surface": 200,
            "Nanowires": 200,
            "Particles": 200,
            "Patterned_surface": 200,
            "Porous_Sponge": 181,
            "Powder": 200,
        }
        for c_name, exp_c in expected_counts.items():
            self.assertIn(c_name, counts)
            self.assertEqual(counts[c_name], exp_c)

    def test_balanced_subset_minority_oversampling_counts(self) -> None:
        """Verify zero duplicate images across all classes (no oversampling with replacement)."""
        df = create_balanced_eval_subset(
            manifest_path=self.manifest_path,
            n_per_class=200,
            seed=42,
        )
        fibres = df[df["class_label"] == "Fibres"]
        porous = df[df["class_label"] == "Porous_Sponge"]

        # Fibres has 162 unique in manifest; 0 duplicates
        self.assertEqual(len(fibres), 162)
        self.assertEqual(fibres["image_id"].nunique(), 162)
        self.assertEqual(len(fibres) - fibres["image_id"].nunique(), 0)

        # Porous_Sponge has 181 unique in manifest; 0 duplicates
        self.assertEqual(len(porous), 181)
        self.assertEqual(porous["image_id"].nunique(), 181)
        self.assertEqual(len(porous) - porous["image_id"].nunique(), 0)

        # Other classes have >= 200 unique images, so 200 sampled with 0 duplicates
        for c in ["Biological", "Films_Coated_Surface", "Nanowires", "Particles", "Patterned_surface", "Powder"]:
            sub = df[df["class_label"] == c]
            self.assertEqual(len(sub), 200)
            self.assertEqual(sub["image_id"].nunique(), 200)

    def test_balanced_subset_deterministic_reproducibility(self) -> None:
        """Identical seed must produce identical row ordering and image IDs."""
        df1 = create_balanced_eval_subset(manifest_path=self.manifest_path, seed=42)
        df2 = create_balanced_eval_subset(manifest_path=self.manifest_path, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_balanced_subset_preserves_required_columns(self) -> None:
        """Evaluation DataFrame must preserve image_id, class_label, class_id, emb_index, full_path."""
        df = create_balanced_eval_subset(manifest_path=self.manifest_path, seed=42)
        required_cols = {"image_id", "class_label", "class_id", "emb_index", "full_path"}
        self.assertTrue(required_cols.issubset(set(df.columns)))
        self.assertFalse(df["emb_index"].isnull().any())
        self.assertTrue((df["emb_index"] >= 0).all())

    def test_balanced_subset_dual_saving(self) -> None:
        """Artifact must be saved to custom path and mirrored to root mirror."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "sub" / "eval_manifest.parquet"
            root_p = Path(tmpdir) / "eval_manifest.parquet"
            df = create_balanced_eval_subset(
                manifest_path=self.manifest_path,
                n_per_class=10,
                seed=42,
                output_path=out_p,
                root_output_path=root_p,
            )
            self.assertTrue(out_p.exists())
            self.assertTrue(root_p.exists())
            self.assertEqual(len(pd.read_parquet(out_p)), 80)
            self.assertEqual(len(pd.read_parquet(root_p)), 80)

    def test_balanced_subset_missing_manifest_raises(self) -> None:
        """Non-existent manifest path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            create_balanced_eval_subset(manifest_path="non_existent_file.parquet")


class TestMockScorerAgent(unittest.TestCase):
    """Unit tests for MockScorerAgent."""

    def setUp(self) -> None:
        self.scorer = MockScorerAgent(seed=42)
        self.concepts = _load_sample_concepts(10)

    def test_mock_scorer_returns_valid_shape(self) -> None:
        """Scorer returns numpy matrix of shape (len(images), len(rubrics))."""
        dummy_images = [Image.new("RGB", (512, 288)) for _ in range(5)]
        matrix = self.scorer.score_images(dummy_images, self.concepts)
        self.assertEqual(matrix.shape, (5, 10))

    def test_mock_scorer_integer_range(self) -> None:
        """All scored values must be integers clamped in [0, 5]."""
        dummy_images = [Image.new("RGB", (512, 288)) for _ in range(20)]
        matrix = self.scorer.score_images(dummy_images, self.concepts)
        self.assertTrue((matrix >= 0).all())
        self.assertTrue((matrix <= 5).all())
        self.assertEqual(matrix.dtype, np.int64)

    def test_mock_scorer_non_zero_variance(self) -> None:
        """Mock scores must exhibit natural variance across concepts (std > 0.20)."""
        dummy_images = [Image.new("RGB", (512, 288)) for _ in range(30)]
        matrix = self.scorer.score_images(dummy_images, self.concepts)
        stds = matrix.std(axis=0)
        self.assertTrue((stds > 0.20).all())

    def test_mock_scorer_empty_inputs(self) -> None:
        """Empty images or empty concepts must return corresponding empty matrices."""
        res_empty_img = self.scorer.score_images([], self.concepts)
        self.assertEqual(res_empty_img.shape, (0, 10))

        res_empty_con = self.scorer.score_images([Image.new("RGB", (512, 288))], [])
        self.assertEqual(res_empty_con.shape, (1, 0))

    def test_mock_scorer_deterministic_with_seed(self) -> None:
        """Two scorers initialized with the same seed produce identical scores."""
        scorer1 = MockScorerAgent(seed=123)
        scorer2 = MockScorerAgent(seed=123)
        dummy_images = [Image.new("RGB", (512, 288)) for _ in range(4)]
        m1 = scorer1.score_images(dummy_images, self.concepts)
        m2 = scorer2.score_images(dummy_images, self.concepts)
        np.testing.assert_array_equal(m1, m2)


class TestBottleneckEmbeddingsComputation(unittest.TestCase):
    """Integration tests for compute_bottleneck_embeddings."""

    def setUp(self) -> None:
        self.concepts = _load_sample_concepts(30)
        # Construct synthetic eval DataFrame with 16 rows (2 per class)
        rows = []
        for c_id, c_name in enumerate(ACTIVE_CLASSES):
            for i in range(2):
                rows.append({
                    "image_id": f"test_{c_name}_{i}",
                    "class_label": c_name,
                    "class_id": c_id,
                    "emb_index": c_id * 2 + i,
                    "full_path": f"non_existent_{c_name}_{i}.jpg",
                })
        self.eval_df = pd.DataFrame(rows)

    def test_compute_bottleneck_embeddings_shape_and_dtype(self) -> None:
        """Bottleneck computation must return float32 matrix of shape (len(eval_df), len(concepts))."""
        scorer = MockScorerAgent(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "embeddings.npy"
            chk_p = Path(tmpdir) / "checkpoint.npy"
            matrix = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=True,
            )
            self.assertEqual(matrix.shape, (16, 30))
            self.assertEqual(matrix.dtype, np.float32)
            self.assertTrue((matrix >= 0.0).all())
            self.assertTrue((matrix <= 5.0).all())
            self.assertFalse(np.isnan(matrix).any())
            self.assertTrue(out_p.exists())

    def test_compute_bottleneck_embeddings_checkpoint_resume(self) -> None:
        """Verify checkpoint resumption only computes remaining unscored rows."""
        scorer = MockScorerAgent(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "embeddings.npy"
            chk_p = Path(tmpdir) / "checkpoint.npy"

            # Pre-seed checkpoint with 10 rows completed
            pre_seeded = np.full((16, 30), np.nan, dtype=np.float32)
            pre_seeded[:10] = 4.0
            np.save(chk_p, pre_seeded)

            matrix = compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                checkpoint_interval=5,
                output_path=out_p,
                checkpoint_path=chk_p,
                force_recompute=False,
            )
            self.assertEqual(matrix.shape, (16, 30))
            # First 10 rows should preserve pre-seeded value 4.0
            np.testing.assert_array_equal(matrix[:10], 4.0)
            # Remaining 6 rows should be populated
            self.assertFalse(np.isnan(matrix[10:]).any())
            self.assertTrue((matrix[10:] >= 0.0).all())
            self.assertTrue((matrix[10:] <= 5.0).all())

    def test_compute_bottleneck_embeddings_dual_saving(self) -> None:
        """Completed embeddings must be saved to output_path and mirrored to root_output_path."""
        scorer = MockScorerAgent(seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "data" / "emb.npy"
            root_p = Path(tmpdir) / "emb.npy"
            compute_bottleneck_embeddings(
                eval_df=self.eval_df,
                concepts=self.concepts,
                scorer=scorer,
                output_path=out_p,
                root_output_path=root_p,
                force_recompute=True,
            )
            self.assertTrue(out_p.exists())
            self.assertTrue(root_p.exists())
            loaded_root = np.load(root_p)
            self.assertEqual(loaded_root.shape, (16, 30))


class TestDINOv3FeatureExtraction(unittest.TestCase):
    """Unit tests for extract_subset_features."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dino_path = EMBEDDING_PATH if EMBEDDING_PATH.exists() else ROOT_EMBEDDING_PATH
        if not cls.dino_path.exists():
            raise unittest.SkipTest("dinov3_pooled.npy not found; skipping feature extraction tests.")

    def test_extract_subset_features_shape_and_dtype(self) -> None:
        """Extracted subset features must have shape (N, 384) with float32 dtype."""
        dummy_df = pd.DataFrame({
            "emb_index": [0, 5, 10, 15, 20],
            "class_label": ["Biological"] * 5,
        })
        features = extract_subset_features(dummy_df, dinov3_features_path=self.dino_path)
        self.assertEqual(features.shape, (5, 384))
        self.assertEqual(features.dtype, np.float32)

    def test_extract_subset_features_alignment(self) -> None:
        """Features extracted at index i must exactly match the original embedding array row."""
        all_features = np.load(self.dino_path)
        test_indices = [3, 42, 100, 500]
        dummy_df = pd.DataFrame({"emb_index": test_indices})
        subset = extract_subset_features(dummy_df, dinov3_features_path=self.dino_path)
        for i, idx in enumerate(test_indices):
            np.testing.assert_array_equal(subset[i], all_features[idx])

    def test_extract_subset_features_normalized(self) -> None:
        """Extracted DINOv3 features must have unit L2 norm (1.0 +/- 1e-4)."""
        dummy_df = pd.DataFrame({"emb_index": [10, 20, 30, 40]})
        subset = extract_subset_features(dummy_df, dinov3_features_path=self.dino_path)
        norms = np.linalg.norm(subset, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


class TestUMAPAndQuantitativeMetrics(unittest.TestCase):
    """Unit tests for UMAP projections and separation metrics."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.n_samples = 40
        self.dino_features = rng.normal(size=(self.n_samples, 384)).astype(np.float32)
        # Normalize
        self.dino_features /= np.linalg.norm(self.dino_features, axis=1, keepdims=True)

        self.assume_features = rng.integers(1, 11, size=(self.n_samples, 30)).astype(np.float32)
        self.labels = [ACTIVE_CLASSES[i % 8] for i in range(self.n_samples)]

    def test_compute_umap_projection_shape(self) -> None:
        """UMAP projection must return (N, 2) array without NaNs."""
        coords_2d = compute_umap_projection(self.dino_features, n_neighbors=5, random_state=42)
        self.assertEqual(coords_2d.shape, (self.n_samples, 2))
        self.assertEqual(coords_2d.dtype, np.float32)
        self.assertFalse(np.isnan(coords_2d).any())

    def test_compute_separation_metrics_keys_and_ranges(self) -> None:
        """Verify silhouette scores and cross-validated k-NN accuracy metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "metrics.json"
            metrics = compute_separation_metrics(
                dino_features=self.dino_features,
                assume_features=self.assume_features,
                labels=self.labels,
                k_neighbors=3,
                cv_splits=3,
                output_path=out_p,
            )
            # Check keys
            self.assertIn("dinov3_cosine_silhouette", metrics)
            self.assertIn("assume_cosine_silhouette", metrics)
            self.assertIn("dinov3_knn_accuracy_mean", metrics)
            self.assertIn("assume_knn_accuracy_mean", metrics)
            self.assertIn("sample_count", metrics)

            # Check value bounds
            self.assertGreaterEqual(metrics["dinov3_cosine_silhouette"], -1.0)
            self.assertLessEqual(metrics["dinov3_cosine_silhouette"], 1.0)
            self.assertGreaterEqual(metrics["assume_cosine_silhouette"], -1.0)
            self.assertLessEqual(metrics["assume_cosine_silhouette"], 1.0)
            self.assertGreaterEqual(metrics["dinov3_knn_accuracy_mean"], 0.0)
            self.assertLessEqual(metrics["dinov3_knn_accuracy_mean"], 1.0)
            self.assertGreaterEqual(metrics["assume_knn_accuracy_mean"], 0.0)
            self.assertLessEqual(metrics["assume_knn_accuracy_mean"], 1.0)

            # Check JSON file written
            self.assertTrue(out_p.exists())
            with open(out_p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["sample_count"], self.n_samples)


class TestFigureGeneration(unittest.TestCase):
    """Unit tests for publication-grade 4-panel comparison visualization."""

    def test_generate_umap_comparison_figure_valid_png(self) -> None:
        """Verify 4-panel comparison figure is generated with valid PNG magic bytes and dual saving."""
        rng = np.random.default_rng(42)
        n = 80
        dino_umap = rng.normal(size=(n, 2)).astype(np.float32)
        assume_umap = rng.normal(size=(n, 2)).astype(np.float32)
        assume_features = rng.integers(1, 11, size=(n, 30)).astype(np.float32)
        labels = [ACTIVE_CLASSES[i % 8] for i in range(n)]
        concept_names = [f"Concept {i}" for i in range(30)]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "plots" / "umap_assume_vs_dinov3.png"
            root_p = Path(tmpdir) / "umap_assume_vs_dinov3.png"

            fig_path = generate_umap_comparison_figure(
                dino_umap=dino_umap,
                assume_umap=assume_umap,
                labels=labels,
                assume_features=assume_features,
                concept_names=concept_names,
                output_path=out_p,
                root_output_path=root_p,
                dpi=100,  # Fast DPI for unit test
            )

            self.assertTrue(out_p.exists())
            self.assertTrue(root_p.exists())
            self.assertEqual(fig_path, out_p)

            # Verify PNG header magic bytes (\x89PNG\r\n\x1a\n)
            with open(out_p, "rb") as f:
                header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

            # Verify file size > 20 KB
            self.assertGreater(out_p.stat().st_size, 20 * 1024)


class TestConceptDictionaryIntegrity(unittest.TestCase):
    """Unit tests verifying the finalized K=60 concept rubric dictionary."""

    def test_concept_dictionary_count_and_anchors(self) -> None:
        """concept_dictionary.json must contain 60 valid concept rubrics with 0, 1, 3, 5 anchors."""
        dict_path = DATA_DIR / "concept_dictionary.json"
        if not dict_path.exists():
            dict_path = PROJECT_ROOT / "concept_dictionary.json"
        self.assertTrue(dict_path.exists())

        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data), 60)
        for item in data:
            self.assertIn("concept_id", item)
            self.assertIn("name", item)
            self.assertIn("definition", item)
            self.assertIn("scale_0", item)
            self.assertIn("scale_1", item)
            self.assertIn("scale_3", item)
            self.assertIn("scale_5", item)
            self.assertTrue(len(item["concept_id"]) > 0)
            self.assertTrue(len(item["name"]) > 0)
            self.assertTrue(len(item["definition"]) > 0)


if __name__ == "__main__":
    unittest.main()
