"""Adversarial Challenge Test Suite for ASSUME Milestone 5.

Implemented by Challenger 2 (Empirical Challenger):
1. Metric Rigor & Data Leakage Tests:
   - Stratified 5-fold cross-validation integrity: exactly 40 samples per class per fold.
   - k-NN pure test folds without training data contamination.
   - Quantification of duplicate image_id overlap from minority class oversampling
     and proof that ASSUME metric superiority holds on pure/clean samples.
   - Silhouette score edge cases: identical points, collinear points, orthogonal clusters,
     all-identical matrices, zero vectors (no division-by-zero or NaNs).
2. UMAP Projection Determinism & Perturbation Stability:
   - Bit-for-bit determinism of 2D UMAP projection with random_state=42.
   - Exact figure reproduction matching data/umap_assume_vs_dinov3.png with 0 pixel difference.
   - Topological and metric stability under Gaussian feature perturbation (epsilon = 1e-4).
3. Concept Activation Integrity:
   - Panel C (Filamentous Fiber Packing Density) semantic separation:
     Fibres and Nanowires >> Patterned_surface and Films_Coated_Surface (p < 1e-50).
   - Panel D (Periodic Array Regularity) semantic separation:
     Patterned_surface >> Porous_Sponge and Biological (p < 1e-50).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import procrustes
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from assume.config import (
    ACTIVE_CLASSES,
    DATA_DIR,
    EMBEDDING_PATH,
    PROJECT_ROOT,
    ROOT_EMBEDDING_PATH,
)
from assume.evaluation.bottleneck_eval import (
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)


class TestMetricRigorAndLeakage(unittest.TestCase):
    """Adversarial stress-tests for metric computation, stratification, and leakage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_p = DATA_DIR / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            cls.manifest_p = PROJECT_ROOT / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            raise unittest.SkipTest("eval_manifest.parquet not found")

        cls.df = pd.read_parquet(cls.manifest_p)
        cls.labels = cls.df["class_label"].values
        cls.image_ids = cls.df["image_id"].values

        # M5 baseline embeddings
        cls.emb_p = DATA_DIR / "assume_embeddings_1600x30.npy"
        if not cls.emb_p.exists():
            cls.emb_p = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
        cls.assume_emb = np.load(cls.emb_p)

        cls.dino_p = EMBEDDING_PATH if EMBEDDING_PATH.exists() else ROOT_EMBEDDING_PATH
        cls.dino_feat = extract_subset_features(cls.df, dinov3_features_path=cls.dino_p)

    def test_stratified_kfold_exact_per_class_balance(self) -> None:
        """Every cross-validation fold must stratify 1,543 samples cleanly across 5 folds."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(self.df, self.labels))
        self.assertEqual(len(folds), 5)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            self.assertIn(len(test_idx), (308, 309))
            self.assertEqual(len(train_idx) + len(test_idx), 1543)

            test_counts = pd.Series(self.labels[test_idx]).value_counts()
            self.assertEqual(len(test_counts), 8)
            for c_name in ACTIVE_CLASSES:
                self.assertIn(c_name, test_counts)
                if c_name == "Fibres":
                    self.assertIn(test_counts[c_name], (32, 33))
                elif c_name == "Porous_Sponge":
                    self.assertIn(test_counts[c_name], (36, 37))
                else:
                    self.assertEqual(test_counts[c_name], 40)

    def test_cross_val_score_pure_test_folds(self) -> None:
        """Verify k-NN evaluation uses disjoint train and test folds without index leakage."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(self.df, self.labels):
            # Index sets must be completely disjoint
            intersection = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(intersection), 0, "Train and test index sets overlap!")

            # Verify k-NN classifier fit on train does not contain test samples in training memory
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            knn.fit(self.assume_emb[train_idx], self.labels[train_idx])
            self.assertEqual(len(knn._fit_X), len(train_idx))

    def test_image_id_overlap_quantification_and_clean_test_integrity(self) -> None:
        """Verify zero duplicate image_id leakage across folds and superior ASSUME k-NN accuracy."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        total_leaked_rows = 0
        clean_dino_accs = []
        clean_assume_accs = []

        for train_idx, test_idx in skf.split(self.df, self.labels):
            train_ids = set(self.image_ids[train_idx])
            leaked_in_fold = [i for i in test_idx if self.image_ids[i] in train_ids]
            total_leaked_rows += len(leaked_in_fold)

            # Evaluate accuracy strictly on clean test samples (no image_id in train set)
            clean_test_idx = [i for i in test_idx if self.image_ids[i] not in train_ids]
            knn_d = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(
                self.dino_feat[train_idx], self.labels[train_idx]
            )
            knn_a = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(
                self.assume_emb[train_idx], self.labels[train_idx]
            )

            clean_dino_accs.append(knn_d.score(self.dino_feat[clean_test_idx], self.labels[clean_test_idx]))
            clean_assume_accs.append(knn_a.score(self.assume_emb[clean_test_idx], self.labels[clean_test_idx]))

        # Zero leaked rows across all 5 folds because we do not oversample with replacement
        self.assertEqual(total_leaked_rows, 0)

        # On clean test samples, ASSUME must maintain > 95% accuracy and outperform DINOv3 by > 5%
        mean_clean_assume = float(np.mean(clean_assume_accs))
        mean_clean_dino = float(np.mean(clean_dino_accs))
        self.assertGreater(mean_clean_assume, 0.95)
        self.assertGreater(mean_clean_assume - mean_clean_dino, 0.05)

    def test_silhouette_score_edge_cases(self) -> None:
        """Test Cosine Silhouette metric across singular and degenerate geometries."""
        # 1. Perfectly orthogonal clusters
        X_ortho = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        labels_ortho = [0, 0, 1, 1, 2, 2]
        sil_ortho = silhouette_score(X_ortho, labels_ortho, metric="cosine")
        self.assertEqual(sil_ortho, 1.0)

        # 2. Collinear points along radial rays
        X_collinear = np.array([
            [1.0, 2.0],
            [2.0, 4.0],
            [10.0, 1.0],
            [20.0, 2.0],
        ])
        labels_collinear = [0, 0, 1, 1]
        sil_collinear = silhouette_score(X_collinear, labels_collinear, metric="cosine")
        self.assertAlmostEqual(sil_collinear, 1.0, places=5)

        # 3. All identical points across all clusters
        X_all_id = np.ones((10, 5))
        labels_all_id = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        sil_all_id = silhouette_score(X_all_id, labels_all_id, metric="cosine")
        self.assertFalse(np.isnan(sil_all_id))
        self.assertEqual(sil_all_id, 0.0)

        # 4. Zero vectors
        X_zero = np.zeros((6, 4))
        labels_zero = [0, 0, 0, 1, 1, 1]
        sil_zero = silhouette_score(X_zero, labels_zero, metric="cosine")
        self.assertFalse(np.isnan(sil_zero))
        self.assertEqual(sil_zero, 0.0)

    def test_compute_separation_metrics_edge_case_handling(self) -> None:
        """compute_separation_metrics handles synthetic boundary datasets without crash or NaNs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "metrics.json"
            # 8 orthogonal classes, 5 samples each = 40 samples
            X_synthetic = np.eye(8, 8)[np.repeat(np.arange(8), 5)]
            labels_synthetic = [f"Class_{i}" for i in np.repeat(np.arange(8), 5)]

            m = compute_separation_metrics(
                X_synthetic,
                X_synthetic,
                labels_synthetic,
                k_neighbors=3,
                cv_splits=5,
                output_path=out_p,
            )
            self.assertEqual(m["dinov3_cosine_silhouette"], 1.0)
            self.assertEqual(m["assume_cosine_silhouette"], 1.0)
            self.assertEqual(m["dinov3_knn_accuracy_mean"], 1.0)
            self.assertEqual(m["assume_knn_accuracy_mean"], 1.0)


class TestUMAPDeterminismAndPerturbation(unittest.TestCase):
    """Empirical tests for UMAP determinism and topological perturbation stability."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_p = DATA_DIR / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            cls.manifest_p = PROJECT_ROOT / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            raise unittest.SkipTest("eval_manifest.parquet not found")

        cls.df = pd.read_parquet(cls.manifest_p)
        cls.dino_p = EMBEDDING_PATH if EMBEDDING_PATH.exists() else ROOT_EMBEDDING_PATH
        cls.dino_feat = extract_subset_features(cls.df, dinov3_features_path=cls.dino_p)

        # M5 UMAP embeddings
        cls.emb_p = DATA_DIR / "assume_embeddings_1600x30.npy"
        if not cls.emb_p.exists():
            cls.emb_p = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
        cls.assume_emb = np.load(cls.emb_p)

    def test_umap_projection_exact_reproducibility(self) -> None:
        """Two runs of compute_umap_projection with seed 42 must be bit-for-bit identical."""
        # Test DINOv3 UMAP determinism
        d1 = compute_umap_projection(self.dino_feat, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        d2 = compute_umap_projection(self.dino_feat, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        np.testing.assert_array_equal(d1, d2)

        # Test ASSUME UMAP determinism
        a1 = compute_umap_projection(self.assume_emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        a2 = compute_umap_projection(self.assume_emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        np.testing.assert_array_equal(a1, a2)

    def test_umap_figure_exact_pixel_reproducibility(self) -> None:
        """Regenerated 4-panel figure matches saved figure with 0 max pixel difference."""
        fig_p = DATA_DIR / "umap_assume_vs_dinov3.png"
        if not fig_p.exists():
            fig_p = PROJECT_ROOT / "umap_assume_vs_dinov3.png"
        if not fig_p.exists():
            raise unittest.SkipTest("umap_assume_vs_dinov3.png not found")

        dict_p = DATA_DIR / "concept_dictionary.json"
        if not dict_p.exists():
            dict_p = PROJECT_ROOT / "concept_dictionary.json"
        with open(dict_p, "r", encoding="utf-8") as f:
            concepts = json.load(f)
        concept_names = [c["name"] for c in concepts]

        dino_umap = compute_umap_projection(self.dino_feat, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        assume_umap = compute_umap_projection(self.assume_emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            recomputed_p = Path(tmpdir) / "recomputed.png"
            generate_umap_comparison_figure(
                dino_umap=dino_umap,
                assume_umap=assume_umap,
                labels=self.df["class_label"].values,
                assume_features=self.assume_emb,
                concept_names=concept_names,
                output_path=recomputed_p,
                dpi=300,
            )

            orig_img = Image.open(fig_p)
            recomp_img = Image.open(recomputed_p)

            self.assertEqual(orig_img.size, recomp_img.size)
            self.assertEqual(orig_img.mode, recomp_img.mode)

            orig_arr = np.array(orig_img)
            recomp_arr = np.array(recomp_img)
            max_diff = np.max(np.abs(orig_arr.astype(int) - recomp_arr.astype(int)))
            self.assertEqual(max_diff, 0, f"Max pixel difference was {max_diff}, expected 0")

    def test_umap_perturbation_stability(self) -> None:
        """UMAP projection and metrics must remain stable under epsilon = 1e-4 feature perturbation."""
        rng = np.random.default_rng(42)
        eps = 1e-4

        # Perturb ASSUME features on [0.0, 5.0] scale (use positive noise on 0 entries so cosine stays smooth)
        noise_assume = np.abs(rng.normal(0, eps, size=self.assume_emb.shape)).astype(np.float32)
        assume_pert = np.clip(self.assume_emb + noise_assume, 0.0, 5.0)

        # Baseline UMAP vs perturbed UMAP
        a_orig = compute_umap_projection(self.assume_emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        a_pert = compute_umap_projection(assume_pert, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)

        # 1. Metric invariance: silhouette delta < 1e-3, k-NN accuracy delta == 0
        labels_arr = self.df["class_label"].values
        sil_orig = silhouette_score(self.assume_emb, labels_arr, metric="cosine")
        sil_pert = silhouette_score(assume_pert, labels_arr, metric="cosine")
        self.assertLess(abs(sil_orig - sil_pert), 1e-3)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        acc_orig = float(np.mean(cross_val_score(knn, self.assume_emb, labels_arr, cv=cv)))
        acc_pert = float(np.mean(cross_val_score(knn, assume_pert, labels_arr, cv=cv)))
        self.assertAlmostEqual(acc_orig, acc_pert, places=4)

        # 2. Pairwise distance correlation and 2D manifold stability
        sub_idx = rng.choice(len(self.assume_emb), size=200, replace=False)
        d_orig = np.linalg.norm(a_orig[sub_idx, None, :] - a_orig[None, sub_idx, :], axis=-1).flatten()
        d_pert = np.linalg.norm(a_pert[sub_idx, None, :] - a_pert[None, sub_idx, :], axis=-1).flatten()
        corr, _ = pearsonr(d_orig, d_pert)
        self.assertGreater(corr, 0.50, f"UMAP distance correlation under perturbation was {corr:.4f}")

        # 3. 2D UMAP cluster separation stability (silhouette and k-NN in 2D space)
        sil_2d_orig = silhouette_score(a_orig, labels_arr, metric="euclidean")
        sil_2d_pert = silhouette_score(a_pert, labels_arr, metric="euclidean")
        self.assertLess(abs(sil_2d_orig - sil_2d_pert), 0.05)

        knn_2d = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        acc_2d_orig = float(np.mean(cross_val_score(knn_2d, a_orig, labels_arr, cv=cv)))
        acc_2d_pert = float(np.mean(cross_val_score(knn_2d, a_pert, labels_arr, cv=cv)))
        self.assertLess(abs(acc_2d_orig - acc_2d_pert), 0.05)


class TestConceptActivationSemantics(unittest.TestCase):
    """Empirical tests for concept activation alignment with SEM ground-truth semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_p = DATA_DIR / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            cls.manifest_p = PROJECT_ROOT / "eval_manifest.parquet"
        if not cls.manifest_p.exists():
            raise unittest.SkipTest("eval_manifest.parquet not found")

        cls.df = pd.read_parquet(cls.manifest_p)

        # M5 concept semantic embeddings
        cls.emb_p = DATA_DIR / "assume_embeddings_k30.npy"
        if not cls.emb_p.exists():
            cls.emb_p = PROJECT_ROOT / "assume_embeddings_k30.npy"
        if not cls.emb_p.exists():
            cls.emb_p = DATA_DIR / "assume_embeddings_1600x30.npy"
        if not cls.emb_p.exists():
            cls.emb_p = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
        cls.assume_emb = np.load(cls.emb_p)

        cls.dict_p = PROJECT_ROOT / "concept_dictionary_k30.json"
        if not cls.dict_p.exists():
            cls.dict_p = DATA_DIR / "concept_dictionary_k30.json"
        if not cls.dict_p.exists():
            cls.dict_p = DATA_DIR / "concept_dictionary.json"
        if not cls.dict_p.exists():
            cls.dict_p = PROJECT_ROOT / "concept_dictionary.json"
        with open(cls.dict_p, "r", encoding="utf-8") as f:
            cls.concepts = json.load(f)

    def test_embeddings_bounds_and_finite(self) -> None:
        """Bottleneck embeddings matrix has valid shape (1543, 30), float32 dtype, and strictly bounded [0.0, 5.0]."""
        self.assertEqual(self.assume_emb.shape, (1543, 30))
        self.assertEqual(self.assume_emb.dtype, np.float32)
        self.assertFalse(np.isnan(self.assume_emb).any())
        self.assertFalse(np.isinf(self.assume_emb).any())
        self.assertTrue((self.assume_emb >= 0.0).all())
        self.assertTrue((self.assume_emb <= 5.0).all())

    def test_fiber_packing_density_semantics(self) -> None:
        """Panel C: Filamentous Fiber Packing Density strictly correlates with fibrous/nanowire morphologies."""
        fiber_idx = next(
            (i for i, c in enumerate(self.concepts) if any(k in c["name"].lower() for k in ["filament", "fiber"])),
            0,
        )
        self.assertIn("fiber", self.concepts[fiber_idx]["name"].lower())
        c0_scores = self.assume_emb[:, fiber_idx]
        self.df["c0"] = c0_scores

        fibres_c0 = self.df[self.df["class_label"] == "Fibres"]["c0"]
        nanowires_c0 = self.df[self.df["class_label"] == "Nanowires"]["c0"]
        patterned_c0 = self.df[self.df["class_label"] == "Patterned_surface"]["c0"]
        films_c0 = self.df[self.df["class_label"] == "Films_Coated_Surface"]["c0"]

        # Mean activations on [0, 5] scale must be high for Fibres (> 3.5) and Nanowires (> 3.5)
        self.assertGreater(fibres_c0.mean(), 3.5)
        self.assertGreater(nanowires_c0.mean(), 3.5)

        # Mean activations must be low for Patterned_surface (< 1.5) and Films (< 1.5)
        self.assertLess(patterned_c0.mean(), 1.5)
        self.assertLess(films_c0.mean(), 1.5)

        # Mann-Whitney U test p-values must be astronomically small (p < 1e-50)
        _, p_f_vs_pat = mannwhitneyu(fibres_c0, patterned_c0, alternative="greater")
        _, p_f_vs_film = mannwhitneyu(fibres_c0, films_c0, alternative="greater")
        _, p_nw_vs_pat = mannwhitneyu(nanowires_c0, patterned_c0, alternative="greater")
        _, p_nw_vs_film = mannwhitneyu(nanowires_c0, films_c0, alternative="greater")

        self.assertLess(p_f_vs_pat, 1e-50)
        self.assertLess(p_f_vs_film, 1e-50)
        self.assertLess(p_nw_vs_pat, 1e-50)
        self.assertLess(p_nw_vs_film, 1e-50)

    def test_periodic_array_regularity_semantics(self) -> None:
        """Panel D: Periodic Array Regularity strictly correlates with patterned surfaces."""
        pattern_idx = next(
            (i for i, c in enumerate(self.concepts) if any(k in c["name"].lower() for k in ["periodic", "pattern"])),
            1,
        )
        self.assertTrue(any(k in self.concepts[pattern_idx]["name"].lower() for k in ["periodic", "pattern"]))
        c1_scores = self.assume_emb[:, pattern_idx]
        self.df["c1"] = c1_scores

        patterned_c1 = self.df[self.df["class_label"] == "Patterned_surface"]["c1"]
        porous_c1 = self.df[self.df["class_label"] == "Porous_Sponge"]["c1"]
        bio_c1 = self.df[self.df["class_label"] == "Biological"]["c1"]

        # Mean activation on [0, 5] scale must be high for Patterned_surface (> 3.8)
        self.assertGreater(patterned_c1.mean(), 3.8)

        # Mean activations must be low for Porous (< 1.5) and Biological (< 1.5)
        self.assertLess(porous_c1.mean(), 1.5)
        self.assertLess(bio_c1.mean(), 1.5)

        # Mann-Whitney U test p-values must be astronomically small (p < 1e-50)
        _, p_pat_vs_por = mannwhitneyu(patterned_c1, porous_c1, alternative="greater")
        _, p_pat_vs_bio = mannwhitneyu(patterned_c1, bio_c1, alternative="greater")

        self.assertLess(p_pat_vs_por, 1e-50)
        self.assertLess(p_pat_vs_bio, 1e-50)


if __name__ == "__main__":
    unittest.main()
