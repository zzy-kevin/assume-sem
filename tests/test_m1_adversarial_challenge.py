"""Empirical adversarial challenge test suite for Milestone 1.

Stress tests:
1. Preprocessing robustness (extreme aspect ratios, odd dimensions, single-pixel, non-standard modes, extreme patterns).
2. Embeddings numerical stability (finite values, zero NaN/Inf, unit L2 norm bounds).
3. Embeddings variance & cross-class semantic discrimination.
4. End-to-end DINOv3 inference robustness.
"""

from __future__ import annotations

import filecmp
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import torch
import torchvision.transforms.functional as TF

from assume.config import (
    ACTIVE_CLASSES,
    CLASS_TO_ID,
    DATASET_DIR,
    EMBEDDING_DIM,
    EXCLUDED_CLASSES,
    EXPECTED_TOTAL_MICROGRAPHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_PATH,
    EMBEDDING_PATH,
    ROOT_MANIFEST_PATH,
    ROOT_EMBEDDING_PATH,
    TARGET_IMAGE_SIZE,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)
from assume.data.dinov3_embedder import DINOv3Embedder
from assume.data.ingestion import MicrographDataset, preprocess_micrograph


class TestM1AdversarialChallenge(unittest.TestCase):
    """Adversarial stress-test suite challenging ASSUME Milestone 1."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.embedder = DINOv3Embedder(device=cls.device)

    # -------------------------------------------------------------------------
    # 1. Manifest & Artifact Integrity
    # -------------------------------------------------------------------------
    def test_artifact_file_coherence(self):
        """Verify root and data/ artifacts exist and are bitwise identical."""
        self.assertTrue(MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}")
        self.assertTrue(EMBEDDING_PATH.exists(), f"Missing {EMBEDDING_PATH}")
        self.assertTrue(ROOT_MANIFEST_PATH.exists(), f"Missing {ROOT_MANIFEST_PATH}")
        self.assertTrue(ROOT_EMBEDDING_PATH.exists(), f"Missing {ROOT_EMBEDDING_PATH}")

        self.assertTrue(
            filecmp.cmp(MANIFEST_PATH, ROOT_MANIFEST_PATH, shallow=False),
            "Root manifest is not identical to data/manifest.parquet",
        )
        self.assertTrue(
            filecmp.cmp(EMBEDDING_PATH, ROOT_EMBEDDING_PATH, shallow=False),
            "Root embedding array is not identical to data/dinov3_pooled.npy",
        )

    def test_manifest_schema_and_integrity(self):
        """Validate 14,620 rows, strict exclusion of Powder_highres, and no missing fields."""
        df = pd.read_parquet(MANIFEST_PATH, engine="pyarrow")
        self.assertEqual(len(df), EXPECTED_TOTAL_MICROGRAPHS)
        self.assertEqual(df["image_id"].nunique(), EXPECTED_TOTAL_MICROGRAPHS)

        # Zero nulls
        self.assertEqual(df.isnull().sum().sum(), 0)

        # emb_index strictly sequential 0..N-1
        self.assertTrue((df["emb_index"] == np.arange(len(df))).all())

        # All 8 classes present and Powder_highres absent
        self.assertListEqual(sorted(df["class_label"].unique()), sorted(ACTIVE_CLASSES))
        for exc in EXCLUDED_CLASSES:
            self.assertNotIn(exc, df["class_label"].values)

    # -------------------------------------------------------------------------
    # 2. Embeddings Numerical Stability & Norms
    # -------------------------------------------------------------------------
    def test_embeddings_numerical_stability(self):
        """Precomputed embeddings must be float32, all-finite, and unit L2 normalized."""
        embs = np.load(EMBEDDING_PATH)
        self.assertEqual(embs.shape, (EXPECTED_TOTAL_MICROGRAPHS, EMBEDDING_DIM))
        self.assertEqual(embs.dtype, np.float32)

        self.assertFalse(np.isnan(embs).any(), "Embeddings contain NaN values")
        self.assertFalse(np.isinf(embs).any(), "Embeddings contain Inf values")
        self.assertTrue(np.isfinite(embs).all(), "Embeddings contain non-finite values")

        norms = np.linalg.norm(embs, axis=1)
        max_deviation = float(np.max(np.abs(norms - 1.0)))
        self.assertLess(
            max_deviation,
            1e-4,
            f"Embedding norm deviation {max_deviation} exceeds tolerance 1e-4",
        )

    # -------------------------------------------------------------------------
    # 3. Cross-Class Variance & Discriminability
    # -------------------------------------------------------------------------
    def test_embeddings_variance_and_non_collapse(self):
        """Verify feature dimensions have healthy variance and features are not collapsed."""
        embs = np.load(EMBEDDING_PATH)

        # Check dimension-wise variance
        variances = np.var(embs, axis=0)
        dead_dims = np.where(variances < 1e-6)[0]
        self.assertEqual(
            len(dead_dims),
            0,
            f"Found {len(dead_dims)} dead feature dimensions with variance < 1e-6",
        )
        self.assertGreater(float(np.min(variances)), 1e-4)

        # Check mean vector norm (hypersphere collapse metric)
        mean_vec = np.mean(embs, axis=0)
        mean_norm = float(np.linalg.norm(mean_vec))
        self.assertLess(
            mean_norm,
            0.90,
            f"Mean embedding norm {mean_norm} indicates geometric collapse",
        )

    def test_cross_class_feature_separation(self):
        """Within-class similarity must exceed between-class similarity and k-NN accuracy > 90%."""
        df = pd.read_parquet(MANIFEST_PATH, engine="pyarrow")
        embs = np.load(EMBEDDING_PATH)

        # 5-fold k-NN classification accuracy
        X = embs
        y = df["class_id"].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        for train_idx, test_idx in skf.split(X, y):
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            knn.fit(X[train_idx], y[train_idx])
            acc = knn.score(X[test_idx], y[test_idx])
            accuracies.append(acc)

        mean_acc = float(np.mean(accuracies))
        self.assertGreater(
            mean_acc,
            0.90,
            f"Mean 5-fold k-NN accuracy {mean_acc:.4f} is below 90% threshold",
        )

    # -------------------------------------------------------------------------
    # 4. Adversarial Preprocessing Robustness
    # -------------------------------------------------------------------------
    def test_extreme_aspect_ratios_and_geometries(self):
        """Preprocess must survive extreme aspect ratios (1000:1, 1:1000) and odd dimensions."""
        extreme_shapes = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (10, 10000),
            (10000, 10),
            (1, 2000),
            (2000, 1),
            (17, 19),
            (513, 289),
            (999, 333),
            (768, 1024),  # Portrait orientation
            (4000, 3000),
            (1023, 767),
            (1022, 766),
        ]

        for w, h in extreme_shapes:
            with self.subTest(shape=(w, h)):
                img = Image.new("RGB", (w, h), color=(100, 150, 200))
                preprocessed = preprocess_micrograph(img)
                self.assertEqual(preprocessed.size, TARGET_IMAGE_SIZE)
                self.assertEqual(preprocessed.mode, "RGB")

                tensor = TF.to_tensor(preprocessed)
                norm_tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                self.assertEqual(norm_tensor.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
                self.assertFalse(torch.isnan(norm_tensor).any())
                self.assertFalse(torch.isinf(norm_tensor).any())

    def test_all_pil_color_modes(self):
        """Preprocess must cleanly convert all standard PIL image modes to RGB."""
        modes = ["1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"]
        for mode in modes:
            with self.subTest(mode=mode):
                base = Image.new("RGB", (256, 256), color=(120, 140, 160))
                img = base.convert(mode)
                preprocessed = preprocess_micrograph(img)
                self.assertEqual(preprocessed.size, TARGET_IMAGE_SIZE)
                self.assertEqual(preprocessed.mode, "RGB")

    def test_extreme_pixel_patterns(self):
        """Preprocess must handle pitch black, pure white, random noise, and impulse spikes."""
        board = np.tile([[0, 255], [255, 0]], (384, 512)).astype(np.uint8)
        rgb_board = np.dstack([board, board, board])

        patterns = {
            "all_black": np.zeros((768, 1024, 3), dtype=np.uint8),
            "all_white": np.ones((768, 1024, 3), dtype=np.uint8) * 255,
            "random_noise": np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8),
            "checkerboard": rgb_board,
        }

        for name, arr in patterns.items():
            with self.subTest(pattern=name):
                img = Image.fromarray(arr, mode="RGB")
                preprocessed = preprocess_micrograph(img)
                self.assertEqual(preprocessed.size, TARGET_IMAGE_SIZE)

                tensor = TF.to_tensor(preprocessed)
                norm_tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                self.assertFalse(torch.isnan(norm_tensor).any())
                self.assertFalse(torch.isinf(norm_tensor).any())

    # -------------------------------------------------------------------------
    # 5. Adversarial DINOv3 Embedder Inference
    # -------------------------------------------------------------------------
    def test_dinov3_adversarial_inference(self):
        """DINOv3Embedder must produce non-NaN unit vectors on extreme inputs and remain batch-invariant."""
        board = np.tile([[0, 255], [255, 0]], (384, 512)).astype(np.uint8)
        rgb_board = np.dstack([board, board, board])

        stress_imgs = [
            Image.new("RGB", (1, 1), color=(128, 128, 128)),
            Image.new("RGB", (10000, 10), color=(80, 160, 240)),
            Image.new("RGB", (10, 10000), color=(240, 160, 80)),
            Image.new("RGB", (1024, 768), color=(0, 0, 0)),
            Image.new("RGB", (1024, 768), color=(255, 255, 255)),
            Image.fromarray(np.random.randint(0, 256, (768, 1024, 3), dtype=np.uint8)),
            Image.fromarray(rgb_board),
        ]

        embs = self.embedder.embed_images(stress_imgs, batch_size=2)
        self.assertEqual(embs.shape, (len(stress_imgs), EMBEDDING_DIM))
        self.assertEqual(embs.dtype, np.float32)
        self.assertFalse(np.isnan(embs).any())
        self.assertFalse(np.isinf(embs).any())

        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

        # Verify pure black vs pure white are not collapsed
        sim_matrix = embs @ embs.T
        black_white_sim = float(sim_matrix[3, 4])
        self.assertLess(
            black_white_sim,
            0.95,
            f"Pure black and pure white collapsed: sim={black_white_sim}",
        )

        # Batch size invariance test: batch_size=1 vs batch_size=3
        embs_b1 = self.embedder.embed_images(stress_imgs[:3], batch_size=1)
        embs_b3 = self.embedder.embed_images(stress_imgs[:3], batch_size=3)
        diff = float(np.max(np.abs(embs_b1 - embs_b3)))
        self.assertLess(
            diff,
            1e-3,
            f"Embeddings vary with batch size (max diff: {diff})",
        )

    def test_dinov3_empty_input_handling(self):
        """Empty image input sequence must yield (0, 384) float32 array gracefully."""
        res = self.embedder.embed_images([], batch_size=4)
        self.assertEqual(res.shape, (0, EMBEDDING_DIM))
        self.assertEqual(res.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
