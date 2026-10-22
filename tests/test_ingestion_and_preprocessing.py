"""Unit tests for ASSUME Milestone 1: Ingestion, Preprocessing, and DINOv3 Embedder."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch

from assume.config import (
    ACTIVE_CLASSES,
    CLASS_TO_ID,
    CROP_RATIO,
    DATASET_DIR,
    EMBEDDING_DIM,
    EXCLUDED_CLASSES,
    EXPECTED_TOTAL_MICROGRAPHS,
    TARGET_IMAGE_SIZE,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)
from assume.data.dinov3_embedder import DINOv3Embedder
from assume.data.ingestion import (
    MicrographDataset,
    build_manifest,
    preprocess_micrograph,
    scan_dataset,
)


class TestIngestionAndPreprocessing(unittest.TestCase):
    """Test suite covering micrograph ingestion, preprocessing geometry, and embeddings."""

    def test_standard_crop_and_resize_geometry(self):
        """Standard 1024x768 images must crop top 75% (576 px) and resize to (512, 288)."""
        raw_img = Image.new("RGB", (1024, 768), color=(128, 128, 128))
        preprocessed = preprocess_micrograph(raw_img)

        self.assertEqual(preprocessed.size, (TARGET_WIDTH, TARGET_HEIGHT))
        self.assertEqual(preprocessed.mode, "RGB")

    def test_preprocess_idempotency_no_double_crop(self):
        """Calling preprocess_micrograph twice must be idempotent and not double-crop."""
        arr = np.zeros((768, 1024, 3), dtype=np.uint8)
        arr[:384, :, 0] = 255       # Top 50%: Red
        arr[384:576, :, 1] = 255    # 50%-75%: Green
        arr[576:, :, 2] = 255       # Bottom 25% (databar): Blue
        raw_img = Image.fromarray(arr, mode="RGB")

        first_pass = preprocess_micrograph(raw_img)
        second_pass = preprocess_micrograph(first_pass)

        self.assertEqual(first_pass.size, (TARGET_WIDTH, TARGET_HEIGHT))
        self.assertEqual(second_pass.size, (TARGET_WIDTH, TARGET_HEIGHT))
        np.testing.assert_array_equal(np.asarray(first_pass), np.asarray(second_pass))

        # Bottom row of 288px image corresponds to row 575 of raw image (Green, not Blue or Red)
        bottom_pixel = second_pass.getpixel((256, TARGET_HEIGHT - 1))
        self.assertEqual(bottom_pixel, (0, 255, 0))

        # Verify path-based reload of an already (512, 288) image does not double-crop or leak file handles on Windows
        tmp_path = Path("tests/tmp_preprocessed_512x288.png")
        first_pass.save(tmp_path)
        try:
            reloaded = preprocess_micrograph(tmp_path)
            self.assertEqual(reloaded.size, (TARGET_WIDTH, TARGET_HEIGHT))
            self.assertEqual(reloaded.getpixel((256, TARGET_HEIGHT - 1)), (0, 255, 0))
            tmp_path.unlink()  # Must succeed on Windows while `reloaded` is still in scope
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_non_standard_resolutions(self):
        """Relative proportional crop must correctly handle non-standard dimensions."""
        test_resolutions = [
            (3072, 2304),  # 3x high resolution
            (2048, 1536),  # 2x high resolution
            (1023, 767),   # 1-pixel boundary truncation
            (1022, 766),   # 2-pixel boundary truncation
            (512, 384),    # 0.5x low resolution
        ]

        for w, h in test_resolutions:
            with self.subTest(resolution=(w, h)):
                raw_img = Image.new("RGB", (w, h), color=(100, 150, 200))
                preprocessed = preprocess_micrograph(raw_img)
                self.assertEqual(
                    preprocessed.size,
                    TARGET_IMAGE_SIZE,
                    f"Failed to resize resolution {(w, h)} to {TARGET_IMAGE_SIZE}",
                )
                self.assertEqual(preprocessed.mode, "RGB")

    def test_greyscale_and_alpha_channel_conversion(self):
        """Greyscale ('L') and RGBA images must be cleanly converted to 3-channel RGB."""
        img_gray = Image.new("L", (1024, 768), color=120)
        prep_gray = preprocess_micrograph(img_gray)
        self.assertEqual(prep_gray.mode, "RGB")
        self.assertEqual(prep_gray.size, TARGET_IMAGE_SIZE)

        img_rgba = Image.new("RGBA", (1024, 768), color=(100, 100, 100, 255))
        prep_rgba = preprocess_micrograph(img_rgba)
        self.assertEqual(prep_rgba.mode, "RGB")
        self.assertEqual(prep_rgba.size, TARGET_IMAGE_SIZE)

    def test_manifest_columns_and_counts(self):
        """Scan real dataset and verify exact file counts, exclusion, and schema."""
        if not DATASET_DIR.exists():
            self.skipTest(f"Dataset directory not found: {DATASET_DIR}")

        manifest = build_manifest(dataset_dir=DATASET_DIR)

        # 1. Total row count assertion
        self.assertEqual(
            len(manifest),
            EXPECTED_TOTAL_MICROGRAPHS,
            f"Expected {EXPECTED_TOTAL_MICROGRAPHS} micrographs, found {len(manifest)}",
        )

        # 2. Schema check
        expected_columns = [
            "image_id",
            "class_label",
            "class_id",
            "rel_path",
            "full_path",
            "emb_index",
            "raw_w",
            "raw_h",
        ]
        self.assertListEqual(list(manifest.columns), expected_columns)

        # 3. Class checks
        unique_classes = sorted(manifest["class_label"].unique())
        self.assertListEqual(unique_classes, sorted(ACTIVE_CLASSES))
        for excluded in EXCLUDED_CLASSES:
            self.assertNotIn(excluded, unique_classes)

        # 4. Sequential emb_index check
        emb_indices = manifest["emb_index"].tolist()
        self.assertListEqual(emb_indices, list(range(EXPECTED_TOTAL_MICROGRAPHS)))

        # 5. Image ID uniqueness check
        self.assertEqual(manifest["image_id"].nunique(), EXPECTED_TOTAL_MICROGRAPHS)

        # 6. File existence check
        first_path = Path(manifest.iloc[0]["full_path"])
        last_path = Path(manifest.iloc[-1]["full_path"])
        self.assertTrue(first_path.is_file(), f"Path does not exist: {first_path}")
        self.assertTrue(last_path.is_file(), f"Path does not exist: {last_path}")

    def test_dataset_class_to_tensor(self):
        """MicrographDataset must yield normalized PyTorch tensors with correct shape."""
        # Create 2 temporary test images
        img1 = Image.new("RGB", (1024, 768), color=(200, 200, 200))
        img2 = Image.new("RGB", (3072, 2304), color=(50, 50, 50))
        tmp_dir = Path("tests/tmp_test_images")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        p1 = tmp_dir / "test1.jpg"
        p2 = tmp_dir / "test2.jpg"
        img1.save(p1)
        img2.save(p2)

        try:
            ds = MicrographDataset([p1, p2])
            self.assertEqual(len(ds), 2)

            t0, idx0 = ds[0]
            self.assertEqual(idx0, 0)
            self.assertEqual(t0.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
            self.assertEqual(t0.dtype, torch.float32)

            t1, idx1 = ds[1]
            self.assertEqual(idx1, 1)
            self.assertEqual(t1.shape, (3, TARGET_HEIGHT, TARGET_WIDTH))
            self.assertEqual(t1.dtype, torch.float32)
        finally:
            if p1.exists():
                p1.unlink()
            if p2.exists():
                p2.unlink()
            if tmp_dir.exists():
                tmp_dir.rmdir()

    def test_dinov3_embedder_tensor_batch(self):
        """DINOv3Embedder must output (B, 384) unit-normalized vectors."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = DINOv3Embedder(device=device)

        batch_size = 3
        # Dummy batch of normalized tensors: (B, 3, 288, 512)
        dummy_tensor = torch.randn(
            (batch_size, 3, TARGET_HEIGHT, TARGET_WIDTH),
            dtype=torch.float32,
        )

        with torch.no_grad():
            embs = embedder.embed_tensor_batch(dummy_tensor)

        self.assertEqual(embs.shape, (batch_size, EMBEDDING_DIM))
        self.assertEqual(embs.dtype, torch.float32)

        # Check L2 normalization: ||v||_2 == 1.0
        norms = torch.norm(embs, p=2, dim=1).cpu().numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_dinov3_embedder_images(self):
        """DINOv3Embedder.embed_images must handle PIL images and return (N, 384) float32."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = DINOv3Embedder(device=device)

        imgs = [
            Image.new("RGB", (1024, 768), color=(100, 100, 100)),
            Image.new("RGB", (2048, 1536), color=(200, 200, 200)),
        ]

        embs = embedder.embed_images(imgs, batch_size=2)
        self.assertIsInstance(embs, np.ndarray)
        self.assertEqual(embs.shape, (2, EMBEDDING_DIM))
        self.assertEqual(embs.dtype, np.float32)

        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
