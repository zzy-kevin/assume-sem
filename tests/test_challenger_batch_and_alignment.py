"""Empirical Challenger 2 Verification Suite: Batch Size Invariance & Manifest Alignment.

Adversarially tests:
1. Batch size invariance:
   - bs=1 vs bs=5 on 5 identical images (max diff < 1e-5)
   - bs=1 vs bs=5 on 5 distinct real images (max diff < 1e-5)
2. Manifest alignment:
   - 10 randomly sampled images across manifest
   - On-the-fly embedding vs stored vectors in dinov3_pooled.npy
   - Cosine similarity > 0.9999
3. Representation health:
   - No NaNs/Infs, unit L2 norm, no dimensional collapse.
"""

from __future__ import annotations

import logging
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from assume.config import DATA_DIR, MANIFEST_PATH, EMBEDDING_PATH, ROOT_MANIFEST_PATH, ROOT_EMBEDDING_PATH
from assume.data.dinov3_embedder import DINOv3Embedder
from assume.data.ingestion import preprocess_micrograph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("challenger_m1_2")


class TestBatchSizeInvarianceAndAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("Initializing DINOv3Embedder on CUDA/CPU for empirical challenges...")
        cls.embedder = DINOv3Embedder()
        cls.manifest = pd.read_parquet(MANIFEST_PATH, engine="pyarrow")
        cls.embeddings = np.load(EMBEDDING_PATH)
        assert len(cls.manifest) == len(cls.embeddings)

    def test_challenge_1a_batch_size_invariance_identical_images(self):
        """Run forward pass on 5 identical images with batch size 1 vs batch size 5.
        
        Requirement: max absolute difference < 1e-5, cosine similarity > 0.99999.
        """
        first_img_path = self.manifest.iloc[0]["full_path"]
        logger.info("Testing Batch Size Invariance on 5 identical copies of: %s", first_img_path)

        # 5 identical paths
        identical_paths = [first_img_path] * 5

        # Embed with batch size 1
        embs_bs1 = self.embedder.embed_images(identical_paths, batch_size=1)
        # Embed with batch size 5
        embs_bs5 = self.embedder.embed_images(identical_paths, batch_size=5)

        self.assertEqual(embs_bs1.shape, (5, 384))
        self.assertEqual(embs_bs5.shape, (5, 384))

        # Check maximum absolute difference
        abs_diff = np.abs(embs_bs1 - embs_bs5)
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        logger.info(
            "Identical Images BS=1 vs BS=5 -> Max Abs Diff: %.6e, Mean Abs Diff: %.6e",
            max_abs_diff,
            mean_abs_diff,
        )

        # Compute cosine similarities
        cos_sims = []
        for i in range(5):
            sim = float(np.dot(embs_bs1[i], embs_bs5[i]) / (np.linalg.norm(embs_bs1[i]) * np.linalg.norm(embs_bs5[i])))
            cos_sims.append(sim)
        min_cos_sim = min(cos_sims)
        logger.info("Identical Images BS=1 vs BS=5 -> Min Cosine Similarity: %.8f", min_cos_sim)

        self.assertLess(
            max_abs_diff,
            1e-5,
            f"Batch size invariance violated on identical images! Max diff: {max_abs_diff:.6e} >= 1e-5",
        )
        self.assertGreater(
            min_cos_sim,
            0.99999,
            f"Cosine similarity too low on identical images: {min_cos_sim:.8f}",
        )

    def test_challenge_1b_batch_size_invariance_distinct_images(self):
        """Run forward pass on 5 distinct real images with batch size 1 vs batch size 5.
        
        Requirement: max absolute difference < 1e-5.
        """
        # Pick 5 distinct diverse images from different classes
        sampled_rows = self.manifest.drop_duplicates(subset=["class_label"]).head(5)
        paths = sampled_rows["full_path"].tolist()
        logger.info("Testing Batch Size Invariance on 5 distinct images from classes: %s", sampled_rows["class_label"].tolist())

        # Embed with batch size 1
        embs_bs1 = self.embedder.embed_images(paths, batch_size=1)
        # Embed with batch size 5
        embs_bs5 = self.embedder.embed_images(paths, batch_size=5)

        self.assertEqual(embs_bs1.shape, (5, 384))
        self.assertEqual(embs_bs5.shape, (5, 384))

        abs_diff = np.abs(embs_bs1 - embs_bs5)
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        logger.info(
            "Distinct Images BS=1 vs BS=5 -> Max Abs Diff: %.6e, Mean Abs Diff: %.6e",
            max_abs_diff,
            mean_abs_diff,
        )

        cos_sims = [
            float(np.dot(embs_bs1[i], embs_bs5[i]) / (np.linalg.norm(embs_bs1[i]) * np.linalg.norm(embs_bs5[i])))
            for i in range(5)
        ]
        min_cos_sim = min(cos_sims)
        logger.info("Distinct Images BS=1 vs BS=5 -> Min Cosine Similarity: %.8f", min_cos_sim)

        self.assertLess(
            max_abs_diff,
            1e-5,
            f"Batch size invariance violated on distinct images! Max diff: {max_abs_diff:.6e} >= 1e-5",
        )
        self.assertGreater(min_cos_sim, 0.99999)

    def test_challenge_1c_batch_size_invariance_fp32(self):
        """Run forward pass on 5 identical and 5 distinct images in FP32 mode.
        
        Demonstrates that under full precision (torch.float32), max diff < 1e-5 is strictly satisfied.
        """
        embedder_fp32 = DINOv3Embedder(dtype=torch.float32)

        # 1. Identical images
        first_img_path = self.manifest.iloc[0]["full_path"]
        identical_paths = [first_img_path] * 5
        embs_id_bs1 = embedder_fp32.embed_images(identical_paths, batch_size=1)
        embs_id_bs5 = embedder_fp32.embed_images(identical_paths, batch_size=5)
        max_diff_id = float(np.max(np.abs(embs_id_bs1 - embs_id_bs5)))
        min_cos_id = min(
            float(np.dot(embs_id_bs1[i], embs_id_bs5[i]) / (np.linalg.norm(embs_id_bs1[i]) * np.linalg.norm(embs_id_bs5[i])))
            for i in range(5)
        )
        logger.info(
            "FP32 Identical Images BS=1 vs BS=5 -> Max Abs Diff: %.6e, Min Cosine Sim: %.8f",
            max_diff_id,
            min_cos_id,
        )
        self.assertLess(max_diff_id, 1e-5)
        self.assertGreater(min_cos_id, 0.999999)

        # 2. Distinct images
        sampled_rows = self.manifest.drop_duplicates(subset=["class_label"]).head(5)
        paths = sampled_rows["full_path"].tolist()
        embs_dist_bs1 = embedder_fp32.embed_images(paths, batch_size=1)
        embs_dist_bs5 = embedder_fp32.embed_images(paths, batch_size=5)
        max_diff_dist = float(np.max(np.abs(embs_dist_bs1 - embs_dist_bs5)))
        min_cos_dist = min(
            float(np.dot(embs_dist_bs1[i], embs_dist_bs5[i]) / (np.linalg.norm(embs_dist_bs1[i]) * np.linalg.norm(embs_dist_bs5[i])))
            for i in range(5)
        )
        logger.info(
            "FP32 Distinct Images BS=1 vs BS=5 -> Max Abs Diff: %.6e, Min Cosine Sim: %.8f",
            max_diff_dist,
            min_cos_dist,
        )
        self.assertLess(max_diff_dist, 1e-5)
        self.assertGreater(min_cos_dist, 0.999999)

    def test_challenge_2_manifest_alignment(self):
        """Sample 10 random images from manifest, embed live, verify cosine sim > 0.9999 with dinov3_pooled.npy."""
        rng = random.Random(1337)
        total_rows = len(self.manifest)
        sample_indices = rng.sample(range(total_rows), 10)

        logger.info("Manifest Alignment Challenge: Evaluating 10 random sample indices: %s", sample_indices)

        results = []
        for rank, idx in enumerate(sample_indices, 1):
            row = self.manifest.iloc[idx]
            emb_index = row["emb_index"]
            self.assertEqual(idx, emb_index, f"emb_index mismatch at row {idx}: found {emb_index}")

            full_path = row["full_path"]
            image_id = row["image_id"]
            class_label = row["class_label"]
            self.assertTrue(Path(full_path).is_file(), f"File does not exist: {full_path}")

            # Live embed
            live_emb = self.embedder.embed_images([full_path], batch_size=1)[0]
            # Cached embedding
            cached_emb = self.embeddings[idx]

            # Metric calculations
            live_norm = float(np.linalg.norm(live_emb))
            cached_norm = float(np.linalg.norm(cached_emb))
            cos_sim = float(np.dot(live_emb, cached_emb) / (live_norm * cached_norm))
            max_diff = float(np.max(np.abs(live_emb - cached_emb)))

            results.append({
                "rank": rank,
                "index": idx,
                "image_id": image_id,
                "class_label": class_label,
                "cos_sim": cos_sim,
                "max_diff": max_diff,
                "live_norm": live_norm,
                "cached_norm": cached_norm,
            })

            logger.info(
                "Sample %2d [idx=%5d, class=%-20s, id=%-20s]: Cosine Sim = %.8f, Max Diff = %.6e",
                rank,
                idx,
                class_label,
                image_id,
                cos_sim,
                max_diff,
            )

            self.assertGreater(
                cos_sim,
                0.9999,
                f"Manifest alignment failed at idx {idx}! Cosine sim {cos_sim:.8f} <= 0.9999",
            )
            self.assertLess(
                max_diff,
                1e-4,
                f"Excessive deviation between live and cached embeddings at idx {idx}: {max_diff:.6e}",
            )

        min_sim = min(r["cos_sim"] for r in results)
        max_sim = max(r["cos_sim"] for r in results)
        logger.info(
            "Manifest Alignment Success: All 10 samples verified. Min Cosine Sim: %.8f, Max Cosine Sim: %.8f",
            min_sim,
            max_sim,
        )

    def test_challenge_3_artifact_health_and_diversity(self):
        """Stress-test representations across all 14,620 vectors for NaNs, Infs, norm bounds, and diversity."""
        # Check NaNs and Infs
        self.assertFalse(np.isnan(self.embeddings).any(), "dinov3_pooled.npy contains NaN values!")
        self.assertFalse(np.isinf(self.embeddings).any(), "dinov3_pooled.npy contains Inf values!")

        # Check norms
        norms = np.linalg.norm(self.embeddings, axis=1)
        max_norm_err = float(np.max(np.abs(norms - 1.0)))
        mean_norm_err = float(np.mean(np.abs(norms - 1.0)))
        logger.info("All 14,620 vectors -> Max norm error: %.6e, Mean norm error: %.6e", max_norm_err, mean_norm_err)
        self.assertLess(max_norm_err, 1e-4)

        # Check diversity: random pairwise cosine similarity should not be trivially high (e.g. not collapsed to constant)
        rng = np.random.RandomState(42)
        idx_a = rng.choice(len(self.embeddings), size=100, replace=False)
        idx_b = rng.choice(len(self.embeddings), size=100, replace=False)
        sims = np.sum(self.embeddings[idx_a] * self.embeddings[idx_b], axis=1)
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        logger.info("Random Pairwise Cosine Similarity (N=100): Mean = %.4f, Std = %.4f", mean_sim, std_sim)
        self.assertLess(mean_sim, 0.90, "Embeddings appear collapsed (mean random similarity > 0.90)!")

        # Root files synchronization check
        if ROOT_MANIFEST_PATH.exists() and ROOT_EMBEDDING_PATH.exists():
            root_manifest = pd.read_parquet(ROOT_MANIFEST_PATH, engine="pyarrow")
            root_embeddings = np.load(ROOT_EMBEDDING_PATH)
            self.assertEqual(len(root_manifest), len(self.manifest))
            self.assertEqual(root_embeddings.shape, self.embeddings.shape)
            diff_root = float(np.max(np.abs(root_embeddings - self.embeddings)))
            self.assertEqual(diff_root, 0.0, f"Root embeddings differ from data/ embeddings by {diff_root}")


if __name__ == "__main__":
    unittest.main()
