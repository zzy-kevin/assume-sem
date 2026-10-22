"""DINOv3 frozen teacher feature extractor with spatial patch mean pooling and L2 normalization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from transformers import AutoModel

from assume.config import (
    DINOV3_MODEL_ID,
    EMBEDDING_DIM,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_SPECIAL_TOKENS,
    PATCH_SLICE,
    TARGET_IMAGE_SIZE,
    TOTAL_TOKENS,
)
from assume.data.ingestion import MicrographDataset, preprocess_micrograph

logger = logging.getLogger(__name__)


class DINOv3Embedder:
    """Wrapper around HuggingFace DINOv3 ViT-S/16+ feature extractor.

    Extracts spatial patch tokens (indices 5:581), performs spatial mean pooling,
    and projects onto the unit hypersphere via L2 normalization.
    """

    def __init__(
        self,
        model_id: str = DINOV3_MODEL_ID,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = True,
    ):
        """Initialize the DINOv3 embedder.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Target torch device. If None, auto-detects CUDA then CPU.
            dtype: Model torch precision (default: torch.float32).
            local_files_only: Whether to restrict to locally cached model weights.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True

        # On CPU, float16 is generally slow or unsupported for some ops; default to float32
        if self.device.type == "cpu" and dtype == torch.float16:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        logger.info(
            "Loading DINOv3 model '%s' on %s (dtype=%s, local_files_only=%s)...",
            model_id,
            self.device,
            self.dtype,
            local_files_only,
        )

        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                dtype=self.dtype,
                local_files_only=local_files_only,
            )
        except Exception as e:
            if local_files_only:
                logger.warning(
                    "Failed to load from local cache (%s); attempting online fetch...", e
                )
                self.model = AutoModel.from_pretrained(
                    model_id,
                    dtype=self.dtype,
                    local_files_only=False,
                )
            else:
                raise

        self.model.to(self.device)
        self.model.eval()

        # Sanity check model config
        hidden_size = getattr(self.model.config, "hidden_size", EMBEDDING_DIM)
        if hidden_size != EMBEDDING_DIM:
            logger.warning(
                "Model hidden_size %d does not match expected EMBEDDING_DIM %d",
                hidden_size,
                EMBEDDING_DIM,
            )

    @torch.no_grad()
    def embed_tensor_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a preprocessed normalized batch tensor.

        Args:
            pixel_values: Tensor of shape (B, 3, H, W) with ImageNet normalization.

        Returns:
            L2-normalized feature tensor of shape (B, 384) in float32.
        """
        x = pixel_values.to(device=self.device, dtype=self.dtype, non_blocking=True)
        outputs = self.model(pixel_values=x)
        last_hidden_state = outputs.last_hidden_state  # Shape: (B, total_tokens, 384)

        num_tokens = last_hidden_state.shape[1]
        if num_tokens < NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"Expected at least {NUM_SPECIAL_TOKENS} tokens, but received {num_tokens}"
            )

        # Spatial patch tokens begin at index 5 (after 1 CLS and 4 register tokens)
        patch_tokens = last_hidden_state[:, NUM_SPECIAL_TOKENS:TOTAL_TOKENS, :]

        # Mean pooling across spatial patch tokens: (B, num_patches, 384) -> (B, 384)
        pooled = patch_tokens.mean(dim=1).to(torch.float32)

        # Global L2 normalization: v / ||v||_2
        normalized = F.normalize(pooled, p=2, dim=1, eps=1e-12)
        return normalized

    def embed_images(
        self,
        images: Sequence[Union[Image.Image, str, Path]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Embed an arbitrary sequence of PIL Images or image file paths.

        Args:
            images: Sequence of PIL images or path strings.
            batch_size: Inference batch size.

        Returns:
            np.ndarray of shape (len(images), 384) with L2-normalized float32 vectors.
        """
        dataset = MicrographDataset(
            image_paths=[p if isinstance(p, (str, Path)) else "" for p in images]
        )
        # If images are already PIL objects, process directly
        embeddings_list: List[np.ndarray] = []

        total = len(images)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_slice = images[start_idx:end_idx]

            tensors = []
            for item in batch_slice:
                if isinstance(item, Image.Image):
                    prep = preprocess_micrograph(item)
                    t = TF.to_tensor(prep)
                    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                else:
                    prep = preprocess_micrograph(Path(item))
                    t = TF.to_tensor(prep)
                    t = TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                tensors.append(t)

            batch_tensor = torch.stack(tensors, dim=0)
            emb = self.embed_tensor_batch(batch_tensor)
            embeddings_list.append(emb.cpu().numpy())

        if not embeddings_list:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        return np.vstack(embeddings_list)

    def embed_manifest(
        self,
        manifest_df: pd.DataFrame,
        batch_size: int = 64,
        num_workers: int = 0,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed all micrographs in manifest_df and return full feature matrix.

        Args:
            manifest_df: DataFrame containing at least 'full_path' and 'emb_index'.
            batch_size: Batch size for DataLoader and forward passes.
            num_workers: Number of workers for DataLoader.
            show_progress: Whether to log periodic progress indicators.

        Returns:
            np.ndarray of shape (len(manifest_df), 384) in float32.
        """
        paths = manifest_df["full_path"].tolist()
        dataset = MicrographDataset(image_paths=paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        n_samples = len(manifest_df)
        out_matrix = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)

        processed = 0
        log_interval = max(1, n_samples // (batch_size * 10))

        for batch_idx, (tensors, indices) in enumerate(loader):
            embs = self.embed_tensor_batch(tensors).cpu().numpy()
            idx_arr = indices.numpy()
            out_matrix[idx_arr] = embs
            processed += len(idx_arr)

            if show_progress and (batch_idx % log_interval == 0 or processed == n_samples):
                pct = 100.0 * processed / n_samples
                logger.info(
                    "Embedded %d / %d micrographs (%.1f%%)", processed, n_samples, pct
                )

        # Verification of unit L2 norms
        norms = np.linalg.norm(out_matrix, axis=1)
        norm_diff = np.abs(norms - 1.0)
        max_diff = float(np.max(norm_diff))
        if max_diff > 1e-4:
            raise ValueError(
                f"Feature L2 normalization verification failed: max deviation {max_diff}"
            )

        logger.info(
            "Successfully embedded %d micrographs. All L2 norms verified within 1.0 ± 1e-4.",
            n_samples,
        )
        return out_matrix
