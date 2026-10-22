"""Scorer Agent for ASSUME.

Evaluates SEM micrographs against atomic concept rubrics with mandatory chunking
(groups of <= 10 concepts per prompt call) to prevent context rot and autoregressive leakage.
Returns integer scores in [0, 5] (0 = absent / not applicable, 1-5 = sparse to dominant)
using ultra-low latency direct inference (thinkingBudget=0).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from assume.agents.proposer import ConceptRubric
from assume.agents.vertex_client import ResilientVertexClient

logger = logging.getLogger(__name__)

SCORER_SYSTEM_PROMPT: str = (
    "You are the ASSUME Scorer Agent. Your role is to evaluate SEM micrographs against a provided set of "
    "atomic microstructural concept rubrics.\n\n"
    "CRITICAL SCORING RULES:\n"
    "1. STRICT INDEPENDENCE: Evaluate each concept strictly and independently based only on its explicit definition and anchors.\n"
    "2. ANCHOR ALIGNMENT (0 to 5 scale):\n"
    "   - 0 = concept or structure is completely absent / not applicable to this micrograph\n"
    "   - 1 = minimal / sparse expression of the property (when present)\n"
    "   - 3 = intermediate / moderate expression of the property\n"
    "   - 5 = maximal / dominant expression of the property\n"
    "3. INTEGER RANGE: Output MUST be an integer between 0 and 5 inclusive for each concept."
)


class ScorerAgent:
    """Evaluates micrographs against concept rubrics chunked into <= 10 concepts per call."""

    def __init__(
        self,
        client: Optional[ResilientVertexClient] = None,
        max_chunk_size: int = 10,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the Scorer Agent.

        Args:
            client: Optional ResilientVertexClient instance.
            max_chunk_size: Maximum concepts per evaluation prompt (hard upper bound 10).
            model: Optional model identifier string override.
        """
        if max_chunk_size > 10:
            raise ValueError(f"max_chunk_size cannot exceed 10 (got {max_chunk_size}).")
        if client is not None:
            self.client = client
        elif model is not None:
            self.client = ResilientVertexClient(model=model)
        else:
            self.client = ResilientVertexClient()
        self.max_chunk_size = max_chunk_size
        self.system_prompt: str = SCORER_SYSTEM_PROMPT

    @staticmethod
    def chunk_rubrics(
        rubrics: Sequence[ConceptRubric],
        chunk_size: int = 10,
    ) -> List[List[ConceptRubric]]:
        """Partition a sequence of rubrics into consecutive chunks of size <= chunk_size.

        Args:
            rubrics: Sequence of ConceptRubric items.
            chunk_size: Maximum items per chunk (default 10).

        Returns:
            List of rubric sub-lists, each of length <= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        rubric_list = list(rubrics)
        return [
            rubric_list[i : i + chunk_size]
            for i in range(0, len(rubric_list), chunk_size)
        ]

    def score_chunk(
        self,
        image: Union[Image.Image, str, Path],
        rubrics: Sequence[ConceptRubric],
    ) -> Dict[str, int]:
        """Score a single micrograph against a chunk of <= 10 concept rubrics.

        Args:
            image: PIL Image or file path.
            rubrics: Sequence of at most 10 ConceptRubric objects.

        Returns:
            Dictionary mapping concept_id to integer score in [0, 5].

        Raises:
            ValueError: If len(rubrics) > max_chunk_size (10).
        """
        if len(rubrics) > self.max_chunk_size:
            raise ValueError(
                f"Rubric chunk size ({len(rubrics)}) exceeds maximum allowed limit of {self.max_chunk_size}."
            )
        if not rubrics:
            return {}

        b64_image = self.client.encode_image(image)
        rubrics_text_blocks = []
        for r in rubrics:
            max_anchor = r.scale_10 if (r.scale_10 and r.scale_10 != r.scale_5) else r.scale_5
            block = (
                f"Concept ID: {r.concept_id}\n"
                f"Name: {r.name}\n"
                f"Definition: {r.definition}\n"
                f"Anchors: 0: {r.scale_0} | 1: {r.scale_1} | 3: {r.scale_3} | 5: {max_anchor}"
            )
            rubrics_text_blocks.append(block)

        rubrics_prompt = "\n\n".join(rubrics_text_blocks)
        user_prompt = (
            f"Examine the SEM micrograph above. Score this image against the following {len(rubrics)} concepts:\n\n"
            f"{rubrics_prompt}\n\n"
            "Return a JSON object with a single key 'scores' mapping each Concept ID to an integer in [0, 5] "
            "(where 0 = absent / not applicable, and 1 to 5 = sparse to dominant):\n"
            "{\n"
            "  \"scores\": {\n"
            "    \"<concept_id>\": <integer_0_to_5>\n"
            "  }\n"
            "}"
        )

        user_parts = [
            {"inlineData": {"mimeType": "image/jpeg", "data": b64_image}},
            {"text": user_prompt},
        ]
        contents = [{"role": "user", "parts": user_parts}]

        text_response = self.client.generate_content(
            contents=contents,
            system_instruction=self.system_prompt,
            thinking_budget=0,  # Zero-overhead fast direct inference
            max_output_tokens=2048,
            temperature=0.0,
            response_mime_type="application/json",
        )

        raw_parsed = self.client.extract_clean_json(text_response)
        scores_map = raw_parsed.get("scores", raw_parsed) if isinstance(raw_parsed, dict) else {}

        # Validate and clamp scores to [0, 5]
        cleaned: Dict[str, int] = {}
        for r in rubrics:
            # Check by concept_id first, then by name
            val = scores_map.get(r.concept_id)
            if val is None:
                val = scores_map.get(r.name, 0)

            try:
                clamped = max(0, min(5, int(round(float(val)))))
            except (ValueError, TypeError):
                clamped = 0

            cleaned[r.concept_id] = clamped

        return cleaned

    def score_single_image(
        self,
        image: Union[Image.Image, str, Path],
        rubrics: Sequence[ConceptRubric],
    ) -> Dict[str, int]:
        """Score a single micrograph across an arbitrary number of rubrics via auto-chunking.

        Args:
            image: PIL Image or file path.
            rubrics: Sequence of ConceptRubric objects (any length).

        Returns:
            Dictionary mapping concept_id to integer score in [0, 5].
        """
        chunks = self.chunk_rubrics(rubrics, chunk_size=self.max_chunk_size)
        all_scores: Dict[str, int] = {}
        for chunk in chunks:
            chunk_res = self.score_chunk(image, chunk)
            all_scores.update(chunk_res)
        return all_scores

    def score_images(
        self,
        images: Sequence[Union[Image.Image, str, Path]],
        rubrics: Sequence[ConceptRubric],
        max_workers: int = 5,
    ) -> np.ndarray:
        """Score multiple micrographs against all rubrics, returning a 2D integer numpy matrix.

        Auto-partitions rubrics into chunks of <= 10 concepts per API call.
        Utilizes concurrency across micrographs when len(images) > 1 and max_workers > 1.

        Args:
            images: List of PIL Images or image file paths.
            rubrics: List of ConceptRubric objects.
            max_workers: Max concurrent worker threads for scoring requests.

        Returns:
            Numpy array of shape (len(images), len(rubrics)) with dtype int64 in range [0, 5].
        """
        n_images = len(images)
        n_rubrics = len(rubrics)

        if n_images == 0:
            return np.zeros((0, n_rubrics), dtype=np.int64)
        if n_rubrics == 0:
            return np.zeros((n_images, 0), dtype=np.int64)

        rubric_list = list(rubrics)
        matrix = np.zeros((n_images, n_rubrics), dtype=np.int64)

        def _score_one(img_idx: int) -> Tuple[int, Dict[str, int]]:
            scores = self.score_single_image(images[img_idx], rubric_list)
            return img_idx, scores

        if max_workers > 1 and n_images > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_score_one, i) for i in range(n_images)]
                for fut in as_completed(futures):
                    img_idx, scores = fut.result()
                    for col_idx, r in enumerate(rubric_list):
                        matrix[img_idx, col_idx] = scores.get(r.concept_id, 0)
        else:
            for i in range(n_images):
                img_idx, scores = _score_one(i)
                for col_idx, r in enumerate(rubric_list):
                    matrix[img_idx, col_idx] = scores.get(r.concept_id, 0)

        return matrix
