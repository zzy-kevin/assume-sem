"""Concept Proposer Agent for ASSUME.

Discovers human-interpretable, quantitative morphological and physical concepts
from sampled micrographs ($B=6$). Enforces strict single-property atomicity and
unambiguous monotonic scoring anchors on the 0-5 scale (0=absent, 1=sparse, 3=moderate, 5=dominant)
in structured JSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from assume.agents.vertex_client import ResilientVertexClient

logger = logging.getLogger(__name__)

DEFAULT_SCALE_0: str = "Concept or structure is absent / not applicable to this micrograph"


class ConceptRubric:
    """Atomic microstructural concept rubric with 0-5 monotonic anchors.

    Scale definition:
        0: Concept / structure is absent or not applicable to this image (scale_0).
        1: Minimal / sparse manifestation when present (scale_1).
        3: Moderate / intermediate manifestation (scale_3).
        5: Maximal / dominant manifestation (scale_5).
    """

    def __init__(
        self,
        name: str,
        definition: str = "",
        scale_1: str = "",
        scale_3: Optional[str] = None,
        scale_5: Optional[str] = None,
        concept_id: str = "",
        scale_0: str = DEFAULT_SCALE_0,
        scale_10: Optional[str] = None,
        description: Optional[str] = None,
        scoring_guide: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.name: str = str(name).strip()
        self.definition: str = str(definition or description or "").strip()
        self.scale_0: str = str(scale_0 or DEFAULT_SCALE_0).strip()
        self.scale_1: str = str(scale_1 or "Minimal / sparse manifestation of property").strip()

        if scale_3 is not None:
            self.scale_3: str = str(scale_3).strip()
            self.scale_5: str = str(scale_5 or scale_10 or "Maximal / dominant manifestation of property").strip()
            self.scale_10: str = str(scale_10 or self.scale_5).strip()
        elif scale_10 is not None:
            # Backwards-compatible initialization when called with (scale_1, scale_5, scale_10)
            self.scale_3 = str(scale_5 or "Moderate / intermediate manifestation of property").strip()
            self.scale_5 = str(scale_5 or "Moderate / intermediate manifestation of property").strip()
            self.scale_10 = str(scale_10).strip()
        else:
            self.scale_3 = str(scale_5 or "Moderate / intermediate manifestation of property").strip()
            self.scale_5 = str(scale_5 or "Maximal / dominant manifestation of property").strip()
            self.scale_10 = self.scale_5

        if concept_id:
            self.concept_id: str = str(concept_id).strip()
        else:
            cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", self.name).strip("_").lower()
            self.concept_id = cleaned or "unnamed_concept"

    @property
    def description(self) -> str:
        """Alias for definition for interface flexibility."""
        return self.definition

    @property
    def scoring_guide(self) -> str:
        """Formatted 0, 1, 3, 5 verbal anchors guide."""
        max_anchor = self.scale_10 if (self.scale_10 and self.scale_10 != self.scale_5) else self.scale_5
        return f"0: {self.scale_0}; 1: {self.scale_1}; 3: {self.scale_3}; 5: {max_anchor}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert rubric to dictionary representation."""
        max_anchor = self.scale_10 if (self.scale_10 and self.scale_10 != self.scale_5) else self.scale_5
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "definition": self.definition,
            "scale_0": self.scale_0,
            "scale_1": self.scale_1,
            "scale_3": self.scale_3,
            "scale_5": max_anchor,
            "scale_10": max_anchor,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConceptRubric):
            return NotImplemented
        max_self = self.scale_10 if (self.scale_10 and self.scale_10 != self.scale_5) else self.scale_5
        max_other = other.scale_10 if (other.scale_10 and other.scale_10 != other.scale_5) else other.scale_5
        return (
            self.concept_id == other.concept_id
            and self.name == other.name
            and self.definition == other.definition
            and self.scale_0 == other.scale_0
            and self.scale_1 == other.scale_1
            and self.scale_3 == other.scale_3
            and max_self == max_other
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like indexing for backwards and cross-module compatibility."""
        if key == "concept_id":
            return self.concept_id
        if key == "name":
            return self.name
        if key in ("definition", "description"):
            return self.definition
        if key == "scale_0":
            return self.scale_0
        if key == "scale_1":
            return self.scale_1
        if key == "scale_3":
            return self.scale_3
        if key == "scale_5":
            return self.scale_5
        if key == "scale_10":
            return self.scale_10
        if key == "scoring_guide":
            return self.scoring_guide
        raise KeyError(f"Invalid ConceptRubric attribute key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in (
            "concept_id",
            "name",
            "definition",
            "description",
            "scale_0",
            "scale_1",
            "scale_3",
            "scale_5",
            "scale_10",
            "scoring_guide",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConceptRubric:
        """Instantiate ConceptRubric from dictionary, handling 0-5 and legacy 1-10 keys gracefully."""
        name = str(data.get("name") or data.get("concept_id") or "Unnamed Concept").strip()
        definition = str(data.get("definition") or data.get("description") or "").strip()
        concept_id = str(data.get("concept_id") or "").strip()

        scale_0 = str(data.get("scale_0") or DEFAULT_SCALE_0).strip()
        scale_1 = str(data.get("scale_1") or "").strip()
        scale_3 = str(data.get("scale_3") or "").strip()
        scale_5 = str(data.get("scale_5") or "").strip()
        scale_10 = str(data.get("scale_10") or "").strip()

        # If individual anchors were not provided, parse from scoring_guide string
        scoring_guide = data.get("scoring_guide")
        if scoring_guide:
            s_str = str(scoring_guide)
            m0 = re.search(r"0\s*:\s*([^;]+)(?:;|$)", s_str)
            m1 = re.search(r"1\s*:\s*([^;]+)(?:;|$)", s_str)
            m3 = re.search(r"3\s*:\s*([^;]+)(?:;|$)", s_str)
            m5 = re.search(r"5\s*:\s*([^;]+)(?:;|$)", s_str)
            m10 = re.search(r"10\s*:\s*([^;]+)(?:;|$)", s_str)
            if m0 and not data.get("scale_0"):
                scale_0 = m0.group(1).strip()
            if m1 and not scale_1:
                scale_1 = m1.group(1).strip()
            if m3 and not scale_3:
                scale_3 = m3.group(1).strip()
            if m5 and not scale_5:
                scale_5 = m5.group(1).strip()
            if m10 and not scale_10:
                scale_10 = m10.group(1).strip()

        if scale_3:
            # New 0-5 schema: scale_1, scale_3, scale_5
            final_3 = scale_3
            final_5 = scale_5 or scale_10 or "Maximal/dominant expression of property"
            return cls(
                name=name,
                definition=definition,
                scale_1=scale_1 or "Minimal/sparse expression of property",
                scale_3=final_3,
                scale_5=final_5,
                concept_id=concept_id,
                scale_0=scale_0,
                scale_10=scale_10 or final_5,
            )
        elif scale_10:
            # Legacy 1-5-10 schema (no scale_3 key): preserve r.scale_5 and r.scale_10
            return cls(
                name=name,
                definition=definition,
                scale_1=scale_1 or "Minimal/sparse expression of property",
                scale_3=None,
                scale_5=scale_5 or "Moderate/intermediate expression of property",
                concept_id=concept_id,
                scale_0=scale_0,
                scale_10=scale_10,
            )
        else:
            return cls(
                name=name,
                definition=definition,
                scale_1=scale_1 or "Minimal/sparse expression of property",
                scale_3="Moderate/intermediate expression of property",
                scale_5=scale_5 or "Maximal/dominant expression of property",
                concept_id=concept_id,
                scale_0=scale_0,
            )


PROPOSER_SYSTEM_PROMPT: str = (
    "You are the Concept Proposer Agent for ASSUME (Auto-Synthesized Semantic Unified Microstructural Encoder).\n"
    "Your mission is to discover human-interpretable, quantitative morphological and physical concepts "
    "from scanning electron micrographs (SEM).\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. SINGLE PROPERTY (ATOMICITY): Each concept must measure strictly ONE independent morphological or physical "
    "axis (e.g., separate grain size from boundary tortuosity, pore area fraction from pore shape). "
    "NEVER combine two separate properties into a double-barreled rubric.\n"
    "2. UNAMBIGUOUS MONOTONIC 0-5 ANCHORS: Each concept is scored on [0, 5] where 0 means the concept/structure is "
    "absent or not applicable to the image, and 1 to 5 measure presence from sparse to dominant:\n"
    "   - scale_0: concept / structure is absent or not applicable to this image\n"
    "   - scale_1: minimal / sparse manifestation (when present)\n"
    "   - scale_3: moderate / intermediate manifestation\n"
    "   - scale_5: maximal / dominant manifestation\n"
    "   Anchors must be monotonically increasing in expression level such that each anchor has exactly ONE objective interpretation.\n"
    "3. RESOLVABLE BY VISION: Only propose features visibly discernable at the given SEM resolution.\n"
    "4. NO REPETITION: Do not propose concepts already present in existing_concepts."
)


class ProposerAgent:
    """Discovers atomic, single-property concept rubrics from batches of micrographs."""

    def __init__(
        self,
        client: Optional[ResilientVertexClient] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the Proposer Agent.

        Args:
            client: Optional ResilientVertexClient instance. If omitted, default client is created.
            model: Optional model identifier string override (e.g. gemini-3.8-flash, gemini-2.5-flash).
        """
        if client is not None:
            self.client = client
        elif model is not None:
            self.client = ResilientVertexClient(model=model)
        else:
            self.client = ResilientVertexClient()
        self.system_prompt: str = PROPOSER_SYSTEM_PROMPT

    def propose(
        self,
        images: Sequence[Union[Image.Image, str, Path]],
        existing_concepts: Optional[Sequence[Union[str, ConceptRubric]]] = None,
        n_proposals: int = 4,
        batch_mode: Optional[str] = None,
        collision_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[ConceptRubric]:
        """Propose n_proposals new atomic concept rubrics based on sampled micrographs.

        Args:
            images: Sequence of B=6 micrographs (PIL Images or file paths).
            existing_concepts: Optional list of already discovered concept names or rubrics.
            n_proposals: Target number of candidate concept rubrics to synthesize.
            batch_mode: Optional batch sampling mode ('fps', 'random', or 'collision').
            collision_pairs: Optional sequence of colliding image index pairs for collision mode.

        Returns:
            List of validated ConceptRubric instances.
        """
        if not images:
            raise ValueError("At least one micrograph image is required to propose concepts.")

        existing_names: List[str] = []
        if existing_concepts:
            for c in existing_concepts:
                if isinstance(c, ConceptRubric):
                    existing_names.append(c.name)
                elif isinstance(c, dict):
                    existing_names.append(c.get("name") or c.get("concept_id", ""))
                else:
                    existing_names.append(str(c))

        is_collision_mode = (
            (batch_mode is not None and str(batch_mode).lower() == "collision")
            or (collision_pairs is not None and len(collision_pairs) > 0)
        )

        num_actual_pairs = 0
        if is_collision_mode and len(images) >= 2:
            if collision_pairs is not None:
                num_actual_pairs = min(len(collision_pairs), len(images) // 2)
            else:
                num_actual_pairs = len(images) // 2

        if num_actual_pairs == 0:
            is_collision_mode = False

        parts: List[Dict[str, Any]] = []
        for idx, img_ref in enumerate(images):
            img_label = (
                os.path.basename(str(img_ref))
                if isinstance(img_ref, (str, Path))
                else f"Micrograph_{idx+1}"
            )
            if is_collision_mode and idx < 2 * num_actual_pairs:
                pair_num = (idx // 2) + 1
                member_role = "A" if (idx % 2 == 0) else "B"
                parts.append({"text": f"[Collision Pair #{pair_num} ({member_role}) - Micrograph #{idx+1}: {img_label}]"})
            else:
                parts.append({"text": f"[Micrograph #{idx+1}: {img_label}]"})
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": self.client.encode_image(img_ref),
                }
            })

        existing_prompt_str = (
            f"Already discovered concepts to avoid duplicating: {existing_names}\n"
            if existing_names
            else "No existing concepts yet (first iteration cold-start).\n"
        )

        collision_instruction = ""
        if is_collision_mode and num_actual_pairs > 0:
            pair_lines: List[str] = []
            for p_idx in range(num_actual_pairs):
                img_a_num = 2 * p_idx + 1
                img_b_num = 2 * p_idx + 2
                if collision_pairs and p_idx < len(collision_pairs):
                    ga, gb = collision_pairs[p_idx]
                    pair_lines.append(
                        f"  - Pair {p_idx + 1}: Micrograph #{img_a_num} (index {ga}) vs Micrograph #{img_b_num} (index {gb})"
                    )
                else:
                    pair_lines.append(
                        f"  - Pair {p_idx + 1}: Micrograph #{img_a_num} vs Micrograph #{img_b_num}"
                    )
            pairs_block = "\n".join(pair_lines)

            context_clause = ""
            if len(images) > 2 * num_actual_pairs:
                context_clause = (
                    f"Micrographs #{2 * num_actual_pairs + 1} to #{len(images)} provide additional "
                    "diverse visual context.\n"
                )

            collision_instruction = (
                "COLLISION DISCREPANCY MODE:\n"
                "The micrographs above contain colliding pairs that currently receive nearly identical student concept "
                "scores despite having very different visual structures in DINOv3 feature space:\n"
                f"{pairs_block}\n"
                f"{context_clause}"
                "Specifically compare the two micrographs within each colliding pair and propose new atomic "
                "concepts that distinguish them.\n\n"
            )

        user_prompt = (
            f"Examine the {len(images)} SEM micrographs above.\n"
            f"{collision_instruction}"
            f"{existing_prompt_str}\n"
            f"Propose exactly {n_proposals} NEW atomic, single-property concept rubrics capturing salient visual "
            "variations across these micrographs that are not covered by existing concepts.\n"
            "Note that score 0 is reserved for when a concept/structure is completely absent or not applicable to an image, "
            "while 1 to 5 measure presence from minimal/sparse (1), moderate/intermediate (3), to maximal/dominant (5).\n\n"
            "Respond ONLY with a JSON array conforming to this exact schema:\n"
            "[\n"
            "  {\n"
            "    \"name\": \"Short Descriptive Name\",\n"
            "    \"definition\": \"1-sentence physical definition of the single property\",\n"
            "    \"scale_1\": \"Verbal description of minimal or sparse manifestation (when present)\",\n"
            "    \"scale_3\": \"Verbal description of intermediate or moderate manifestation\",\n"
            "    \"scale_5\": \"Verbal description of maximal or dominant manifestation\"\n"
            "  }\n"
            "]"
        )
        parts.append({"text": user_prompt})

        contents = [{"role": "user", "parts": parts}]

        text_response = self.client.generate_content(
            contents=contents,
            system_instruction=self.system_prompt,
            thinking_budget=1024,
            max_output_tokens=8192,
            temperature=0.3,
            response_mime_type="application/json",
        )

        parsed_json = self.client.extract_clean_json(text_response)
        if isinstance(parsed_json, dict) and "concepts" in parsed_json:
            parsed_json = parsed_json["concepts"]

        if not isinstance(parsed_json, list):
            raise ValueError(f"Expected JSON array from proposer, got {type(parsed_json).__name__}: {parsed_json}")

        rubrics: List[ConceptRubric] = []
        for item in parsed_json:
            if isinstance(item, dict):
                rubric = ConceptRubric.from_dict(item)
                # Ensure it has basic non-empty attributes
                if rubric.name and rubric.definition:
                    rubrics.append(rubric)

        logger.info("ProposerAgent generated %d candidate concept rubrics.", len(rubrics))
        return rubrics
