"""Visual Merger Agent for ASSUME.

Inspects colliding concept pairs (|r| > 0.80) along with 2-3 distinguishing micrographs
to decide whether to MERGE redundant concepts into a single atomic rubric or KEEP_BOTH
orthogonal concepts that merely co-occur in the current sample set.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from assume.agents.proposer import ConceptRubric
from assume.agents.vertex_client import ResilientVertexClient

logger = logging.getLogger(__name__)


@dataclass
class MergeDecision:
    """Outcome of visual merger evaluation on colliding concept pairs.

    Attributes:
        action: "MERGE" if concepts are semantically redundant, or "KEEP_BOTH" if orthogonal.
        rationale: Physical / morphological reasoning explaining the decision.
        merged_rubric: Synthesized unified ConceptRubric (only present if action == "MERGE").
    """

    action: str  # "MERGE" or "KEEP_BOTH"
    rationale: str = ""
    merged_rubric: Optional[ConceptRubric] = None

    def __post_init__(self) -> None:
        # Standardize action string to uppercase
        clean_action = str(self.action).strip().upper()
        if "MERGE" in clean_action:
            self.action = "MERGE"
        elif "KEEP" in clean_action:
            self.action = "KEEP_BOTH"
        else:
            self.action = clean_action

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary representation."""
        return {
            "action": self.action,
            "decision": self.action,
            "rationale": self.rationale,
            "merged_rubric": self.merged_rubric.to_dict() if self.merged_rubric else None,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backwards compatibility."""
        if key in ("action", "decision"):
            return self.action
        if key == "rationale":
            return self.rationale
        if key in ("merged_rubric", "merged_concept"):
            return self.merged_rubric
        raise KeyError(f"Invalid MergeDecision key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in ("action", "decision", "rationale", "merged_rubric", "merged_concept")


MERGER_SYSTEM_PROMPT: str = (
    "You are the ASSUME Visual Merger Agent. Your mission is to maintain orthogonality and prevent semantic bloat "
    "in an autonomous Concept Bottleneck dictionary for scanning electron micrographs (SEM).\n\n"
    "When two concepts exhibit high empirical collinearity (|r| > 0.80), you inspect both concept rubrics and "
    "representative micrographs where the correlation occurs. You must decide between:\n"
    "1. 'MERGE': If the two concepts essentially measure the same underlying physical or visual manifestation in 2D SEM "
    "(e.g., surface roughness vs nanoscale granular relief; or pore area fraction vs interconnected void density). "
    "Synthesize a single, cleaner, strictly atomic ConceptRubric with unambiguous 0, 1, 3, 5 monotonic anchors.\n"
    "2. 'KEEP_BOTH': If the two concepts represent genuinely independent physical phenomena that merely co-occur "
    "in the current sample set and can be visibly distinguished (e.g., fiber diameter vs fiber straightness)."
)


class VisualMergerAgent:
    """Evaluates collinear concept pairs (|r| > 0.80) to determine MERGE or KEEP_BOTH."""

    def __init__(
        self,
        client: Optional[ResilientVertexClient] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the Visual Merger Agent.

        Args:
            client: Optional ResilientVertexClient instance.
            model: Optional model identifier string override.
        """
        if client is not None:
            self.client = client
        elif model is not None:
            self.client = ResilientVertexClient(model=model)
        else:
            self.client = ResilientVertexClient()
        self.system_prompt: str = MERGER_SYSTEM_PROMPT

    def evaluate_collision(
        self,
        rubric_a: Union[ConceptRubric, Dict[str, Any]],
        rubric_b: Union[ConceptRubric, Dict[str, Any]],
        images: Sequence[Union[Image.Image, str, Path]],
        correlation_r: Optional[float] = None,
    ) -> MergeDecision:
        """Evaluate a pair of colliding concepts on 2-3 representative micrographs.

        Args:
            rubric_a: First ConceptRubric (or dict).
            rubric_b: Second ConceptRubric (or dict).
            images: 2 to 3 representative micrographs exhibiting the high correlation.
            correlation_r: Observed Pearson correlation coefficient (e.g. 0.85).

        Returns:
            MergeDecision with action ('MERGE' or 'KEEP_BOTH') and optional merged_rubric.
        """
        if not images:
            raise ValueError("At least one representative micrograph is required to evaluate collision.")

        # Normalize rubrics to ConceptRubric
        ra = rubric_a if isinstance(rubric_a, ConceptRubric) else ConceptRubric.from_dict(rubric_a)
        rb = rubric_b if isinstance(rubric_b, ConceptRubric) else ConceptRubric.from_dict(rubric_b)

        parts: List[Dict[str, Any]] = []
        for idx, img_ref in enumerate(images):
            img_label = (
                os.path.basename(str(img_ref))
                if isinstance(img_ref, (str, Path))
                else f"Representative_Micrograph_{idx+1}"
            )
            parts.append({"text": f"[Representative Micrograph #{idx+1}: {img_label}]"})
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": self.client.encode_image(img_ref),
                }
            })

        corr_str = f"(observed Pearson correlation r = {correlation_r:.2f})" if correlation_r is not None else ""
        max_a = ra.scale_10 if (ra.scale_10 and ra.scale_10 != ra.scale_5) else ra.scale_5
        max_b = rb.scale_10 if (rb.scale_10 and rb.scale_10 != rb.scale_5) else rb.scale_5

        prompt = (
            f"A statistical collinearity collision {corr_str} was detected between the following two concepts:\n\n"
            f"--- Concept A ---\n"
            f"ID: {ra.concept_id}\n"
            f"Name: {ra.name}\n"
            f"Definition: {ra.definition}\n"
            f"Anchors: 0: {ra.scale_0} | 1: {ra.scale_1} | 3: {ra.scale_3} | 5: {max_a}\n\n"
            f"--- Concept B ---\n"
            f"ID: {rb.concept_id}\n"
            f"Name: {rb.name}\n"
            f"Definition: {rb.definition}\n"
            f"Anchors: 0: {rb.scale_0} | 1: {rb.scale_1} | 3: {rb.scale_3} | 5: {max_b}\n\n"
            f"Examine the {len(images)} representative micrographs above and decide whether to MERGE these concepts "
            "or KEEP_BOTH.\n\n"
            "Return JSON matching this exact schema:\n"
            "{\n"
            "  \"action\": \"MERGE\" | \"KEEP_BOTH\",\n"
            "  \"rationale\": \"1-2 sentence physical / morphological reasoning for the decision\",\n"
            "  \"merged_rubric\": {\n"
            "    \"name\": \"Unified Concept Name\",\n"
            "    \"definition\": \"1-sentence physical definition of the unified single property\",\n"
            "    \"scale_1\": \"Verbal description of level 1 (minimal / sparse when present)\",\n"
            "    \"scale_3\": \"Verbal description of level 3 (intermediate / moderate)\",\n"
            "    \"scale_5\": \"Verbal description of level 5 (maximal / dominant)\"\n"
            "  }\n"
            "}\n"
            "Note: If action is 'KEEP_BOTH', set 'merged_rubric' to null."
        )
        parts.append({"text": prompt})

        contents = [{"role": "user", "parts": parts}]

        text_response = self.client.generate_content(
            contents=contents,
            system_instruction=self.system_prompt,
            thinking_budget=512,  # Bounded reasoning for morphological adjudication
            max_output_tokens=4096,
            temperature=0.2,
            response_mime_type="application/json",
        )

        parsed_json = self.client.extract_clean_json(text_response)
        if not isinstance(parsed_json, dict):
            raise ValueError(f"Expected JSON object from merger agent, got {type(parsed_json).__name__}")

        raw_action = parsed_json.get("action") or parsed_json.get("decision") or "KEEP_BOTH"
        rationale = str(parsed_json.get("rationale") or "").strip()

        merged_rubric: Optional[ConceptRubric] = None
        raw_merged = parsed_json.get("merged_rubric") or parsed_json.get("merged_concept")

        action_str = "MERGE" if "MERGE" in str(raw_action).upper() else "KEEP_BOTH"

        if action_str == "MERGE" and isinstance(raw_merged, dict):
            merged_rubric = ConceptRubric.from_dict(raw_merged)

        decision = MergeDecision(
            action=action_str,
            rationale=rationale,
            merged_rubric=merged_rubric,
        )
        logger.info(
            "VisualMerger decision: %s between '%s' and '%s'.",
            decision.action, ra.name, rb.name
        )
        return decision
