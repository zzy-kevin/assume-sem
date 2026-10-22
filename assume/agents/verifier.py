"""Literature Verifier Agent for ASSUME.

Checks each discovered concept against retrieved literature (Valency Bond paper search)
and decides whether the concept describes specimen morphology (KEEP) or the imaging
conditions / something unrelated to morphology (PRUNE). Kept concepts carry the
supporting references forward into the concept dictionary.

Retrieval and judgement are separated: the evidence (papers per concept) is produced by
any retriever and passed in as plain dicts, and this agent only does the judgement on
Gemini via Vertex AI Express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from assume.agents.proposer import ConceptRubric
from assume.agents.vertex_client import ResilientVertexClient

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ("specimen_morphology", "imaging_condition", "other")


@dataclass
class VerificationDecision:
    """Outcome of literature verification for one concept.

    Attributes:
        concept_id: ID of the verified concept.
        action: "KEEP" if the concept describes specimen morphology, else "PRUNE".
        category: One of "specimen_morphology", "imaging_condition", "other".
        rationale: 1-2 sentence reasoning grounded in the retrieved papers.
        references: Supporting papers (title, url) selected from the provided evidence.
    """

    concept_id: str
    action: str
    category: str = "other"
    rationale: str = ""
    references: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.action = "PRUNE" if "PRUNE" in str(self.action).upper() else "KEEP"
        if self.category not in VALID_CATEGORIES:
            self.category = "other"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "action": self.action,
            "category": self.category,
            "rationale": self.rationale,
            "references": self.references,
        }


VERIFIER_SYSTEM_PROMPT: str = (
    "You are the ASSUME Verifier Agent. You check whether a concept in a concept bottleneck dictionary for "
    "scanning electron microscopy (SEM) micrographs describes the physical morphology of the specimen.\n\n"
    "You are given one concept rubric and a short list of papers retrieved from the literature for it. Classify "
    "the concept into one of:\n"
    "- 'specimen_morphology': a property of the material itself that a materials scientist would measure or "
    "describe from the micrograph (e.g. particle size, porosity, fiber alignment, grain boundaries, surface roughness).\n"
    "- 'imaging_condition': a property of how the image was acquired rather than of the specimen (e.g. focus, "
    "sharpness, shadowing, detector contrast, brightness, charging, noise, depth of field).\n"
    "- 'other': anything else unrelated to specimen morphology (e.g. image composition, background clutter).\n\n"
    "KEEP only 'specimen_morphology' concepts, PRUNE the rest. Use the retrieved papers as evidence of how the "
    "property is treated in the literature, but they may be noisy; ignore papers that are off-topic. Only cite "
    "papers from the provided list, and only ones that actually support your decision."
)


class VerifierAgent:
    """Classifies concepts as specimen morphology vs imaging condition using retrieved literature."""

    def __init__(
        self,
        client: Optional[ResilientVertexClient] = None,
        model: Optional[str] = None,
        max_papers: int = 5,
    ) -> None:
        """Initialize the Verifier Agent.

        Args:
            client: Optional ResilientVertexClient instance.
            model: Optional model identifier string override.
            max_papers: Maximum number of retrieved papers shown to the model per concept.
        """
        if client is not None:
            self.client = client
        elif model is not None:
            self.client = ResilientVertexClient(model=model)
        else:
            self.client = ResilientVertexClient()
        self.max_papers = max_papers
        self.system_prompt: str = VERIFIER_SYSTEM_PROMPT

    def verify(
        self,
        rubric: Union[ConceptRubric, Dict[str, Any]],
        papers: Sequence[Dict[str, Any]],
    ) -> VerificationDecision:
        """Verify one concept against its retrieved papers.

        Args:
            rubric: ConceptRubric (or dict) to verify.
            papers: Retrieved papers, each a dict with 'title', 'url', and optional 'abstract', 'year'.

        Returns:
            VerificationDecision with action ('KEEP' or 'PRUNE'), category, rationale and references.
        """
        r = rubric if isinstance(rubric, ConceptRubric) else ConceptRubric.from_dict(rubric)
        shown = list(papers)[: self.max_papers]

        paper_blocks = []
        for i, p in enumerate(shown):
            abstract = str(p.get("abstract") or "").strip()[:500]
            paper_blocks.append(
                f"[P{i+1}] {p.get('title', '')} ({p.get('year', '')})\n"
                f"URL: {p.get('url', '')}\n"
                f"Abstract: {abstract}"
            )
        papers_text = "\n\n".join(paper_blocks) if paper_blocks else "(no papers retrieved)"

        prompt = (
            f"--- Concept ---\n"
            f"ID: {r.concept_id}\n"
            f"Name: {r.name}\n"
            f"Definition: {r.definition}\n"
            f"Anchors: 0: {r.scale_0} | 1: {r.scale_1} | 3: {r.scale_3} | 5: {r.scale_5}\n\n"
            f"--- Retrieved papers ---\n{papers_text}\n\n"
            "Return JSON matching this exact schema:\n"
            "{\n"
            "  \"category\": \"specimen_morphology\" | \"imaging_condition\" | \"other\",\n"
            "  \"action\": \"KEEP\" | \"PRUNE\",\n"
            "  \"rationale\": \"1-2 sentences, referring to papers as P1, P2, ... where relevant\",\n"
            "  \"supporting_papers\": [\"P1\", ...]\n"
            "}"
        )
        contents = [{"role": "user", "parts": [{"text": prompt}]}]

        text_response = self.client.generate_content(
            contents=contents,
            system_instruction=self.system_prompt,
            thinking_budget=512,
            max_output_tokens=2048,
            temperature=0.0,
            response_mime_type="application/json",
        )

        parsed = self.client.extract_clean_json(text_response)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from verifier agent, got {type(parsed).__name__}")

        category = str(parsed.get("category") or "other").strip().lower()
        action = parsed.get("action") or ("KEEP" if category == "specimen_morphology" else "PRUNE")

        references: List[Dict[str, str]] = []
        for tag in parsed.get("supporting_papers") or []:
            try:
                idx = int(str(tag).strip().upper().lstrip("P")) - 1
            except ValueError:
                continue
            if 0 <= idx < len(shown):
                references.append({"title": str(shown[idx].get("title", "")), "url": str(shown[idx].get("url", ""))})

        decision = VerificationDecision(
            concept_id=r.concept_id,
            action=action,
            category=category,
            rationale=str(parsed.get("rationale") or "").strip(),
            references=references,
        )
        logger.info("Verifier decision: %s (%s) for '%s'.", decision.action, decision.category, r.name)
        return decision
