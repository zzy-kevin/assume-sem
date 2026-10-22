"""ASSUME Agent Module: Resilient Gemini 3.8 Flash Vertex AI Express Harness.

Provides:
- ResilientVertexClient: Exponential backoff with full jitter and HTTP 429 / 5xx handling.
- ConceptRubric: Structured atomic rubric with 1, 5, 10 monotonic verbal anchors.
- ProposerAgent: Discovers atomic single-property concept rubrics from microstructural batches.
- ScorerAgent: Evaluates micrographs against rubrics with <= 10 concept chunking.
- VisualMergerAgent: Evaluates collinear pairs (|r| > 0.80) to MERGE or KEEP_BOTH.
- MergeDecision: Outcome object encapsulating merge actions and unified rubrics.
"""

from __future__ import annotations

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient

__all__ = [
    "ResilientVertexClient",
    "ConceptRubric",
    "ProposerAgent",
    "ScorerAgent",
    "VisualMergerAgent",
    "MergeDecision",
]
