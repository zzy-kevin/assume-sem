"""ASSUME Loop Module: Iterative Discovery, Pruning, and Collinear Resolution.

Provides:
- StatisticalPruner: Prunes dead dimensions (std < 0.20 or >90% identical),
  computes Pearson correlation matrix, and detects collinear pairs (|r| > 0.80).
- ConceptDiscoveryLoop: Orchestrates active sampling, LLM proposal, scoring,
  and diversity evaluation every 3 batches to discover K=30 validated atomic concepts.
"""

from __future__ import annotations

from assume.loop.discovery_loop import ConceptDiscoveryLoop
from assume.loop.pruner import StatisticalPruner

__all__ = [
    "StatisticalPruner",
    "ConceptDiscoveryLoop",
]
