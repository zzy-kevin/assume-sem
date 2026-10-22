#!/usr/bin/env python3
"""CLI Runner for Milestone 4: Autonomous Iterative Concept Discovery & Pruning Loop.

Discovers K=30 validated, human-interpretable microstructural concepts using
Tri-Modal Active Batch Sampling and Gemini 3.8 Flash on Vertex AI Express,
with periodic diversity evaluation on DINOv3 FPS micrographs, dead dimension pruning,
and collinear concept merging.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

# Ensure project root is on Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assume.agents.merger import VisualMergerAgent
from assume.agents.proposer import ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient
from assume.config import DATA_DIR
from assume.loop.discovery_loop import ConceptDiscoveryLoop
from assume.loop.pruner import StatisticalPruner
from assume.sampling.tri_modal_sampler import TriModalBatchSampler

logger = logging.getLogger("assume.m4_runner")


def setup_logging(verbose: bool = False) -> None:
    """Configure stdout logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="Run ASSUME Milestone 4 Autonomous Concept Discovery Loop."
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=30,
        help="Target number of active validated concept rubrics to discover (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Micrographs sampled per discovery batch (default: 6).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=24,
        help="Number of diverse DINOv3 FPS micrographs for evaluation (default: 24).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=3,
        help="Evaluation interval in batches (default: 3).",
    )
    parser.add_argument(
        "--proposals-per-batch",
        type=int,
        default=4,
        help="Candidate concepts proposed per batch (default: 4).",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=0.20,
        help="Minimum standard deviation to avoid dead dimension pruning (default: 0.20).",
    )
    parser.add_argument(
        "--collinear-threshold",
        type=float,
        default=0.80,
        help="Collinear correlation threshold |r| > threshold (default: 0.80).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Safety ceiling on discovery iterations (default: 30).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DATA_DIR / "concept_dictionary.json"),
        help="Path for exported concept dictionary JSON (default: data/concept_dictionary.json).",
    )
    parser.add_argument(
        "--root-output-path",
        type=str,
        default=str(PROJECT_ROOT / "concept_dictionary.json"),
        help="Root path mirror for concept dictionary JSON (default: concept_dictionary.json).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume discovery from existing concept dictionary checkpoint if available.",
    )
    parser.add_argument(
        "--scorer-workers",
        type=int,
        default=2,
        help="Concurrency workers for ScorerAgent (default: 2).",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=1.0,
        help="Delay in seconds between discovery steps (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible active sampling (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute M4 active discovery runner."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 70)
    logger.info("ASSUME Milestone 4: Autonomous Iterative Concept Discovery Loop")
    logger.info("=" * 70)
    logger.info("Target Concepts (K):        %d", args.target_k)
    logger.info("Batch Size (B):             %d", args.batch_size)
    logger.info("Proposals Per Batch:        %d", args.proposals_per_batch)
    logger.info("Evaluation FPS Batch Size:  %d", args.eval_batch_size)
    logger.info("Evaluation Interval:        Every %d batches", args.eval_interval)
    logger.info("Dead Dimension Min Std:     %.2f", args.min_std)
    logger.info("Collinear Threshold (|r|):  %.2f", args.collinear_threshold)
    logger.info("Scorer Workers:             %d", args.scorer_workers)
    logger.info("Step Delay:                 %.1fs", args.step_delay)
    logger.info("Resume from Checkpoint:     %s", args.resume)
    logger.info("Primary Output:             %s", args.output_path)
    logger.info("Root Output:                %s", args.root_output_path)
    logger.info("=" * 70)

    start_time = time.time()

    # 1. Initialize Vertex AI client and agents with high retry budget and backoff
    logger.info("Initializing ResilientVertexClient with Gemini 3.8 Flash (retries=25, timeout=120s)...")
    client = ResilientVertexClient(
        max_retries=25,
        base_delay=3.0,
        max_delay=60.0,
        timeout=120.0,
    )
    logger.info("Vertex AI Project: %s | Endpoint: %s", client.project_id, client.endpoint)

    proposer = ProposerAgent(client=client)
    scorer = ScorerAgent(client=client, max_chunk_size=10)
    merger = VisualMergerAgent(client=client)

    # 2. Initialize Tri-Modal Sampler and Statistical Pruner
    logger.info("Initializing TriModalBatchSampler...")
    sampler = TriModalBatchSampler(seed=args.seed)
    logger.info("Loaded sampler with %d dataset micrographs.", len(sampler))

    pruner = StatisticalPruner(
        min_std=args.min_std,
        collinear_threshold=args.collinear_threshold,
    )

    # 3. Construct and run ConceptDiscoveryLoop
    loop = ConceptDiscoveryLoop(
        sampler=sampler,
        proposer=proposer,
        scorer=scorer,
        merger=merger,
        pruner=pruner,
        target_k=args.target_k,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        proposals_per_batch=args.proposals_per_batch,
        min_std=args.min_std,
        collinear_threshold=args.collinear_threshold,
        max_iterations=args.max_iterations,
        output_path=args.output_path,
        root_output_path=args.root_output_path,
        seed=args.seed,
        scorer_max_workers=args.scorer_workers,
        step_delay=args.step_delay,
        resume=args.resume,
    )

    concepts = loop.run()
    elapsed = time.time() - start_time

    # 4. Final summary and verification report
    logger.info("=" * 70)
    logger.info("DISCOVERY COMPLETE in %.1f seconds", elapsed)
    logger.info("Validated Concepts Active: %d (Target: %d)", len(concepts), args.target_k)
    logger.info("Pruned Dead Dimensions:    %d", len(loop.pruned_log))
    logger.info("Merged Collinear Pairs:    %d", len(loop.merged_log))
    logger.info("=" * 70)

    print("\n--- Finalized Concept Rubric Dictionary (K={}) ---".format(len(concepts)))
    for idx, c in enumerate(concepts, start=1):
        print(f"[{idx:02d}] {c.name} ({c.concept_id})")
        print(f"     Definition: {c.definition}")
        print(f"     0:  {c.scale_0}")
        print(f"     1:  {c.scale_1}")
        print(f"     3:  {c.scale_3}")
        print(f"     5:  {c.scale_5}\n")

    if len(concepts) < args.target_k:
        logger.error(
            "Loop failed to discover target K=%d concepts (found %d).",
            args.target_k,
            len(concepts),
        )
        return 1

    logger.info("SUCCESS: Concept dictionary saved and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
