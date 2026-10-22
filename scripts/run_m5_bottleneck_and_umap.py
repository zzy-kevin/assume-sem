#!/usr/bin/env python3
"""CLI Runner for Milestone 5: Final Zero-Shot Bottleneck Encoding & UMAP Evaluation.

Executes the end-to-end ASSUME evaluation pipeline:
1. Balanced evaluation subset extraction (1,600 micrographs, exactly 200/class across 8 classes).
2. Concept bottleneck matrix computation (1,600 x 30) against validated rubrics.
3. DINOv3 feature subset extraction (1,600 x 384).
4. 2D UMAP projections for DINOv3 and ASSUME concept bottleneck spaces.
5. Quantitative separation metrics (Cosine Silhouette & 5-fold k-NN accuracy).
6. Publication-grade 4-panel comparison visualization at 300 DPI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

# Ensure project root is on Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from assume.agents.proposer import ConceptRubric
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient
from assume.config import DATA_DIR
from assume.evaluation.bottleneck_eval import (
    MockScorerAgent,
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)

logger = logging.getLogger("assume.m5_runner")


def setup_logging(verbose: bool = False) -> None:
    """Configure stdout logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(
        description="Run ASSUME Milestone 5 Bottleneck Encoding and UMAP Evaluation."
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=200,
        help="Number of micrographs per class in evaluation subset (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and UMAP projections (default: 42).",
    )
    parser.add_argument(
        "--scorer-workers",
        type=int,
        default=2,
        help="Concurrency workers for ScorerAgent (default: 2).",
    )
    parser.add_argument(
        "--mock-scoring",
        action="store_true",
        help="Use synthetic mock scorer for fast verification test runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of bottleneck embeddings even if cached file exists.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Interval in images between disk checkpoints during scoring (default: 50).",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DATA_DIR / "manifest.parquet"),
        help="Path to full dataset manifest (default: data/manifest.parquet).",
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        default=str(DATA_DIR / "concept_dictionary.json"),
        help="Path to concept rubric dictionary JSON (default: data/concept_dictionary.json).",
    )
    parser.add_argument(
        "--dinov3-path",
        type=str,
        default=str(DATA_DIR / "dinov3_pooled.npy"),
        help="Path to precomputed DINOv3 features (default: data/dinov3_pooled.npy).",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default=str(DATA_DIR / "assume_embeddings_1600x30.npy"),
        help="Output path for bottleneck embeddings matrix (default: data/assume_embeddings_1600x30.npy).",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(DATA_DIR / "umap_assume_vs_dinov3.png"),
        help="Output path for 4-panel comparison figure (default: data/umap_assume_vs_dinov3.png).",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=str(DATA_DIR / "eval_metrics.json"),
        help="Output path for evaluation metrics JSON (default: data/eval_metrics.json).",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default=str(DATA_DIR / "eval_manifest.parquet"),
        help="Output path for evaluation manifest parquet (default: data/eval_manifest.parquet).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute M5 evaluation runner."""
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 70)
    logger.info("ASSUME Milestone 5: Zero-Shot Bottleneck Encoding & UMAP Evaluation")
    logger.info("=" * 70)
    logger.info("Samples Per Class:        %d", args.n_per_class)
    logger.info("Random Seed:              %d", args.seed)
    logger.info("Scorer Workers:           %d", args.scorer_workers)
    logger.info("Mock Scoring Mode:        %s", args.mock_scoring)
    logger.info("Force Recompute:          %s", args.force)
    logger.info("Checkpoint Interval:      %d", args.checkpoint_interval)
    logger.info("Manifest Path:            %s", args.manifest_path)
    logger.info("Concept Dictionary:       %s", args.dict_path)
    logger.info("DINOv3 Features:          %s", args.dinov3_path)
    logger.info("Output Embeddings:        %s", args.output_embeddings)
    logger.info("Output UMAP Figure:       %s", args.output_plot)
    logger.info("Output Metrics JSON:      %s", args.output_metrics)
    logger.info("Output Eval Manifest:     %s", args.output_manifest)
    logger.info("=" * 70)

    start_time = time.time()

    # 1. Balanced evaluation subset extraction
    logger.info("Step 1: Extracting balanced evaluation subset (n=%d per class)...", args.n_per_class)
    eval_df = create_balanced_eval_subset(
        manifest_path=args.manifest_path,
        n_per_class=args.n_per_class,
        seed=args.seed,
        output_path=args.output_manifest,
        root_output_path=PROJECT_ROOT / "eval_manifest.parquet",
    )
    logger.info("Evaluation subset extracted: %d rows across %d classes.", len(eval_df), eval_df["class_label"].nunique())

    # 2. Load concept rubric dictionary
    dict_p = Path(args.dict_path)
    if not dict_p.exists():
        root_dict_p = PROJECT_ROOT / "concept_dictionary.json"
        if root_dict_p.exists():
            dict_p = root_dict_p
        else:
            raise FileNotFoundError(f"Concept dictionary not found at {args.dict_path} or {root_dict_p}")

    with open(dict_p, "r", encoding="utf-8") as f:
        raw_concepts = json.load(f)
    concepts = [ConceptRubric.from_dict(d) for d in raw_concepts]
    logger.info("Loaded %d validated concept rubrics from %s.", len(concepts), dict_p)

    # 3. Initialize Scorer
    embeddings_file = Path(args.output_embeddings)
    needs_scoring = args.force or not embeddings_file.exists()

    scorer = None
    if needs_scoring:
        if args.mock_scoring:
            logger.info("Initializing MockScorerAgent (seed=%d)...", args.seed)
            scorer = MockScorerAgent(seed=args.seed)
        else:
            api_key = os.getenv("VERTEX_API_KEY")
            project_id = os.getenv("VERTEX_PROJECT_ID")
            if not api_key or not project_id:
                logger.warning(
                    "VERTEX_API_KEY / VERTEX_PROJECT_ID not found in environment; using MockScorerAgent."
                )
                scorer = MockScorerAgent(seed=args.seed)
            else:
                logger.info("Initializing live ScorerAgent with ResilientVertexClient...")
                client = ResilientVertexClient(
                    max_retries=25,
                    base_delay=3.0,
                    max_delay=60.0,
                    timeout=120.0,
                )
                scorer = ScorerAgent(client=client, max_chunk_size=10)

    # 4. Compute / load bottleneck embeddings
    logger.info("Step 2: Obtaining ASSUME bottleneck embeddings...")
    assume_embeddings = compute_bottleneck_embeddings(
        eval_df=eval_df,
        concepts=concepts,
        scorer=scorer,
        max_workers=args.scorer_workers,
        checkpoint_interval=args.checkpoint_interval,
        output_path=args.output_embeddings,
        root_output_path=PROJECT_ROOT / "assume_embeddings_1600x30.npy",
        force_recompute=args.force,
    )
    logger.info("ASSUME Bottleneck Matrix ready: shape=%s, dtype=%s", assume_embeddings.shape, assume_embeddings.dtype)

    # 5. Extract corresponding DINOv3 features
    logger.info("Step 3: Extracting DINOv3 teacher features for evaluation subset...")
    dino_features = extract_subset_features(eval_df, dinov3_features_path=args.dinov3_path)
    logger.info("DINOv3 Features extracted: shape=%s, dtype=%s", dino_features.shape, dino_features.dtype)

    # 6. Compute 2D UMAP projections
    logger.info("Step 4: Computing 2D UMAP projections (metric=cosine, n_neighbors=15, min_dist=0.1)...")
    dino_umap = compute_umap_projection(
        dino_features,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=args.seed,
    )
    assume_umap = compute_umap_projection(
        assume_embeddings,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=args.seed,
    )
    logger.info("UMAP 2D projections computed successfully.")

    # 7. Compute quantitative separation metrics
    logger.info("Step 5: Computing quantitative cluster separation metrics...")
    metrics = compute_separation_metrics(
        dino_features=dino_features,
        assume_features=assume_embeddings,
        labels=eval_df["class_label"].values,
        output_path=args.output_metrics,
        root_output_path=PROJECT_ROOT / "eval_metrics.json",
        seed=args.seed,
    )

    # 8. Generate publication-grade 4-panel comparison figure
    logger.info("Step 6: Generating publication-grade 4-panel comparison figure (DPI=%d)...", args.dpi)
    concept_names = [c.name for c in concepts]
    fig_path = generate_umap_comparison_figure(
        dino_umap=dino_umap,
        assume_umap=assume_umap,
        labels=eval_df["class_label"].values,
        assume_features=assume_embeddings,
        concept_names=concept_names,
        output_path=args.output_plot,
        root_output_path=PROJECT_ROOT / "umap_assume_vs_dinov3.png",
        dpi=args.dpi,
    )
    logger.info("Comparison figure exported to %s", fig_path)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("EVALUATION PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("Deliverables:")
    logger.info("  1. Eval Manifest:      %s", args.output_manifest)
    logger.info("  2. Assume Embeddings:  %s (shape %s)", args.output_embeddings, assume_embeddings.shape)
    logger.info("  3. Comparison Figure:  %s", args.output_plot)
    logger.info("  4. Metrics JSON:       %s", args.output_metrics)
    logger.info("-" * 70)
    logger.info("Metrics Summary:")
    logger.info("  DINOv3 Cosine Silhouette:  %.4f", metrics["dinov3_cosine_silhouette"])
    logger.info("  ASSUME Cosine Silhouette:  %.4f", metrics["assume_cosine_silhouette"])
    logger.info("  DINOv3 5-fold k-NN Acc:    %.2f%% +/- %.2f%%", metrics["dinov3_knn_accuracy_mean"] * 100, metrics["dinov3_knn_accuracy_std"] * 100)
    logger.info("  ASSUME 5-fold k-NN Acc:    %.2f%% +/- %.2f%%", metrics["assume_knn_accuracy_mean"] * 100, metrics["assume_knn_accuracy_std"] * 100)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
