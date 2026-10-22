#!/usr/bin/env python3
"""Live Gemini scoring of the evaluation subset against the verified concept dictionary.

Same steps as run_m5_bottleneck_and_umap.py, but:
- never falls back to the mock scorer (exits if Vertex credentials are missing),
- writes everything under data/real/ and never mirrors to the project root,
  so no existing file is overwritten.

Outputs (data/real/):
    eval_manifest.parquet, assume_embeddings_real.npy, eval_metrics_real.json, umap_real_vs_dinov3.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assume.config import DATA_DIR  # also loads .env
from assume.agents.proposer import ConceptRubric
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient
from assume.evaluation.bottleneck_eval import (
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    extract_subset_features,
    compute_separation_metrics,
    compute_umap_projection,
    generate_umap_comparison_figure,
)

logger = logging.getLogger("assume.real_scoring")

OUT_DIR = DATA_DIR / "real"


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Gemini scoring of the eval subset (no mock, no root mirroring).")
    parser.add_argument("--dict-path", type=Path, default=DATA_DIR / "verifier" / "concept_dictionary_verified.json")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N images (timing test).")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent images per batch of 10 (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gemini-3.8-flash", help="Vertex model id (default: gemini-3.8-flash).")
    parser.add_argument("--backup-key", action="store_true",
                        help="Use VERTEX_API_KEY_BACKUP from .env against the AI Studio Gemini API endpoint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    if not os.getenv("VERTEX_API_KEY") or not os.getenv("VERTEX_PROJECT_ID"):
        logger.error("VERTEX_API_KEY / VERTEX_PROJECT_ID missing; refusing to run without live Gemini.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_first{args.limit}" if args.limit else ""

    eval_df = create_balanced_eval_subset(
        manifest_path=DATA_DIR / "manifest.parquet",
        n_per_class=200,
        seed=args.seed,
        output_path=OUT_DIR / "eval_manifest.parquet",
        root_output_path=OUT_DIR / "eval_manifest.parquet",
    )
    if args.limit:
        # interleave classes so a short test run still covers all of them
        eval_df = eval_df.groupby("class_label", group_keys=False).head(args.limit // 8 + 1).head(args.limit)
        eval_df = eval_df.reset_index(drop=True)

    raw = json.loads(args.dict_path.read_text(encoding="utf-8"))
    concepts = [ConceptRubric.from_dict(d) for d in raw]
    logger.info("Scoring %d images x %d concepts from %s", len(eval_df), len(concepts), args.dict_path)

    client = ResilientVertexClient(model=args.model, max_retries=25, base_delay=3.0, max_delay=60.0, timeout=120.0)
    if args.backup_key:
        # prepaid AI Studio key: same request body, Gemini API endpoint instead of Vertex
        client.api_key = os.environ["VERTEX_API_KEY_BACKUP"]
        client.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent"
        client.headers["x-goog-api-key"] = client.api_key
        vertex_call = client.call

        def ai_studio_call(payload):
            # AI Studio rejects thinkingBudget=0 for Gemini 3.x; the lowest accepted setting is thinkingLevel "low"
            cfg = payload.get("generationConfig", {})
            if "thinkingConfig" in cfg:
                cfg["thinkingConfig"] = {"thinkingLevel": "low"}
            return vertex_call(payload)

        client.call = ai_studio_call
        logger.info("Using backup AI Studio key and endpoint.")
    logger.info("Model: %s", client.endpoint.rsplit("/", 1)[-1])
    scorer = ScorerAgent(client=client, max_chunk_size=10)

    t0 = time.time()
    emb = compute_bottleneck_embeddings(
        eval_df=eval_df,
        concepts=concepts,
        scorer=scorer,
        max_workers=args.workers,
        checkpoint_interval=50,
        output_path=OUT_DIR / f"assume_embeddings_real{tag}.npy",
        root_output_path=OUT_DIR / f"assume_embeddings_real{tag}.npy",
        checkpoint_path=OUT_DIR / f"assume_embeddings_real{tag}_checkpoint.npy",
    )
    dt = time.time() - t0
    logger.info("Scored %d images in %.1f s (%.2f s/image)", len(eval_df), dt, dt / max(len(eval_df), 1))
    if args.limit:
        logger.info("Projected time for 1543 images: %.1f min", dt / len(eval_df) * 1543 / 60)
        return 0

    eval_df.to_parquet(OUT_DIR / "eval_manifest.parquet")
    dino = extract_subset_features(eval_df, dinov3_features_path=DATA_DIR / "dinov3_pooled.npy")
    labels = eval_df["class_label"].values
    metrics = compute_separation_metrics(
        dino_features=dino, assume_features=emb, labels=labels,
        output_path=OUT_DIR / "eval_metrics_real.json", root_output_path=OUT_DIR / "eval_metrics_real.json",
        seed=args.seed,
    )
    generate_umap_comparison_figure(
        dino_umap=compute_umap_projection(dino, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=args.seed),
        assume_umap=compute_umap_projection(emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=args.seed),
        labels=labels, assume_features=emb, concept_names=[c.name for c in concepts],
        output_path=OUT_DIR / "umap_real_vs_dinov3.png", root_output_path=OUT_DIR / "umap_real_vs_dinov3.png",
        dpi=200,
    )
    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if "silhouette" in k or "knn_accuracy_mean" in k})
    return 0


if __name__ == "__main__":
    sys.exit(main())
