#!/usr/bin/env python3
"""Keep only the verifier-approved concept columns of the scored embeddings.

Reads data/real/assume_embeddings_real.npy (scored on data/verifier/concepts_snapshot.json),
keeps the columns of data/verifier/concept_dictionary_verified.json in that order, and writes
data/real/assume_embeddings_verified.npy plus metrics and the UMAP comparison figure.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assume.evaluation.umap_visualizer import (
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)

REAL = PROJECT_ROOT / "data" / "real"
VER = PROJECT_ROOT / "data" / "verifier"


def main() -> int:
    snapshot = json.loads((VER / "concepts_snapshot.json").read_text(encoding="utf-8"))
    verified = json.loads((VER / "concept_dictionary_verified.json").read_text(encoding="utf-8"))
    emb_all = np.load(REAL / "assume_embeddings_real.npy")
    assert emb_all.shape[1] == len(snapshot), (emb_all.shape, len(snapshot))
    col = {c["concept_id"]: j for j, c in enumerate(snapshot)}
    keep = [col[c["concept_id"]] for c in verified]
    emb = emb_all[:, keep]
    np.save(REAL / "assume_embeddings_verified.npy", emb)
    print(f"kept {len(keep)} / {len(snapshot)} columns -> {emb.shape}")

    df = pd.read_parquet(REAL / "eval_manifest.parquet")
    labels = df["class_label"].values
    dino = extract_subset_features(df, dinov3_features_path=PROJECT_ROOT / "data" / "dinov3_pooled.npy")

    for tag, e in (("all59", emb_all), ("verified", emb)):
        m = compute_separation_metrics(dino_features=dino, assume_features=e, labels=labels,
                                       output_path=REAL / f"eval_metrics_{tag}.json",
                                       root_output_path=REAL / f"eval_metrics_{tag}.json", seed=42)
        print(tag, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    names = [c["name"] for c in verified]
    generate_umap_comparison_figure(
        dino_umap=compute_umap_projection(dino, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42),
        assume_umap=compute_umap_projection(emb, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42),
        labels=labels, assume_features=emb, concept_names=names,
        output_path=REAL / "umap_verified_vs_dinov3.png", root_output_path=REAL / "umap_verified_vs_dinov3.png", dpi=200,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
