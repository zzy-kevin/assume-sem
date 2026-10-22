#!/usr/bin/env python3
"""Single-concept latent traversals: 5 micrographs per row where one ASSUME concept climbs
through 5 of its integer score levels while the other 45 concepts stay as close as possible.

For concept j and every anchor image a, each level L picks the image with score_j == L that is
nearest to a on the other concepts (RMS score difference), with DINOv3 cosine distance as a small
tie-breaker so ties among integer scores resolve to the most visually similar micrograph.
"Drift" = mean RMS difference of the other concepts from the anchor along the path (lower = cleaner).

  python scripts/make_fig_latent_traversal.py rank                 # rank concepts + contact sheet
  python scripts/make_fig_latent_traversal.py figure c1 c2 c3      # final figure for chosen concepts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from concept_plane_thumbnail_grids import load, probe_r2  # noqa: E402
from visualize_latent_analysis import preprocess_micrograph_to_square  # noqa: E402

FIG_DIR = PROJECT_ROOT / "figures"
OUT_DIR = PROJECT_ROOT / "visualizations" / "traversals"
TIE_W = 0.3  # weight of DINOv3 cosine distance (0-2) vs RMS concept difference (0-5)
MIN_PER_LEVEL = 10
SHORT = {
    "feature_elongation": "Elongation",
    "feature_occupancy_density": "Occupancy\ndensity",
    "feature_relief_height": "Relief\nheight",
    "spatial_pattern_uniformity_and_periodicity": "Pattern\nperiodicity",
}


def levels_for(col):
    """Five of the six integer levels, dropping the sparsest one."""
    counts = np.bincount(col.astype(int), minlength=6)
    keep = sorted(np.argsort(counts)[1:])
    return keep if counts[keep].min() >= MIN_PER_LEVEL else None


def best_paths(emb, zdino, j, n_paths=3):
    levels = levels_for(emb[:, j])
    if levels is None:
        return None, []
    others = np.delete(emb, j, axis=1)
    groups = [np.where(emb[:, j] == L)[0] for L in levels]
    results = []
    for a in range(len(emb)):
        path, drift = [], []
        for g in groups:
            rms = np.sqrt(((others[g] - others[a]) ** 2).mean(1))
            cos = 1 - zdino[g] @ zdino[a]
            k = np.argmin(rms + TIE_W * cos)
            path.append(int(g[k])); drift.append(rms[k])
        results.append((float(np.mean(drift)), path, a))
    results.sort(key=lambda t: t[0])
    picked, used = [], set()
    for d, p, a in results:  # distinct paths (no shared images)
        if used.isdisjoint(p):
            picked.append(dict(drift=d, path=p, anchor=a)); used.update(p)
        if len(picked) == n_paths:
            break
    return levels, picked


def rank(df, emb, ids, names, zdino, r2):
    rows, paths = [], {}
    for j, cid in enumerate(ids):
        levels, picked = best_paths(emb, zdino, j)
        if not picked:
            continue
        paths[cid] = dict(levels=[int(x) for x in levels], paths=picked)
        rows.append(dict(concept_id=cid, name=names[j], probe_r2=float(r2[j]), drift=picked[0]["drift"],
                         score=float(r2[j]) / (0.5 + picked[0]["drift"])))
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "traversal_ranking.csv", index=False)
    (OUT_DIR / "traversal_paths.json").write_text(json.dumps(paths, indent=1))
    print(table.to_string())

    top = table.head(12)
    fp = df["full_path"].tolist()
    fig, axes = plt.subplots(len(top), 5, figsize=(10, 2.1 * len(top)), dpi=110)
    for r, row in enumerate(top.itertuples()):
        info = paths[row.concept_id]
        for c, (L, i) in enumerate(zip(info["levels"], info["paths"][0]["path"])):
            a = axes[r, c]
            a.imshow(preprocess_micrograph_to_square(fp[i], 192)); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{L}", fontsize=9)
        axes[r, 0].set_ylabel(f"{row.Index}. {row.name}\nR²={row.probe_r2:.2f} drift={row.drift:.2f}", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "traversal_candidates.png")
    print("saved", OUT_DIR / "traversal_candidates.png")


def figure(df, emb, ids, names, chosen, which=None):
    paths = json.loads((OUT_DIR / "traversal_paths.json").read_text())
    which = which or [0] * len(chosen)
    fp = df["full_path"].tolist()
    fig, axes = plt.subplots(len(chosen), 5, figsize=(2.2 * 5 + 1.6, 2.35 * len(chosen)), dpi=200)
    axes = np.atleast_2d(axes)
    for r, (cid, w) in enumerate(zip(chosen, which)):
        info = paths[cid]
        for c, (L, i) in enumerate(zip(info["levels"], info["paths"][w]["path"])):
            a = axes[r, c]
            a.imshow(preprocess_micrograph_to_square(fp[i], 256)); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            a.set_title(f"{L}", fontsize=16)
        axes[r, 0].set_ylabel(SHORT.get(cid, names[ids.index(cid)]), fontsize=15, fontweight="bold")
    fig.suptitle("Single-concept latent traversal (other concepts held ~fixed)", fontsize=20, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG_DIR / "fig_latent_traversal.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


def main():
    df, emb, concepts, dino = load()
    ids = [c["concept_id"] for c in concepts]
    names = [c["name"] for c in concepts]
    zdino = dino / np.linalg.norm(dino, axis=1, keepdims=True)
    if sys.argv[1] == "rank":
        rank(df, emb, ids, names, zdino, probe_r2(emb, dino))
    else:
        chosen = [a.split(":")[0] for a in sys.argv[2:]]
        which = [int(a.split(":")[1]) if ":" in a else 0 for a in sys.argv[2:]]
        figure(df, emb, ids, names, chosen, which)


if __name__ == "__main__":
    main()
