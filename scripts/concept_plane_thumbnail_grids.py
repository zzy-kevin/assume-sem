#!/usr/bin/env python3
"""10x10 thumbnail grids on 2D planes spanned by two ASSUME concept axes.

Gemini concept scores are integers 0-5, so a raw 10x10 binning leaves most cells empty and
puts arbitrary images in each tie block. Each score is therefore smoothed over its k nearest
neighbours in DINOv3 space (a manifold-aware denoise) to give continuous axes. Cells show the
image whose smoothed (x, y) is closest to the cell centre.

Also ranks every concept pair by how well it slices a dense, well-spread 2D plane:
  plane_score = occupancy(10x10, >=3 imgs per cell) * (1 - |r_xy|) * sqrt(R2_x * R2_y)
where R2 is the 5-fold DINOv3 linear-probe R^2 of each concept (how visually grounded it is).

Usage:
  python scripts/concept_plane_thumbnail_grids.py                   # elongation x occupancy + top 5
  python scripts/concept_plane_thumbnail_grids.py --pair a b        # one specific pair
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from visualize_latent_analysis import build_thumbnail_latent_grid  # noqa: E402

REAL = PROJECT_ROOT / "data" / "real"
EMB_PATH = REAL / "assume_embeddings_verified.npy"
MANIFEST_PATH = REAL / "eval_manifest.parquet"
DICT_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
DINO_PATH = PROJECT_ROOT / "data" / "dinov3_pooled.npy"
OUT_DIR = PROJECT_ROOT / "visualizations" / "concept_planes"

GRID = 10
K_SMOOTH = 10
MIN_PER_CELL = 3
DEFAULT_PAIR = ("feature_elongation", "feature_occupancy_density")


def load():
    df = pd.read_parquet(MANIFEST_PATH)
    emb = np.load(EMB_PATH).astype(np.float32)
    concepts = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    dino = np.load(DINO_PATH)[df["emb_index"].to_numpy()]
    assert len(df) == len(emb) == len(dino) and emb.shape[1] == len(concepts)
    return df, emb, concepts, dino


def smooth_scores(emb, dino, k=K_SMOOTH):
    z = dino / np.linalg.norm(dino, axis=1, keepdims=True)
    _, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(z).kneighbors(z)
    return emb[idx].mean(axis=1)


def probe_r2(emb, dino):
    """5-fold ridge R^2 of each concept from DINOv3 features."""
    model = RidgeCV(alphas=np.logspace(-1, 4, 12))
    pred = cross_val_predict(model, dino, emb, cv=5)
    ss_res = ((emb - pred) ** 2).sum(0)
    ss_tot = ((emb - emb.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-9)


def occupancy(x, y, grid=GRID, min_n=MIN_PER_CELL):
    h, _, _ = np.histogram2d(x, y, bins=grid)
    return float((h >= min_n).mean())


def rank_pairs(sm, r2, ids):
    rows = []
    for i, j in itertools.combinations(range(sm.shape[1]), 2):
        r = float(np.corrcoef(sm[:, i], sm[:, j])[0, 1])
        occ = occupancy(sm[:, i], sm[:, j])
        g = float(np.sqrt(max(r2[i], 0) * max(r2[j], 0)))
        rows.append(dict(x=ids[i], y=ids[j], occupancy=occ, r_xy=r, grounding=g,
                         plane_score=occ * (1 - abs(r)) * g))
    return pd.DataFrame(rows).sort_values("plane_score", ascending=False).reset_index(drop=True)


def pick_diverse(ranked, n=5, exclude=(), max_uses=1):
    """Greedy top-n pairs, each concept used at most max_uses times so the planes differ."""
    uses, picks = {c: max_uses for c in exclude}, []
    for row in ranked.itertuples():
        if uses.get(row.x, 0) >= max_uses or uses.get(row.y, 0) >= max_uses:
            continue
        picks.append(row)
        uses[row.x] = uses.get(row.x, 0) + 1
        uses[row.y] = uses.get(row.y, 0) + 1
        if len(picks) == n:
            break
    return picks


def render(df, sm, emb, names, ids, cx, cy, out_path, stats):
    i, j = ids.index(cx), ids.index(cy)
    coords = np.stack([sm[:, i], sm[:, j]], axis=1)
    canvas, occ, extent = build_thumbnail_latent_grid(coords, df["full_path"].tolist(), grid_size=GRID, cell_size=160)

    fig = plt.figure(figsize=(11.5, 9), dpi=200)
    ax = fig.add_axes([0.07, 0.08, 0.66, 0.84])
    xmin, xmax, ymin, ymax = extent
    ax.imshow(canvas, extent=[xmin, xmax, ymin, ymax], origin="upper", aspect="auto")
    ax.set_xlabel(f"{names[i]}  →  (kNN-smoothed score, 0–5)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{names[j]}  →  (kNN-smoothed score, 0–5)", fontsize=12, fontweight="bold")
    ax.set_title(f"{names[i]}  ×  {names[j]}", fontsize=14, fontweight="bold", pad=10)

    # Side panel: density of the plane + stats
    ax2 = fig.add_axes([0.77, 0.52, 0.21, 0.40])
    ax2.hist2d(coords[:, 0], coords[:, 1], bins=GRID, range=[[xmin, xmax], [ymin, ymax]], cmap="Greys")
    ax2.set_title("images per cell", fontsize=9)
    ax2.tick_params(labelsize=7)
    txt = (f"n = {len(df)}\n"
           f"cells ≥{MIN_PER_CELL} imgs: {stats['occupancy']:.0%}\n"
           f"cells filled: {occ.mean():.0%}\n"
           f"r(x, y) = {stats['r_xy']:+.2f}\n"
           f"DINOv3 probe R²\n  x: {stats['r2x']:.2f}   y: {stats['r2y']:.2f}\n"
           f"raw-score r(x,y) = {np.corrcoef(emb[:, i], emb[:, j])[0, 1]:+.2f}\n"
           f"smoothing: k={K_SMOOTH} DINOv3 NN")
    fig.text(0.77, 0.08, txt, fontsize=9, family="monospace", va="bottom")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("X", "Y"))
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    df, emb, concepts, dino = load()
    ids = [c["concept_id"] for c in concepts]
    names = [c["name"] for c in concepts]
    sm = smooth_scores(emb, dino)
    r2 = probe_r2(emb, dino)

    ranked = rank_pairs(sm, r2, ids)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT_DIR / "pair_ranking.csv", index=False)
    pd.DataFrame({"concept_id": ids, "name": names, "probe_r2": r2, "raw_std": emb.std(0)}) \
        .sort_values("probe_r2", ascending=False).to_csv(OUT_DIR / "concept_probe_r2.csv", index=False)

    def stats_for(cx, cy):
        row = ranked[((ranked.x == cx) & (ranked.y == cy)) | ((ranked.x == cy) & (ranked.y == cx))].iloc[0]
        return dict(occupancy=row.occupancy, r_xy=row.r_xy, r2x=r2[ids.index(cx)], r2y=r2[ids.index(cy)])

    pairs = [tuple(args.pair)] if args.pair else [DEFAULT_PAIR]
    if not args.pair:
        pairs += [(p.x, p.y) for p in pick_diverse(ranked, n=args.n, exclude=DEFAULT_PAIR)]
    summary = []
    for k, (cx, cy) in enumerate(pairs):
        s = stats_for(cx, cy)
        summary.append(dict(x=cx, y=cy, **{key: float(v) for key, v in s.items()}))
        render(df, sm, emb, names, ids, cx, cy, OUT_DIR / f"plane{k}_{cx}__vs__{cy}.png", s)
    (OUT_DIR / "planes_summary.json").write_text(json.dumps(summary, indent=2))
    print(ranked.head(25).to_string())


if __name__ == "__main__":
    main()
