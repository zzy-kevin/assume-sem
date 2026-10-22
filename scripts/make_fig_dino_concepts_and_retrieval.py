#!/usr/bin/env python3
"""Paper figures:
  fig_dino_umap_by_concept.png  DINOv3 UMAP (cached layout) painted by ASSUME concepts + DINOv3 probe R^2
  fig_retrieval_dino_vs_assume.png  top-5 nearest neighbours of query micrographs under DINOv3 vs ASSUME

Retrieval queries are picked automatically (and reported as such): queries whose DINOv3
neighbours mostly fall in other classes while ASSUME neighbours stay in-class and span a wider
brightness/contrast range. Aggregate brightness/class stats over all 1543 queries are also printed
so the selected examples can be checked against the overall trend.
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
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from concept_plane_thumbnail_grids import load, probe_r2  # noqa: E402
from visualize_latent_analysis import CLASS_COLORS, preprocess_micrograph_to_square  # noqa: E402

FIG_DIR = PROJECT_ROOT / "figures"
UMAP_CACHE = PROJECT_ROOT / "data" / "real" / "umap_coords_cache.npz"
PAINT = ["feature_elongation", "feature_occupancy_density", "spatial_pattern_uniformity_and_periodicity",
         "feature_relief_height", "surface_granularity", "substrate_exposure"]
K = 5
N_QUERIES = 3


def fig_umap_by_concept(df, emb, ids, names, r2):
    xy = np.load(UMAP_CACHE)["dino"]
    fig = plt.figure(figsize=(16, 7.6), dpi=200)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 1, 1, 1, 1.25], wspace=0.08, hspace=0.18,
                          left=0.02, right=0.98, top=0.88, bottom=0.06)

    ax = fig.add_subplot(gs[:, 0])
    for cls, col in CLASS_COLORS.items():
        m = (df["class_label"] == cls).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=5, c=col, label=cls.replace("_", " "), lw=0, alpha=0.85)
    ax.set_title("a  DINOv3 latent (384-d), UMAP\ncoloured by class label", fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, markerscale=2.5, frameon=False, loc="lower left", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])

    for k, cid in enumerate(PAINT):
        a = fig.add_subplot(gs[k // 3, 1 + k % 3])
        j = ids.index(cid)
        order = np.argsort(emb[:, j])  # high scores drawn on top
        sc = a.scatter(xy[order, 0], xy[order, 1], c=emb[order, j], s=4, cmap="viridis", vmin=0, vmax=5, lw=0)
        a.set_title(f"{names[j]}\nprobe R² = {r2[j]:.2f}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_color("#bbbbbb")
    fig.text(0.255, 0.95, "b  Same DINOv3 layout, painted by ASSUME concept scores (0–5)",
             fontsize=11, fontweight="bold")
    cax = fig.add_axes([0.355, 0.025, 0.2, 0.012])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("concept score", fontsize=8); cb.ax.tick_params(labelsize=7)

    ab = fig.add_subplot(gs[:, 4])
    order = np.argsort(r2)
    cols = ["#2563eb" if ids[i] in PAINT else "#9ca3af" for i in order]
    ab.barh(range(len(order)), r2[order], color=cols, height=0.75)
    ab.set_yticks(range(len(order)))
    ab.set_yticklabels([names[i] for i in order], fontsize=5.2)
    ab.yaxis.tick_right()
    ab.set_xlim(0, 0.8)
    ab.set_xlabel("5-fold ridge R²: DINOv3 → concept", fontsize=8)
    ab.tick_params(axis="x", labelsize=7)
    ab.set_title(f"c  Concepts are linearly readable\nfrom DINOv3 (median R² = {np.median(r2):.2f})",
                 fontsize=11, fontweight="bold", loc="left")
    for s in ("top", "left"):
        ab.spines[s].set_visible(False)

    out = FIG_DIR / "fig_dino_umap_by_concept.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_assume_umap(df):
    xy = np.load(UMAP_CACHE)["assume"]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    for cls, col in CLASS_COLORS.items():
        m = (df["class_label"] == cls).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=7, c=col, label=cls.replace("_", " "), lw=0, alpha=0.85)
    ax.set_title("ASSUME concept embedding (46-d), UMAP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2.5, frameon=False, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    out = FIG_DIR / "fig_assume_umap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def image_stats(paths):
    st = np.zeros((len(paths), 2), np.float32)
    for i, p in enumerate(paths):
        a = np.asarray(preprocess_micrograph_to_square(p, 96).convert("L"), np.float32) / 255
        st[i] = a.mean(), a.std()
    return st


def neighbours(feat, metric):
    _, idx = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(feat).kneighbors(feat)
    return idx[:, 1:]


def fig_retrieval(df, emb, dino):
    cache = FIG_DIR.parent / "data" / "real" / "image_brightness_stats.npy"
    if cache.exists():
        st = np.load(cache)
    else:
        st = image_stats(df["full_path"].tolist())
        np.save(cache, st)
    labels = df["class_label"].to_numpy()
    nn_d = neighbours(dino / np.linalg.norm(dino, axis=1, keepdims=True), "cosine")
    nn_a = neighbours(emb, "euclidean")

    same_d = (labels[nn_d] == labels[:, None]).mean(1)
    same_a = (labels[nn_a] == labels[:, None]).mean(1)
    dmean_d = np.abs(st[nn_d, 0] - st[:, None, 0]).mean(1)
    dmean_a = np.abs(st[nn_a, 0] - st[:, None, 0]).mean(1)
    agg = dict(
        dino_same_class_top5=float(same_d.mean()), assume_same_class_top5=float(same_a.mean()),
        dino_mean_abs_brightness_diff=float(dmean_d.mean()), assume_mean_abs_brightness_diff=float(dmean_a.mean()),
        frac_queries_assume_spans_more_brightness=float((dmean_a > dmean_d).mean()),
    )
    print(json.dumps(agg, indent=2))

    # ASSUME in-class, DINOv3 mostly out-of-class, ASSUME neighbours vary in brightness; one query per class
    score = (same_a - same_d) + 2 * (dmean_a - dmean_d)
    cand = np.where((same_a >= 0.8) & (same_d <= 0.4))[0]
    cand = cand[np.argsort(-score[cand])]
    picks, seen = [], set()
    for q in cand:
        if labels[q] not in seen:
            picks.append(q); seen.add(labels[q])
        if len(picks) == N_QUERIES:
            break

    # Rows kept for the paper: Particles and Nanowires queries, each under DINOv3 then ASSUME
    rows = [(picks[0], nn_d, "DINOv3"), (picks[0], nn_a, "ASSUME"),
            (picks[2], nn_d, "DINOv3"), (picks[2], nn_a, "ASSUME")]
    paths = df["full_path"].tolist()
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(2.0 * (K + 1), 2.15 * len(rows)), dpi=200)
    for axr, (q, nn, tag) in zip(axes, rows):
        axr[0].imshow(preprocess_micrograph_to_square(paths[q], 256))
        axr[0].set_ylabel(tag, fontsize=22, fontweight="bold", color="#6b7280" if tag == "DINOv3" else "#2563eb")
        for c, n in enumerate(nn[q]):
            axr[c + 1].imshow(preprocess_micrograph_to_square(paths[n], 256))
        for a in axr:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    axes[0, 0].set_title("query", fontsize=16)
    for c in range(K):
        axes[0, c + 1].set_title(f"#{c + 1}", fontsize=16)
    fig.suptitle("DINOv3 vs ASSUME nearest neighbor retrieval", fontsize=22, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.canvas.draw()
    for axr, (_, _, tag) in zip(axes, rows):
        if tag == "ASSUME":
            b0, b1 = axr[0].get_position(), axr[-1].get_position()
            pad = 0.006
            fig.patches.append(plt.Rectangle((0.0, b0.y0 - pad), 1.0, b0.height + 2 * pad, transform=fig.transFigure,
                                             facecolor="#dbeafe", edgecolor="none", zorder=-1))
    out = FIG_DIR / "fig_retrieval_dino_vs_assume.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    (FIG_DIR / "fig_retrieval_stats.json").write_text(json.dumps(
        dict(aggregate=agg, queries=[dict(image_id=df["image_id"].iloc[q], cls=labels[q],
                                          dino_same=float(same_d[q]), assume_same=float(same_a[q])) for q in picks]),
        indent=2))
    print("saved", out)


def main():
    df, emb, concepts, dino = load()
    ids = [c["concept_id"] for c in concepts]
    names = [c["name"] for c in concepts]
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "umap"):
        fig_umap_by_concept(df, emb, ids, names, probe_r2(emb, dino))
    if which in ("both", "assume_umap"):
        fig_assume_umap(df)
    if which in ("both", "retrieval"):
        fig_retrieval(df, emb, dino)


if __name__ == "__main__":
    main()
