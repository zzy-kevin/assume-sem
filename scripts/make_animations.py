#!/usr/bin/env python3
"""Video animations (1920x1080, 30 fps, written to figures/anim_*.mp4).

    python scripts/make_animations.py morph recolor loop
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (AGENT, AGENT_EDGE, INK, MODEL, MODEL_EDGE, MUTED, OUTC, OUTC_EDGE, thumb)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "figures"
REAL = PROJECT_ROOT / "data" / "real"
VER = PROJECT_ROOT / "data" / "verifier"
CACHE = REAL / "umap_coords_cache.npz"
FPS = 30
CLASS_COLORS = {
    "Biological": "#4E79A7", "Fibres": "#F28E2B", "Films_Coated_Surface": "#E15759", "Nanowires": "#76B7B2",
    "Particles": "#59A14F", "Patterned_surface": "#EDC948", "Porous_Sponge": "#B07AA1", "Powder": "#9C755F",
}
plt.rcParams.update({"font.family": "DejaVu Sans"})


def bezier_ease(t, p1=(0.42, 0.0), p2=(0.58, 1.0)):
    """CSS-style cubic-bezier easing: solve x(s) = t for s, return y(s)."""
    t = float(np.clip(t, 0, 1))
    s = t
    for _ in range(8):
        x = 3 * (1 - s) ** 2 * s * p1[0] + 3 * (1 - s) * s ** 2 * p2[0] + s ** 3 - t
        dx = 3 * (1 - s) ** 2 * p1[0] + 6 * (1 - s) * s * (p2[0] - p1[0]) + 3 * s ** 2 * (1 - p2[0])
        if abs(dx) < 1e-6:
            break
        s = float(np.clip(s - x / dx, 0, 1))
    return 3 * (1 - s) ** 2 * s * p1[1] + 3 * (1 - s) * s ** 2 * p2[1] + s ** 3


def ramp(t, t0, dur):
    return float(np.clip((t - t0) / dur, 0, 1))


def load():
    df = pd.read_parquet(REAL / "eval_manifest.parquet")
    emb = np.load(REAL / "assume_embeddings_verified.npy")
    concepts = json.loads((VER / "concept_dictionary_verified.json").read_text(encoding="utf-8"))
    if CACHE.exists():
        z = np.load(CACHE)
        return df, emb, concepts, z["dino"], z["assume"]
    import umap
    dino = np.load(PROJECT_ROOT / "data" / "dinov3_pooled.npy")[df["emb_index"].values]
    kw = dict(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    ud = umap.UMAP(**kw).fit_transform(dino)
    ua = umap.UMAP(**kw).fit_transform(emb)
    # standardize both, then rotate/flip ASSUME onto DINOv3 (UMAP orientation is arbitrary) to keep the morph short
    ud = (ud - ud.mean(0)) / ud.std(0)
    ua = (ua - ua.mean(0)) / ua.std(0)
    u, _, vt = np.linalg.svd(ua.T @ ud)
    ua = ua @ (u @ vt)
    np.savez(CACHE, dino=ud, assume=ua)
    return df, emb, concepts, ud, ua


def new_fig():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    return fig


def scatter_axes(fig, pts):
    ax = fig.add_axes([0.05, 0.06, 0.72, 0.8])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    lo, hi = pts.min(0), pts.max(0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    return ax


def class_legend(fig):
    lx = fig.add_axes([0.79, 0.25, 0.2, 0.5])
    lx.axis("off")
    for k, (c, col) in enumerate(CLASS_COLORS.items()):
        lx.scatter([0.05], [1 - k * 0.12], s=160, color=col)
        lx.text(0.14, 1 - k * 0.12, c.replace("_", " "), va="center", fontsize=15, color=INK)
    lx.set_xlim(0, 1)
    lx.set_ylim(0.05, 1.05)
    return lx


def anim_morph():
    df, emb, concepts, ud, ua = load()
    cols = np.array([CLASS_COLORS[c] for c in df["class_label"]])
    fig = new_fig()
    ax = scatter_axes(fig, np.vstack([ud, ua]))
    class_legend(fig)
    sc = ax.scatter(ud[:, 0], ud[:, 1], c=cols, s=16, alpha=0.85, linewidths=0)
    t1 = fig.text(0.41, 0.92, "DINOv3 latent space", ha="center", fontsize=30, weight="bold", color=INK)
    t2 = fig.text(0.41, 0.92, "ASSUME latent space", ha="center", fontsize=30, weight="bold", color=INK, alpha=0)
    hold0, morph, hold1 = 1.5, 2.0, 2.5
    n = int((hold0 + morph + hold1) * FPS)
    w = FFMpegWriter(fps=FPS, bitrate=8000)
    with w.saving(fig, str(OUT / "anim_umap_morph.mp4"), dpi=120):
        for f in range(n):
            t = f / FPS
            p = bezier_ease(ramp(t, hold0, morph))
            sc.set_offsets(ud * (1 - p) + ua * p)
            t1.set_alpha(1 - ramp(t, hold0, 0.5))
            t2.set_alpha(ramp(t, hold0 + morph - 0.5, 0.5))
            w.grab_frame()
    plt.close(fig)


def pick_concepts(emb, ua, names, k=6):
    r1 = np.array([abs(np.corrcoef(emb[:, j], ua[:, 0])[0, 1]) if emb[:, j].std() > 0 else 0 for j in range(emb.shape[1])])
    r2 = np.array([abs(np.corrcoef(emb[:, j], ua[:, 1])[0, 1]) if emb[:, j].std() > 0 else 0 for j in range(emb.shape[1])])
    score = np.maximum(r1, r2)
    cc = np.abs(np.corrcoef(emb.T))
    chosen = []
    for j in np.argsort(-score):
        if all(cc[j, i] < 0.6 for i in chosen):
            chosen.append(j)
        if len(chosen) == k:
            break
    for j in chosen:
        print(f"  {names[j]}: |r| UMAP1={r1[j]:.2f} UMAP2={r2[j]:.2f}")
    return chosen


def anim_recolor():
    df, emb, concepts, ud, ua = load()
    names = [c["name"] for c in concepts]
    chosen = pick_concepts(emb, ua, names)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, 5)
    base = matplotlib.colors.to_rgba_array([CLASS_COLORS[c] for c in df["class_label"]])
    states = [base] + [cmap(norm(emb[:, j])) for j in chosen]
    titles = ["ASSUME latent space"] + [names[j] for j in chosen]
    order = np.random.default_rng(0).permutation(len(ua))

    fig = new_fig()
    ax = scatter_axes(fig, ua)
    sc = ax.scatter(ua[order, 0], ua[order, 1], c=base[order], s=16, linewidths=0)
    leg = class_legend(fig)
    cax = fig.add_axes([0.82, 0.2, 0.025, 0.55])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    cb.ax.tick_params(labelsize=14)
    cb.set_label("concept score", fontsize=16)
    cax.set_visible(False)
    ttl = [fig.text(0.41, 0.92, s, ha="center", fontsize=30, weight="bold", color=INK, alpha=0) for s in titles]
    ttl[0].set_alpha(1)

    first_hold, trans, hold = 1.5, 2.0, 4.0
    total = first_hold + len(chosen) * (trans + hold)
    w = FFMpegWriter(fps=FPS, bitrate=8000)
    with w.saving(fig, str(OUT / "anim_concept_recolor.mp4"), dpi=120):
        for f in range(int(total * FPS)):
            t = f / FPS
            seg = int(max(0, t - first_hold) // (trans + hold))
            seg = min(seg, len(chosen) - 1)
            t0 = first_hold + seg * (trans + hold)
            p = bezier_ease(ramp(t, t0, trans))
            c = states[seg] * (1 - p) + states[seg + 1] * p
            sc.set_facecolors(c[order])
            for i, tx in enumerate(ttl):
                a = 0.0
                if i == seg:
                    a = 1 - ramp(t, t0, trans / 2)
                elif i == seg + 1:
                    a = ramp(t, t0 + trans / 2, trans / 2)
                tx.set_alpha(a)
            if seg == 0:
                leg.set_visible(p < 0.5)
                cax.set_visible(p >= 0.5)
            w.grab_frame()
    plt.close(fig)


def anim_loop():
    """Build the pipeline flowchart step by step (same steps as the PowerPoint)."""
    from flowchart_spec import COLORS, steps

    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    hexc = lambda h: "#" + h
    fonts = {"title": dict(fontsize=24, weight="bold", color=hexc(COLORS["ink"])),
             "phase": dict(fontsize=17, weight="bold", color=hexc(COLORS["ink"])),
             "quote": dict(fontsize=13, style="italic", color=hexc(COLORS["quote"])),
             "label": dict(fontsize=13, style="italic", color=hexc(COLORS["rubric"]))}
    groups = []
    for step in steps():
        arts = []
        for prim in step:
            kind = prim[0]
            if kind == "box":
                _, x, y, w, h, title, sub, style = prim
                fc, ec = COLORS[style]
                arts.append(ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                                        fc=hexc(fc), ec=hexc(ec), lw=2)))
                big = h > 1.0
                arts.append(ax.text(x + w / 2, y + h - (0.28 if big else 0.2), title, ha="center", va="top",
                                    fontsize=17 if big else 15, weight="bold", color=hexc(COLORS["ink"])))
                arts.append(ax.text(x + w / 2, y + h - (0.75 if big else 0.55), sub, ha="center", va="top",
                                    fontsize=12.5 if big else 11.5, color=hexc(COLORS["muted"]), linespacing=1.3))
            elif kind == "arrow":
                _, pts, double = prim
                color = hexc(COLORS["rubric"]) if len(pts) > 2 else hexc(COLORS["ink"])
                for a, b in zip(pts[:-2], pts[1:-1]):
                    arts.append(ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=2.2)[0])
                arts.append(ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="<|-|>" if double else "-|>",
                                                         mutation_scale=20, color=color, lw=2.2)))
            elif kind == "line":
                (x0, y0), (x1, y1) = prim[1]
                arts.append(ax.plot([x0, x1], [y0, y1], color=hexc(COLORS["sep"]), lw=1.5, ls="--")[0])
            elif kind == "text":
                _, x, y, s, k = prim
                arts.append(ax.text(x, y, s, ha="center", va="top", **fonts[k]))
        for a in arts:
            a.set_alpha(0)
        groups.append(arts)

    gap, fade, hold_end = 1.3, 0.4, 3.0
    end = gap * (len(groups) - 1) + fade + hold_end
    w = FFMpegWriter(fps=FPS, bitrate=8000)
    with w.saving(fig, str(OUT / "anim_agent_loop.mp4"), dpi=120):
        for f in range(int(end * FPS)):
            tt = f / FPS
            for k, arts in enumerate(groups):
                a = ramp(tt, k * gap, fade)
                for art in arts:
                    art.set_alpha(a)
            w.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["morph", "recolor", "loop"]:
        globals()[f"anim_{name}"]()
        print("wrote", name)
