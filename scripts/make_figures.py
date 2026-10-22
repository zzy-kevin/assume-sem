#!/usr/bin/env python3
"""Figures for the writeup and video (written to figures/).

    python scripts/make_figures.py pipeline verifier        # no scores needed
    python scripts/make_figures.py rubric fingerprint       # need data/real/assume_embeddings_verified.npy
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "figures"
OUT.mkdir(exist_ok=True)
REAL = PROJECT_ROOT / "data" / "real"
VERIFIER = PROJECT_ROOT / "data" / "verifier"

INK = "#1f2933"
MUTED = "#616e7c"
AGENT = "#dbeafe"      # LLM agents
AGENT_EDGE = "#2563eb"
MODEL = "#e5e7eb"      # frozen models / data
MODEL_EDGE = "#6b7280"
OUTC = "#dcfce7"       # outputs
OUTC_EDGE = "#16a34a"
PRUNE = "#dc2626"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def thumb(path: str, size: int = 256) -> Image.Image:
    with Image.open(resolve_micrograph(path)) as im:
        im = im.convert("L")
        w, h = im.size
        im = im.crop((0, 0, w, int(h * 0.75)))
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return im.resize((size, size), Image.Resampling.LANCZOS)


def box(ax, x, y, w, h, title, sub="", fc=MODEL, ec=MODEL_EDGE, title_size=12.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h - 0.68, sub, ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.35)


def arrow(ax, p, q, text="", color=INK, rad=0.0, text_xy=None, style="-|>", lw=1.6, fs=9.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        tx, ty = text_xy if text_xy else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.18)
        ax.text(tx, ty, text, ha="center", va="bottom", fontsize=fs, color=color, style="italic")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    ax.text(0.25, 9.3, "Concept discovery (agentic loop, label-free)", fontsize=14, weight="bold", color=INK, va="top")
    ax.text(0.25, 3.05, "Embedding (run time)", fontsize=14, weight="bold", color=INK, va="top")
    ax.plot([0.25, 15.75], [3.3, 3.3], color="#d1d5db", lw=1, ls="--")

    # legend, top right
    for i, (fc, ec, t) in enumerate([(AGENT, AGENT_EDGE, "LLM agent (Gemini)"), (MODEL, MODEL_EDGE, "frozen model / tool / data"),
                                     (OUTC, OUTC_EDGE, "output")]):
        ax.add_patch(FancyBboxPatch((12.9, 9.05 - 0.36 * i), 0.3, 0.2, boxstyle="round,pad=0.01", fc=fc, ec=ec))
        ax.text(13.35, 9.15 - 0.36 * i, t, va="center", fontsize=9.5, color=INK)

    bw, bh, y = 2.55, 1.4, 5.6
    xs = [0.3, 3.4, 6.5, 9.6, 12.7]
    box(ax, xs[0], y, bw, bh, "Tri-modal sampler", "random  |  DINOv3 farthest-\npoint  |  collision pairs")
    box(ax, xs[1], y, bw, bh, "Proposer agent", "writes atomic concepts\nwith 0 / 1 / 3 / 5 anchors", AGENT, AGENT_EDGE)
    box(ax, xs[2], y, bw, bh, "Merger agent", "prunes dead concepts;\n|r| > 0.8: merge or keep", AGENT, AGENT_EDGE)
    box(ax, xs[3], y, bw, bh, "Verifier agent", "specimen morphology or\nimaging condition?", AGENT, AGENT_EDGE)
    box(ax, xs[4], y, bw + 0.3, bh, "Concept dictionary", "named rubrics +\nliterature references", OUTC, OUTC_EDGE)
    quotes = ["", "\"these two differ in\nparticle density\"", "\"is this different from\nparticle area fraction?\"",
              "\"is this about the specimen\nor about the image?\""]
    for x, q in zip(xs, quotes):
        if q:
            ax.text(x + bw / 2, y - 0.1, q, ha="center", va="top", fontsize=9.5, color=AGENT_EDGE, style="italic")
    for i in range(4):
        arrow(ax, (xs[i] + bw + 0.03, y + bh / 2), (xs[i + 1] - 0.03, y + bh / 2))

    box(ax, 0.3, 7.65, 2.55, 0.95, "DINOv3 teacher", "frozen ViT features", MODEL, MODEL_EDGE, title_size=12)
    arrow(ax, (1.575, 7.62), (1.575, y + bh + 0.03))
    box(ax, 9.6, 7.65, 2.55, 0.95, "Literature search", "Valency Bond MCP", MODEL, MODEL_EDGE, title_size=11.5)
    arrow(ax, (10.875, 7.62), (10.875, y + bh + 0.03), style="<|-|>")

    # collision feedback, between the rows
    arrow(ax, (xs[4] + 1.0, y - 0.03), (xs[0] + 1.9, y - 0.03), color="#b45309", rad=-0.22)
    ax.text(7.85, 4.02, "collision: DINOv3 says two micrographs differ, current concepts say they are the same"
            "  ->  sample them and ask the proposer what differs", ha="center", va="top", fontsize=10, color="#b45309",
            style="italic")

    # run-time row
    by, bh2 = 0.35, 1.45
    box(ax, 0.3, by, 2.55, bh2, "New micrograph", "unlabelled, any\nmagnification / class")
    box(ax, 4.0, by, 2.9, bh2, "Embedding agent", "scores the micrograph\nagainst every rubric (0-5)", AGENT, AGENT_EDGE)
    box(ax, 8.05, by, 3.0, bh2, "Concept vector", "every dimension is a named\nmorphological concept", OUTC, OUTC_EDGE)
    box(ax, 12.2, by, 3.5, bh2, "Use", "clustering  |  retrieval  |  edit\na concept and find nearest\nmicrograph  |  explanation", MODEL, MODEL_EDGE)
    arrow(ax, (2.88, by + bh2 / 2), (3.97, by + bh2 / 2))
    arrow(ax, (6.93, by + bh2 / 2), (8.02, by + bh2 / 2))
    arrow(ax, (11.08, by + bh2 / 2), (12.17, by + bh2 / 2))
    ax.text(4.95, 1.88, "\"it has particles: describe it\nwith particle density\"", ha="center", va="bottom",
            fontsize=9.5, color=AGENT_EDGE, style="italic")

    # rubrics: dictionary -> embedding agent, routed around the right edge
    ax.plot([15.55, 15.85, 15.85, 6.6], [y + bh / 2, y + bh / 2, 2.6, 2.6], color=OUTC_EDGE, lw=1.6)
    arrow(ax, (6.6, 2.6), (6.6, by + bh2 + 0.03), color=OUTC_EDGE)
    ax.text(12.0, 2.65, "rubrics from the dictionary", ha="center", va="bottom", fontsize=9.5, color=OUTC_EDGE, style="italic")

    fig.savefig(OUT / "fig_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_verifier():
    report = json.loads((VERIFIER / "verifier_report.json").read_text(encoding="utf-8"))
    pruned = [r for r in report if r["action"] == "PRUNE"]
    kept = [r for r in report if r["action"] == "KEEP" and r["references"] and r["concept_id"] != "collinear_fiber_orientation_dup"]
    show_kept = [r for r in kept if r["name"] in ("Surface Relief Texturing", "Fiber Alignment",
                                                   "Pore Shape Uniformity", "Feature Branching Complexity")][:4]
    if len(show_kept) < 4:
        show_kept += [r for r in kept if r not in show_kept][: 4 - len(show_kept)]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    n_total = len(report)
    n_dup = 1
    ax.text(0.3, 8.75, f"Verifier agent: {len(pruned)} of {n_total} discovered concepts pruned as imaging conditions",
            fontsize=16, weight="bold", color=INK, va="top")
    ax.text(0.3, 8.2, "Papers retrieved per concept with the Valency Bond literature MCP; Gemini reads the abstracts and decides.",
            fontsize=11.5, color=MUTED, va="top")

    ax.text(0.3, 7.5, "Pruned (imaging condition)", fontsize=13.5, weight="bold", color=PRUNE, va="top")
    yy = 7.0
    for r in pruned:
        first = r["rationale"].split(". ")[0].rstrip(".") + "."
        ax.text(0.3, yy, f"x  {r['name']}", fontsize=11.5, weight="bold", color=INK, va="top")
        ax.text(0.6, yy - 0.32, textwrap.shorten(first, 88, placeholder="..."), fontsize=9, color=MUTED, va="top")
        yy -= 0.82

    ax.text(8.4, 7.5, "Kept (specimen morphology), with literature", fontsize=13.5, weight="bold", color=OUTC_EDGE, va="top")
    yy = 7.0
    for r in show_kept:
        ax.text(8.4, yy, f"+  {r['name']}", fontsize=11.5, weight="bold", color=INK, va="top")
        for ref in r["references"][:2]:
            yy -= 0.36
            ax.text(8.7, yy, textwrap.shorten(ref["title"], 82, placeholder="..."), fontsize=9.3, color=MUTED, va="top",
                    style="italic")
        yy -= 0.6

    fig.savefig(OUT / "fig_verifier.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_real():
    df = pd.read_parquet(REAL / "eval_manifest.parquet")
    emb = np.load(REAL / "assume_embeddings_verified.npy")
    concepts = json.loads((VERIFIER / "concept_dictionary_verified.json").read_text(encoding="utf-8"))
    assert emb.shape == (len(df), len(concepts)), (emb.shape, len(df), len(concepts))
    return df, emb, concepts


def fig_rubric(names=("Feature Elongation", "Feature Occupancy Density", "Feature Branching Complexity")):
    """One row per concept: rubric anchors + real micrographs the embedding agent scored 0, 1, 3, 5."""
    df, emb, concepts = _load_real()
    by_name = {c["name"]: i for i, c in enumerate(concepts)}
    names = [n for n in names if n in by_name]
    rng = np.random.default_rng(0)
    levels = [0, 1, 3, 5]
    fig, axes = plt.subplots(len(names), 5, figsize=(17, 3.9 * len(names)),
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1]})
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        j = by_name[name]
        c = concepts[j]
        ax = axes[r, 0]
        ax.axis("off")
        txt = f"{name}\n\n" + "\n".join(
            textwrap.fill(f"{lv}: {c.get(f'scale_{lv}', '')}", 46) for lv in levels)
        ax.text(0, 1, txt, va="top", fontsize=9.6, color=INK, transform=ax.transAxes, linespacing=1.3)
        for k, lv in enumerate(levels):
            a = axes[r, k + 1]
            a.axis("off")
            idx = np.where(emb[:, j] == lv)[0]
            if len(idx) == 0:
                a.set_title(f"score {lv}: none", fontsize=10, color=MUTED)
                continue
            i = rng.choice(idx)
            a.imshow(thumb(df.iloc[i]["full_path"]), cmap="gray")
            a.set_title(f"score {lv}   ({df.iloc[i]['class_label']})", fontsize=10.5, color=INK)
    fig.suptitle("Rubric anchors and micrographs the embedding agent scored at each level", fontsize=14, weight="bold",
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rubric_cards.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_fingerprint(n_examples=3, top=8):
    """Micrograph next to its strongest concepts: the concept vector read as a description."""
    df, emb, concepts = _load_real()
    names = [c["name"] for c in concepts]
    rng = np.random.default_rng(3)
    classes = ["Powder", "Fibres", "Porous_Sponge", "Patterned_surface", "Nanowires", "Films_Coated_Surface"]
    picks = [rng.choice(np.where(df["class_label"].values == c)[0]) for c in classes[:n_examples]]
    fig, axes = plt.subplots(n_examples, 2, figsize=(14, 4.3 * n_examples), gridspec_kw={"width_ratios": [1, 1.5]})
    for r, i in enumerate(picks):
        axes[r, 0].imshow(thumb(df.iloc[i]["full_path"], 384), cmap="gray")
        axes[r, 0].axis("off")
        axes[r, 0].set_title(df.iloc[i]["class_label"] + " (label not used)", fontsize=11, color=MUTED)
        order = np.argsort(-emb[i])[:top][::-1]
        ax = axes[r, 1]
        ax.barh(range(top), emb[i, order], color=AGENT_EDGE, height=0.65)
        ax.set_yticks(range(top))
        ax.set_yticklabels([names[k] for k in order], fontsize=10.5)
        ax.set_xlim(0, 5)
        ax.set_xticks([0, 1, 3, 5])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("concept score (0-5)", fontsize=10)
    fig.suptitle("Reading a micrograph's embedding: its strongest concepts", fontsize=14, weight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_concept_fingerprint.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["pipeline", "verifier"]
    for name in todo:
        globals()[f"fig_{name}"]()
        print("wrote", name)
