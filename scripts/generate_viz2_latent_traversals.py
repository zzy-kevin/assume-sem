#!/usr/bin/env python3
"""Generate Viz 2: Monotonic Latent Traversal Filmstrips along Manifold Geodesics.

Finds constrained manifold paths in the ASSUME latent space where
a chosen target concept varies monotonically while non-target latent dimensions
remain tightly constrained. Visualizes the resulting filmstrips with micrograph
thumbnails, quantitative scores, and latent stability response plots.
Supports K=60 and K=30 concepts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assume.viz2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
EMBEDDINGS_60_PATH = PROJECT_ROOT / "data" / "real" / "assume_embeddings_verified.npy"
EMBEDDINGS_30_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_60_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
CONCEPTS_30_PATH = PROJECT_ROOT / "concept_dictionary_k30.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "viz2_latent_traversal_filmstrips.png"
OUTPUT_JSON = OUTPUT_DIR / "viz2_traversal_paths.json"


def load_data(k: int = 60):
    df = pd.read_parquet(EVAL_MANIFEST_PATH)
    if k == 60 and EMBEDDINGS_60_PATH.exists() and CONCEPTS_60_PATH.exists():
        embeddings = np.load(EMBEDDINGS_60_PATH)
        with open(CONCEPTS_60_PATH, "r", encoding="utf-8") as f:
            concept_dict = json.load(f)
    else:
        embeddings = np.load(EMBEDDINGS_30_PATH)
        with open(CONCEPTS_30_PATH, "r", encoding="utf-8") as f:
            concept_dict = json.load(f)
    concept_names = [c["name"] for c in concept_dict]
    concept_ids = [c["concept_id"] for c in concept_dict]
    return df, embeddings, concept_names, concept_ids


def preprocess_micrograph(image_path: str, target_size: int = 260) -> Image.Image:
    with Image.open(resolve_micrograph(image_path)) as raw_img:
        img = raw_img.convert("RGB")
        w, h = img.size
        h_crop = max(1, int(h * 0.75))
        img_cropped = img.crop((0, 0, w, h_crop))
        min_dim = min(w, h_crop)
        left = (w - min_dim) // 2
        top = (h_crop - min_dim) // 2
        img_sq = img_cropped.crop((left, top, left + min_dim, top + min_dim))
        return img_sq.resize((target_size, target_size), Image.Resampling.LANCZOS)


def find_constrained_manifold_path(
    target_cid: str,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    concept_ids: List[str],
    n_steps: int = 6
) -> Dict:
    k = concept_ids.index(target_cid)
    scores = embeddings[:, k]
    non_k = [i for i in range(len(concept_ids)) if i != k]

    s_min, s_max = np.percentile(scores, 2), np.percentile(scores, 98)
    bin_edges = np.linspace(s_min, s_max, n_steps + 1)
    bin_indices = []
    for b in range(n_steps):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == n_steps - 1:
            idx = np.where((scores >= low) & (scores <= high))[0]
        else:
            idx = np.where((scores >= low) & (scores < high))[0]
        bin_indices.append(idx)

    best_path = None
    min_drift = float("inf")
    mode_name = "Global"

    # Intra-class search
    for cls in df["class_label"].unique():
        cls_idx = np.where(df["class_label"] == cls)[0]
        cls_bins = [np.intersect1d(b, cls_idx) for b in bin_indices]
        if all(len(b) > 0 for b in cls_bins):
            mean_non_k = embeddings[cls_idx][:, non_k].mean(axis=0)
            path = []
            for b in cls_bins:
                dists = np.sum((embeddings[b][:, non_k] - mean_non_k) ** 2, axis=1)
                path.append(int(b[np.argmin(dists)]))
            drift = float(np.sum(np.var(embeddings[path][:, non_k], axis=0)))
            if drift < min_drift:
                min_drift = drift
                best_path = path
                mode_name = f"Intra-Class ({cls})"

    # Cross-manifold anchor points search
    if all(len(b) > 0 for b in bin_indices):
        for a in range(0, len(df), 3):
            a_vec = embeddings[a, non_k]
            path = []
            for b in bin_indices:
                dists = np.sum((embeddings[b][:, non_k] - a_vec) ** 2, axis=1)
                path.append(int(b[np.argmin(dists)]))
            drift = float(np.sum(np.var(embeddings[path][:, non_k], axis=0)))
            if drift < min_drift:
                min_drift = drift
                best_path = path
                mode_name = f"Manifold Geodesic ({df.iloc[a]['class_label']})"

    if best_path is None:
        # Robust fallback: select points evenly spaced across the sorted activation spectrum
        sorted_order = np.argsort(scores)
        step_indices = np.linspace(0, len(sorted_order) - 1, n_steps).astype(int)
        best_path = [int(sorted_order[i]) for i in step_indices]
        min_drift = float(np.sum(np.var(embeddings[best_path][:, non_k], axis=0)))
        mode_name = "Quantile Traversal"

    return {
        "target_cid": target_cid,
        "target_idx": k,
        "mode": mode_name,
        "path_indices": best_path,
        "target_scores": [float(embeddings[p, k]) for p in best_path],
        "non_target_drift_var": min_drift,
        "classes": [df.iloc[p]["class_label"] for p in best_path],
        "image_ids": [df.iloc[p]["image_id"] for p in best_path],
        "full_paths": [df.iloc[p]["full_path"] for p in best_path],
    }


def create_viz2(k: int = 60):
    df, embeddings, concept_names, concept_ids = load_data(k=k)
    logger.info("Loaded embeddings shape: %s with %d concepts", embeddings.shape, len(concept_ids))

    def resolve(cids, fallback_idx=0):
        for c in cids:
            if c in concept_ids:
                return c
        return concept_ids[fallback_idx % len(concept_ids)]

    # Select 4 representative continuous concepts
    target_cids = [
        resolve(["surface_irregularity_and_texture", "surface_roughness", "depth_of_field_focal_degradation"], 0),
        resolve(["feature_occupancy_density", "feature_density", "discrete_particle_spatial_areal_density"], 1),
        resolve(["feature_elongation", "sub_micron_surface_granular_roughness"], 2),
        resolve(["spatial_pattern_uniformity_and_periodicity", "topographical_relief_and_edge_contrast_prominence"], 3)
    ]

    traversal_results = []
    for cid in target_cids:
        res = find_constrained_manifold_path(cid, embeddings, df, concept_ids, n_steps=6)
        traversal_results.append(res)
        logger.info("Found path for %s (drift=%.3f, scores=%s)",
                    cid, res["non_target_drift_var"], res["target_scores"])

    # Save metadata to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        clean_json = [{k: v for k, v in t.items() if k != "full_paths"} for t in traversal_results]
        json.dump(clean_json, f, indent=2)

    # Plot Filmstrip Figure
    n_rows = len(target_cids)
    n_cols = 6

    fig = plt.figure(figsize=(26, 20), dpi=300)
    fig.patch.set_facecolor("#0F172A")

    col_widths = [1.0] * n_cols + [1.9]
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=col_widths,
                  wspace=0.14, hspace=0.58, left=0.035, right=0.955, top=0.91, bottom=0.04)

    colors = ["#38BDF8", "#F59E0B", "#10B981", "#EC4899"]

    for row_idx, res in enumerate(traversal_results):
        k_idx = res["target_idx"]
        cid = res["target_cid"]
        cname = concept_names[k_idx]
        path_indices = res["path_indices"]
        target_scores = res["target_scores"]
        color = colors[row_idx % len(colors)]

        # Plot 6 micrograph frames
        for col_idx in range(n_cols):
            ax_img = fig.add_subplot(gs[row_idx, col_idx])
            img_path = res["full_paths"][col_idx]
            cls_name = res["classes"][col_idx]
            img_id = res["image_ids"][col_idx]
            score = target_scores[col_idx]

            try:
                thumb = preprocess_micrograph(img_path, target_size=240)
                ax_img.imshow(thumb)
            except Exception as e:
                logger.warning("Could not load image %s: %s", img_path, e)
                ax_img.set_facecolor("#1e293b")
                ax_img.text(0.5, 0.5, "Image Missing", color="white", ha="center", va="center")

            ax_img.set_xticks([])
            ax_img.set_yticks([])

            for spine in ax_img.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.2 if col_idx in [0, n_cols - 1] else 1.2)

            step_lbl = f"Step {col_idx + 1}"
            ax_img.set_title(f"{step_lbl}\nScore: {score:.1f} / 5",
                             fontsize=11, fontweight="bold", color="white", pad=8)
            ax_img.set_xlabel(f"{cls_name}", fontsize=9, color="#94A3B8", labelpad=6)

        # Plot stability response curve in rightmost column
        ax_plot = fig.add_subplot(gs[row_idx, n_cols])
        ax_plot.set_facecolor("#1E293B")
        for spine in ax_plot.spines.values():
            spine.set_edgecolor("#334155")

        path_embs = embeddings[path_indices]
        non_k = [i for i in range(len(concept_names)) if i != k_idx]
        non_target_vals = path_embs[:, non_k]

        steps = np.arange(1, n_cols + 1)

        # Plot non-target traces
        for j in range(non_target_vals.shape[1]):
            ax_plot.plot(steps, non_target_vals[:, j], color="#64748B", alpha=0.30, linewidth=0.7)

        # Plot non-target mean ± 1σ
        mean_non_target = np.mean(non_target_vals, axis=1)
        std_non_target = np.std(non_target_vals, axis=1)
        ax_plot.fill_between(steps, mean_non_target - std_non_target, mean_non_target + std_non_target,
                             color="#94A3B8", alpha=0.25, label=f"Other {len(non_k)} Latents Mean ± 1σ")

        # Plot Target concept activation
        ax_plot.plot(steps, target_scores, color=color, linewidth=3.2, marker="o", markersize=6,
                     label=f"Target: {cname}")

        ax_plot.set_ylim(-0.2, 5.5)
        ax_plot.set_xlim(0.8, n_cols + 0.2)
        ax_plot.set_xticks(steps)
        ax_plot.set_xticklabels([f"t={s}" for s in steps], fontsize=8.5, color="#CBD5E1")
        ax_plot.tick_params(colors="#CBD5E1", labelsize=8.5)
        ax_plot.grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

        ax_plot.yaxis.tick_right()
        ax_plot.yaxis.set_label_position("right")
        ax_plot.set_title(f"Disentanglement ({res['mode']})",
                          fontsize=11, fontweight="bold", color="white", pad=8)
        ax_plot.set_ylabel("Activation (0 - 5)", fontsize=9.5, color="#CBD5E1", labelpad=8)
        ax_plot.legend(loc="upper left", fontsize=7.5, facecolor="#0F172A", edgecolor="#334155",
                       labelcolor="white", framealpha=0.85)

        row_y_pos = 0.945 - (row_idx * 0.232)
        fig.text(0.04, row_y_pos,
                 f"Traversal {row_idx + 1}: {cname} (K={len(concept_ids)} Manifold Geodesic)",
                 fontsize=14, fontweight="bold", color=color, ha="left", va="bottom")

    fig.suptitle(f"ASSUME Monotonic Latent Traversals: Geodesic Manifold Filmstrips with Constrained Non-Target Latents (K={len(concept_ids)})",
                 fontsize=18, fontweight="heavy", color="#F8FAFC", y=0.985)

    logger.info("Saving Viz 2 filmstrips to %s ...", OUTPUT_IMG)
    plt.savefig(OUTPUT_IMG, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    if len(concept_ids) == 60:
        k60_path = OUTPUT_DIR / "viz2_latent_traversal_filmstrips_k60.png"
        plt.savefig(k60_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
        logger.info("Also mirrored to %s", k60_path)
    plt.close()
    logger.info("Successfully generated Viz 2: %s", OUTPUT_IMG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Viz 2 Latent Traversals")
    parser.add_argument("--k", type=int, default=60, choices=[30, 60], help="Number of concepts (30 or 60, default: 60)")
    args = parser.parse_args()
    create_viz2(k=args.k)
