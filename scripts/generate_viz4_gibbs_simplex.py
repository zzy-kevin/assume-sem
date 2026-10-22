#!/usr/bin/env python3
"""Generate Viz 4: Gibbs Morphological Simplex (Ternary Phase Diagram).

Maps the 1,600 SEM micrographs onto a 3-axis barycentric ternary phase diagram
governed by fundamental microstructural dimensionality:
  - 0D Granular / Discrete Particulate (Particles, Powder)
  - 1D Elongated / Interlaced Filamentous (Fibres, Nanowires)
  - 3D Open-Cell / Cellular Porous Foam (Porous_Sponge, Biological)

Supports both K=60 and K=30 concepts with dynamic concept resolution.
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
from matplotlib.patches import Polygon, ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assume.viz4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
EMBEDDINGS_60_PATH = PROJECT_ROOT / "data" / "real" / "assume_embeddings_verified.npy"
EMBEDDINGS_30_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_60_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
CONCEPTS_30_PATH = PROJECT_ROOT / "concept_dictionary_k30.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "viz4_gibbs_morphological_simplex.png"


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


def preprocess_micrograph(image_path: str, target_size: int = 160) -> Image.Image:
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


def compute_barycentric_coords(embeddings: np.ndarray, concept_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 0D, 1D, and 3D morphological phase components."""
    # 0D Granular candidate concepts
    c_0d_cands = [
        "discrete_particle_spatial_areal_density",
        "dispersed_phase_cluster_aggregation_degree",
        "nanoparticle_monolayer_island_aggregation_saliency",
        "sub_feature_nodular_clumping_degree"
    ]
    idx_0d = [concept_ids.index(c) for c in c_0d_cands if c in concept_ids]
    s_0d = np.mean(embeddings[:, idx_0d], axis=1)

    # 1D Elongated candidate concepts
    c_1d_cands = [
        "filamentous_fiber_packing_density",
        "aspect_ratio_of_primary_microstructural_units",
        "directional_alignment_of_elongated_structures",
        "acicular_whisker_random_interlocking_web_fraction"
    ]
    idx_1d = [concept_ids.index(c) for c in c_1d_cands if c in concept_ids]
    s_1d = np.mean(embeddings[:, idx_1d], axis=1)

    # 3D Porous / Cellular Network candidate concepts
    c_3d_cands = [
        "interconnected_strut_ligament_thickness",
        "pore_cell_wall_interconnectivity_thickness",
        "inter_bundle_void_volume_fraction",
        "irregular_boundary_etch_pit_cavity_area_fraction",
        "structural_branching_junction_density"
    ]
    idx_3d = [concept_ids.index(c) for c in c_3d_cands if c in concept_ids]
    s_3d = np.mean(embeddings[:, idx_3d], axis=1)

    # Normalize into positive fractions summing to 1
    w_0d = np.maximum(0.1, s_0d - 1.0) ** 1.8
    w_1d = np.maximum(0.1, s_1d - 1.0) ** 1.8
    w_3d = np.maximum(0.1, s_3d - 1.0) ** 1.8

    total = w_0d + w_1d + w_3d
    f_0d = w_0d / total
    f_1d = w_1d / total
    f_3d = w_3d / total

    return f_0d, f_1d, f_3d


def barycentric_to_cartesian(f_0d: np.ndarray, f_1d: np.ndarray, f_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform barycentric fractions to 2D Cartesian plane:
    Vertex A (0D Granular) = (0, 0) - Bottom Left
    Vertex B (1D Elongated) = (1, 0) - Bottom Right
    Vertex C (3D Porous) = (0.5, sqrt(3)/2) - Top Apex
    """
    x = f_1d + 0.5 * f_3d
    y = (np.sqrt(3.0) / 2.0) * f_3d
    return x, y


def create_viz4(k: int = 60):
    df, embeddings, concept_names, concept_ids = load_data(k=k)
    logger.info("Loaded embeddings shape: %s with %d concepts", embeddings.shape, len(concept_ids))
    f_0d, f_1d, f_3d = compute_barycentric_coords(embeddings, concept_ids)
    x_pts, y_pts = barycentric_to_cartesian(f_0d, f_1d, f_3d)

    h_tri = np.sqrt(3.0) / 2.0

    fig = plt.figure(figsize=(24, 20), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_axes([0.15, 0.10, 0.70, 0.80])
    ax.set_facecolor("#F8FAFC")

    # Draw Triangular Boundaries
    triangle = Polygon([[0, 0], [1, 0], [0.5, h_tri]], closed=True,
                       facecolor="#F1F5F9", edgecolor="#334155", linewidth=3.0, zorder=1)
    ax.add_patch(triangle)

    # Draw Ternary Gridlines at 20%, 40%, 60%, 80%
    ticks = np.array([0.2, 0.4, 0.6, 0.8])
    for t in ticks:
        # Constant 3D lines
        y_val = t * h_tri
        x_left = 0.5 * t
        x_right = 1.0 - 0.5 * t
        ax.plot([x_left, x_right], [y_val, y_val], color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=2)
        ax.text(x_left - 0.02, y_val, f"{int(round(t*100))}%", fontsize=9, color="#64748B", ha="right", va="center")

        # Constant 1D lines
        ax.plot([t, 0.5 + 0.5 * t], [0, (1.0 - t) * h_tri], color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=2)
        ax.text(t, -0.022, f"{int(round(t*100))}%", fontsize=9, color="#64748B", ha="center", va="top")

        # Constant 0D lines
        ax.plot([1.0 - t, 0.5 * (1.0 - t)], [0, (1.0 - t) * h_tri], color="#CBD5E1", linestyle="--", linewidth=1.1, zorder=2)
        x_bc = 1.0 - 0.5 * t
        ax.text(x_bc + 0.02, y_val, f"{int(round((1.0 - t)*100))}%", fontsize=9, color="#64748B", ha="left", va="center")

    # Phase Shading Contours (KDE background)
    try:
        xy = np.vstack([x_pts, y_pts])
        kde = gaussian_kde(xy)
        gx, gy = np.mgrid[0:1:200j, 0:h_tri:200j]
        mask_in_tri = (gy >= 0) & (gy <= 2 * h_tri * gx) & (gy <= 2 * h_tri * (1 - gx))
        gz = kde(np.vstack([gx.flatten(), gy.flatten()])).reshape(gx.shape)
        gz[~mask_in_tri] = np.nan
        ax.contourf(gx, gy, gz, levels=12, cmap="Blues", alpha=0.25, zorder=3)
    except Exception as e:
        logger.warning("KDE contour failed: %s", e)

    # Class Colors
    class_palette = {
        "Biological": "#059669",
        "Fibres": "#2563EB",
        "Films_Coated_Surface": "#D97706",
        "Nanowires": "#0284C7",
        "Particles": "#DC2626",
        "Patterned_surface": "#7C3AED",
        "Porous_Sponge": "#0D9488",
        "Powder": "#EA580C"
    }

    # Plot Scatter Points by Class
    for cls in sorted(df["class_label"].unique()):
        mask = (df["class_label"] == cls).to_numpy()
        ax.scatter(
            x_pts[mask], y_pts[mask],
            c=class_palette.get(cls, "#4B5563"),
            label=f"{cls} (n={mask.sum()})",
            s=42, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=4
        )

    # Label the Vertices
    ax.text(-0.04, -0.05, "0D Granular / Discrete Particulate\n(Particles, Powder)",
            fontsize=13, fontweight="bold", color="#DC2626", ha="center", va="top")

    ax.text(1.04, -0.05, "1D Elongated / Interlaced Filaments\n(Fibres, Nanowires)",
            fontsize=13, fontweight="bold", color="#2563EB", ha="center", va="top")

    ax.text(0.5, h_tri + 0.04, "3D Open-Cell Cellular Porous Foam\n(Porous_Sponge, Biological)",
            fontsize=13, fontweight="bold", color="#0D9488", ha="center", va="bottom")

    # Add Callout Thumbnails for Micrographs
    medoid_0d_idx = int(np.argmax(f_0d))
    medoid_1d_idx = int(np.argmax(f_1d))
    medoid_3d_idx = int(np.argmax(f_3d))

    hybrid_01_dist = (f_0d - 0.45)**2 + (f_1d - 0.45)**2 + (f_3d - 0.10)**2
    hybrid_01_idx = int(np.argmin(hybrid_01_dist))

    hybrid_13_dist = (f_0d - 0.05)**2 + (f_1d - 0.45)**2 + (f_3d - 0.50)**2
    hybrid_13_idx = int(np.argmin(hybrid_13_dist))

    callouts = [
        {
            "idx": medoid_0d_idx,
            "box_pos": (-0.18, 0.18),
            "title": f"Pure 0D Phase ({df.iloc[medoid_0d_idx]['class_label']})\n$f_{{0D}}={f_0d[medoid_0d_idx]:.2f}$"
        },
        {
            "idx": medoid_1d_idx,
            "box_pos": (1.18, 0.18),
            "title": f"Pure 1D Phase ({df.iloc[medoid_1d_idx]['class_label']})\n$f_{{1D}}={f_1d[medoid_1d_idx]:.2f}$"
        },
        {
            "idx": medoid_3d_idx,
            "box_pos": (0.84, h_tri + 0.14),
            "title": f"Pure 3D Phase ({df.iloc[medoid_3d_idx]['class_label']})\n$f_{{3D}}={f_3d[medoid_3d_idx]:.2f}$"
        },
        {
            "idx": hybrid_01_idx,
            "box_pos": (0.50, -0.16),
            "title": f"0D/1D Transition ({df.iloc[hybrid_01_idx]['class_label']})\n$f_{{0D}}={f_0d[hybrid_01_idx]:.2f},\\ f_{{1D}}={f_1d[hybrid_01_idx]:.2f}$"
        },
        {
            "idx": hybrid_13_idx,
            "box_pos": (1.18, 0.60),
            "title": f"1D/3D Transition ({df.iloc[hybrid_13_idx]['class_label']})\n$f_{{1D}}={f_1d[hybrid_13_idx]:.2f},\\ f_{{3D}}={f_3d[hybrid_13_idx]:.2f}$"
        }
    ]

    for c in callouts:
        idx = c["idx"]
        img_path = df.iloc[idx]["full_path"]
        try:
            thumb = preprocess_micrograph(img_path, target_size=130)
            imagebox = OffsetImage(thumb, zoom=0.75)
            pt_x, pt_y = x_pts[idx], y_pts[idx]
            bx, by = c["box_pos"]

            ab = AnnotationBbox(
                imagebox, (bx, by),
                xybox=(bx, by),
                frameon=True,
                pad=0.3,
                bboxprops=dict(edgecolor="#334155", linewidth=1.5, facecolor="white")
            )
            ax.add_artist(ab)

            arrow = ConnectionPatch(
                xyA=(pt_x, pt_y), xyB=(bx, by),
                coordsA="data", coordsB="data",
                axesA=ax, axesB=ax,
                arrowstyle="-|>", connectionstyle="arc3,rad=0.15",
                color="#1E293B", linewidth=1.5, zorder=10
            )
            ax.add_artist(arrow)

            ax.text(bx, by - 0.08 if by < 0 else by + 0.08, c["title"],
                    fontsize=9, fontweight="bold", ha="center", va="center", color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9))

        except Exception as err:
            logger.warning("Callout thumbnail error: %s", err)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.22, h_tri + 0.32)
    ax.set_aspect("equal")
    ax.axis("off")

    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.98),
        title="SEM Morphological Classes", title_fontsize=11, fontsize=9.5,
        frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95
    )
    legend.get_title().set_fontweight("bold")

    fig.suptitle(f"ASSUME Gibbs Morphological Simplex: Unsupervised 3-Axis Barycentric Phase Diagram (K={len(concept_ids)})",
                 fontsize=19, fontweight="heavy", color="#0F172A", y=0.965)
    ax.text(0.5, h_tri + 0.30,
            f"Mapping Microstructural Dimensionality across 0D Granular, 1D Elongated, and 3D Open-Cell Regimes (N=1,600, K={len(concept_ids)})",
            fontsize=12, color="#475569", ha="center", va="center")

    logger.info("Saving Viz 4 to %s ...", OUTPUT_IMG)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
    if len(concept_ids) == 60:
        k60_path = OUTPUT_DIR / "viz4_gibbs_morphological_simplex_k60.png"
        plt.savefig(k60_path, dpi=300, bbox_inches="tight")
        logger.info("Also mirrored to %s", k60_path)
    plt.close()
    logger.info("Successfully generated Viz 4: %s", OUTPUT_IMG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Viz 4 Gibbs Simplex")
    parser.add_argument("--k", type=int, default=60, choices=[30, 60], help="Number of concepts (30 or 60, default: 60)")
    args = parser.parse_args()
    create_viz4(k=args.k)
