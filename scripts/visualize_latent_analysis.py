#!/usr/bin/env python3
"""ASSUME Latent Analysis and Visualization Suite.

Generates:
1. Global & Per-Class PCA (PC1 vs PC2) and UMAP (UMAP1 vs UMAP2) projections.
2. 10x10 Thumbnail Latent Grids (closest micrograph to grid center, center-cropped 1:1).
3. Feature Importance & Correlation Analysis (PC1, PC2, UMAP1, UMAP2 vs. 30 semantic concepts)
   both globally and per SEM class.
4. Publication-grade figures and structured JSON metrics.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
import umap
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assume.latent_viz")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "eval_manifest.parquet"
EMBEDDINGS_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_PATH = PROJECT_ROOT / "concept_dictionary.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette for 8 SEM classes
CLASS_COLORS = {
    "Biological": "#4E79A7",            # Muted blue
    "Fibres": "#F28E2B",                # Tangerine orange
    "Films_Coated_Surface": "#E15759", # Carmine red
    "Nanowires": "#76B7B2",             # Teal
    "Particles": "#59A14F",            # Sage green
    "Patterned_surface": "#EDC948",     # Gold yellow
    "Porous_Sponge": "#B07AA1",         # Violet
    "Powder": "#9C755F",                # Earth brown
}


def load_data():
    """Load evaluation manifest, semantic embeddings, and concept dictionary."""
    df = pd.read_parquet(EVAL_MANIFEST_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        concept_dict = json.load(f)
    concept_names = [c["name"] for c in concept_dict]
    concept_ids = [c["concept_id"] for c in concept_dict]
    logger.info(f"Loaded {len(df)} samples, embeddings {embeddings.shape}, {len(concept_names)} concepts.")
    return df, embeddings, concept_dict, concept_names, concept_ids


def preprocess_micrograph_to_square(image_path: str, target_size: int = 160) -> Image.Image:
    """Preprocess SEM micrograph:
    1. Load image
    2. Crop bottom 25% (remove instrument metadata bar) -> top 75% height
    3. Center crop to 1:1 square
    4. Resize to target thumbnail dimension
    """
    try:
        with Image.open(resolve_micrograph(image_path)) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            # Crop bottom 25%
            h_crop = max(1, int(h * 0.75))
            img_cropped = img.crop((0, 0, w, h_crop))
            # Center crop to 1:1 square
            cw, ch = img_cropped.size
            sq_dim = min(cw, ch)
            left = (cw - sq_dim) // 2
            top = (ch - sq_dim) // 2
            img_square = img_cropped.crop((left, top, left + sq_dim, top + sq_dim))
            # Resize
            thumb = img_square.resize((target_size, target_size), Image.Resampling.LANCZOS)
            return thumb
    except Exception as e:
        logger.warning(f"Error loading {image_path}: {e}")
        # Return blank placeholder image
        return Image.new("RGB", (target_size, target_size), color=(200, 200, 200))


def build_thumbnail_latent_grid(
    coords: np.ndarray,
    image_paths: List[str],
    grid_size: int = 10,
    cell_size: int = 160,
    bg_color: Tuple[int, int, int] = (240, 242, 245),
    border_color: Tuple[int, int, int] = (210, 214, 220),
) -> Tuple[Image.Image, np.ndarray, Tuple[float, float, float, float]]:
    """Build a 10x10 square thumbnail latent grid.
    
    For each cell with >= 1 point, embeds the micrograph closest to the cell center.
    Returns:
        grid_image: PIL Image of the composed grid
        occupancy: 10x10 boolean array of occupied cells
        extent: (xmin, xmax, ymin, ymax) coordinate bounds
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    
    # 2% padding margin
    x_span = max(xmax - xmin, 1e-6)
    y_span = max(ymax - ymin, 1e-6)
    x_pad = 0.02 * x_span
    y_pad = 0.02 * y_span
    
    xmin_p, xmax_p = xmin - x_pad, xmax + x_pad
    ymin_p, ymax_p = ymin - y_pad, ymax + y_pad
    
    x_edges = np.linspace(xmin_p, xmax_p, grid_size + 1)
    y_edges = np.linspace(ymin_p, ymax_p, grid_size + 1)
    
    # Canvas
    total_w = grid_size * cell_size
    total_h = grid_size * cell_size
    canvas = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)
    
    occupancy = np.zeros((grid_size, grid_size), dtype=bool)
    selected_indices = set()
    
    # Process each cell: row r from top (0) to bottom (grid_size - 1)
    # col c from left (0) to right (grid_size - 1)
    for r in range(grid_size):
        # r=0 is topmost Y bin (highest y) -> b = grid_size - 1 - r
        b = grid_size - 1 - r
        y_low = y_edges[b]
        y_high = y_edges[b + 1]
        y_mid = (y_low + y_high) / 2.0
        
        # Disjoint partitioning along Y (r=0 is top row containing ymax_p)
        y_mask = (y >= y_low) & (y <= y_high) if r == 0 else (y >= y_low) & (y < y_high)
        
        for c in range(grid_size):
            x_low = x_edges[c]
            x_high = x_edges[c + 1]
            x_mid = (x_low + x_high) / 2.0
            
            # Disjoint partitioning along X (c=grid_size-1 is rightmost col containing xmax_p)
            mask = (x >= x_low) & (x <= x_high) if c == grid_size - 1 else (x >= x_low) & (x < x_high)
            in_cell = np.where(mask & y_mask)[0]
            
            if len(in_cell) > 0:
                occupancy[r, c] = True
                # Closest point to (x_mid, y_mid)
                dists = (x[in_cell] - x_mid) ** 2 + (y[in_cell] - y_mid) ** 2
                best_idx = in_cell[np.argmin(dists)]
                if best_idx in selected_indices:
                    logger.error(f"Duplicate image index {best_idx} encountered at cell ({r}, {c})!")
                selected_indices.add(best_idx)
                thumb = preprocess_micrograph_to_square(image_paths[best_idx], target_size=cell_size)
                
                # Paste into canvas
                canvas.paste(thumb, (c * cell_size, r * cell_size))
            else:
                # Draw empty cell subtle pattern/cross
                px = c * cell_size
                py = r * cell_size
                draw.rectangle([px, py, px + cell_size - 1, py + cell_size - 1], fill=bg_color)
                # Subtle center dot
                draw.ellipse(
                    [px + cell_size // 2 - 2, py + cell_size // 2 - 2, px + cell_size // 2 + 2, py + cell_size // 2 + 2],
                    fill=(200, 204, 210),
                )
            
            # Draw subtle cell border
            px = c * cell_size
            py = r * cell_size
            draw.rectangle([px, py, px + cell_size - 1, py + cell_size - 1], outline=border_color, width=1)
            
    extent = (xmin_p, xmax_p, ymin_p, ymax_p)
    return canvas, occupancy, extent


def compute_axis_correlations(
    axis_values: np.ndarray,
    embeddings: np.ndarray,
    concept_names: List[str],
) -> List[Dict[str, float]]:
    """Compute Pearson and Spearman correlation between a 1D projection axis and all 30 concepts."""
    results = []
    for k in range(embeddings.shape[1]):
        feat = embeddings[:, k]
        # Check variance
        if np.std(axis_values) < 1e-8 or np.std(feat) < 1e-8:
            r, p_val = 0.0, 1.0
            rho, s_pval = 0.0, 1.0
        else:
            r, p_val = pearsonr(axis_values, feat)
            rho, s_pval = spearmanr(axis_values, feat)
        results.append({
            "concept_index": k + 1,
            "concept_name": concept_names[k],
            "pearson_r": float(r),
            "pearson_pval": float(p_val),
            "spearman_rho": float(rho),
            "spearman_pval": float(s_pval),
            "abs_r": float(abs(r)),
        })
    # Sort descending by absolute Pearson r
    results.sort(key=lambda item: item["abs_r"], reverse=True)
    return results


def save_thumbnail_plot_with_axes(
    canvas: Image.Image,
    extent: Tuple[float, float, float, float],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
    dpi: int = 300,
):
    """Render the PIL thumbnail canvas into a matplotlib figure with exact coordinate axes and ticks."""
    xmin, xmax, ymin, ymax = extent
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    
    # Display canvas with extent matching coordinates
    ax.imshow(canvas, extent=[xmin, xmax, ymin, ymax], origin="upper", aspect="equal")
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(False)
    
    # Style ticks
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=1.2)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.2)
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved thumbnail latent plot to {save_path}")


def create_composite_class_figure(
    class_name: str,
    class_pca_coords: np.ndarray,
    class_umap_coords: np.ndarray,
    pca_canvas: Image.Image,
    umap_canvas: Image.Image,
    pca_extent: Tuple[float, float, float, float],
    umap_extent: Tuple[float, float, float, float],
    pca_var_ratio: Tuple[float, float],
    top_pca1_concept: str,
    top_pca2_concept: str,
    top_umap1_concept: str,
    top_umap2_concept: str,
    save_path: Path,
    dpi: int = 300,
):
    """Generate publication-grade 4-panel composite figure for a class:
    Panel A: PCA Scatter (PC1 vs PC2)
    Panel B: PCA 10x10 Thumbnail Latent Grid
    Panel C: UMAP Scatter (UMAP1 vs UMAP2)
    Panel D: UMAP 10x10 Thumbnail Latent Grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    
    color = CLASS_COLORS.get(class_name, "#1F77B4")
    
    # Panel A: PCA Scatter
    ax_a = axes[0, 0]
    ax_a.scatter(
        class_pca_coords[:, 0],
        class_pca_coords[:, 1],
        c=color,
        s=45,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
    )
    ax_a.set_title(
        f"A. PCA Scatter ? {class_name} (N=200)\nPC1: {top_pca1_concept[:28]}... | PC2: {top_pca2_concept[:28]}...",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_a.set_xlabel(f"PC1 ({pca_var_ratio[0]*100:.1f}% variance)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel(f"PC2 ({pca_var_ratio[1]*100:.1f}% variance)", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.3)
    ax_a.set_aspect("equal", adjustable="datalim")
    
    # Panel B: PCA Thumbnail Grid
    ax_b = axes[0, 1]
    pxmin, pxmax, pymin, pymax = pca_extent
    ax_b.imshow(pca_canvas, extent=[pxmin, pxmax, pymin, pymax], origin="upper", aspect="equal")
    ax_b.set_title(
        f"B. PCA 10x10 Thumbnail Latent Grid ? {class_name}\n(Closest Micrograph to Cell Center)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_b.set_xlabel(f"PC1 ({pca_var_ratio[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_b.set_ylabel(f"PC2 ({pca_var_ratio[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    
    # Panel C: UMAP Scatter
    ax_c = axes[1, 0]
    ax_c.scatter(
        class_umap_coords[:, 0],
        class_umap_coords[:, 1],
        c=color,
        s=45,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
    )
    ax_c.set_title(
        f"C. UMAP Scatter ? {class_name} (N=200)\nUMAP1: {top_umap1_concept[:28]}... | UMAP2: {top_umap2_concept[:28]}...",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_c.set_xlabel("UMAP Dimension 1", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("UMAP Dimension 2", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.3)
    ax_c.set_aspect("equal", adjustable="datalim")
    
    # Panel D: UMAP Thumbnail Grid
    ax_d = axes[1, 1]
    uxmin, uxmax, uymin, uymax = umap_extent
    ax_d.imshow(umap_canvas, extent=[uxmin, uxmax, uymin, uymax], origin="upper", aspect="equal")
    ax_d.set_title(
        f"D. UMAP 10x10 Thumbnail Latent Grid ? {class_name}\n(Closest Micrograph to Cell Center)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_d.set_xlabel("UMAP Dimension 1", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("UMAP Dimension 2", fontsize=11, fontweight="bold")
    
    plt.suptitle(f"ASSUME Latent Morphological Space: {class_name}", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 4-panel composite figure for {class_name} to {save_path}")


def create_global_composite_figure(
    df: pd.DataFrame,
    global_coords: np.ndarray,
    grid_canvas: Image.Image,
    extent: Tuple[float, float, float, float],
    proj_name: str,
    xlabel: str,
    ylabel: str,
    save_path: Path,
    dpi: int = 300,
):
    """Create side-by-side composite for Global dataset (Scatter colored by 8 classes + 10x10 Thumbnail Grid)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    
    # Scatter colored by 8 classes
    ax_sc = axes[0]
    for class_name, color in CLASS_COLORS.items():
        mask = (df["class_label"] == class_name).values
        ax_sc.scatter(
            global_coords[mask, 0],
            global_coords[mask, 1],
            c=color,
            label=class_name,
            s=28,
            alpha=0.75,
            edgecolors="none",
        )
    ax_sc.set_title(f"A. Global {proj_name} Scatter (N=1,600, 8 SEM Classes)", fontsize=13, fontweight="bold", pad=10)
    ax_sc.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax_sc.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax_sc.grid(True, linestyle="--", alpha=0.3)
    ax_sc.legend(title="SEM Class", fontsize=9, title_fontsize=10, loc="upper right", framealpha=0.9)
    ax_sc.set_aspect("equal", adjustable="datalim")
    
    # Thumbnail Grid
    ax_th = axes[1]
    xmin, xmax, ymin, ymax = extent
    ax_th.imshow(grid_canvas, extent=[xmin, xmax, ymin, ymax], origin="upper", aspect="equal")
    ax_th.set_title(f"B. Global {proj_name} 10x10 Thumbnail Latent Grid", fontsize=13, fontweight="bold", pad=10)
    ax_th.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax_th.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    
    plt.suptitle(f"Global ASSUME Concept Space ? {proj_name} Representation", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved global {proj_name} composite to {save_path}")


def main():
    logger.info("Starting ASSUME Latent Analysis and Visualization Suite...")
    df, embeddings, concept_dict, concept_names, concept_ids = load_data()
    image_paths = df["full_path"].tolist()
    
    # -------------------------------------------------------------
    # 1. Global Projections (PCA and UMAP on all 1600 micrographs)
    # -------------------------------------------------------------
    logger.info("Computing Global PCA (PC1 vs PC2)...")
    global_pca = PCA(n_components=2, random_state=42)
    global_pca_coords = global_pca.fit_transform(embeddings)
    pca_var = global_pca.explained_variance_ratio_
    logger.info(f"Global PCA explained variance: PC1={pca_var[0]*100:.2f}%, PC2={pca_var[1]*100:.2f}%")
    
    logger.info("Computing Global UMAP (UMAP1 vs UMAP2)...")
    global_umap_reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    global_umap_coords = global_umap_reducer.fit_transform(embeddings)
    
    # -------------------------------------------------------------
    # 2. Global Thumbnail Grids
    # -------------------------------------------------------------
    logger.info("Building Global PCA 10x10 Thumbnail Grid...")
    pca_thumb_canvas, pca_occ, pca_extent = build_thumbnail_latent_grid(
        global_pca_coords, image_paths, grid_size=10, cell_size=160
    )
    save_thumbnail_plot_with_axes(
        pca_thumb_canvas,
        pca_extent,
        xlabel=f"PC1 ({pca_var[0]*100:.1f}%)",
        ylabel=f"PC2 ({pca_var[1]*100:.1f}%)",
        title="Global ASSUME PCA 10x10 Thumbnail Latent Plot",
        save_path=OUTPUT_DIR / "global_pca_thumbnail_grid.png",
    )
    
    logger.info("Building Global UMAP 10x10 Thumbnail Grid...")
    umap_thumb_canvas, umap_occ, umap_extent = build_thumbnail_latent_grid(
        global_umap_coords, image_paths, grid_size=10, cell_size=160
    )
    save_thumbnail_plot_with_axes(
        umap_thumb_canvas,
        umap_extent,
        xlabel="UMAP Dimension 1",
        ylabel="UMAP Dimension 2",
        title="Global ASSUME UMAP 10x10 Thumbnail Latent Plot",
        save_path=OUTPUT_DIR / "global_umap_thumbnail_grid.png",
    )
    
    # Global composite figures
    create_global_composite_figure(
        df,
        global_pca_coords,
        pca_thumb_canvas,
        pca_extent,
        proj_name="PCA",
        xlabel=f"PC1 ({pca_var[0]*100:.1f}% explained variance)",
        ylabel=f"PC2 ({pca_var[1]*100:.1f}% explained variance)",
        save_path=OUTPUT_DIR / "global_pca_composite.png",
    )
    create_global_composite_figure(
        df,
        global_umap_coords,
        umap_thumb_canvas,
        umap_extent,
        proj_name="UMAP",
        xlabel="UMAP Dimension 1",
        ylabel="UMAP Dimension 2",
        save_path=OUTPUT_DIR / "global_umap_composite.png",
    )
    
    # -------------------------------------------------------------
    # 3. Global Feature Importance / Correlation Analysis
    # -------------------------------------------------------------
    logger.info("Computing Global Axis-to-Concept Correlations...")
    global_corr_pc1 = compute_axis_correlations(global_pca_coords[:, 0], embeddings, concept_names)
    global_corr_pc2 = compute_axis_correlations(global_pca_coords[:, 1], embeddings, concept_names)
    global_corr_umap1 = compute_axis_correlations(global_umap_coords[:, 0], embeddings, concept_names)
    global_corr_umap2 = compute_axis_correlations(global_umap_coords[:, 1], embeddings, concept_names)
    
    # Build Global Correlation Matrix (30 concepts x 4 axes)
    global_corr_matrix = np.zeros((len(concept_names), 4))
    for i, cname in enumerate(concept_names):
        # Find Pearson r
        r_pc1 = next(item["pearson_r"] for item in global_corr_pc1 if item["concept_name"] == cname)
        r_pc2 = next(item["pearson_r"] for item in global_corr_pc2 if item["concept_name"] == cname)
        r_umap1 = next(item["pearson_r"] for item in global_corr_umap1 if item["concept_name"] == cname)
        r_umap2 = next(item["pearson_r"] for item in global_corr_umap2 if item["concept_name"] == cname)
        global_corr_matrix[i, :] = [r_pc1, r_pc2, r_umap1, r_umap2]
        
    global_corr_df = pd.DataFrame(
        global_corr_matrix,
        index=[f"{i+1}. {name}" for i, name in enumerate(concept_names)],
        columns=["PC1", "PC2", "UMAP1", "UMAP2"],
    )
    global_corr_df.to_csv(OUTPUT_DIR / "global_latent_correlations.csv")
    
    # Plot Global Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    cax = ax.imshow(global_corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation (r)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["PC1", "PC2", "UMAP1", "UMAP2"], fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(concept_names)))
    ax.set_yticklabels(
        [f"C{i+1:02d}: {name[:32]}" for i, name in enumerate(concept_names)],
        fontsize=9,
    )
    ax.set_title("Global Latent Projections vs. 30 ASSUME Semantic Concepts\n(Pearson Correlation r across 1,600 Micrographs)", fontsize=13, fontweight="bold", pad=15)
    
    # Overlay correlation values text
    for i in range(len(concept_names)):
        for j in range(4):
            val = global_corr_matrix[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")
            
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "global_concept_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved global correlation heatmap.")

    # -------------------------------------------------------------
    # 4. Per-Class Latent Analysis (Within each of the 8 classes)
    # -------------------------------------------------------------
    per_class_summary = {}
    class_names = sorted(list(CLASS_COLORS.keys()))
    
    for cls in class_names:
        logger.info(f"Processing Class: {cls}...")
        mask = (df["class_label"] == cls).values
        cls_df = df[mask].reset_index(drop=True)
        cls_emb = embeddings[mask]
        cls_paths = cls_df["full_path"].tolist()
        
        # Fit Class PCA
        cls_pca = PCA(n_components=2, random_state=42)
        cls_pca_coords = cls_pca.fit_transform(cls_emb)
        cls_pca_var = cls_pca.explained_variance_ratio_
        
        # Fit Class UMAP
        cls_umap_reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        cls_umap_coords = cls_umap_reducer.fit_transform(cls_emb)
        
        # Build 10x10 Thumbnail Grids for this class
        cls_pca_canvas, _, cls_pca_extent = build_thumbnail_latent_grid(
            cls_pca_coords, cls_paths, grid_size=10, cell_size=160
        )
        cls_umap_canvas, _, cls_umap_extent = build_thumbnail_latent_grid(
            cls_umap_coords, cls_paths, grid_size=10, cell_size=160
        )
        
        # Save individual thumbnail grids
        save_thumbnail_plot_with_axes(
            cls_pca_canvas,
            cls_pca_extent,
            xlabel=f"PC1 ({cls_pca_var[0]*100:.1f}%)",
            ylabel=f"PC2 ({cls_pca_var[1]*100:.1f}%)",
            title=f"{cls} ? PCA 10x10 Thumbnail Latent Plot",
            save_path=OUTPUT_DIR / f"{cls}_pca_thumbnail_grid.png",
        )
        save_thumbnail_plot_with_axes(
            cls_umap_canvas,
            cls_umap_extent,
            xlabel="UMAP Dimension 1",
            ylabel="UMAP Dimension 2",
            title=f"{cls} ? UMAP 10x10 Thumbnail Latent Plot",
            save_path=OUTPUT_DIR / f"{cls}_umap_thumbnail_grid.png",
        )
        
        # Compute correlations for this class
        cls_corr_pc1 = compute_axis_correlations(cls_pca_coords[:, 0], cls_emb, concept_names)
        cls_corr_pc2 = compute_axis_correlations(cls_pca_coords[:, 1], cls_emb, concept_names)
        cls_corr_umap1 = compute_axis_correlations(cls_umap_coords[:, 0], cls_emb, concept_names)
        cls_corr_umap2 = compute_axis_correlations(cls_umap_coords[:, 1], cls_emb, concept_names)
        
        # Save composite 4-panel figure
        create_composite_class_figure(
            class_name=cls,
            class_pca_coords=cls_pca_coords,
            class_umap_coords=cls_umap_coords,
            pca_canvas=cls_pca_canvas,
            umap_canvas=cls_umap_canvas,
            pca_extent=cls_pca_extent,
            umap_extent=cls_umap_extent,
            pca_var_ratio=(cls_pca_var[0], cls_pca_var[1]),
            top_pca1_concept=f"{cls_corr_pc1[0]['concept_name']} (r={cls_corr_pc1[0]['pearson_r']:+.2f})",
            top_pca2_concept=f"{cls_corr_pc2[0]['concept_name']} (r={cls_corr_pc2[0]['pearson_r']:+.2f})",
            top_umap1_concept=f"{cls_corr_umap1[0]['concept_name']} (r={cls_corr_umap1[0]['pearson_r']:+.2f})",
            top_umap2_concept=f"{cls_corr_umap2[0]['concept_name']} (r={cls_corr_umap2[0]['pearson_r']:+.2f})",
            save_path=OUTPUT_DIR / f"{cls}_composite_4panel.png",
        )
        
        per_class_summary[cls] = {
            "pca_variance_ratio": [float(cls_pca_var[0]), float(cls_pca_var[1])],
            "top_pc1_concepts": cls_corr_pc1[:5],
            "top_pc2_concepts": cls_corr_pc2[:5],
            "top_umap1_concepts": cls_corr_umap1[:5],
            "top_umap2_concepts": cls_corr_umap2[:5],
        }
        
    # Save complete JSON summary
    full_report = {
        "global_pca_variance_ratio": [float(pca_var[0]), float(pca_var[1])],
        "global_top_pc1": global_corr_pc1[:5],
        "global_top_pc2": global_corr_pc2[:5],
        "global_top_umap1": global_corr_umap1[:5],
        "global_top_umap2": global_corr_umap2[:5],
        "per_class": per_class_summary,
    }
    with open(OUTPUT_DIR / "latent_correlations_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info("Exported latent_correlations_summary.json successfully.")

    # -------------------------------------------------------------
    # 5. Master 8-Class Overview Gallery Figure (All 8 classes UMAP thumbnails)
    # -------------------------------------------------------------
    logger.info("Generating 8-Class Master Overview Gallery...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=250)
    fig.patch.set_facecolor("#FFFFFF")
    
    for idx, cls in enumerate(class_names):
        r, c = divmod(idx, 4)
        ax = axes[r, c]
        img_path = OUTPUT_DIR / f"{cls}_umap_thumbnail_grid.png"
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{cls}", fontsize=14, fontweight="bold", pad=8, color=CLASS_COLORS[cls])
        
    plt.suptitle("ASSUME UMAP 10x10 Micrograph Latent Grids across all 8 SEM Classes", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_8_classes_umap_gallery.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved all_8_classes_umap_gallery.png successfully!")
    print("\n>>> ALL ANALYSES AND VISUALIZATIONS GENERATED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
