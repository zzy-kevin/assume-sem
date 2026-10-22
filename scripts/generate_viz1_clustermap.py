#!/usr/bin/env python3
"""Generate Viz 1: Hierarchical Concept Clustermap & Materials Ontology Dendrogram.

This script performs:
1. Computation of Pearson correlation matrix across all K=60 (or K=30) ASSUME concepts.
2. Ward hierarchical clustering with dendrograms revealing modular microstructural concept groups.
3. Class-by-Concept activation profiling with biclustering, establishing the
   unsupervised Materials Microstructure Ontology Tree across all 8 SEM classes.
4. Generates a high-resolution publication-quality figure at 300 DPI.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
# Use matplotlib native colormaps without seaborn dependency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assume.viz1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
EMBEDDINGS_60_PATH = PROJECT_ROOT / "data" / "real" / "assume_embeddings_verified.npy"
EMBEDDINGS_30_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_60_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
CONCEPTS_30_PATH = PROJECT_ROOT / "concept_dictionary_k30.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "viz1_concept_clustermap_dendrogram.png"


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


def create_viz1(k: int = 60):
    df, embeddings, concept_names, concept_ids = load_data(k=k)
    n_samples, n_concepts = embeddings.shape
    logger.info("Loaded embeddings shape: %s with %d concepts", embeddings.shape, n_concepts)

    # 1. Compute Pearson Correlation Matrix
    corr_matrix = np.corrcoef(embeddings, rowvar=False)

    # Ward hierarchical clustering on concepts using correlation distance
    condensed_corr_dist = pdist(embeddings.T, metric="correlation")
    linkage_concepts = sch.linkage(condensed_corr_dist, method="ward")
    dendro_concepts = sch.dendrogram(linkage_concepts, no_plot=True)
    concept_order = dendro_concepts["leaves"]

    # Reorder correlation matrix and names
    reordered_corr = corr_matrix[concept_order, :][:, concept_order]
    reordered_concept_names = [concept_names[i] for i in concept_order]

    # 2. Compute Class-Average Concept Activation Profiles
    classes = sorted(df["class_label"].unique())
    class_means = np.zeros((len(classes), n_concepts))
    for i, cls in enumerate(classes):
        mask = (df["class_label"] == cls).to_numpy()
        class_means[i, :] = embeddings[mask].mean(axis=0)

    # Hierarchical clustering on Class profiles
    class_dist = pdist(class_means, metric="euclidean")
    linkage_classes = sch.linkage(class_dist, method="ward")
    dendro_classes = sch.dendrogram(linkage_classes, no_plot=True)
    class_order = dendro_classes["leaves"]

    reordered_class_means = class_means[class_order, :][:, concept_order]
    reordered_classes = [classes[i] for i in class_order]

    # Plot Figure
    fig_w = 34 if n_concepts > 30 else 26
    fig_h = 28 if n_concepts > 30 else 22
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # GridSpec layout
    gs = GridSpec(2, 2, width_ratios=[0.07, 0.93], height_ratios=[0.24, 0.76],
                  wspace=0.015, hspace=0.20, left=0.04, right=0.76, top=0.93, bottom=0.12)

    # ---------------- PANEL A: Materials Ontology Heatmap ----------------
    ax_class_dendro = fig.add_subplot(gs[0, 0])
    sch.dendrogram(
        linkage_classes,
        orientation="left",
        labels=reordered_classes,
        ax=ax_class_dendro,
        color_threshold=0,
        above_threshold_color="#1e40af"
    )
    ax_class_dendro.invert_yaxis()
    ax_class_dendro.set_axis_off()

    ax_class_heat = fig.add_subplot(gs[0, 1])
    cmap_scores = plt.colormaps["YlGnBu"]
    im_cls = ax_class_heat.imshow(reordered_class_means, aspect="auto", cmap=cmap_scores, vmin=0.0, vmax=5.0)

    # Labels for Class Heatmap
    ax_class_heat.set_yticks(np.arange(len(reordered_classes)))
    ax_class_heat.set_yticklabels(reordered_classes, fontsize=13, fontweight="bold", color="#111827")
    ax_class_heat.yaxis.tick_right()
    ax_class_heat.set_xticks(np.arange(n_concepts))
    ax_class_heat.set_xticklabels([])
    ax_class_heat.set_title(f"A. Unsupervised Materials Ontology Dendrogram & Mean Concept Profiles across 8 SEM Classes (K={n_concepts})",
                            fontsize=16, fontweight="bold", pad=14, loc="left", color="#111827")

    # Annotate numeric values inside class heatmap cells
    cell_font = 6.5 if n_concepts > 30 else 8.0
    for r in range(len(reordered_classes)):
        for c in range(n_concepts):
            val = reordered_class_means[r, c]
            color = "white" if val > 3.0 else "black"
            ax_class_heat.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=cell_font, color=color, fontweight="semibold")

    # Colorbar for Class Profiles
    cbar_ax_cls = fig.add_axes([0.94, 0.70, 0.015, 0.18])
    cbar_cls = fig.colorbar(im_cls, cax=cbar_ax_cls)
    cbar_cls.set_label("Mean Concept Score (0-5 Rubric)", fontsize=11, fontweight="bold")
    cbar_cls.ax.tick_params(labelsize=9)

    # ---------------- PANEL B: Concept Correlation Clustermap ----------------
    ax_corr_dendro = fig.add_subplot(gs[1, 0])
    sch.dendrogram(
        linkage_concepts,
        orientation="left",
        ax=ax_corr_dendro,
        color_threshold=0,
        above_threshold_color="#0e7490"
    )
    ax_corr_dendro.invert_yaxis()
    ax_corr_dendro.set_axis_off()

    ax_corr_heat = fig.add_subplot(gs[1, 1])
    cmap_corr = plt.colormaps["coolwarm"]
    im_corr = ax_corr_heat.imshow(reordered_corr, aspect="auto", cmap=cmap_corr, vmin=-1.0, vmax=1.0)

    label_font = 7.5 if n_concepts > 30 else 9.0
    ax_corr_heat.set_yticks(np.arange(n_concepts))
    ax_corr_heat.set_yticklabels(reordered_concept_names, fontsize=label_font, color="#1f2937")
    ax_corr_heat.yaxis.tick_right()
    ax_corr_heat.set_xticks(np.arange(n_concepts))
    ax_corr_heat.set_xticklabels(reordered_concept_names, rotation=60, ha="right", fontsize=label_font, color="#1f2937")
    ax_corr_heat.set_title(f"B. Concept-to-Concept Pearson Correlation Matrix with Ward Hierarchical Clustering (K={n_concepts}, N={len(df):,} Micrographs)",
                           fontsize=16, fontweight="bold", pad=14, loc="left", color="#111827")

    # Colorbar for Correlation
    cbar_ax_corr = fig.add_axes([0.94, 0.18, 0.015, 0.42])
    cbar_corr = fig.colorbar(im_corr, cax=cbar_ax_corr)
    cbar_corr.set_label("Pearson Correlation Coefficient (r)", fontsize=11, fontweight="bold")
    cbar_corr.ax.tick_params(labelsize=9)

    # Overall title
    fig.suptitle(f"ASSUME Semantic Architecture: Hierarchical Concept Clustermap & Materials Ontology Dendrogram (K={n_concepts} Latents)",
                 fontsize=21, fontweight="heavy", y=0.985, color="#0f172a")

    logger.info("Saving figure to %s ...", OUTPUT_IMG)
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches="tight")
    if n_concepts == 60:
        k60_path = OUTPUT_DIR / "viz1_concept_clustermap_dendrogram_k60.png"
        plt.savefig(k60_path, dpi=300, bbox_inches="tight")
        logger.info("Also mirrored to %s", k60_path)
    plt.close()
    logger.info("Successfully generated Viz 1: %s", OUTPUT_IMG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Viz 1 Clustermap")
    parser.add_argument("--k", type=int, default=60, choices=[30, 60], help="Number of concepts (30 or 60, default: 60)")
    args = parser.parse_args()
    create_viz1(k=args.k)
