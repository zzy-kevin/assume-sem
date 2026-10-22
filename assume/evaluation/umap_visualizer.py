"""Publication-grade UMAP Projections & Quantitative Evaluation Visualizer.

Implements:
1. Extraction of balanced 1,600 subset DINOv3 teacher features (1600 x 384).
2. 2D UMAP projections for DINOv3 teacher space and ASSUME concept bottleneck space.
3. Quantitative cluster separation metrics:
   - Cosine Silhouette Score for DINOv3 vs ASSUME concept space.
   - 5-Fold cross-validated k-NN classification accuracy (k=5) for DINOv3 vs ASSUME.
   - Dual-path JSON export to data/eval_metrics.json and project root eval_metrics.json.
4. Publication-quality 4-panel comparison figure (300 DPI, publication aesthetic):
   - Panel A: DINOv3 2D UMAP colored by 8 SEM classes.
   - Panel B: ASSUME Concept Space 2D UMAP colored by 8 SEM classes.
   - Panel C: ASSUME Concept Space colored by Top Concept Activation 1 (Filamentous Fiber Packing Density).
   - Panel D: ASSUME Concept Space colored by Top Concept Activation 2 (Periodic Array Regularity).
   - Dual-path PNG export to data/umap_assume_vs_dinov3.png and project root umap_assume_vs_dinov3.png.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import umap

from assume.config import (
    ACTIVE_CLASSES,
    DATA_DIR,
    EMBEDDING_PATH,
    PROJECT_ROOT,
    ROOT_EMBEDDING_PATH,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH: Path = DATA_DIR / "eval_metrics.json"
DEFAULT_ROOT_METRICS_PATH: Path = PROJECT_ROOT / "eval_metrics.json"
DEFAULT_PLOT_PATH: Path = DATA_DIR / "umap_assume_vs_dinov3.png"
DEFAULT_ROOT_PLOT_PATH: Path = PROJECT_ROOT / "umap_assume_vs_dinov3.png"

# Publication-grade palette for 8 SEM classes
CLASS_COLORS: Dict[str, str] = {
    "Biological": "#4E79A7",            # Muted blue
    "Fibres": "#F28E2B",                # Tangerine orange
    "Films_Coated_Surface": "#E15759", # Carmine red
    "Nanowires": "#76B7B2",             # Teal
    "Particles": "#59A14F",            # Sage green
    "Patterned_surface": "#EDC948",     # Gold yellow
    "Porous_Sponge": "#B07AA1",         # Violet
    "Powder": "#9C755F",                # Earth brown
}


def _mirror_file(src: Path, dst: Path) -> None:
    """Safely mirror a file from src to dst."""
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        logger.warning("Failed to mirror %s to %s: %s", src, dst, e)


def extract_subset_features(
    eval_df: pd.DataFrame,
    dinov3_features_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Extract DINOv3 features for the evaluation subset using emb_index.

    Args:
        eval_df: Evaluation subset DataFrame containing 'emb_index' column.
        dinov3_features_path: Path to dinov3_pooled.npy (default: data/dinov3_pooled.npy).

    Returns:
        np.ndarray of shape (len(eval_df), 384) with float32 dtype.
    """
    f_path = Path(dinov3_features_path) if dinov3_features_path else EMBEDDING_PATH
    if not f_path.exists():
        if ROOT_EMBEDDING_PATH.exists():
            f_path = ROOT_EMBEDDING_PATH
        else:
            raise FileNotFoundError(f"DINOv3 features not found at {f_path} or {ROOT_EMBEDDING_PATH}")

    all_features = np.load(f_path)
    indices = eval_df["emb_index"].values.astype(int)

    subset_features = all_features[indices]
    if subset_features.shape != (len(eval_df), 384):
        raise ValueError(
            f"Expected feature shape ({len(eval_df)}, 384), got {subset_features.shape}"
        )

    return subset_features.astype(np.float32)


def compute_umap_projection(
    features: np.ndarray,
    metric: str = "cosine",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Fit and transform 2D UMAP projection on feature matrix.

    Args:
        features: (N, D) numpy feature matrix.
        metric: Distance metric for UMAP (default: 'cosine').
        n_neighbors: Number of nearest neighbors (default: 15).
        min_dist: Minimum distance between points (default: 0.1).
        random_state: Random state seed for reproducibility (default: 42).

    Returns:
        np.ndarray of shape (N, 2) containing 2D UMAP coordinates.
    """
    logger.info(
        "Computing 2D UMAP projection: shape=%s, metric=%s, n_neighbors=%d, min_dist=%.2f, seed=%d",
        features.shape,
        metric,
        n_neighbors,
        min_dist,
        random_state,
    )
    reducer = umap.UMAP(
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=2,
    )
    embedding_2d = reducer.fit_transform(features)

    if np.isnan(embedding_2d).any():
        raise RuntimeError("UMAP projection returned NaN values.")

    return embedding_2d.astype(np.float32)


def compute_separation_metrics(
    dino_features: np.ndarray,
    assume_features: np.ndarray,
    labels: Sequence[Any],
    k_neighbors: int = 5,
    cv_splits: int = 5,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
    root_output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Calculate quantitative cluster separation and classification metrics.

    Computes:
    - Cosine Silhouette Score for DINOv3 space vs ASSUME concept space.
    - 5-Fold cross-validated k-NN classification accuracy (k=5) for DINOv3 vs ASSUME.

    Exports results to data/eval_metrics.json and mirrors to root eval_metrics.json.

    Args:
        dino_features: (N, 384) float32 DINOv3 features.
        assume_features: (N, 30) float32 ASSUME concept ratings.
        labels: Class labels sequence of length N.
        k_neighbors: Nearest neighbors for k-NN classifier (default: 5).
        cv_splits: Number of stratified cross-validation folds (default: 5).
        seed: Random seed for CV splitting (default: 42).
        output_path: Primary output path (default: data/eval_metrics.json).
        root_output_path: Root mirror path (default: eval_metrics.json).

    Returns:
        Dictionary of calculated quantitative metrics.
    """
    out_p = Path(output_path) if output_path else DEFAULT_METRICS_PATH
    if root_output_path:
        root_out_p = Path(root_output_path)
    elif output_path is None:
        root_out_p = DEFAULT_ROOT_METRICS_PATH
    else:
        root_out_p = None

    labels_arr = np.array(labels)
    unique_classes = sorted(list(set(labels_arr)))

    logger.info("Evaluating Cosine Silhouette Scores (N=%d, classes=%d)...", len(labels_arr), len(unique_classes))
    dino_sil = float(silhouette_score(dino_features, labels_arr, metric="cosine"))
    assume_sil = float(silhouette_score(assume_features, labels_arr, metric="cosine"))
    logger.info("Cosine Silhouette -> DINOv3: %.4f | ASSUME: %.4f", dino_sil, assume_sil)

    logger.info("Evaluating %d-fold cross-validated k-NN accuracy (k=%d, metric=cosine)...", cv_splits, k_neighbors)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    knn_dino = KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine")
    dino_scores = cross_val_score(knn_dino, dino_features, labels_arr, cv=cv, n_jobs=1)
    dino_acc_mean = float(np.mean(dino_scores))
    dino_acc_std = float(np.std(dino_scores))

    knn_assume = KNeighborsClassifier(n_neighbors=k_neighbors, metric="cosine")
    assume_scores = cross_val_score(knn_assume, assume_features, labels_arr, cv=cv, n_jobs=1)
    assume_acc_mean = float(np.mean(assume_scores))
    assume_acc_std = float(np.std(assume_scores))

    logger.info(
        "k-NN (k=%d) Accuracy -> DINOv3: %.4f +/- %.4f | ASSUME: %.4f +/- %.4f",
        k_neighbors,
        dino_acc_mean,
        dino_acc_std,
        assume_acc_mean,
        assume_acc_std,
    )

    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample_count": int(len(labels_arr)),
        "class_count": int(len(unique_classes)),
        "classes": unique_classes,
        "dinov3_feature_dim": int(dino_features.shape[1]),
        "assume_feature_dim": int(assume_features.shape[1]),
        "dinov3_cosine_silhouette": dino_sil,
        "assume_cosine_silhouette": assume_sil,
        "knn_k_neighbors": int(k_neighbors),
        "cv_folds": int(cv_splits),
        "dinov3_knn_accuracy_mean": dino_acc_mean,
        "dinov3_knn_accuracy_std": dino_acc_std,
        "assume_knn_accuracy_mean": assume_acc_mean,
        "assume_knn_accuracy_std": assume_acc_std,
    }

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", out_p)

    if root_out_p:
        _mirror_file(out_p, root_out_p)
        logger.info("Mirrored evaluation metrics to %s", root_out_p)

    return metrics


def generate_umap_comparison_figure(
    dino_umap: np.ndarray,
    assume_umap: np.ndarray,
    labels: Sequence[str],
    assume_features: np.ndarray,
    concept_names: Sequence[str],
    output_path: Optional[Union[str, Path]] = None,
    root_output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Generate high-resolution 4-panel publication-grade UMAP comparison figure.

    Panels:
    - Panel A: DINOv3 2D UMAP colored by 8 SEM classes.
    - Panel B: ASSUME Concept Space 2D UMAP colored by 8 SEM classes.
    - Panel C: ASSUME Concept Space colored by Top Concept Activation 1 (Filamentous Fiber Packing Density).
    - Panel D: ASSUME Concept Space colored by Top Concept Activation 2 (Periodic Array Regularity).

    Saves figure to data/umap_assume_vs_dinov3.png and mirrors to root umap_assume_vs_dinov3.png.

    Args:
        dino_umap: (1600, 2) 2D coordinates for DINOv3 space.
        assume_umap: (1600, 2) 2D coordinates for ASSUME concept space.
        labels: Class labels of length 1600.
        assume_features: (1600, 30) float32 concept rating matrix.
        concept_names: List of 30 concept display names.
        output_path: Primary output path (default: data/umap_assume_vs_dinov3.png).
        root_output_path: Root mirror path (default: umap_assume_vs_dinov3.png).
        dpi: Export resolution in dots per inch (default: 300).

    Returns:
        Path to generated figure.
    """
    out_p = Path(output_path) if output_path else DEFAULT_PLOT_PATH
    if root_output_path:
        root_out_p = Path(root_output_path)
    elif output_path is None:
        root_out_p = DEFAULT_ROOT_PLOT_PATH
    else:
        root_out_p = None

    labels_arr = np.array(labels)
    unique_classes = list(ACTIVE_CLASSES)

    # Style configuration for Nature / Science publication aesthetic
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5), dpi=dpi)
    ((ax_a, ax_b), (ax_c, ax_d)) = axes

    point_size = 18
    point_alpha = 0.78

    # ----------------------------------------------------
    # Panel A: DINOv3 2D UMAP by SEM Class
    # ----------------------------------------------------
    for c_name in unique_classes:
        mask = labels_arr == c_name
        color = CLASS_COLORS.get(c_name, "#333333")
        clean_label = c_name.replace("_", " ")
        ax_a.scatter(
            dino_umap[mask, 0],
            dino_umap[mask, 1],
            s=point_size,
            c=color,
            label=clean_label,
            alpha=point_alpha,
            edgecolors="none",
        )
    ax_a.set_title("a  DINOv3 ViT-S/16+ Feature Space (384-d)", weight="bold", loc="left", pad=10)
    ax_a.set_xlabel("UMAP Dimension 1")
    ax_a.set_ylabel("UMAP Dimension 2")
    ax_a.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5, markerscale=1.6)

    # ----------------------------------------------------
    # Panel B: ASSUME Concept Space 2D UMAP by SEM Class
    # ----------------------------------------------------
    for c_name in unique_classes:
        mask = labels_arr == c_name
        color = CLASS_COLORS.get(c_name, "#333333")
        clean_label = c_name.replace("_", " ")
        ax_b.scatter(
            assume_umap[mask, 0],
            assume_umap[mask, 1],
            s=point_size,
            c=color,
            label=clean_label,
            alpha=point_alpha,
            edgecolors="none",
        )
    ax_b.set_title(f"b  ASSUME Concept Space ({assume_features.shape[1]}-d)", weight="bold", loc="left", pad=10)
    ax_b.set_xlabel("UMAP Dimension 1")
    ax_b.set_ylabel("UMAP Dimension 2")
    ax_b.grid(True, linestyle="--", alpha=0.3, color="#cccccc")

    # ----------------------------------------------------
    # Panel C: ASSUME Concept Space colored by Concept 0
    # (Filamentous Fiber Packing Density)
    # ----------------------------------------------------
    concept_0_idx = 0
    concept_0_name = concept_names[concept_0_idx] if len(concept_names) > concept_0_idx else "Concept 1"
    c0_values = assume_features[:, concept_0_idx]

    is_0_5_scale = bool(np.nanmax(assume_features) <= 5.0)
    vmin_val = 0.0 if is_0_5_scale else 1.0
    vmax_val = 5.0 if is_0_5_scale else 10.0
    cbar_label = "Semantic Score [0 – 5]" if is_0_5_scale else "Semantic Score [1 – 10]"
    cbar_ticks = range(0, 6) if is_0_5_scale else range(1, 11)

    sc_c = ax_c.scatter(
        assume_umap[:, 0],
        assume_umap[:, 1],
        s=point_size,
        c=c0_values,
        cmap="viridis",
        vmin=vmin_val,
        vmax=vmax_val,
        alpha=point_alpha,
        edgecolors="none",
    )
    ax_c.set_title(f"c  Top Concept 1: {concept_0_name}", weight="bold", loc="left", pad=10)
    ax_c.set_xlabel("UMAP Dimension 1")
    ax_c.set_ylabel("UMAP Dimension 2")
    ax_c.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
    cbar_c = plt.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label(cbar_label, fontsize=10.5)
    cbar_c.set_ticks(cbar_ticks)

    # ----------------------------------------------------
    # Panel D: ASSUME Concept Space colored by Concept 1
    # (Periodic Array Regularity)
    # ----------------------------------------------------
    concept_1_idx = 1
    concept_1_name = concept_names[concept_1_idx] if len(concept_names) > concept_1_idx else "Concept 2"
    c1_values = assume_features[:, concept_1_idx]

    sc_d = ax_d.scatter(
        assume_umap[:, 0],
        assume_umap[:, 1],
        s=point_size,
        c=c1_values,
        cmap="plasma",
        vmin=vmin_val,
        vmax=vmax_val,
        alpha=point_alpha,
        edgecolors="none",
    )
    ax_d.set_title(f"d  Top Concept 2: {concept_1_name}", weight="bold", loc="left", pad=10)
    ax_d.set_xlabel("UMAP Dimension 1")
    ax_d.set_ylabel("UMAP Dimension 2")
    ax_d.grid(True, linestyle="--", alpha=0.3, color="#cccccc")
    cbar_d = plt.colorbar(sc_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label(cbar_label, fontsize=10.5)
    cbar_d.set_ticks(cbar_ticks)

    # Clean styling
    for ax in [ax_a, ax_b, ax_c, ax_d]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_aspect("auto")

    fig.suptitle(
        "DINOv3 embedding vs ASSUME concept embedding (UMAP, colored by class; labels not used by ASSUME)",
        weight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved 4-panel comparison figure to %s (%d DPI)", out_p, dpi)

    if root_out_p:
        _mirror_file(out_p, root_out_p)
        logger.info("Mirrored comparison figure to %s", root_out_p)

    return out_p
