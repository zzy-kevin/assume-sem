#!/usr/bin/env python3
"""ASSUME KMeans Clustering & Micrograph Strip Visualizer.

For classes: Porous_Sponge, Powder, and Films_Coated_Surface:
1. Runs KMeans with K=10 on ASSUME 30-dim concept embeddings.
2. For each cluster (row), displays 5 representative micrographs (closest to centroid).
3. Annotates each cluster row with its size and top signature semantic concepts.
4. Generates both per-class 10x5 figures and a combined 3-class 10x5 figure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assume.clustering")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "eval_manifest.parquet"
EMBEDDINGS_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_PATH = PROJECT_ROOT / "concept_dictionary.json"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["Porous_Sponge", "Powder", "Films_Coated_Surface"]


def load_data():
    df = pd.read_parquet(EVAL_MANIFEST_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(CONCEPTS_PATH, "r", encoding="utf-8") as f:
        concept_dict = json.load(f)
    concept_names = [c["name"] for c in concept_dict]
    return df, embeddings, concept_names


def preprocess_micrograph(image_path: str, target_size: int = 200) -> Image.Image:
    try:
        with Image.open(resolve_micrograph(image_path)) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            # Crop bottom 25% to remove databar
            h_crop = max(1, int(h * 0.75))
            img_cropped = img.crop((0, 0, w, h_crop))
            # Center crop to 1:1 square
            cw, ch = img_cropped.size
            sq_dim = min(cw, ch)
            left = (cw - sq_dim) // 2
            top = (ch - sq_dim) // 2
            img_square = img_cropped.crop((left, top, left + sq_dim, top + sq_dim))
            return img_square.resize((target_size, target_size), Image.Resampling.LANCZOS)
    except Exception as e:
        logger.warning(f"Error loading {image_path}: {e}")
        return Image.new("RGB", (target_size, target_size), color=(220, 220, 220))


def plot_cluster_grid(
    cluster_data: List[Dict],
    title: str,
    save_path: Path,
    target_size: int = 200,
    dpi: int = 250,
):
    """Plot a 10-row x 5-column figure where each row is one cluster.
    
    Each row has:
    - Left text label: Cluster ID, count, and top distinctive concept.
    - 5 image columns: Micrographs closest to the centroid.
    """
    K = len(cluster_data)
    n_cols = 5
    
    fig, axes = plt.subplots(K, n_cols, figsize=(15, 2.6 * K), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    
    for r, cl_info in enumerate(cluster_data):
        c_id = cl_info["cluster_id"]
        size = cl_info["size"]
        top_concept = cl_info["signature_concept"]
        score_val = cl_info["signature_score"]
        imgs = cl_info["images"]
        
        for c in range(n_cols):
            ax = axes[r, c] if K > 1 else axes[c]
            if c < len(imgs):
                thumb = preprocess_micrograph(imgs[c]["path"], target_size=target_size)
                ax.imshow(thumb)
                dist = imgs[c]["distance"]
                ax.set_title(f"Rank {c+1} (d={dist:.2f})", fontsize=9, pad=4)
            else:
                ax.axis("off")
                continue
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#CCCCCC")
                spine.set_linewidth(1)
                
            # Add cluster descriptor label on the leftmost plot
            if c == 0:
                ax.set_ylabel(
                    f"Cluster {c_id}\n(N={size})\nTop: {top_concept[:22]}\n(mean={score_val:.1f})",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    labelpad=95,
                    ha="center",
                    va="center",
                )
                
    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved cluster grid to {save_path}")


def run_kmeans_and_extract_clusters(
    subset_df: pd.DataFrame,
    subset_emb: np.ndarray,
    concept_names: List[str],
    k: int = 10,
    random_state: int = 42,
) -> Tuple[List[Dict], KMeans]:
    """Fit KMeans and extract top 5 closest micrographs per cluster."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(subset_emb)
    centroids = kmeans.cluster_centers_
    
    # Calculate population mean of each concept to find what's distinctive for each cluster
    pop_mean = np.mean(subset_emb, axis=0)
    
    cluster_results = []
    # Sort clusters by size descending for clean presentation
    cluster_sizes = [(c_idx, np.sum(labels == c_idx)) for c_idx in range(k)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (c_idx, count) in enumerate(cluster_sizes, start=1):
        c_mask = (labels == c_idx)
        c_indices = np.where(c_mask)[0]
        c_points = subset_emb[c_indices]
        c_paths = subset_df["full_path"].iloc[c_indices].tolist()
        centroid = centroids[c_idx]
        
        # Distances to centroid
        dists = np.linalg.norm(c_points - centroid, axis=1)
        sort_order = np.argsort(dists)
        
        # Pick top 5 closest
        top_k_imgs = []
        for i in sort_order[:5]:
            top_k_imgs.append({
                "path": c_paths[i],
                "distance": float(dists[i]),
            })
            
        # Determine signature concept (highest difference from population mean)
        diff = centroid - pop_mean
        top_c_feat_idx = int(np.argmax(np.abs(diff)))
        sig_name = concept_names[top_c_feat_idx]
        sig_score = float(centroid[top_c_feat_idx])
        
        cluster_results.append({
            "cluster_id": rank,
            "original_kmeans_id": c_idx,
            "size": int(count),
            "signature_concept": sig_name,
            "signature_score": sig_score,
            "centroid": centroid.tolist(),
            "images": top_k_imgs,
        })
        
    return cluster_results, kmeans


def main():
    logger.info("Starting KMeans K=10 clustering on target classes...")
    df, embeddings, concept_names = load_data()
    
    summary_report = {}
    
    # 1. Individual Per-Class Clustering
    for cls in TARGET_CLASSES:
        logger.info(f"Clustering class: {cls} (K=10)...")
        mask = (df["class_label"] == cls).values
        cls_df = df[mask].reset_index(drop=True)
        cls_emb = embeddings[mask]
        
        clusters, _ = run_kmeans_and_extract_clusters(cls_df, cls_emb, concept_names, k=10)
        
        save_path = OUTPUT_DIR / f"{cls}_kmeans_k10_clusters.png"
        plot_cluster_grid(
            clusters,
            title=f"ASSUME Semantic Latent Clustering (K=10) ? {cls} (N=200)",
            save_path=save_path,
        )
        
        summary_report[cls] = [
            {
                "cluster_rank": c["cluster_id"],
                "size": c["size"],
                "signature_concept": c["signature_concept"],
                "signature_score": round(c["signature_score"], 2),
            }
            for c in clusters
        ]
        
    # 2. Combined 3-Class Clustering
    logger.info("Clustering combined Porous_Sponge + Powder + Films_Coated_Surface (N=600, K=10)...")
    comb_mask = df["class_label"].isin(TARGET_CLASSES).values
    comb_df = df[comb_mask].reset_index(drop=True)
    comb_emb = embeddings[comb_mask]
    
    comb_clusters, _ = run_kmeans_and_extract_clusters(comb_df, comb_emb, concept_names, k=10)
    comb_save_path = OUTPUT_DIR / "combined_3classes_kmeans_k10_clusters.png"
    plot_cluster_grid(
        comb_clusters,
        title="ASSUME Semantic Latent Clustering (K=10) ? Porous Sponge + Powder + Films (N=600)",
        save_path=comb_save_path,
    )
    
    summary_report["Combined_3Classes"] = [
        {
            "cluster_rank": c["cluster_id"],
            "size": c["size"],
            "signature_concept": c["signature_concept"],
            "signature_score": round(c["signature_score"], 2),
        }
        for c in comb_clusters
    ]
    
    with open(OUTPUT_DIR / "kmeans_k10_clusters_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
        
    logger.info("Exported kmeans_k10_clusters_summary.json successfully.")
    print("\n>>> KMEANS K=10 CLUSTERING COMPLETED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
