#!/usr/bin/env python3
"""Run KMeans K=5 for individual classes: Porous_Sponge, Powder, and Films_Coated_Surface.

For each class:
- Fits KMeans with K=5 on the 30-dim ASSUME concept embeddings (N=200).
- Generates a 5-row x 5-column publication-grade figure.
- Each row represents one cluster, showing 5 representative micrographs (closest to centroid).
- Annotates each cluster row with its population size (N), top positive signature concept,
  and top negative signature concept.
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
logger = logging.getLogger("assume.k5_clustering")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "real" / "assume_embeddings_verified.npy"
CONCEPTS_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
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


def preprocess_micrograph(image_path: str, target_size: int = 240) -> Image.Image:
    try:
        with Image.open(resolve_micrograph(image_path)) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            # Crop bottom 25% to remove instrument databar
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


def plot_k5_cluster_grid(
    cluster_data: List[Dict],
    class_name: str,
    save_path: Path,
    target_size: int = 240,
    dpi: int = 300,
):
    K = 5
    n_cols = 5
    
    fig, axes = plt.subplots(K, n_cols, figsize=(16, 17), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    
    for r, cl_info in enumerate(cluster_data):
        c_id = cl_info["cluster_id"]
        size = cl_info["size"]
        top_pos = cl_info["top_positive_concept"]
        pos_val = cl_info["top_positive_score"]
        pos_delta = cl_info["top_positive_delta"]
        imgs = cl_info["images"]
        
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(imgs):
                thumb = preprocess_micrograph(imgs[c]["path"], target_size=target_size)
                ax.imshow(thumb)
                dist = imgs[c]["distance"]
                ax.set_title(f"Rank {c+1} (d={dist:.2f})", fontsize=10, pad=4)
            else:
                ax.axis("off")
                continue
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#BBBBBB")
                spine.set_linewidth(1.2)
                
            # Cluster label on leftmost column
            if c == 0:
                ax.set_ylabel(
                    f"Cluster {c_id}\n(N={size})\n\nPrimary Driver:\n{top_pos[:34]}\n(score {pos_val:.1f}, {pos_delta:+.1f} vs class mean)",
                    fontsize=9.5,
                    fontweight="bold",
                    rotation=0,
                    labelpad=105,
                    ha="center",
                    va="center",
                )
                
    plt.suptitle(
        f"ASSUME Semantic Latent Clustering (K=5): {class_name} (N=200 Micrographs)\n(Each row displays 5 representative micrographs closest to cluster centroid)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved K=5 cluster grid to {save_path}")


def run_k5_clustering(
    subset_df: pd.DataFrame,
    subset_emb: np.ndarray,
    concept_names: List[str],
    random_state: int = 42,
) -> List[Dict]:
    K = 5
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(subset_emb)
    centroids = kmeans.cluster_centers_
    
    pop_mean = np.mean(subset_emb, axis=0)
    
    # Sort clusters descending by size
    cluster_sizes = [(c_idx, np.sum(labels == c_idx)) for c_idx in range(K)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for rank, (c_idx, count) in enumerate(cluster_sizes, start=1):
        c_indices = np.where(labels == c_idx)[0]
        c_points = subset_emb[c_indices]
        c_paths = subset_df["full_path"].iloc[c_indices].tolist()
        centroid = centroids[c_idx]
        
        # Distances to centroid
        dists = np.linalg.norm(c_points - centroid, axis=1)
        sort_order = np.argsort(dists)
        
        # Top 5 closest
        top_5_imgs = []
        for i in sort_order[:5]:
            top_5_imgs.append({
                "path": c_paths[i],
                "distance": float(dists[i]),
            })
            
        # Top positive deviation from population mean
        diff = centroid - pop_mean
        pos_idx = int(np.argmax(diff))
        neg_idx = int(np.argmin(diff))
        
        # Also top 3 active concepts in absolute terms
        top_active_indices = np.argsort(centroid)[::-1][:3]
        top_active = [
            {"concept": concept_names[idx], "score": float(centroid[idx])}
            for idx in top_active_indices
        ]
        
        results.append({
            "cluster_id": rank,
            "original_id": c_idx,
            "size": int(count),
            "top_positive_concept": concept_names[pos_idx],
            "top_positive_score": float(centroid[pos_idx]),
            "top_positive_delta": float(diff[pos_idx]),
            "top_negative_concept": concept_names[neg_idx],
            "top_negative_score": float(centroid[neg_idx]),
            "top_negative_delta": float(diff[neg_idx]),
            "top_active_concepts": top_active,
            "images": top_5_imgs,
        })
        
    return results


def main():
    logger.info("Starting KMeans K=5 analysis for Porous_Sponge, Powder, and Films_Coated_Surface...")
    df, embeddings, concept_names = load_data()
    
    report = {}
    for cls in TARGET_CLASSES:
        logger.info(f"Clustering {cls} with K=5...")
        mask = (df["class_label"] == cls).values
        cls_df = df[mask].reset_index(drop=True)
        cls_emb = embeddings[mask]
        
        clusters = run_k5_clustering(cls_df, cls_emb, concept_names)
        
        save_path = OUTPUT_DIR / f"{cls}_kmeans_k5_clusters.png"
        plot_k5_cluster_grid(clusters, class_name=cls, save_path=save_path)
        
        report[cls] = clusters
        
    with open(OUTPUT_DIR / "kmeans_k5_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Saved kmeans_k5_summary.json successfully.")
    print("\n>>> ALL K=5 CLUSTERINGS COMPLETED! <<<")


if __name__ == "__main__":
    main()
