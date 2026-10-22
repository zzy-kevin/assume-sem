#!/usr/bin/env python3
"""Remake K=5 and K=10 cluster plots using strictly the closest micrographs to each cluster centroid.

For each of Porous_Sponge, Powder, and Films_Coated_Surface:
1. Runs KMeans on 30-dim ASSUME concept embeddings.
2. For each cluster (row):
   - Computes cluster centroid ?_c = (1/N_c) sum_{i in C_c} x_i
   - Computes Euclidean distance d(x_i, ?_c) = ||x_i - ?_c||_2
   - Sorts points ascending by distance
   - Plots the top 5 closest micrographs (Rank 1 = Medoid, Rank 2, Rank 3, Rank 4, Rank 5)
3. Clearly titles each column with "Rank 1 (Medoid, d=...)", "Rank 2 (d=...)", etc.
4. Generates both K=5 and K=10 figures at 300 DPI.
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
logger = logging.getLogger("assume.cluster_remake")

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


def preprocess_micrograph(image_path: str, target_size: int = 240) -> Image.Image:
    try:
        with Image.open(resolve_micrograph(image_path)) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            # Crop bottom 25% (remove instrument metadata databar)
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


def compute_clusters_closest(
    subset_df: pd.DataFrame,
    subset_emb: np.ndarray,
    concept_names: List[str],
    k: int,
    random_state: int = 42,
) -> List[Dict]:
    """Fit KMeans and extract the top 5 strictly closest micrographs to each centroid."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(subset_emb)
    centroids = kmeans.cluster_centers_
    
    pop_mean = np.mean(subset_emb, axis=0)
    
    # Sort clusters descending by size
    cluster_sizes = [(c_idx, np.sum(labels == c_idx)) for c_idx in range(k)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    cluster_info = []
    for rank, (c_idx, count) in enumerate(cluster_sizes, start=1):
        c_mask = (labels == c_idx)
        c_indices = np.where(c_mask)[0]
        c_points = subset_emb[c_indices]
        c_paths = subset_df["full_path"].iloc[c_indices].tolist()
        centroid = centroids[c_idx]
        
        # Euclidean distances to centroid in 30-dim concept space
        dists = np.linalg.norm(c_points - centroid, axis=1)
        sort_order = np.argsort(dists)
        
        # Top 5 strictly closest points
        top_5 = []
        for i in sort_order[:5]:
            top_5.append({
                "path": c_paths[i],
                "filename": Path(c_paths[i]).name,
                "distance": float(dists[i]),
            })
            
        # Signature concept
        diff = centroid - pop_mean
        pos_idx = int(np.argmax(diff))
        
        cluster_info.append({
            "cluster_rank": rank,
            "size": int(count),
            "signature_concept": concept_names[pos_idx],
            "signature_score": float(centroid[pos_idx]),
            "signature_delta": float(diff[pos_idx]),
            "images": top_5,
        })
        
    return cluster_info


def plot_cluster_grid_closest(
    clusters: List[Dict],
    class_name: str,
    k: int,
    save_path: Path,
    target_size: int = 240,
    dpi: int = 300,
):
    n_cols = 5
    fig, axes = plt.subplots(k, n_cols, figsize=(16, 2.7 * k), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    
    col_labels = [
        "1st Closest\n(Cluster Medoid)",
        "2nd Closest",
        "3rd Closest",
        "4th Closest",
        "5th Closest",
    ]
    
    for r, cl in enumerate(clusters):
        c_rank = cl["cluster_rank"]
        size = cl["size"]
        sig_name = cl["signature_concept"]
        sig_score = cl["signature_score"]
        sig_delta = cl["signature_delta"]
        imgs = cl["images"]
        
        for c in range(n_cols):
            ax = axes[r, c] if k > 1 else axes[c]
            if c < len(imgs):
                thumb = preprocess_micrograph(imgs[c]["path"], target_size=target_size)
                ax.imshow(thumb)
                dist = imgs[c]["distance"]
                # Column header on row 0
                if r == 0:
                    ax.set_title(f"{col_labels[c]}\n(d={dist:.2f})", fontsize=10, fontweight="bold", pad=6)
                else:
                    ax.set_title(f"Rank {c+1} (d={dist:.2f})", fontsize=9, pad=4)
            else:
                ax.axis("off")
                continue
                
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#BBBBBB")
                spine.set_linewidth(1.2)
                
            if c == 0:
                ax.set_ylabel(
                    f"Cluster {c_rank} (N={size})\nTop Archetype:\n{sig_name[:22]}\n(mean={sig_score:.1f}, ?={sig_delta:+.1f})",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    labelpad=95,
                    ha="center",
                    va="center",
                )
                
    plt.suptitle(
        f"ASSUME Semantic Latent Clustering (K={k}) ? {class_name} (N=200 Micrographs)\n(Each row strictly displays the 5 micrographs closest to the 30-dim cluster centroid)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved K={k} cluster grid to {save_path}")


def main():
    logger.info("Regenerating K=5 and K=10 cluster plots with strict centroid-closest micrographs...")
    df, embeddings, concept_names = load_data()
    
    for k in [5, 10]:
        for cls in TARGET_CLASSES:
            logger.info(f"Processing {cls} with K={k}...")
            mask = (df["class_label"] == cls).values
            cls_df = df[mask].reset_index(drop=True)
            cls_emb = embeddings[mask]
            
            clusters = compute_clusters_closest(cls_df, cls_emb, concept_names, k=k)
            save_path = OUTPUT_DIR / f"{cls}_kmeans_k{k}_clusters.png"
            plot_cluster_grid_closest(clusters, class_name=cls, k=k, save_path=save_path)
            
    print("\n>>> ALL K=5 AND K=10 CLUSTER PLOTS REGENERATED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    main()
