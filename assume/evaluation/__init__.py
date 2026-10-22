"""Evaluation and visualization module for ASSUME Milestone 5.

Provides:
- Balanced evaluation subset creation (1,600 micrographs, 200/class across 8 classes).
- Zero-shot semantic concept bottleneck matrix evaluation and checkpointing.
- Mock and live micrograph scoring interfaces.
- DINOv3 feature subset extraction and 2D UMAP projection.
- Quantitative cluster separation metrics (Cosine Silhouette and 5-fold k-NN accuracy).
- Publication-quality 4-panel comparison visualization.
"""

from __future__ import annotations

from assume.evaluation.bottleneck_eval import (
    MockScorerAgent,
    compute_bottleneck_embeddings,
    create_balanced_eval_subset,
)
from assume.evaluation.umap_visualizer import (
    compute_separation_metrics,
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)

__all__ = [
    "create_balanced_eval_subset",
    "compute_bottleneck_embeddings",
    "MockScorerAgent",
    "extract_subset_features",
    "compute_umap_projection",
    "compute_separation_metrics",
    "generate_umap_comparison_figure",
]
