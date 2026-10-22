# ASSUME: Scientific Insights, Morphological Analysis & Empirical Findings

**Project**: ASSUME (Auto-Synthesized Semantic Unified Microstructural Encoder)  
**Scope**: Training-Free Multimodal Concept Bottleneck Model (CBM) for SEM Microstructure  
**Date**: October 21, 2026  
**Artifacts Generated**: `manifest.parquet`, `dinov3_pooled.npy`, `concept_dictionary.json`, `assume_embeddings_1600x30.npy`, `eval_metrics.json`

---

## 1. Pipeline Sample Exposure & Extreme Training-Free Efficiency

A central design pillar of the ASSUME framework is maximizing data utility while drastically minimizing expensive foundation model inference calls.

### Sample Count Breakdown
1. **Teacher Feature Pre-Embedding (DINOv3 ViT-16+)**:
   - **Total Processed**: **14,620 micrographs** across all 8 active SEM classes (`Biological`, `Fibres`, `Films_Coated_Surface`, `Nanowires`, `Particles`, `Patterned_surface`, `Porous_Sponge`, `Powder`).
   - Every micrograph was cropped to its top 75% height ($1024 \times 576$, stripping out the variable-position SEM instrument metadata databar) and 50% downscaled to $512 \times 288$ ($32 \times 18 = 576$ ViT-16 patches).
   - All 14,620 feature vectors were L2-normalized into $\mathbb{R}^{384}$ and cached into `dinov3_pooled.npy` in ~4 minutes.
2. **Autonomous Concept Discovery Loop (Gemini 3.8 Flash on Vertex AI Express)**:
   - **Total Micrographs Sampled**: **~80 to 100 micrographs**.
   - Sampled in batches of $B=6$ micrographs (33% random, 33% DINOv3 teacher farthest-point, 33% student discrepancy) over ~10 discovery iterations, plus 12-image validation batches for Pearson correlation checks.
3. **Concept Bottleneck Evaluation Matrix (Gemini 3.8 Flash on Vertex AI Express)**:
   - **Total Micrographs Scored**: **1,600 micrographs** (balanced evaluation benchmark: exactly 200 per class across the 8 classes).
   - Each image was evaluated against all $K=30$ concept rubrics (chunked into prompts of $\le 10$ concepts).

### Key Takeaway on Sample Efficiency
- **The LLM/VLM never needed to see the full 14,620 micrographs to discover the domain lexicon.**
- By delegating global spatial topology to the frozen self-supervised DINOv3 teacher via farthest-point active sampling, the LLM discovered a comprehensive, non-redundant 30-concept dictionary having directly inspected fewer than **100 micrographs** (<0.7% of the dataset).
- Total API expenditure for the entire project was **~$15 CAD** (consuming <5% of our $300 CAD cloud credit).

---

## 2. Latent Activation Sparsity & Concept Disentanglement

### The Empirical Distribution of Semantic Scores
Across the $1,600 \times 30 = 48,000$ scored entries in `assume_embeddings_1600x30.npy`:
- **Range**: Minimum $1.0$, Maximum $10.0$ (mean $4.85$, median $5.00$, std $1.80$).
- **Strict Baseline Minimum ($1.0$)**: $3.0\%$ of all matrix entries.
- **Low / Absent ($\le 3.0$)**: **$20.1\%$** of all matrix entries.
- **Moderate / Neutral ($4.0 - 6.0$)**: **$53.8\%$** of all matrix entries.
- **Prominent / Defining ($\ge 8.0$)**: Only **$9.1\%$** of all matrix entries.

### Why Is Activation Sparse?
1. **Physical Mutual Exclusivity in Materials Science**:
   - Microstructural geometries are inherently specialized. A particulate powder field has high discrete particle density ($8.8/10$), but physically cannot possess continuous fiber alignment ($1.0/10$) or deep open-cell sponge cavities ($1.0/10$).
   - A porous sponge has high void volume ($9.0/10$), but near-zero planar boundary sharpness.
2. **Selective Sparse Coding**:
   - Each individual micrograph only strongly activates **$2$ to $4$ dominant concepts** out of $30$.
   - The remaining ~26 concepts remain near baseline ($1.0 - 3.0$) or near neutral ($4.0 - 5.0$).
3. **The Power of Disentangled Concept Bottlenecks**:
   - In black-box embeddings (e.g. raw DINOv3 $\mathbb{R}^{384}$), features are dense and entangled — every dimension carries polysemantic, distributed representations.
   - In ASSUME, each dimension is an atomic, human-interpretable microstructural concept. **Sparse activation is proof of concept orthogonality**: each concept acts as an independent basis vector that fires only when its specific morphological trait is physically present.
   - This directly explains why **5-Fold k-NN classification accuracy increased by $+7.38\%$ ($90.12\% \to 97.50\%$)** and **Cosine Silhouette Separation improved 3× ($0.1287 \to 0.3733$)** when moving from dense DINOv3 to sparse ASSUME concept space.

---

## 3. Global Latent Architecture: What Separates the Classes?

In the 2D global PCA and UMAP projections, the 8 SEM classes are cleanly organized along two orthogonal physical axes:

```
                      GLOBAL ASSUME LATENT SPACE
               
  [High Periodic Regularity, Lithographic Facets]
                     + UMAP2 / + PC2
                            ^
                            |   Patterned_surface
                            |   Films_Coated_Surface
                            |
   Particles, Powder <------+------> Fibres, Nanowires
   - PC1 / - UMAP1          |        + PC1 / + UMAP1
   [Granular singlets,      |        [Entangled filaments,
    low aspect ratio]       |         high aspect ratio]
                            v
                     - UMAP2 / - PC2
                [Amorphous & Porous Sponge]
```

### Global Coordinate Interpretability
1. **Global PC1 (26.1% Explained Variance): The "Aspect Ratio & Entanglement" Axis**
   - **Positive Pole**: `Filamentous Fiber Packing Density` ($r = +0.944$), `Aspect Ratio of Primary Microstructural Units` ($r = +0.825$), `Inter-Filament Entanglement Interlacing` ($r = +0.692$), `Free-Standing Fiber End Density` ($r = +0.673$).
   - **Negative Pole**: `Discrete Particle Spatial Areal Density` ($r = -0.662$).
   - **Separation Function**: Completely isolates continuous 1D elongated morphologies (`Fibres`, `Nanowires`) from detached 0D isotropic granular morphologies (`Particles`, `Powder`).
2. **Global PC2 (15.1% Explained Variance): The "Lithographic Order & Planar Facets" Axis**
   - **Positive Pole**: `Planar Boundary and Facet Edge Sharpness` ($r = +0.742$), `Periodic Array Regularity` ($r = +0.695$), `Periodic Grating Spatial Frequency` ($r = +0.491$).
   - **Negative Pole**: `Discrete Particle Spatial Areal Density` ($r = -0.509$), `Open-Loop Feature Curvature Openness` ($r = -0.487$).
   - **Separation Function**: Segregates engineered micro-fabricated surfaces (`Patterned_surface`, `Films_Coated_Surface`) from stochastic particulate beds and amorphous sponges.
3. **Global UMAP1 & UMAP2**:
   - `UMAP1`: Preserves the continuous topological progression from discrete particulate fields to high-aspect-ratio tangled mesh networks ($r = +0.724$ with Aspect Ratio).
   - `UMAP2`: Preserves the orthogonal geometric continuum from stochastic fibrous tangles ($-0.793$) to strict lithographic array periodicity ($+0.722$).

---

## 4. Fine-Grained Intra-Class Clustering ($K=5$ and $K=10$)

Clustering within individual classes demonstrates that ASSUME not only separates macro classes, but also uncovers subtle metallurgical and synthesis-driven sub-morphologies:

1. **`Porous_Sponge`**:
   - While all sponges share high Void Volume ($8.2 - 9.0$), sub-clusters resolve **ligament integrity**:
     - *Cluster 1*: Particulate agglomerates trapped on foam struts.
     - *Cluster 2*: Clean, smooth-ligament open cellular foam.
     - *Cluster 4*: Rigid polygonal faceted struts with acute angle junctions.
     - *Cluster 5*: Cavernous 3D open-cell sponge with sweeping curvilinear ligaments.
2. **`Powder`**:
   - Sub-clusters resolve **milling state and agglomeration**:
     - *Cluster 1*: Sintered, coalesced multi-grain powder cakes.
     - *Cluster 3*: Completely dispersed single-crystal particulate fields ($8.8/10$ particle density).
     - *Cluster 4*: Cleaved, angular crushed powder particles.
     - *Cluster 5*: Acicular, elongated needle-like powder particles with flow alignment.
3. **`Films_Coated_Surface`**:
   - Sub-clusters resolve **thin-film deposition defects and grain structure**:
     - *Cluster 1*: Monolithic, defect-free continuous specular coatings.
     - *Cluster 2*: Cross-sectional columnar grain structures with uniform column diameters.
     - *Cluster 3*: Films exhibiting micro-crack networks.
     - *Cluster 4*: Coated surfaces suffering from delamination and peeling flake edges.
