# Project: ASSUME (Auto-Synthesized Semantic Unified Microstructural Encoder)

## Architecture
ASSUME is an agentic, training-free Concept Bottleneck framework for scanning electron microscopy (SEM) micrographs that extracts human-interpretable microstructural concepts using DINOv3 active sampling and Gemini 3.8 Flash on Vertex AI Express.

### System Pipeline & Data Flow
1. **Ingestion & Teacher Pre-Embedding**:
   - 14,620 in-scope SEM JPGs (1024x768 4:3) across 8 classes from `D:\UOFT\Grad1_local\SEM\online\t2v4p-2x965` (`Powder_highres` excluded).
   - Proportional crop: top 75% height `(0, 0, W, int(round(H * 0.75)))` -> bilinear resize to `(512, 288)`.
   - Feature Extraction: `facebook/dinov3-vits16plus-pretrain-lvd1689m` (patch mean pooling of tokens 5:581 -> L2 normalized 384-d vector).
   - Cached artifacts: `data/manifest.parquet` and `data/dinov3_pooled.npy` (also available at root `manifest.parquet` and `dinov3_pooled.npy`).
2. **Tri-Modal Active Sampler**:
   - Batch size $B=6$ per iteration with 33% Uniform Random, 33% Teacher Farthest-Point (in DINOv3 space), and 33% Student Discrepancy/Collision ($\text{sim}_{\text{ASSUME}} > 0.85$ and $d_{\text{DINO}} > 0.40$).
   - Step 0: 100% Random + FPS initialization.
3. **Resilient Gemini Vertex AI Harness**:
   - Direct REST client using `VERTEX_API_KEY` and `VERTEX_PROJECT_ID` with full-jitter exponential backoff and HTTP 429 / RESOURCE_EXHAUSTED retry.
   - Proposer Agent: Synthesizes atomic, single-property rubrics with 1, 5, 10 monotonic anchors in structured JSON.
   - Scorer Agent: Evaluates micrographs against rubrics chunked into $\le 10$ concepts per prompt, returning integer scores $[1, 10]$.
   - Visual Merger Agent: Evaluates collinear concept pairs ($|r| > 0.80$) with 2-3 distinguishing micrographs to decide Merge or Keep Both.
4. **Iterative Discovery & Pruning Loop**:
   - Discovers candidate concepts until target $K=30$ validated, non-redundant concepts are reached.
   - Every 3 batches: evaluates on DINOv3 FPS images, prunes dead dimensions ($\sigma < 0.20$), and resolves collinear pairs ($|r| > 0.80$).
5. **Final Evaluation & UMAP Projections**:
   - Balanced evaluation subset of 1,600 micrographs (200 per class across 8 classes, with deterministic resampling seed 42 for `Fibres` and `Porous_Sponge`).
   - Scores 1,600 images against finalized $K=30$ rubrics -> exports `assume_embeddings_1600x30.npy` and `concept_dictionary.json`.
   - 2D UMAP projection and quantitative metrics (`umap_assume_vs_dinov3.png`, silhouette score, 5-fold k-NN classification accuracy).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Micrograph Ingestion & Filtering | Scan 14,620 JPGs across 8 classes, exclude `Powder_highres` | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Robust Proportional Preprocessing | Top 75% crop removing databar + resize to 512x288 (handles 133 non-standard res) | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Frozen DINOv3 Teacher Embedding | ViT-S/16+ feature extraction with patch mean pooling (tokens 5:581) + L2 normalization | M1 | ORIGINAL_REQUEST §R1 |
| 4 | Cache Manifest & Pooled Features | Write `manifest.parquet` and `dinov3_pooled.npy` (14,620 x 384 float32) | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Uniform Random Stream (33%) | Unbiased sampling of 2 micrographs per batch | M2 | ORIGINAL_REQUEST §R2 |
| 6 | Teacher Farthest-Point Stream (33%) | Vectorized cosine FPS in DINOv3 space (2 micrographs per batch) | M2 | ORIGINAL_REQUEST §R2 |
| 7 | Student Collision Stream (33%) | Discrepancy sampling: $\text{sim}_{\text{ASSUME}} > 0.85$ and $d_{\text{DINO}} > 0.40$ (2 micrographs) | M2 | ORIGINAL_REQUEST §R2 |
| 8 | Step 0 Cold-Start Policy | 100% Random + FPS initialization before student concepts exist | M2 | ORIGINAL_REQUEST §R2 |
| 9 | Resilient Vertex AI Express Client | REST client with exponential backoff, full jitter, 429 / RESOURCE_EXHAUSTED retry | M3 | ORIGINAL_REQUEST §R3 |
| 10 | Proposer Agent | B=6 micrographs -> atomic single-property rubrics with 1, 5, 10 monotonic anchors | M3 | ORIGINAL_REQUEST §R3 |
| 11 | Scorer Agent | Micrograph scoring with $\le 10$ concept chunking -> integer scores [1, 10] | M3 | ORIGINAL_REQUEST §R3 |
| 12 | Visual Merger Agent | Inspects collinear pairs ($|r| > 0.80$) with 2-3 images -> Merge or Keep Both | M3 | ORIGINAL_REQUEST §R3 |
| 13 | Active Discovery Orchestration | Iterates sampling, proposing, scoring to discover candidate concepts toward K=30 | M4 | ORIGINAL_REQUEST §R4 |
| 14 | Diversity Evaluation Batches | Evaluates candidate concepts every 3 batches on DINOv3 FPS images | M4 | ORIGINAL_REQUEST §R4 |
| 15 | Dead Dimension Pruner | Prunes zero-variance dimensions ($\sigma < 0.20$ or >90% identical) | M4 | ORIGINAL_REQUEST §R4 |
| 16 | Pearson Correlation Pruning Trigger | Calculates Pearson $R$ matrix on FPS evaluation batches, routing $|r| > 0.80$ to Merger | M4 | ORIGINAL_REQUEST §R4 |
| 17 | Balanced 1,600 Evaluation Sampler | Selects exactly 200 per class across 8 classes (seed 42 resampling for Fibres & Porous_Sponge) | M5 | ORIGINAL_REQUEST §R5 |
| 18 | Zero-Shot Bottleneck Scorer | Fully scores 1,600 images on K=30 concepts -> `assume_embeddings_1600x30.npy` | M5 | ORIGINAL_REQUEST §R5 |
| 19 | Concept Dictionary JSON Exporter | Exports finalized validated rubrics to `concept_dictionary.json` | M5 | ORIGINAL_REQUEST §R5 |
| 20 | 2D UMAP Projections Visualizer | Publication-quality 4-panel comparison figure `umap_assume_vs_dinov3.png` | M5 | ORIGINAL_REQUEST §R5 |
| 21 | Quantitative Separation Metrics | Evaluates silhouette score (cosine) and 5-fold k-NN accuracy (DINOv3 vs. ASSUME) | M5 | ORIGINAL_REQUEST §R5 |
| 22 | Comprehensive E2E Test Suite | 4-tier test harness validating all requirements independently | M-TEST | Dual Track |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Micrograph Ingestion & Frozen DINOv3 Teacher Pre-Embedding | Ingestion, proportional cropping, downscaling, DINOv3 feature extraction, caching `manifest.parquet` and `dinov3_pooled.npy` | none | DONE |
| M2 | Tri-Modal Active Batch Sampler | 33% Random, 33% DINOv3 FPS, 33% Student Collision Sampler with step 0 cold-start | M1 | DONE |
| M3 | Resilient Gemini Vertex API Harness | Vertex Express REST client, Proposer Agent, Scorer Agent, Visual Merger Agent with backoff | none | DONE |
| M4 | Iterative Discovery & Pruning Loop | Autonomous active loop to K=30 concepts: evaluation every 3 batches, dead dimension pruner, visual merger | M1, M2, M3 | DONE |
| M5 | Final Zero-Shot Bottleneck Encoding & UMAP Evaluation | 1,600 balanced eval subset scoring, `assume_embeddings_1600x30.npy`, `concept_dictionary.json`, UMAP figure & metrics | M1, M3, M4 | DONE |
| M-TEST | E2E Testing Suite (Tiers 1-4) & Adversarial Hardening (Tier 5) | Comprehensive opaque-box and hermetic test suite verifying all 22 features, publishing `TEST_READY.md` | M1-M5 | DONE |

---

## Interface Contracts

### M1 ↔ M2, M4, M5 (Manifest and Teacher Embeddings)
- `manifest.parquet`: pyarrow table with columns `image_id` (str), `class_label` (str), `class_id` (int), `full_path` (str), `emb_index` (int 0..14619).
- `dinov3_pooled.npy`: numpy array shape `(14620, 384)`, dtype `float32`, each row $\|v\|_2 = 1.0 \pm 1e-5$. Row $i$ corresponds to `emb_index = i`.

### M2 ↔ M4 (Tri-Modal Batch Sampler)
- `sample_batch(step: int, batch_size: int = 6, concept_scores: Optional[np.ndarray] = None) -> List[int]`:
  - Returns list of image indices in $[0, 14619]$.
  - If `step == 0` or `concept_scores is None`: returns $B/2$ random + $B/2$ DINOv3 FPS.
  - If `step > 0`: returns 2 random, 2 DINOv3 FPS, 2 student collision ($\text{sim}_{\text{ASSUME}} > 0.85, d_{\text{DINO}} > 0.40$).

### M3 ↔ M4, M5 (Gemini Agents)
- `ProposerAgent.propose(images: List[Image.Image], existing_concepts: List[str]) -> List[ConceptRubric]`:
  - Returns atomic `ConceptRubric` objects with `name`, `definition`, `scale_1`, `scale_5`, `scale_10`.
- `ScorerAgent.score_images(images: List[Image.Image], rubrics: List[ConceptRubric]) -> np.ndarray`:
  - Chunks rubrics into groups of $\le 10$.
  - Returns integer matrix in $[1, 10]$ of shape `(len(images), len(rubrics))`.
- `VisualMergerAgent.evaluate_collision(rubric_a: ConceptRubric, rubric_b: ConceptRubric, images: List[Image.Image]) -> MergeDecision`:
  - Returns `action: "MERGE" | "KEEP_BOTH"`, and if `"MERGE"`, includes `merged_rubric`.

### M4 ↔ M5 (Concept Dictionary)
- `concept_dictionary.json`: JSON list of finalized $K=30$ `ConceptRubric` dicts.

---

## Code Layout
```
assume/
├── __init__.py
├── config.py                 # Paths, constants, thresholds
├── data/
│   ├── __init__.py
│   ├── ingestion.py          # Ingestion & relative proportional crop (1024x768 -> 512x288)
│   └── dinov3_embedder.py    # ViT-S/16+ feature extraction & caching
├── sampling/
│   ├── __init__.py
│   └── tri_modal_sampler.py  # 33% Random, 33% FPS, 33% Student Collision Sampler
├── agents/
│   ├── __init__.py
│   ├── vertex_client.py      # Resilient Vertex Express HTTP client with backoff
│   ├── proposer.py           # Proposer Agent with atomic rubric enforcement
│   ├── scorer.py             # Scorer Agent with 1-10 scoring & 10-concept chunking
│   └── merger.py             # Visual Merger Agent for collinear pairs (|r| > 0.80)
├── loop/
│   ├── __init__.py
│   ├── discovery_loop.py     # Main active discovery loop (target K=30)
│   └── pruner.py             # Dead dimension pruner & Pearson correlation matrix
└── evaluation/
    ├── __init__.py
    ├── bottleneck_eval.py    # 1,600 subset scoring -> assume_embeddings_1600x30.npy
    └── umap_visualizer.py    # UMAP projections, silhouette scores, k-NN accuracy, plots
tests/
├── test_ingestion_and_preprocessing.py
├── test_tri_modal_sampler.py
├── test_vertex_client_and_agents.py
├── test_discovery_and_pruner.py
├── test_bottleneck_and_umap.py
└── test_e2e_pipeline.py
```
