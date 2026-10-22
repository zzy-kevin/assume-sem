# E2E Test Infra: ASSUME

## Test Philosophy
- Requirement-driven, opaque-box, derived strictly from `ORIGINAL_REQUEST.md`.
- No reliance on internal module hacks: tests interact via defined CLI/API interfaces and file artifacts.
- Methodology: 4-Tier testing strategy:
  1. Tier 1: Feature Coverage (>=5 per feature) - Fast, hermetic, verifying each feature in isolation.
  2. Tier 2: Boundary & Corner Cases (>=5 per feature) - Testing edge conditions (133 non-standard image sizes, zero-variance concepts, high collinearity |r| > 0.80, rate limit HTTP 429 backoff).
  3. Tier 3: Cross-Feature Combinations (Pairwise) - Feature interactions (e.g. Ingestion -> Sampler -> Agents -> Pruning).
  4. Tier 4: Real-World Application Scenarios - Complete pipeline execution on SEM dataset producing valid bottleneck representations and UMAP figures.

## Feature Inventory Coverage Matrix
| # | Feature | Requirement | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---------|-------------|:------:|:------:|:------:|:------:|
| 1 | Micrograph Ingestion & Filtering | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 2 | Robust Proportional Preprocessing | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 3 | Frozen DINOv3 Teacher Embedding | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 4 | Cache Manifest & Pooled Features | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ | ✓ |
| 5 | Uniform Random Stream (33%) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 6 | Teacher Farthest-Point Stream (33%) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 7 | Student Collision Stream (33%) | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 8 | Step 0 Cold-Start Policy | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ | ✓ |
| 9 | Resilient Vertex AI Express Client | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 10 | Proposer Agent | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 11 | Scorer Agent | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 12 | Visual Merger Agent | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ | ✓ |
| 13 | Active Discovery Orchestration | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 14 | Diversity Evaluation Batches | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 15 | Dead Dimension Pruner | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 16 | Pearson Correlation Pruning Trigger | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ | ✓ |
| 17 | Balanced 1,600 Evaluation Sampler | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |
| 18 | Zero-Shot Bottleneck Scorer | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |
| 19 | Concept Dictionary JSON Exporter | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |
| 20 | 2D UMAP Projections Visualizer | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |
| 21 | Quantitative Separation Metrics | ORIGINAL_REQUEST §R5 | 5 | 5 | ✓ | ✓ |
| 22 | Comprehensive E2E Test Suite | Acceptance Criteria | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Primary Test Runner**: `D:\UOFT\Grad1_local\DINO_SEM\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"`
- **Test Categories**:
  - `tests/test_ingestion_and_preprocessing.py`: Validates 14,620 JPG counts, exclusion of `Powder_highres`, databar proportional crop formula, and output shapes.
  - `tests/test_tri_modal_sampler.py`: Validates 33/33/33 proportions, step 0 initialization, FPS distance calculation, and collision filtering.
  - `tests/test_vertex_client_and_agents.py`: Validates exponential backoff with full jitter, structured JSON schemas for Proposer, Scorer chunking <= 10, and Visual Merger decisions.
  - `tests/test_discovery_and_pruner.py`: Validates Pearson correlation matrix calculation, dead dimension pruning, and merger triggering.
  - `tests/test_bottleneck_and_umap.py`: Validates 1,600 balanced subset selection with seed 42 oversampling, shape $(1600, 30)$, UMAP cosine projections, silhouette score, and k-NN cross-validation.
  - `tests/test_e2e_pipeline.py`: Comprehensive smoke and integration tests exercising the end-to-end pipeline.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Ingestion & Caching | Features 1, 2, 3, 4 | High |
| 2 | Active Discovery Step with Live Vertex API | Features 5, 6, 7, 8, 9, 10, 11 | High |
| 3 | Statistical Pruning & Visual Merger on Colliding Micrographs | Features 12, 14, 15, 16 | High |
| 4 | Balanced 1,600 Micrograph Evaluation & Artifact Generation | Features 17, 18, 19 | High |
| 5 | Publication-Ready UMAP Generation & Metric Logging | Features 20, 21 | Medium |
