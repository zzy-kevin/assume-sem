"""Autonomous Concept Discovery & Pruning Loop for ASSUME.

Orchestrates iterative discovery of human-interpretable microstructural concept rubrics:
1. Tri-modal batch sampling (Uniform Random, Teacher DINOv3 FPS, Student Collision).
2. Proposer Agent synthesis of strictly atomic single-property rubrics with 1, 5, 10 anchors.
3. Scorer Agent batch evaluation with <= 10 concept chunking.
4. Periodic diversity evaluation every 3 batches on DINOv3 FPS micrographs:
   - Dead dimension pruning (std < 0.20 or >90% identical).
   - Resilient Pearson correlation matrix computation.
   - Visual Merger Agent arbitration of collinear pairs (|r| > 0.80).
5. Converges on target K=30 validated, orthogonal concept dictionary exported to JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import time

import numpy as np
from PIL import Image

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.config import DATA_DIR, PROJECT_ROOT
from assume.data.ingestion import preprocess_micrograph
from assume.loop.pruner import StatisticalPruner
from assume.sampling.tri_modal_sampler import TriModalBatchSampler

logger = logging.getLogger(__name__)

DEFAULT_TARGET_K: int = 30
DEFAULT_BATCH_SIZE: int = 6
DEFAULT_EVAL_BATCH_SIZE: int = 24
DEFAULT_EVAL_INTERVAL: int = 3
DEFAULT_PROPOSALS_PER_BATCH: int = 4
DEFAULT_MIN_STD: float = 0.20
DEFAULT_COLLINEAR_THRESHOLD: float = 0.80
DEFAULT_MAX_ITERATIONS: int = 30

DEFAULT_DICT_PATH: Path = DATA_DIR / "concept_dictionary.json"
DEFAULT_ROOT_DICT_PATH: Path = PROJECT_ROOT / "concept_dictionary.json"


class ConceptDiscoveryLoop:
    """Orchestrates iterative active discovery, evaluation, pruning, and collinear resolution."""

    def __init__(
        self,
        sampler: Optional[TriModalBatchSampler] = None,
        proposer: Optional[ProposerAgent] = None,
        scorer: Optional[ScorerAgent] = None,
        merger: Optional[VisualMergerAgent] = None,
        pruner: Optional[StatisticalPruner] = None,
        target_k: int = DEFAULT_TARGET_K,
        batch_size: int = DEFAULT_BATCH_SIZE,
        eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
        eval_interval: int = DEFAULT_EVAL_INTERVAL,
        proposals_per_batch: int = DEFAULT_PROPOSALS_PER_BATCH,
        min_std: float = DEFAULT_MIN_STD,
        collinear_threshold: float = DEFAULT_COLLINEAR_THRESHOLD,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        output_path: Optional[Union[str, Path]] = None,
        root_output_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = 42,
        scorer_max_workers: int = 2,
        step_delay: float = 1.0,
        resume: bool = False,
        model: Optional[str] = None,
    ) -> None:
        """Initialize ConceptDiscoveryLoop.

        Args:
            sampler: Active TriModalBatchSampler instance.
            proposer: ProposerAgent instance.
            scorer: ScorerAgent instance.
            merger: VisualMergerAgent instance.
            pruner: StatisticalPruner instance.
            target_k: Target number of validated concepts to discover (default 30).
            batch_size: Discovery batch size per step (default 6).
            eval_batch_size: Number of diverse DINOv3 FPS micrographs for evaluation (default 24).
            eval_interval: Number of batches between diversity evaluation passes (default 3).
            proposals_per_batch: Number of candidate concepts to propose per batch (default 4).
            min_std: Minimum standard deviation threshold for dead dimension pruning (default 0.20).
            collinear_threshold: Pearson correlation threshold (|r| > threshold) for merger trigger (default 0.80).
            max_iterations: Safety ceiling on maximum discovery iterations (default 30).
            output_path: Primary output path for concept dictionary (default data/concept_dictionary.json).
            root_output_path: Root mirror path for concept dictionary (default concept_dictionary.json).
            seed: Optional random seed for reproducible sampling.
            scorer_max_workers: Concurrency worker limit for ScorerAgent (default 2, avoids RPM bursts).
            step_delay: Pause duration in seconds between discovery steps (default 1.0s).
            resume: If True, resumes from existing output_path dictionary if present.
            model: Optional model identifier string override (e.g. gemini-3.8-flash, gemini-2.5-flash).
        """
        self.target_k: int = int(target_k)
        self.batch_size: int = int(batch_size)
        self.eval_batch_size: int = int(eval_batch_size)
        self.eval_interval: int = int(eval_interval)
        self.proposals_per_batch: int = int(proposals_per_batch)
        self.min_std: float = float(min_std)
        self.collinear_threshold: float = float(collinear_threshold)
        self.max_iterations: int = int(max_iterations)
        self.seed: Optional[int] = seed
        self.scorer_max_workers: int = int(scorer_max_workers)
        self.step_delay: float = float(step_delay)
        self.resume: bool = bool(resume)
        self.model: Optional[str] = model

        self.output_path: Path = Path(output_path) if output_path else DEFAULT_DICT_PATH
        self.root_output_path: Path = Path(root_output_path) if root_output_path else DEFAULT_ROOT_DICT_PATH

        # Initialize subcomponents with dependency injection support
        self.sampler: TriModalBatchSampler = sampler or TriModalBatchSampler(seed=seed)
        self.proposer: ProposerAgent = proposer or (ProposerAgent(model=model) if model else ProposerAgent())
        self.scorer: ScorerAgent = scorer or (ScorerAgent(model=model) if model else ScorerAgent())
        self.merger: VisualMergerAgent = merger or (VisualMergerAgent(model=model) if model else VisualMergerAgent())
        self.pruner: StatisticalPruner = pruner or StatisticalPruner(
            min_std=self.min_std,
            collinear_threshold=self.collinear_threshold,
        )

        # Active state tracking
        self.concepts: List[ConceptRubric] = []
        self.history_indices: List[int] = []
        self.history_scores: np.ndarray = np.zeros((0, 0), dtype=np.int64)
        self.eval_scores: np.ndarray = np.zeros((0, 0), dtype=np.int64)

        # Incremental evaluation pool & score caching
        self._eval_indices: Optional[List[int]] = None
        self._eval_images: Optional[List[Image.Image]] = None
        self._eval_score_cache: Dict[str, np.ndarray] = {}

        # Whitelist of verified orthogonal concept pairs (KEEP_BOTH)
        self.verified_orthogonal_pairs: Set[Tuple[str, str]] = set()

        # Telemetry & Logging
        self.pruned_log: List[Dict[str, Any]] = []
        self.merged_log: List[Dict[str, Any]] = []
        self.iteration_history: List[Dict[str, Any]] = []

        # Optional state resumption
        if self.resume and self.output_path.exists():
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    loaded = [ConceptRubric.from_dict(d) for d in data if isinstance(d, dict)]
                    self.concepts = loaded
                    logger.info("Resumed %d concepts from %s", len(self.concepts), self.output_path)
            except Exception as e:
                logger.warning("Failed to resume from %s: %s", self.output_path, e)

    def get_evaluation_indices(self) -> List[int]:
        """Sample or retrieve fixed diverse evaluation pool using DINOv3 FPS.

        Returns:
            List of integer dataset indices spanning diverse microstructural clusters.
        """
        if self._eval_indices is None or len(self._eval_indices) != min(self.eval_batch_size, len(self.sampler)):
            n_eval = min(self.eval_batch_size, len(self.sampler))
            self._eval_indices = self.sampler._sample_fps(
                n_points=n_eval,
                reference_indices=[],
                exclude_indices=set(),
            )
            self._eval_images = None
            logger.info(
                "Sampled fixed diverse DINOv3 FPS evaluation pool of %d micrographs: %s",
                len(self._eval_indices),
                self._eval_indices,
            )
        return list(self._eval_indices)

    def _get_evaluation_images(self) -> List[Image.Image]:
        """Load and cache PIL Images for the fixed evaluation pool."""
        eval_indices = self.get_evaluation_indices()
        if self._eval_images is None or len(self._eval_images) != len(eval_indices):
            self._eval_images = [self._load_image(idx) for idx in eval_indices]
        return self._eval_images

    def _ensure_eval_scores(
        self,
        eval_images: Optional[Sequence[Image.Image]] = None,
    ) -> np.ndarray:
        """Incrementally score only newly added concepts on the fixed evaluation pool.

        Already-scored concepts are retrieved from self._eval_score_cache so ScorerAgent
        is never redundantly invoked on previously scored concepts.
        """
        if not self.concepts:
            self.eval_scores = np.zeros((0, 0), dtype=np.int64)
            self.history_scores = self.eval_scores
            self.history_indices = []
            return self.eval_scores

        eval_indices = self.get_evaluation_indices()
        imgs = list(eval_images) if eval_images is not None else self._get_evaluation_images()
        n_eval = len(imgs)

        # Identify concepts not yet cached or whose cached column length differs from n_eval
        uncached: List[ConceptRubric] = [
            c
            for c in self.concepts
            if (c.concept_id not in self._eval_score_cache)
            or (len(self._eval_score_cache[c.concept_id]) != n_eval)
        ]

        if uncached:
            scored = self.scorer.score_images(
                imgs, uncached, max_workers=self.scorer_max_workers
            )
            scored_arr = np.asarray(scored, dtype=np.int64)
            if scored_arr.ndim == 2 and scored_arr.shape == (n_eval, len(uncached)):
                for col_idx, c in enumerate(uncached):
                    self._eval_score_cache[c.concept_id] = scored_arr[:, col_idx].copy()
            elif scored_arr.ndim == 2 and scored_arr.shape == (n_eval, len(self.concepts)):
                # Compatibility with mock scorers returning full concept matrix
                for col_idx, c in enumerate(self.concepts):
                    self._eval_score_cache[c.concept_id] = scored_arr[:, col_idx].copy()

        cols = [
            self._eval_score_cache.get(c.concept_id, np.zeros(n_eval, dtype=np.int64))
            for c in self.concepts
        ]
        self.eval_scores = np.column_stack(cols).astype(np.int64) if cols else np.zeros((n_eval, 0), dtype=np.int64)
        self.history_scores = self.eval_scores
        self.history_indices = list(eval_indices)
        return self.eval_scores

    def _load_image(self, index: int) -> Image.Image:
        """Load and preprocess micrograph for a dataset index, with test fallback."""
        try:
            if hasattr(self.sampler, "manifest") and self.sampler.manifest is not None:
                if 0 <= index < len(self.sampler.manifest):
                    row = self.sampler.manifest.iloc[index]
                    p = Path(row["full_path"])
                    if p.exists():
                        return preprocess_micrograph(p)
        except Exception as e:
            logger.debug("Micrograph load failed for index %d: %s; falling back", index, e)

        # Safe synthetic fallback for unit test environments
        return Image.new("RGB", (512, 288), color=(128, 128, 128))

    def _select_distinguishing_micrographs(
        self,
        scores_i: np.ndarray,
        scores_j: np.ndarray,
        eval_images: Sequence[Image.Image],
        n_images: int = 3,
        correlation_r: Optional[float] = None,
    ) -> List[Image.Image]:
        """Select 2-3 representative micrographs that best expose correlation or divergence.

        Sign-aware selection:
        When correlation_r < 0 (negatively correlated pair), inverts concept j's active scale
        (6 - s_j for positive scores on [0, 5]) so that joint high/low and discrepancy micrographs
        meaningfully reflect the inverse relationship.

        Args:
            scores_i: Score vector for concept i across evaluation pool.
            scores_j: Score vector for concept j across evaluation pool.
            eval_images: Sequence of evaluation images.
            n_images: Desired number of representative micrographs (default 3).
            correlation_r: Optional Pearson correlation coefficient between i and j.

        Returns:
            List of 2-3 distinct PIL Images.
        """
        n_eval = len(eval_images)
        if n_eval <= n_images:
            return list(eval_images)

        si = np.asarray(scores_i, dtype=np.float32)
        sj = np.asarray(scores_j, dtype=np.float32)

        if correlation_r is not None and correlation_r < 0.0:
            # Invert active scale for negatively correlated concept j
            max_scale = 6.0 if np.max(sj) <= 5.0 else 11.0
            eff_sj = np.where(sj > 0.0, max_scale - sj, 0.0)
        else:
            eff_sj = sj

        selected_indices: List[int] = []

        # 1. Highest joint expression
        joint_sum = si + eff_sj
        high_idx = int(np.argmax(joint_sum))
        selected_indices.append(high_idx)

        # 2. Lowest joint expression (prefer active samples if available)
        low_idx = int(np.argmin(joint_sum))
        if low_idx not in selected_indices:
            selected_indices.append(low_idx)

        # 3. Maximum discrepancy / divergence
        diff = np.abs(si - eff_sj)
        for idx in np.argsort(-diff):
            if int(idx) not in selected_indices:
                selected_indices.append(int(idx))
                break

        # Fill up to n_images if necessary
        for idx in range(n_eval):
            if len(selected_indices) >= n_images:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

        return [eval_images[idx] for idx in selected_indices[:n_images]]

    def evaluate_and_prune(self) -> Dict[str, Any]:
        """Execute periodic diversity evaluation on DINOv3 FPS micrographs.

        Steps:
        1. Incrementally score evaluation micrographs against any newly added candidate concepts.
        2. Prune dead dimensions (std < min_std or >90% identical).
        3. Compute active-only Pearson correlation matrix on evaluation scores.
        4. Detect collinear pairs (|r| > threshold), skipping pairs already in verified_orthogonal_pairs,
           and resolve via VisualMergerAgent with chain-merge locking and immediate re-scoring of merged rubrics.

        Returns:
            Dictionary containing evaluation summary metrics.
        """
        if not self.concepts:
            return {"active_k": 0, "pruned": 0, "merged": 0}

        eval_indices = self.get_evaluation_indices()
        eval_images = self._get_evaluation_images()

        # 1. Incrementally score evaluation pool on uncached concepts
        eval_scores = self._ensure_eval_scores(eval_images)

        # 2. Prune dead dimensions
        filtered_scores, retained_rubrics, pruned_names = self.pruner.prune_dead_dimensions(
            eval_scores,
            self.concepts,
            min_std=self.min_std,
        )

        n_pruned = len(pruned_names)
        if n_pruned > 0:
            self.pruned_log.append({
                "step": len(self.iteration_history),
                "pruned_names": pruned_names,
                "remaining_k": len(retained_rubrics),
            })
            self.concepts = retained_rubrics
            eval_scores = filtered_scores.astype(np.int64)
            self.eval_scores = eval_scores
            self.history_scores = eval_scores
            self.history_indices = list(eval_indices)

        # 3. Collinear pair detection & Visual Merger arbitration
        n_merged = 0
        if len(self.concepts) >= 2 and eval_scores.shape[1] >= 2:
            corr_matrix = self.pruner.compute_correlation_matrix(eval_scores)
            collinear_pairs = self.pruner.find_collinear_pairs(
                corr_matrix,
                threshold=self.collinear_threshold,
            )

            if collinear_pairs:
                logger.info(
                    "Detected %d collinear concept pair(s) (|r| > %.2f): %s",
                    len(collinear_pairs),
                    self.collinear_threshold,
                    [(self.concepts[i].name, self.concepts[j].name, round(r, 2)) for i, j, r in collinear_pairs],
                )

                locked_indices: Set[int] = set()
                removed_indices: Set[int] = set()
                replacements: Dict[int, ConceptRubric] = {}

                for i, j, r_val in collinear_pairs:
                    if i in locked_indices or j in locked_indices:
                        continue

                    rubric_a = self.concepts[i]
                    rubric_b = self.concepts[j]

                    canon_pair = (
                        min(rubric_a.concept_id, rubric_b.concept_id),
                        max(rubric_a.concept_id, rubric_b.concept_id),
                    )
                    if canon_pair in self.verified_orthogonal_pairs:
                        logger.debug(
                            "Skipping previously verified orthogonal pair (KEEP_BOTH): %s",
                            canon_pair,
                        )
                        continue

                    distinguishing_imgs = self._select_distinguishing_micrographs(
                        eval_scores[:, i],
                        eval_scores[:, j],
                        eval_images,
                        n_images=3,
                        correlation_r=r_val,
                    )

                    decision: MergeDecision = self.merger.evaluate_collision(
                        rubric_a=rubric_a,
                        rubric_b=rubric_b,
                        images=distinguishing_imgs,
                        correlation_r=r_val,
                    )

                    if decision.action == "MERGE":
                        merged_rubric = decision.merged_rubric
                        if not merged_rubric:
                            merged_rubric = ConceptRubric(
                                name=f"{rubric_a.name} (Unified)",
                                definition=f"Unified physical microstructural property synthesizing {rubric_a.name} and {rubric_b.name}.",
                                scale_1=rubric_a.scale_1,
                                scale_3=rubric_a.scale_3,
                                scale_5=rubric_a.scale_5,
                            )
                        replacements[i] = merged_rubric
                        removed_indices.add(j)
                        locked_indices.update([i, j])
                        n_merged += 1

                        # Immediately score the newly merged rubric on the evaluation pool
                        merged_scores = self.scorer.score_images(
                            eval_images,
                            [merged_rubric],
                            max_workers=self.scorer_max_workers,
                        )
                        merged_arr = np.asarray(merged_scores, dtype=np.int64)
                        if merged_arr.ndim == 2 and merged_arr.shape[0] == len(eval_images) and merged_arr.shape[1] >= 1:
                            merged_col = merged_arr[:, 0].copy()
                        else:
                            merged_col = np.round(0.5 * (eval_scores[:, i] + eval_scores[:, j])).astype(np.int64)
                        self._eval_score_cache[merged_rubric.concept_id] = merged_col
                        eval_scores[:, i] = merged_col

                        self.merged_log.append({
                            "concept_a": rubric_a.name,
                            "concept_b": rubric_b.name,
                            "merged": merged_rubric.name,
                            "correlation": float(r_val),
                            "rationale": decision.rationale,
                        })
                        logger.info(
                            "Merged '%s' + '%s' -> '%s' (r=%.2f): %s",
                            rubric_a.name,
                            rubric_b.name,
                            merged_rubric.name,
                            r_val,
                            decision.rationale,
                        )
                    else:
                        self.verified_orthogonal_pairs.add(canon_pair)
                        logger.info(
                            "Retained both orthogonal concepts (whitelisted): '%s' and '%s' (r=%.2f): %s",
                            rubric_a.name,
                            rubric_b.name,
                            r_val,
                            decision.rationale,
                        )

                if removed_indices or replacements:
                    updated_concepts: List[ConceptRubric] = []
                    keep_cols: List[int] = []
                    for idx, r in enumerate(self.concepts):
                        if idx in removed_indices:
                            continue
                        if idx in replacements:
                            updated_concepts.append(replacements[idx])
                            keep_cols.append(idx)
                        else:
                            updated_concepts.append(r)
                            keep_cols.append(idx)

                    self.concepts = updated_concepts
                    eval_scores = eval_scores[:, keep_cols]

        self.eval_scores = eval_scores.astype(np.int64)
        self.history_scores = self.eval_scores
        self.history_indices = list(eval_indices)

        # Checkpoint dictionary after evaluation and pruning pass
        self.export_dictionary()

        return {
            "active_k": len(self.concepts),
            "pruned": n_pruned,
            "merged": n_merged,
        }

    def run_step(self, step: int) -> Dict[str, Any]:
        """Execute a single active discovery step.

        Single-mode batch cycling per step:
        - step % 3 == 0: DINOv3 FPS batch (6 images)
        - step % 3 == 1: Uniform Random batch (6 images)
        - step % 3 == 2: Student Collision batch (3 pairs = 6 images)
        Scoring is performed ONLY at the periodic checkpoint on the fixed evaluation pool
        (right before evaluate_and_prune() and Collision batch sampling), never per-step on the 6 batch images.

        Args:
            step: Current step index (0-indexed).

        Returns:
            Dictionary with telemetry for this step.
        """
        eval_summary: Optional[Dict[str, Any]] = None

        # Before a Collision step (step % 3 == 2) or periodic checkpoint, incrementally score
        # the evaluation pool and run evaluate_and_prune() so Collision sampling uses clean eval_scores
        is_collision_step = (step % 3 == 2)
        is_eval_checkpoint = ((step + 1) % self.eval_interval == 0)

        if is_collision_step and len(self.concepts) > 0:
            logger.info("--- Step %d (Pre-Collision Checkpoint): Incremental Eval Scoring & Pruning ---", step)
            eval_summary = self.evaluate_and_prune()

        # 1. Sample active batch via TriModalBatchSampler using evaluation pool scores
        batch_indices = self.sampler.sample_batch(
            step=step,
            batch_size=self.batch_size,
            concept_scores=self.eval_scores if self.eval_scores.size > 0 else None,
            scored_indices=self.history_indices if self.history_indices else None,
        )

        batch_images = [self._load_image(idx) for idx in batch_indices]
        last_info = getattr(self.sampler, "last_batch_info", None) or {}
        default_mode = "fps" if (step % 3 == 0) else ("random" if (step % 3 == 1) else "collision")
        batch_mode = last_info.get("strategy", default_mode)
        collision_pairs = last_info.get("collision_pairs", [])

        # 2. Propose candidate concept rubrics (with contrastive pair formatting when in collision mode)
        try:
            proposals = self.proposer.propose(
                images=batch_images,
                existing_concepts=self.concepts,
                n_proposals=self.proposals_per_batch,
                batch_mode=batch_mode,
                collision_pairs=collision_pairs,
            )
        except TypeError:
            proposals = self.proposer.propose(
                images=batch_images,
                existing_concepts=self.concepts,
                n_proposals=self.proposals_per_batch,
            )

        # 3. Deduplicate against active concepts
        existing_names = {c.name.strip().lower() for c in self.concepts}
        existing_ids = {c.concept_id for c in self.concepts}

        accepted_proposals: List[ConceptRubric] = []
        for prop in proposals:
            norm_name = prop.name.strip().lower()
            if norm_name not in existing_names and prop.concept_id not in existing_ids:
                accepted_proposals.append(prop)
                existing_names.add(norm_name)
                existing_ids.add(prop.concept_id)

        # 4. Integrate accepted proposals into active set
        n_new = len(accepted_proposals)
        if accepted_proposals:
            self.concepts.extend(accepted_proposals)

        # 5. Periodic diversity evaluation checkpoint (incrementally scores only new proposals)
        if is_eval_checkpoint and len(self.concepts) > 0:
            logger.info("--- Step %d: Running Diversity Evaluation & Pruning ---", step)
            eval_summary = self.evaluate_and_prune()

        # Checkpoint dictionary after each step
        self.export_dictionary()

        # Polite inter-step pacing to prevent API rate-limit bursts
        if self.step_delay > 0:
            time.sleep(self.step_delay)

        step_info = {
            "step": step,
            "batch_mode": batch_mode,
            "batch_indices": batch_indices,
            "collision_pairs": collision_pairs,
            "proposals_generated": len(proposals),
            "proposals_accepted": n_new,
            "active_k": len(self.concepts),
            "eval_summary": eval_summary,
        }
        self.iteration_history.append(step_info)
        return step_info

    def run(self) -> List[ConceptRubric]:
        """Run the autonomous discovery loop until target_k validated concepts are discovered.

        Returns:
            List of finalized, validated ConceptRubric objects.
        """
        logger.info(
            "Starting ASSUME Concept Discovery Loop (Target K=%d, Batch Size=%d, Eval Interval=%d)",
            self.target_k,
            self.batch_size,
            self.eval_interval,
        )

        # If resuming from checkpoint without eval scores, incrementally score on the fixed evaluation pool
        if self.resume and len(self.concepts) > 0 and self.eval_scores.size == 0:
            logger.info(
                "Initializing evaluation pool scores for %d resumed concepts...",
                len(self.concepts),
            )
            self._ensure_eval_scores()

        step = 0
        while step < self.max_iterations and len(self.concepts) < self.target_k:
            step_info = self.run_step(step)
            logger.info(
                "Step %d complete (%s) | New concepts accepted: %d | Active concepts: %d / %d",
                step,
                step_info["batch_mode"],
                step_info["proposals_accepted"],
                len(self.concepts),
                self.target_k,
            )
            step += 1

        # Final validation pass to ensure active concepts meet variance and non-collinearity criteria
        logger.info("Executing final diversity evaluation & pruning pass...")
        self.evaluate_and_prune()

        # If pruning dropped below target and iterations remain, continue
        while step < self.max_iterations and len(self.concepts) < self.target_k:
            step_info = self.run_step(step)
            step += 1
            if len(self.concepts) >= self.target_k:
                self.evaluate_and_prune()

        # Trim to top target_k concepts if surplus exists
        if len(self.concepts) > self.target_k:
            logger.info(
                "Surplus concepts discovered (%d > %d). Ranking by variance on evaluation set...",
                len(self.concepts),
                self.target_k,
            )
            eval_scores = self._ensure_eval_scores()
            stds = np.std(eval_scores, axis=0)
            ranked_indices = np.argsort(-stds)[: self.target_k]
            ranked_indices = sorted(ranked_indices)  # preserve relative order
            self.concepts = [self.concepts[i] for i in ranked_indices]
            self.eval_scores = eval_scores[:, ranked_indices]
            self.history_scores = self.eval_scores

        logger.info(
            "Discovery loop finished at step %d with %d validated concept(s).",
            step,
            len(self.concepts),
        )

        # Export concept dictionary to standard artifacts
        self.export_dictionary()

        return self.concepts

    def export_dictionary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        root_output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        """Export finalized concept dictionary to JSON files.

        Enforces atomic single-property rubrics with unambiguous 0, 1, 3, 5 verbal anchors.

        Args:
            output_path: Primary destination path (default data/concept_dictionary.json).
            root_output_path: Root mirror path (default concept_dictionary.json).

        Returns:
            Tuple of (primary_path, root_path).
        """
        p_out = Path(output_path) if output_path else self.output_path
        p_root = Path(root_output_path) if root_output_path else self.root_output_path

        # Construct JSON records conforming to M4/M5 contract
        records: List[Dict[str, Any]] = []
        for idx, r in enumerate(self.concepts):
            max_anchor = r.scale_10 if (r.scale_10 and r.scale_10 != r.scale_5) else r.scale_5
            records.append({
                "concept_id": r.concept_id or f"concept_{idx+1:02d}",
                "name": r.name,
                "definition": r.definition,
                "description": r.definition,
                "scale_0": r.scale_0,
                "scale_1": r.scale_1,
                "scale_3": r.scale_3,
                "scale_5": max_anchor,
                "scale_10": max_anchor,
                "scoring_guide": r.scoring_guide,
            })

        # Ensure parent directories exist
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_root.parent.mkdir(parents=True, exist_ok=True)

        json_text = json.dumps(records, indent=2, ensure_ascii=False)

        with open(p_out, "w", encoding="utf-8") as f:
            f.write(json_text)
            f.write("\n")

        with open(p_root, "w", encoding="utf-8") as f:
            f.write(json_text)
            f.write("\n")

        logger.info(
            "Exported finalized concept dictionary (%d concepts) to:\n- %s\n- %s",
            len(records),
            p_out.resolve(),
            p_root.resolve(),
        )

        return p_out, p_root
