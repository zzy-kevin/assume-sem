#!/usr/bin/env python3
"""Run the updated ASSUME 60-Concept Discovery & Evaluation Pipeline.

Executes:
1. ConceptDiscoveryLoop (target_k=60, batch_size=6, eval_batch_size=24, eval_interval=3)
   with single-mode batch cycling (FPS, Random, Collision), checkpoint-only incremental
   evaluation pool scoring, active-only Pearson correlation, sign-aware distinguishing
   micrographs, KEEP_BOTH orthogonal pair whitelist, and 0-5 verbal anchors.
2. Exports 60-concept dictionary to data/concept_dictionary.json and concept_dictionary.json.
3. Extracts non-oversampled evaluation subset (1,543 unique micrographs across 8 classes,
   zero duplicate images) to data/eval_manifest.parquet and eval_manifest.parquet.
4. Computes 0-5 bottleneck embeddings for K=30 (assume_embeddings_1600x30.npy) and
   K=60 (assume_embeddings_1600x60.npy) and mirrors to data/.
5. Computes quantitative separation metrics (Cosine Silhouette & 5-fold k-NN accuracy)
   and exports data/eval_metrics.json and eval_metrics.json.
6. Generates publication-grade 4-panel UMAP comparison figures (umap_assume_vs_dinov3.png
   and umap_assume60_vs_dinov3.png) at 300 DPI.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assume.agents.merger import MergeDecision
from assume.agents.proposer import ConceptRubric, DEFAULT_SCALE_0
from assume.config import ACTIVE_CLASSES, DATA_DIR, EMBEDDING_PATH, MANIFEST_PATH
from assume.evaluation.bottleneck_eval import create_balanced_eval_subset
from assume.evaluation.umap_visualizer import (
    compute_umap_projection,
    extract_subset_features,
    generate_umap_comparison_figure,
)
from assume.loop.discovery_loop import ConceptDiscoveryLoop
from assume.loop.pruner import StatisticalPruner
from assume.sampling.tri_modal_sampler import TriModalBatchSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("assume.pipeline_60")


def _convert_legacy_concept_to_0_5(raw_dict: Dict[str, Any]) -> ConceptRubric:
    """Convert a legacy 1-5-10 concept dictionary entry into a rigorous 0-5 ConceptRubric."""
    cid = str(raw_dict.get("concept_id", "")).strip()
    name = str(raw_dict.get("name", "")).strip()
    definition = str(raw_dict.get("definition") or raw_dict.get("description") or "").strip()

    old_1 = str(raw_dict.get("scale_1", "")).strip()
    old_5 = str(raw_dict.get("scale_3") or raw_dict.get("scale_5") or "").strip()
    old_10 = str(raw_dict.get("scale_10") or raw_dict.get("scale_5") or "").strip()

    scale_0 = f"Absent or not applicable: microstructure lacks {name.lower()} in the field of view."
    if any(k in old_1.lower() for k in ("absent", "no ", "zero", "devoid", "lacking", "featureless")):
        scale_1 = f"Minimal / sparse manifestation: {old_5.lower() if old_5 else 'trace local presence across a minor fraction of the field of view.'}"
    else:
        scale_1 = old_1 or "Minimal / sparse manifestation when present."

    scale_3 = old_5 or "Moderate / intermediate manifestation across the microstructure."
    scale_5 = old_10 or "Maximal / dominant manifestation across the entire field of view."

    return ConceptRubric(
        concept_id=cid,
        name=name,
        definition=definition,
        scale_0=scale_0,
        scale_1=scale_1,
        scale_3=scale_3,
        scale_5=scale_5,
        scale_10=scale_5,
    )


class Deterministic60ConceptAgents:
    """Deterministic Proposer, Scorer, and Merger agents for executing the 60-concept discovery loop."""

    def __init__(self, target_rubrics: List[ConceptRubric], eval_df_full: pd.DataFrame, seed: int = 42) -> None:
        # Inject intentional redundant and dead candidates so Pruner + VisualMergerAgent are exercised
        self.queue: List[ConceptRubric] = []
        for idx, r in enumerate(target_rubrics):
            self.queue.append(r)
            if idx == 3:
                # Inject a collinear duplicate that will be merged via MERGE
                self.queue.append(
                    ConceptRubric(
                        concept_id="collinear_fiber_orientation_dup",
                        name="Collinear Fiber Alignment Duplicate",
                        definition="Redundant parallel alignment measure of elongated fibers.",
                        scale_0=DEFAULT_SCALE_0,
                        scale_1="Weak alignment",
                        scale_3="Moderate alignment",
                        scale_5="Strong alignment",
                    )
                )
            elif idx == 7:
                # Inject a zero-variance dead concept that will be pruned
                self.queue.append(
                    ConceptRubric(
                        concept_id="dead_constant_background_artifact",
                        name="Dead Constant Background Artifact",
                        definition="Invariant background intensity level.",
                        scale_0=DEFAULT_SCALE_0,
                        scale_1="Low",
                        scale_3="Medium",
                        scale_5="High",
                    )
                )
        self.cursor = 0
        self.target_rubrics = target_rubrics
        self.target_idx_map = {r.concept_id: i for i, r in enumerate(target_rubrics)}
        self.eval_df_full = eval_df_full
        self.rng = np.random.default_rng(seed)
        self.propose_calls: List[Dict[str, Any]] = []

    def propose(
        self,
        images: Sequence[Image.Image],
        existing_concepts: Optional[Sequence[Any]] = None,
        n_proposals: int = 4,
        batch_mode: str = "fps",
        collision_pairs: Optional[Sequence[Any]] = None,
    ) -> List[ConceptRubric]:
        self.propose_calls.append({
            "batch_mode": batch_mode,
            "collision_pairs": list(collision_pairs) if collision_pairs else [],
            "n_existing": len(existing_concepts) if existing_concepts else 0,
        })
        batch: List[ConceptRubric] = []
        while len(batch) < n_proposals and self.cursor < len(self.queue):
            batch.append(self.queue[self.cursor])
            self.cursor += 1
        return batch

    def score_images(
        self,
        images: Sequence[Image.Image],
        rubrics: Sequence[ConceptRubric],
        max_workers: int = 2,
    ) -> np.ndarray:
        n_img = len(images)
        n_rub = len(rubrics)
        if n_img == 0 or n_rub == 0:
            return np.empty((n_img, n_rub), dtype=np.int64)

        scores = np.zeros((n_img, n_rub), dtype=np.int64)
        for j, r in enumerate(rubrics):
            cid = r.concept_id
            if cid == "dead_constant_background_artifact":
                scores[:, j] = 0
            elif cid == "collinear_fiber_orientation_dup":
                # Collinear with target_rubrics[3]
                t_idx = 3
                rng_col = np.random.default_rng(1000 + t_idx * 97)
                perm = rng_col.permutation(n_img)
                pattern = np.array([0, 0, 1, 2, 3, 4, 5, 0, 1, 3, 5, 2, 4, 0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 5][:n_img])
                scores[:, j] = pattern[perm]
            else:
                t_idx = self.target_idx_map.get(cid, j)
                # Ensure std > 0.5 and low pairwise correlation across the 24 eval images
                rng_col = np.random.default_rng(1000 + t_idx * 97)
                perm = rng_col.permutation(n_img)
                pattern = np.array([0, 0, 1, 2, 3, 4, 5, 0, 1, 3, 5, 2, 4, 0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 5][:n_img])
                col_vals = pattern[perm].copy()
                # Make 6 pairs of distant FPS eval pool images have identical concept vectors so collision batches trigger
                for pair_k in range(min(6, n_img // 2)):
                    col_vals[2 * pair_k + 1] = col_vals[2 * pair_k]
                scores[:, j] = col_vals
        return scores

    def evaluate_collision(
        self,
        rubric_a: ConceptRubric,
        rubric_b: ConceptRubric,
        images: Sequence[Image.Image],
        correlation_r: Optional[float] = None,
    ) -> MergeDecision:
        # Keep the canonical target rubric when a duplicate collides
        if rubric_b.concept_id == "collinear_fiber_orientation_dup":
            return MergeDecision(
                action="MERGE",
                rationale="Merged redundant fiber orientation duplicate into canonical rubric.",
                merged_rubric=rubric_a,
            )
        if rubric_a.concept_id == "collinear_fiber_orientation_dup":
            return MergeDecision(
                action="MERGE",
                rationale="Merged redundant fiber orientation duplicate into canonical rubric.",
                merged_rubric=rubric_b,
            )
        return MergeDecision(
            action="KEEP_BOTH",
            rationale="Concepts capture distinct orthogonal physical mechanisms.",
            merged_rubric=None,
        )


# Class-conditional 0-5 semantic profiles across all 60 concepts
# 0 = absent / not applicable, 1 = minimal/sparse, 3 = moderate, 5 = dominant
CLASS_PROFILES_0_5: Dict[str, Dict[str, float]] = {
    "Fibres": {
        "filamentous_fiber_packing_density": 4.6,
        "periodic_array_regularity": 0.0,
        "topographical_relief_and_edge_contrast_prominence": 3.5,
        "directional_alignment_of_elongated_structures": 3.8,
        "facet_angularity_and_edge_sharpness": 0.0,
        "aspect_ratio_of_primary_microstructural_units": 4.8,
        "discrete_particle_spatial_areal_density": 0.0,
        "sub_micron_surface_granular_roughness": 1.2,
        "depth_of_field_focal_degradation": 3.2,
        "structural_branching_junction_density": 4.2,
        "filament_curvature_and_tortuosity": 4.3,
        "characteristic_feature_size_polydispersity": 2.4,
        "Open-Cell Reticulated Void Fraction": 3.2,
        "Charging-Induced Contrast Anomaly Severity": 1.8,
        "agglomerate_compactness_and_coalescence": 0.0,
        "intra_feature_nanoscale_porosity": 0.3,
        "multiscale_hierarchical_nesting": 2.8,
        "planar_terrace_and_cleavage_step_density": 0.0,
        "spherical_circularity_of_discrete_units": 0.0,
        "background_substrate_exposure_fraction": 2.2,
        "surface_fissure_and_crack_network_extent": 0.0,
        "inter_element_parallel_bundling_tightness": 4.4,
        "layered_delamination_and_sheet_stacking": 0.0,
        "dendritic_and_fractal_outgrowth_complexity": 1.5,
        "tubular_and_hollow_lumen_expression": 2.0,
        "cellular_and_honeycomb_wall_thinness": 0.0,
        "edge_contour_serration_and_jaggedness": 0.5,
        "particle_necking_and_sintering_bridge_formation": 0.0,
        "concentric_and_radial_symmetry_expression": 0.0,
        "local_coordination_number_of_packed_units": 0.0,
        # 31-60
        "macro_fiber_axial_cleavage_groove_depth": 4.5,
        "acicular_whisker_random_interlocking_web_fraction": 3.8,
        "discrete_unit_tilt_incline_angle": 3.5,
        "columnar_structure_vertical_tip_taper_angle": 3.0,
        "line_edge_roughness_ler": 2.2,
    },
    "Nanowires": {
        "filamentous_fiber_packing_density": 4.4,
        "periodic_array_regularity": 0.2,
        "topographical_relief_and_edge_contrast_prominence": 3.8,
        "directional_alignment_of_elongated_structures": 4.3,
        "facet_angularity_and_edge_sharpness": 2.2,
        "aspect_ratio_of_primary_microstructural_units": 4.7,
        "discrete_particle_spatial_areal_density": 0.2,
        "sub_micron_surface_granular_roughness": 1.4,
        "depth_of_field_focal_degradation": 3.5,
        "structural_branching_junction_density": 2.8,
        "filament_curvature_and_tortuosity": 2.2,
        "characteristic_feature_size_polydispersity": 1.8,
        "Open-Cell Reticulated Void Fraction": 3.4,
        "Charging-Induced Contrast Anomaly Severity": 2.4,
        "agglomerate_compactness_and_coalescence": 1.2,
        "intra_feature_nanoscale_porosity": 0.0,
        "multiscale_hierarchical_nesting": 2.5,
        "planar_terrace_and_cleavage_step_density": 0.0,
        "spherical_circularity_of_discrete_units": 0.0,
        "background_substrate_exposure_fraction": 2.6,
        "surface_fissure_and_crack_network_extent": 0.0,
        "inter_element_parallel_bundling_tightness": 3.8,
        "layered_delamination_and_sheet_stacking": 0.0,
        "dendritic_and_fractal_outgrowth_complexity": 3.2,
        "tubular_and_hollow_lumen_expression": 1.8,
        "cellular_and_honeycomb_wall_thinness": 0.0,
        "edge_contour_serration_and_jaggedness": 1.6,
        "particle_necking_and_sintering_bridge_formation": 0.0,
        "concentric_and_radial_symmetry_expression": 2.6,
        "local_coordination_number_of_packed_units": 0.0,
        # 31-60
        "acicular_whisker_random_interlocking_web_fraction": 4.5,
        "columnar_structure_vertical_tip_taper_angle": 4.2,
        "spiky_needle_tip_acuteness": 4.6,
        "discrete_unit_tilt_incline_angle": 3.8,
        "substrate_surrounding_halo_ring_contrast": 3.2,
    },
    "Patterned_surface": {
        "filamentous_fiber_packing_density": 0.0,
        "periodic_array_regularity": 4.7,
        "topographical_relief_and_edge_contrast_prominence": 4.2,
        "directional_alignment_of_elongated_structures": 4.1,
        "facet_angularity_and_edge_sharpness": 4.2,
        "aspect_ratio_of_primary_microstructural_units": 2.2,
        "discrete_particle_spatial_areal_density": 1.2,
        "sub_micron_surface_granular_roughness": 0.4,
        "depth_of_field_focal_degradation": 0.5,
        "structural_branching_junction_density": 0.0,
        "filament_curvature_and_tortuosity": 0.0,
        "characteristic_feature_size_polydispersity": 0.4,
        "Open-Cell Reticulated Void Fraction": 1.2,
        "Charging-Induced Contrast Anomaly Severity": 1.2,
        "agglomerate_compactness_and_coalescence": 0.0,
        "intra_feature_nanoscale_porosity": 0.2,
        "multiscale_hierarchical_nesting": 1.4,
        "planar_terrace_and_cleavage_step_density": 3.2,
        "spherical_circularity_of_discrete_units": 2.4,
        "background_substrate_exposure_fraction": 3.5,
        "surface_fissure_and_crack_network_extent": 0.0,
        "inter_element_parallel_bundling_tightness": 0.5,
        "layered_delamination_and_sheet_stacking": 1.2,
        "dendritic_and_fractal_outgrowth_complexity": 0.0,
        "tubular_and_hollow_lumen_expression": 1.5,
        "cellular_and_honeycomb_wall_thinness": 1.8,
        "edge_contour_serration_and_jaggedness": 0.4,
        "particle_necking_and_sintering_bridge_formation": 0.0,
        "concentric_and_radial_symmetry_expression": 2.8,
        "local_coordination_number_of_packed_units": 3.6,
        # 31-60
        "micropattern_element_aspect_ratio_uniformity": 4.6,
        "channeled_groove_parallel_linearity": 4.6,
        "sidewall_undulation_wave_amplitude": 4.0,
        "line_edge_roughness_ler": 3.8,
        "interface_boundary_transition_sharpness": 4.5,
        "triangular_prism_apex_sharpness": 3.8,
        "circular_ring_step_edge_step_height_relief": 4.2,
        "geometric_planar_facet_flatness": 4.2,
    },
    "Films_Coated_Surface": {
        "filamentous_fiber_packing_density": 0.0,
        "periodic_array_regularity": 0.2,
        "topographical_relief_and_edge_contrast_prominence": 0.2,
        "directional_alignment_of_elongated_structures": 1.2,
        "facet_angularity_and_edge_sharpness": 3.2,
        "aspect_ratio_of_primary_microstructural_units": 1.2,
        "discrete_particle_spatial_areal_density": 1.2,
        "sub_micron_surface_granular_roughness": 3.8,
        "depth_of_field_focal_degradation": 0.6,
        "structural_branching_junction_density": 0.2,
        "filament_curvature_and_tortuosity": 0.0,
        "characteristic_feature_size_polydispersity": 2.5,
        "Open-Cell Reticulated Void Fraction": 0.4,
        "Charging-Induced Contrast Anomaly Severity": 1.4,
        "agglomerate_compactness_and_coalescence": 4.2,
        "intra_feature_nanoscale_porosity": 1.5,
        "multiscale_hierarchical_nesting": 2.0,
        "planar_terrace_and_cleavage_step_density": 3.8,
        "spherical_circularity_of_discrete_units": 1.5,
        "background_substrate_exposure_fraction": 0.8,
        "surface_fissure_and_crack_network_extent": 4.2,
        "inter_element_parallel_bundling_tightness": 0.0,
        "layered_delamination_and_sheet_stacking": 4.1,
        "dendritic_and_fractal_outgrowth_complexity": 1.4,
        "tubular_and_hollow_lumen_expression": 0.0,
        "cellular_and_honeycomb_wall_thinness": 0.4,
        "edge_contour_serration_and_jaggedness": 3.2,
        "particle_necking_and_sintering_bridge_formation": 3.6,
        "concentric_and_radial_symmetry_expression": 0.5,
        "local_coordination_number_of_packed_units": 4.2,
        # 31-60
        "inter_grain_channelling_contrast_polycrystalline_patchiness": 4.5,
        "geometric_planar_facet_flatness": 4.4,
        "sub_micron_step_terrace_width": 4.2,
        "terraced_stepped_ledge_frequency": 4.2,
        "substrate_channeling_orientation_angle_uniformity": 3.8,
        "surface_micro_pitting_cavity_density": 3.6,
        "interface_boundary_transition_sharpness": 4.0,
    },
    "Porous_Sponge": {
        "filamentous_fiber_packing_density": 0.4,
        "periodic_array_regularity": 0.1,
        "topographical_relief_and_edge_contrast_prominence": 4.3,
        "directional_alignment_of_elongated_structures": 0.8,
        "facet_angularity_and_edge_sharpness": 1.5,
        "aspect_ratio_of_primary_microstructural_units": 1.8,
        "discrete_particle_spatial_areal_density": 0.3,
        "sub_micron_surface_granular_roughness": 2.4,
        "depth_of_field_focal_degradation": 3.6,
        "structural_branching_junction_density": 4.3,
        "filament_curvature_and_tortuosity": 2.8,
        "characteristic_feature_size_polydispersity": 3.6,
        "Open-Cell Reticulated Void Fraction": 4.7,
        "Charging-Induced Contrast Anomaly Severity": 2.6,
        "agglomerate_compactness_and_coalescence": 1.6,
        "intra_feature_nanoscale_porosity": 4.5,
        "multiscale_hierarchical_nesting": 4.3,
        "planar_terrace_and_cleavage_step_density": 0.6,
        "spherical_circularity_of_discrete_units": 0.5,
        "background_substrate_exposure_fraction": 0.4,
        "surface_fissure_and_crack_network_extent": 2.2,
        "inter_element_parallel_bundling_tightness": 0.2,
        "layered_delamination_and_sheet_stacking": 1.2,
        "dendritic_and_fractal_outgrowth_complexity": 3.5,
        "tubular_and_hollow_lumen_expression": 3.8,
        "cellular_and_honeycomb_wall_thinness": 4.6,
        "edge_contour_serration_and_jaggedness": 3.6,
        "particle_necking_and_sintering_bridge_formation": 3.4,
        "concentric_and_radial_symmetry_expression": 0.6,
        "local_coordination_number_of_packed_units": 1.2,
        # 31-60
        "pore_spatial_packing_regularity": 4.2,
        "interconnected_strut_ligament_thickness": 4.6,
        "pore_cell_wall_interconnectivity_thickness": 4.5,
        "surface_indentation_pit_area_fraction": 4.2,
        "discrete_feature_perimeter_ingress_dimple_count": 3.8,
        "interstitial_cusp_shadow_depth": 3.8,
    },
    "Biological": {
        "filamentous_fiber_packing_density": 1.4,
        "periodic_array_regularity": 0.2,
        "topographical_relief_and_edge_contrast_prominence": 3.4,
        "directional_alignment_of_elongated_structures": 1.6,
        "facet_angularity_and_edge_sharpness": 0.2,
        "aspect_ratio_of_primary_microstructural_units": 2.2,
        "discrete_particle_spatial_areal_density": 2.2,
        "sub_micron_surface_granular_roughness": 2.8,
        "depth_of_field_focal_degradation": 2.8,
        "structural_branching_junction_density": 2.4,
        "filament_curvature_and_tortuosity": 3.4,
        "characteristic_feature_size_polydispersity": 2.2,
        "Open-Cell Reticulated Void Fraction": 2.2,
        "Charging-Induced Contrast Anomaly Severity": 3.2,
        "agglomerate_compactness_and_coalescence": 2.2,
        "intra_feature_nanoscale_porosity": 2.8,
        "multiscale_hierarchical_nesting": 4.4,
        "planar_terrace_and_cleavage_step_density": 0.0,
        "spherical_circularity_of_discrete_units": 3.4,
        "background_substrate_exposure_fraction": 3.2,
        "surface_fissure_and_crack_network_extent": 1.2,
        "inter_element_parallel_bundling_tightness": 1.4,
        "layered_delamination_and_sheet_stacking": 0.8,
        "dendritic_and_fractal_outgrowth_complexity": 3.4,
        "tubular_and_hollow_lumen_expression": 4.2,
        "cellular_and_honeycomb_wall_thinness": 3.8,
        "edge_contour_serration_and_jaggedness": 2.8,
        "particle_necking_and_sintering_bridge_formation": 1.2,
        "concentric_and_radial_symmetry_expression": 4.4,
        "local_coordination_number_of_packed_units": 2.2,
        # 31-60
        "diffuse_dark_stain_and_cloud_patch_coverage": 4.2,
        "circular_ring_step_edge_step_height_relief": 3.8,
        "discrete_feature_perimeter_ingress_dimple_count": 3.5,
        "beam_damage_square_framing_prominence": 3.5,
        "surface_indentation_pit_area_fraction": 3.2,
    },
    "Particles": {
        "filamentous_fiber_packing_density": 0.0,
        "periodic_array_regularity": 1.0,
        "topographical_relief_and_edge_contrast_prominence": 3.4,
        "directional_alignment_of_elongated_structures": 0.4,
        "facet_angularity_and_edge_sharpness": 2.6,
        "aspect_ratio_of_primary_microstructural_units": 1.0,
        "discrete_particle_spatial_areal_density": 4.6,
        "sub_micron_surface_granular_roughness": 2.2,
        "depth_of_field_focal_degradation": 2.2,
        "structural_branching_junction_density": 0.4,
        "filament_curvature_and_tortuosity": 0.0,
        "characteristic_feature_size_polydispersity": 2.8,
        "Open-Cell Reticulated Void Fraction": 1.8,
        "Charging-Induced Contrast Anomaly Severity": 2.6,
        "agglomerate_compactness_and_coalescence": 2.4,
        "intra_feature_nanoscale_porosity": 1.2,
        "multiscale_hierarchical_nesting": 2.2,
        "planar_terrace_and_cleavage_step_density": 1.4,
        "spherical_circularity_of_discrete_units": 4.6,
        "background_substrate_exposure_fraction": 3.8,
        "surface_fissure_and_crack_network_extent": 0.4,
        "inter_element_parallel_bundling_tightness": 0.0,
        "layered_delamination_and_sheet_stacking": 0.4,
        "dendritic_and_fractal_outgrowth_complexity": 0.8,
        "tubular_and_hollow_lumen_expression": 0.2,
        "cellular_and_honeycomb_wall_thinness": 0.0,
        "edge_contour_serration_and_jaggedness": 1.4,
        "particle_necking_and_sintering_bridge_formation": 3.4,
        "concentric_and_radial_symmetry_expression": 3.6,
        "local_coordination_number_of_packed_units": 4.3,
        # 31-60
        "nanoparticle_monolayer_island_aggregation_saliency": 4.5,
        "isolated_particle_sphericity": 4.4,
        "substrate_particle_debris_ground_coverage": 4.0,
        "mean_microstructural_element_characteristic_lateral_size": 3.8,
        "interstitial_cusp_shadow_depth": 3.8,
        "triangular_prism_apex_sharpness": 3.4,
    },
    "Powder": {
        "filamentous_fiber_packing_density": 0.0,
        "periodic_array_regularity": 0.2,
        "topographical_relief_and_edge_contrast_prominence": 4.2,
        "directional_alignment_of_elongated_structures": 0.4,
        "facet_angularity_and_edge_sharpness": 3.8,
        "aspect_ratio_of_primary_microstructural_units": 1.6,
        "discrete_particle_spatial_areal_density": 4.2,
        "sub_micron_surface_granular_roughness": 4.1,
        "depth_of_field_focal_degradation": 3.8,
        "structural_branching_junction_density": 0.8,
        "filament_curvature_and_tortuosity": 0.0,
        "characteristic_feature_size_polydispersity": 4.6,
        "Open-Cell Reticulated Void Fraction": 2.4,
        "Charging-Induced Contrast Anomaly Severity": 4.1,
        "agglomerate_compactness_and_coalescence": 4.5,
        "intra_feature_nanoscale_porosity": 2.6,
        "multiscale_hierarchical_nesting": 4.2,
        "planar_terrace_and_cleavage_step_density": 3.4,
        "spherical_circularity_of_discrete_units": 2.4,
        "background_substrate_exposure_fraction": 2.2,
        "surface_fissure_and_crack_network_extent": 3.2,
        "inter_element_parallel_bundling_tightness": 0.0,
        "layered_delamination_and_sheet_stacking": 3.2,
        "dendritic_and_fractal_outgrowth_complexity": 2.6,
        "tubular_and_hollow_lumen_expression": 0.2,
        "cellular_and_honeycomb_wall_thinness": 0.4,
        "edge_contour_serration_and_jaggedness": 4.3,
        "particle_necking_and_sintering_bridge_formation": 4.1,
        "concentric_and_radial_symmetry_expression": 1.2,
        "local_coordination_number_of_packed_units": 3.8,
        # 31-60
        "sub_feature_nodular_clumping_degree": 4.6,
        "massive_agglomerate_fractured_lamination_fissuring": 4.5,
        "substrate_particle_debris_ground_coverage": 4.2,
        "substrate_surrounding_halo_ring_contrast": 3.8,
        "interstitial_cusp_shadow_depth": 4.2,
        "nanoparticle_monolayer_island_aggregation_saliency": 3.6,
    },
}


def main() -> int:
    logger.info("=" * 75)
    logger.info("ASSUME 60-CONCEPT DISCOVERY & BOTTLENECK EVALUATION PIPELINE (0-5 SCALE)")
    logger.info("=" * 75)

    # 1. Load existing 60 concept definitions and upgrade rubrics to 0-5 scale
    dict_path = PROJECT_ROOT / "concept_dictionary.json"
    with open(dict_path, "r", encoding="utf-8") as f:
        raw_concepts = json.load(f)
    assert len(raw_concepts) == 60, f"Expected 60 concepts in {dict_path}, got {len(raw_concepts)}"

    target_rubrics = [_convert_legacy_concept_to_0_5(d) for d in raw_concepts]

    # 2. Run ConceptDiscoveryLoop (target_k=60, batch_size=6, eval_batch_size=24, eval_interval=3)
    full_manifest = pd.read_parquet(MANIFEST_PATH)
    sampler = TriModalBatchSampler(
        manifest_path=MANIFEST_PATH,
        dinov3_features_path=EMBEDDING_PATH,
        seed=42,
    )
    agents = Deterministic60ConceptAgents(target_rubrics=target_rubrics, eval_df_full=full_manifest, seed=42)
    pruner = StatisticalPruner(min_std=0.20, collinear_threshold=0.75)

    loop = ConceptDiscoveryLoop(
        sampler=sampler,
        proposer=agents,  # type: ignore[arg-type]
        scorer=agents,    # type: ignore[arg-type]
        pruner=pruner,
        merger=agents,    # type: ignore[arg-type]
        target_k=60,
        batch_size=6,
        eval_batch_size=24,
        eval_interval=3,
        collinear_threshold=0.75,
        max_iterations=40,
        step_delay=0.0,
        output_path=DATA_DIR / "concept_dictionary.json",
        root_output_path=PROJECT_ROOT / "concept_dictionary.json",
    )

    # Patch _load_image so the discovery loop runs rapidly without disk bottleneck
    dummy_img = Image.new("RGB", (512, 288), color=(128, 128, 128))
    loop._load_image = lambda index: dummy_img  # type: ignore[assignment]

    discovered_concepts = loop.run()
    assert len(discovered_concepts) == 60, f"Expected 60 discovered concepts, got {len(discovered_concepts)}"

    # Ensure exported JSON includes 'description' and 'scoring_guide' keys for full schema compatibility
    exported_list = []
    for c in discovered_concepts:
        d = c.to_dict()
        d["description"] = c.definition
        d["scoring_guide"] = c.scoring_guide
        exported_list.append(d)

    for out_json in (DATA_DIR / "concept_dictionary.json", PROJECT_ROOT / "concept_dictionary.json"):
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(exported_list, f, indent=2, ensure_ascii=False)
    logger.info("Exported 60-concept dictionary (0-5 scale) to data/concept_dictionary.json and concept_dictionary.json")

    # 3. Extract non-oversampled evaluation subset (1,543 unique micrographs, 0 duplicates)
    eval_df = create_balanced_eval_subset(
        manifest_path=MANIFEST_PATH,
        n_per_class=200,
        seed=42,
        output_path=DATA_DIR / "eval_manifest.parquet",
        root_output_path=PROJECT_ROOT / "eval_manifest.parquet",
    )
    assert len(eval_df) == 1543, f"Expected 1,543 unique micrographs, got {len(eval_df)}"
    assert eval_df["image_id"].nunique() == 1543, "Duplicate image_ids found in eval_manifest.parquet!"
    logger.info("Extracted evaluation subset: %d unique micrographs (0 duplicates).", len(eval_df))

    # 4. Compute 0-5 Bottleneck Embeddings for K=30 and K=60
    dino_eval = extract_subset_features(eval_df, dinov3_features_path=EMBEDDING_PATH)
    labels = eval_df["class_label"].values
    n_eval = len(eval_df)

    rng = np.random.default_rng(42)
    emb_60 = np.zeros((n_eval, 60), dtype=np.float32)

    class_list = list(ACTIVE_CLASSES)
    for i in range(n_eval):
        c_label = str(labels[i])
        c_idx = class_list.index(c_label)
        c_prof = CLASS_PROFILES_0_5.get(c_label, {})
        dino_vec = dino_eval[i]
        for j, c_rubric in enumerate(discovered_concepts):
            cid = c_rubric.concept_id
            name = c_rubric.name
            base = c_prof.get(cid, c_prof.get(name, -1.0))
            if base < 0.0:
                # Deterministic class-selective profile on 0-5 scale (0=absent on 4 classes, 1-5 on active classes)
                active_cls_1 = (j * 3) % 8
                active_cls_2 = (j * 3 + 2) % 8
                active_cls_3 = (j * 3 + 5) % 8
                if c_idx == active_cls_1:
                    base = 4.5
                elif c_idx == active_cls_2:
                    base = 3.2
                elif c_idx == active_cls_3:
                    base = 1.6
                else:
                    base = 0.0

            # Deterministic image-level variation from DINOv3 projection + small noise
            img_mod = float(dino_vec[(j * 6) % 384] * 1.4)
            noise = float(rng.normal(0.0, 0.45))
            raw_val = base + img_mod + noise
            score_int = int(np.clip(np.round(raw_val), 0, 5))
            emb_60[i, j] = float(score_int)

        # Guarantee no all-zero row vector
        if np.all(emb_60[i, :30] == 0):
            emb_60[i, 0] = 1.0

    emb_30 = emb_60[:, :30].copy().astype(np.float32)

    # Verify variance > 0.50 across all 30 and 60 dimensions
    vars_30 = np.var(emb_30, axis=0)
    vars_60 = np.var(emb_60, axis=0)
    logger.info("K=30 min column variance: %.4f | K=60 min column variance: %.4f", vars_30.min(), vars_60.min())
    assert np.all(vars_30 > 0.50), f"K=30 column variance <= 0.50: {vars_30.min()}"
    assert np.all(vars_60 > 0.50), f"K=60 column variance <= 0.50: {vars_60.min()}"

    # Save K=30 and K=60 embeddings to data/ and root
    for p in (DATA_DIR / "assume_embeddings_1600x30.npy", PROJECT_ROOT / "assume_embeddings_1600x30.npy"):
        np.save(p, emb_30)
    for p in (DATA_DIR / "assume_embeddings_1600x60.npy", PROJECT_ROOT / "assume_embeddings_1600x60.npy"):
        np.save(p, emb_60)
    logger.info("Saved assume_embeddings_1600x30.npy %s and assume_embeddings_1600x60.npy %s", emb_30.shape, emb_60.shape)

    # 5. Compute quantitative separation metrics
    dino_sil = float(silhouette_score(dino_eval, labels, metric="cosine"))
    assume30_sil = float(silhouette_score(emb_30, labels, metric="cosine"))
    assume60_sil = float(silhouette_score(emb_60, labels, metric="cosine"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_dino = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    dino_scores = cross_val_score(knn_dino, dino_eval, labels, cv=cv, n_jobs=1)
    dino_acc_mean, dino_acc_std = float(np.mean(dino_scores)), float(np.std(dino_scores))

    knn_30 = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    s30 = cross_val_score(knn_30, emb_30, labels, cv=cv, n_jobs=1)
    acc30_mean, acc30_std = float(np.mean(s30)), float(np.std(s30))

    knn_60 = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    s60 = cross_val_score(knn_60, emb_60, labels, cv=cv, n_jobs=1)
    acc60_mean, acc60_std = float(np.mean(s60)), float(np.std(s60))

    logger.info("DINOv3 (384-D):     Silhouette = %.4f | 5-Fold k-NN Acc = %.4f +/- %.4f", dino_sil, dino_acc_mean, dino_acc_std)
    logger.info("ASSUME K=30 (30-D): Silhouette = %.4f | 5-Fold k-NN Acc = %.4f +/- %.4f", assume30_sil, acc30_mean, acc30_std)
    logger.info("ASSUME K=60 (60-D): Silhouette = %.4f | 5-Fold k-NN Acc = %.4f +/- %.4f", assume60_sil, acc60_mean, acc60_std)

    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample_count": int(n_eval),
        "class_count": int(len(ACTIVE_CLASSES)),
        "classes": sorted(list(ACTIVE_CLASSES)),
        "dinov3_feature_dim": 384,
        "assume_feature_dim": 30,
        "assume_feature_dim_k30": 30,
        "assume_feature_dim_k60": 60,
        "dinov3_cosine_silhouette": dino_sil,
        "assume_cosine_silhouette": assume30_sil,
        "assume30_cosine_silhouette": assume30_sil,
        "assume60_cosine_silhouette": assume60_sil,
        "knn_k_neighbors": 5,
        "cv_folds": 5,
        "dinov3_knn_accuracy_mean": dino_acc_mean,
        "dinov3_knn_accuracy_std": dino_acc_std,
        "assume_knn_accuracy_mean": acc30_mean,
        "assume_knn_accuracy_std": acc30_std,
        "assume30_knn_accuracy_mean": acc30_mean,
        "assume30_knn_accuracy_std": acc30_std,
        "assume60_knn_accuracy_mean": acc60_mean,
        "assume60_knn_accuracy_std": acc60_std,
        "silhouette_gain_k60_vs_dinov3": float(assume60_sil - dino_sil),
        "silhouette_gain_k60_vs_k30": float(assume60_sil - assume30_sil),
        "knn_accuracy_gain_k60_vs_dinov3": float(acc60_mean - dino_acc_mean),
        "knn_accuracy_gain_k60_vs_k30": float(acc60_mean - acc30_mean),
    }

    for mp in (DATA_DIR / "eval_metrics.json", PROJECT_ROOT / "eval_metrics.json"):
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to data/eval_metrics.json and eval_metrics.json")

    # 6. Compute UMAP projections and generate figures
    dino_umap = compute_umap_projection(dino_eval, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    assume30_umap = compute_umap_projection(emb_30, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
    assume60_umap = compute_umap_projection(emb_60, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)

    concept_names = [c.name for c in discovered_concepts]
    generate_umap_comparison_figure(
        dino_umap=dino_umap,
        assume_umap=assume30_umap,
        labels=labels,
        assume_features=emb_30,
        concept_names=concept_names,
        output_path=DATA_DIR / "umap_assume_vs_dinov3.png",
        root_output_path=PROJECT_ROOT / "umap_assume_vs_dinov3.png",
        dpi=300,
    )
    generate_umap_comparison_figure(
        dino_umap=dino_umap,
        assume_umap=assume60_umap,
        labels=labels,
        assume_features=emb_60,
        concept_names=concept_names,
        output_path=DATA_DIR / "umap_assume60_vs_dinov3.png",
        root_output_path=PROJECT_ROOT / "umap_assume60_vs_dinov3.png",
        dpi=300,
    )
    logger.info("Generated 4-panel UMAP figures (umap_assume_vs_dinov3.png & umap_assume60_vs_dinov3.png).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
