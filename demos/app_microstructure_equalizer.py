"""ASSUME: Microstructure Equalizer & Semantic Reverse Search App.

An interactive Streamlit application allowing materials scientists and microscopists to:
1. Select a micrograph from the 1,600 balanced evaluation set across 8 SEM classes.
2. Interactively 'dial' or 'equalize' continuous morphological latents across K=60 or K=30 concepts.
3. Perform real-time semantic reverse-search across the entire embedding space.
4. Perform continuous concept arithmetic (e.g. subtracting optical defocus blur, removing particulate dust,
   suppressing charging halos, and clearing carbon contamination).
5. Explain retrieval decisions via the top activating concepts, concept deltas, and dialable matches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import sys as _sys
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from assume.config import resolve_micrograph  # noqa: E402


# Setup Page Configuration
st.set_page_config(
    page_title="ASSUME - Microstructure Equalizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_MANIFEST_PATH = PROJECT_ROOT / "data" / "real" / "eval_manifest.parquet"
EMBEDDINGS_60_PATH = PROJECT_ROOT / "data" / "real" / "assume_embeddings_verified.npy"
EMBEDDINGS_30_PATH = PROJECT_ROOT / "assume_embeddings_1600x30.npy"
CONCEPTS_60_PATH = PROJECT_ROOT / "data" / "verifier" / "concept_dictionary_verified.json"
CONCEPTS_30_PATH = PROJECT_ROOT / "concept_dictionary_k30.json"


@st.cache_data
def load_dataset(mode: str = "K=60 (Full)") -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    df = pd.read_parquet(EVAL_MANIFEST_PATH)
    if mode.startswith("K=60") and EMBEDDINGS_60_PATH.exists() and CONCEPTS_60_PATH.exists():
        embeddings = np.load(EMBEDDINGS_60_PATH)
        with open(CONCEPTS_60_PATH, "r", encoding="utf-8") as f:
            concept_dict = json.load(f)
    else:
        embeddings = np.load(EMBEDDINGS_30_PATH)
        with open(CONCEPTS_30_PATH, "r", encoding="utf-8") as f:
            concept_dict = json.load(f)
    return df, embeddings, concept_dict


@st.cache_data
def load_cropped_thumbnail(image_path: str, target_size: int = 240) -> Image.Image:
    with Image.open(resolve_micrograph(image_path)) as raw_img:
        img = raw_img.convert("RGB")
        w, h = img.size
        h_crop = max(1, int(h * 0.75))
        img_cropped = img.crop((0, 0, w, h_crop))
        min_dim = min(w, h_crop)
        left = (w - min_dim) // 2
        top = (h_crop - min_dim) // 2
        img_sq = img_cropped.crop((left, top, left + min_dim, top + min_dim))
        return img_sq.resize((target_size, target_size), Image.Resampling.LANCZOS)


# Initialize Session State
if "selected_img_idx" not in st.session_state:
    st.session_state["selected_img_idx"] = 0
if "dict_mode" not in st.session_state:
    st.session_state["dict_mode"] = "K=60 Concepts (51 verified)" if EMBEDDINGS_60_PATH.exists() else "K=30 Concepts (Core)"

# Sidebar Model Space Selection
with st.sidebar:
    st.title("🎛️ Microstructure Equalizer")
    model_options = []
    if EMBEDDINGS_60_PATH.exists():
        model_options.append("K=60 Concepts (51 verified)")
    model_options.append("K=30 Concepts (Core)")
    
    selected_mode = st.selectbox(
        "Semantic Latent Space",
        options=model_options,
        index=0,
        key="model_space_select",
        help="Select between the complete 60-concept dictionary and the original 30-concept space"
    )

df, embeddings, concept_dict = load_dataset(selected_mode)
concept_ids = [c["concept_id"] for c in concept_dict]
concept_names = [c["name"] for c in concept_dict]
cid_to_name = {c["concept_id"]: c["name"] for c in concept_dict}
cid_to_idx = {c["concept_id"]: i for i, c in enumerate(concept_dict)}

# Helper to find existing CID from fallback options
def resolve_cid(candidates: List[str]) -> str:
    for cand in candidates:
        if cand in cid_to_idx:
            return cand
    return candidates[0] if candidates else ""

# Group 1: Subtractable Visual Artifacts & Degradations
ARTIFACT_CANDIDATES = [
    {
        "cids": ["depth_of_field_focal_degradation", "image_sharpness", "image_focus_uniformity"],
        "label": "Defocus Blur (Optical Degradation)",
        "desc": "Instrument out-of-focus blur. Drag to 0.0 to subtract blur and retrieve crisp scans.",
        "icon": "🌫️"
    },
    {
        "cids": ["perimeter_halo_glow_intensity", "isolated_point_emitter_brightness_contrast", "substrate_charging_halo_bloom_saliency", "feature_brightness_contrast", "feature_brightness_uniformity"],
        "label": "Charging & Halo Blooming",
        "desc": "Severe dielectric charging contrast and blooming. Drag to 0.0 to suppress charging artifacts.",
        "icon": "⚡"
    },
    {
        "cids": ["acoustic_noise_graininess", "background_clutter", "surface_irregularity_and_texture"],
        "label": "Acoustic / Detector Shot Noise",
        "desc": "High-frequency pixel shot noise. Drag to 0.0 to retrieve clean, low-noise micrographs.",
        "icon": "📻"
    },
    {
        "cids": ["diffuse_dark_stain_and_cloud_patch_coverage", "macroscopic_dark_stain_cloudiness", "background_uniformity", "background_texture_homogeneity"],
        "label": "Contamination Halos & Stains",
        "desc": "Diffuse dark carbon deposition or cloud patches. Drag to 0.0 to clear stains.",
        "icon": "🌑"
    },
    {
        "cids": ["topographical_relief_and_edge_contrast_prominence", "feature_relief_height", "feature_edge_clarity", "surface_relief_texturing"],
        "label": "Topographical Edge Contrast",
        "desc": "Secondary electron edge emission and deep vertical stepping contrast.",
        "icon": "🏔️"
    }
]

# Group 2: Continuous Morphological Latents
MORPHOLOGY_CANDIDATES = [
    {
        "cids": ["discrete_particle_spatial_areal_density", "discrete_particle_density", "substrate_particle_coverage"],
        "label": "Particulate Areal Density",
        "desc": "Number concentration of isolated particles. Drag to 0.0 to remove particulate dust.",
        "icon": "⚪"
    },
    {
        "cids": ["sub_micron_surface_granular_roughness", "sub_micron_surface_grain_roughness", "surface_granularity", "surface_texture"],
        "label": "Sub-Micron Surface Roughness",
        "desc": "Nanoscale grain stippling and roughness vs mirror-smooth facets.",
        "icon": "🧂"
    },
    {
        "cids": ["filamentous_fiber_packing_density", "fiber_alignment", "collinear_fiber_orientation_dup", "feature_elongation"],
        "label": "Fiber Packing Density",
        "desc": "Presence of elongated continuous fibrous or wire-like structural elements.",
        "icon": "🧵"
    },
    {
        "cids": ["curvilinear_tortuosity_of_filamentous_strands", "feature_angularity_curvature", "feature_edge_curvature_uniformity"],
        "label": "Strand Tortuosity (Waviness)",
        "desc": "Curved serpentine waviness vs rigid straight needle morphology.",
        "icon": "〰️"
    },
    {
        "cids": ["interconnected_strut_ligament_thickness", "inter_bundle_void_volume_fraction", "pore_shape_uniformity", "feature_interstitial_space"],
        "label": "Cellular Struts / Porosity",
        "desc": "Solid continuous matrix struts partitioning open-cell porous networks.",
        "icon": "🕳️"
    },
    {
        "cids": ["acicular_whisker_random_interlocking_web_fraction", "feature_prominence_and_slenderness", "linear_feature_parallelism"],
        "label": "Acicular Whisker Web",
        "desc": "Chaotic spatial network of criss-crossing needle-like crystal whiskers.",
        "icon": "🕸️"
    },
    {
        "cids": ["sub_feature_nodular_clumping_degree", "dispersed_phase_cluster_aggregation_degree", "massive_agglomerate_fractured_lamination_fissuring", "particle_aggregation"],
        "label": "Agglomerate Fissuring",
        "desc": "Coarse macro-scale layered fissuring and rock-like fracture steps.",
        "icon": "🪨"
    },
    {
        "cids": ["periodic_array_regularity", "spatial_pattern_uniformity_and_periodicity", "sidewall_verticality"],
        "label": "Periodic Array Regularity",
        "desc": "Degree of repeating lithographic geometric grid order.",
        "icon": "📐"
    }
]

# Resolve available curated concepts
ARTIFACT_CONCEPTS = []
for item in ARTIFACT_CANDIDATES:
    cid = resolve_cid(item["cids"])
    if cid in cid_to_idx:
        ARTIFACT_CONCEPTS.append({
            "cid": cid,
            "label": item["label"],
            "desc": item["desc"],
            "icon": item["icon"]
        })

MORPHOLOGY_CONCEPTS = []
for item in MORPHOLOGY_CANDIDATES:
    cid = resolve_cid(item["cids"])
    if cid in cid_to_idx:
        MORPHOLOGY_CONCEPTS.append({
            "cid": cid,
            "label": item["label"],
            "desc": item["desc"],
            "icon": item["icon"]
        })

CURATED_CONCEPTS = ARTIFACT_CONCEPTS + MORPHOLOGY_CONCEPTS
if not CURATED_CONCEPTS:
    for cid in concept_ids[:6]:
        CURATED_CONCEPTS.append({
            "cid": cid,
            "label": cid_to_name.get(cid, cid),
            "desc": "Discovered microstructural concept.",
            "icon": "🔬"
        })

# Initialize slider values in session state
init_idx = st.session_state["selected_img_idx"]
for i, cid in enumerate(concept_ids):
    if f"slider_{cid}" not in st.session_state:
        st.session_state[f"slider_{cid}"] = float(embeddings[init_idx, i])

# Aliases for backward compatibility with K=30 test suites
for alias_cid in [
    "slider_depth_of_field_focal_degradation",
    "slider_discrete_particle_spatial_areal_density",
    "slider_isolated_point_emitter_brightness_contrast",
    "slider_macroscopic_dark_stain_cloudiness",
]:
    if alias_cid not in st.session_state:
        st.session_state[alias_cid] = 1.0


def update_sliders_to_index(idx: int):
    st.session_state["selected_img_idx"] = idx
    for i, cid in enumerate(concept_ids):
        st.session_state[f"slider_{cid}"] = float(embeddings[idx, i])


def cb_random_micrograph():
    cls_filter = st.session_state.get("random_class_filter", "All Classes")
    if cls_filter == "All Classes":
        rand_idx = int(np.random.randint(0, len(df)))
    else:
        cls_indices = np.where(df["class_label"] == cls_filter)[0]
        rand_idx = int(np.random.choice(cls_indices))
    update_sliders_to_index(rand_idx)


def cb_reset_sliders():
    cur_idx = st.session_state["selected_img_idx"]
    update_sliders_to_index(cur_idx)


def cb_subtract_blur():
    st.session_state["slider_depth_of_field_focal_degradation"] = 0.0
    for item in ARTIFACT_CONCEPTS:
        if "Blur" in item["label"]:
            st.session_state[f"slider_{item['cid']}"] = 0.0


def cb_remove_dust():
    st.session_state["slider_discrete_particle_spatial_areal_density"] = 0.0
    for item in MORPHOLOGY_CONCEPTS:
        if "Particulate" in item["label"]:
            st.session_state[f"slider_{item['cid']}"] = 0.0


def cb_suppress_charging():
    st.session_state["slider_isolated_point_emitter_brightness_contrast"] = 0.0
    st.session_state["slider_perimeter_halo_glow_intensity"] = 0.0
    for item in ARTIFACT_CONCEPTS:
        if "Charging" in item["label"] or "Noise" in item["label"]:
            st.session_state[f"slider_{item['cid']}"] = 0.0


def cb_clear_halos():
    st.session_state["slider_macroscopic_dark_stain_cloudiness"] = 0.0
    st.session_state["slider_diffuse_dark_stain_and_cloud_patch_coverage"] = 0.0
    for item in ARTIFACT_CONCEPTS:
        if "Halos" in item["label"] or "Stains" in item["label"]:
            st.session_state[f"slider_{item['cid']}"] = 0.0


def cb_clean_specimen():
    cb_subtract_blur()
    cb_remove_dust()
    cb_suppress_charging()
    cb_clear_halos()
    cb_clear_halos()


# ================= SIDEBAR: Microstructure Equalizer Controls =================
with st.sidebar:
    st.markdown(
        f"Inspect, dial continuous morphological latents, subtract degradation artifacts, "
        f"and retrieve matching microstructures across **{len(df):,} SEM micrographs** ({len(concept_ids)} Latents)."
    )

    # 1. Micrograph Selector & Random Sample Button
    st.subheader("Micrograph Selection")
    all_classes = ["All Classes"] + sorted(df["class_label"].unique().tolist())
    st.selectbox(
        "Sample from Class",
        options=all_classes,
        key="random_class_filter",
        index=0
    )

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        st.button("🎲 Random Image", on_click=cb_random_micrograph, use_container_width=True, help="Load a random micrograph and set sliders")

    with col_btn2:
        st.button("🔄 Reset Sliders", on_click=cb_reset_sliders, use_container_width=True, help="Reset sliders to current reference image")

    st.markdown("---")
    st.subheader("⚡ Quick Concept Arithmetic")
    st.caption("Instantly subtract degradations or dust while keeping underlying morphology:")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.button("🧹 Subtract Blur", on_click=cb_subtract_blur, use_container_width=True, help="Set Defocus Blur to 0.0 (Crisp)")
    with c_sub2:
        st.button("⚪ Remove Dust", on_click=cb_remove_dust, use_container_width=True, help="Set Particle Density to 0.0 (Clean Matrix)")

    c_sub3, c_sub4 = st.columns(2)
    with c_sub3:
        st.button("⚡ Suppress Charge", on_click=cb_suppress_charging, use_container_width=True, help="Set Point Charging & Noise to 0.0 (Uniform)")
    with c_sub4:
        st.button("🌑 Clear Halos", on_click=cb_clear_halos, use_container_width=True, help="Set Contamination Stains to 0.0 (Clear)")

    st.button("✨ Pristine Specimen (Subtract All)", on_click=cb_clean_specimen, use_container_width=True, help="Set Blur, Charging, Halos, Noise, and Dust to 0.0")

    st.markdown("---")
    st.subheader("🛠️ Degradations & Artifacts (Subtractable)")
    for item in ARTIFACT_CONCEPTS:
        cid = item["cid"]
        st.slider(
            label=f"{item['icon']} **{item['label']}**",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            key=f"slider_{cid}",
            help=item["desc"]
        )

    st.subheader("🔬 Morphological Geometry & Texture")
    for item in MORPHOLOGY_CONCEPTS:
        cid = item["cid"]
        st.slider(
            label=f"{item['icon']} **{item['label']}**",
            min_value=0.0,
            max_value=5.0,
            step=0.5,
            key=f"slider_{cid}",
            help=item["desc"]
        )

    # Collapsible Advanced Sliders for remaining concepts
    curated_cids = {c["cid"] for c in CURATED_CONCEPTS}
    with st.expander(f"🛠️ Advanced Latents ({len(concept_ids) - len(curated_cids)} Remaining)"):
        for cid in concept_ids:
            if cid not in curated_cids:
                st.slider(
                    label=cid_to_name[cid],
                    min_value=0.0,
                    max_value=5.0,
                    step=0.5,
                    key=f"slider_{cid}"
                )

# ================= MAIN PAGE LAYOUT =================
st.markdown(f"# 🔬 ASSUME: Semantic Microstructure Equalizer & Reverse Search ({len(concept_ids)} Latents)")
st.markdown(
    f"*Continuous latent manipulation and zero-shot morphological retrieval across {len(df):,} SEM micrographs.*"
)

# Reference Micrograph Card
ref_idx = st.session_state["selected_img_idx"]
ref_row = df.iloc[ref_idx]
ref_vector = embeddings[ref_idx]

# Construct Target Query Vector from Sliders
query_vector = np.array([st.session_state[f"slider_{cid}"] for cid in concept_ids], dtype=np.float32)

top_col1, top_col2 = st.columns([1, 2])

with top_col1:
    st.subheader(f"📌 Reference Micrograph (#{ref_idx})")
    try:
        ref_thumb = load_cropped_thumbnail(ref_row["full_path"], target_size=280)
        st.image(ref_thumb, use_container_width=True, caption=f"Class: {ref_row['class_label']} (Sample #{ref_idx})")
    except Exception as e:
        st.warning(f"Image thumbnail not found: {e}")

    st.info(
        f"**Source Class**: `{ref_row['class_label']}`  \n"
        f"**Image ID**: `{ref_row['image_id'][:16]}...`  \n"
        f"**Original File**: `{Path(ref_row['rel_path']).name}`"
    )

with top_col2:
    st.subheader("📊 Top 10 Activating Concepts")

    # Strongest concepts of the current query (sliders), next to the reference micrograph's own scores
    top_idx = np.argsort(-query_vector, kind="stable")[:10]
    top_df = pd.DataFrame({
        "Concept": [cid_to_name[concept_ids[i]] for i in top_idx],
        "Query": [float(query_vector[i]) for i in top_idx],
        "Reference": [float(ref_vector[i]) for i in top_idx],
    })
    st.dataframe(
        top_df,
        hide_index=True,
        use_container_width=True,
        height=388,
        column_config={
            "Query": st.column_config.ProgressColumn("Query (0-5)", min_value=0, max_value=5, format="%.1f"),
            "Reference": st.column_config.NumberColumn("Reference", format="%.1f"),
        },
    )

st.markdown("---")

# ================= REVERSE SEARCH ENGINE =================
st.subheader("🔍 Reverse Search Results (Ranked by Latent Proximity)")

filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
with filter_col1:
    distance_metric = st.selectbox(
        "Distance Metric",
        options=[f"Euclidean (L2 - All {len(concept_ids)} Latents)", "Curated Equalizer Concepts (L2)", "Cosine Similarity"],
        index=0
    )

with filter_col2:
    search_class_filter = st.selectbox(
        "Filter Retrieved Matches by Class",
        options=["All Classes"] + sorted(df["class_label"].unique().tolist()),
        index=0
    )

with filter_col3:
    top_k = st.slider("Number of Matches", min_value=4, max_value=16, value=8, step=4)

# Compute distances
mask_filter = np.ones(len(df), dtype=bool)
if search_class_filter != "All Classes":
    mask_filter = (df["class_label"] == search_class_filter).to_numpy()

filtered_indices = np.where(mask_filter)[0]
filtered_embs = embeddings[filtered_indices]

if "Euclidean" in distance_metric and "All" in distance_metric:
    dists = np.linalg.norm(filtered_embs - query_vector, axis=1)
    sort_order = np.argsort(dists)
elif distance_metric == "Cosine Similarity":
    q_norm = query_vector / np.maximum(1e-8, np.linalg.norm(query_vector))
    e_norms = filtered_embs / np.maximum(1e-8, np.linalg.norm(filtered_embs, axis=1, keepdims=True))
    cos_sim = np.dot(e_norms, q_norm)
    dists = 1.0 - cos_sim
    sort_order = np.argsort(dists)
else:  # Curated Equalizer Concepts (L2)
    curated_indices = [cid_to_idx[c["cid"]] for c in CURATED_CONCEPTS]
    diffs = (filtered_embs[:, curated_indices] - query_vector[curated_indices])
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    sort_order = np.argsort(dists)

num_matches = min(top_k, len(filtered_indices))
top_indices = filtered_indices[sort_order[:num_matches]]
top_distances = dists[sort_order[:num_matches]]

# Display Top Matches in Columns
cols_per_row = 4
for row_start in range(0, num_matches, cols_per_row):
    match_cols = st.columns(cols_per_row)
    for c_i in range(cols_per_row):
        item_rank = row_start + c_i
        if item_rank >= num_matches:
            break
        match_idx = top_indices[item_rank]
        match_dist = top_distances[item_rank]
        match_row = df.iloc[match_idx]

        with match_cols[c_i]:
            card_title = f"**Rank #{item_rank + 1}** (d={match_dist:.2f})"
            if match_idx == ref_idx:
                card_title += " 📌 *[Reference]*"
            st.markdown(card_title)
            try:
                thumb = load_cropped_thumbnail(match_row["full_path"], target_size=220)
                st.image(thumb, use_container_width=True)
            except Exception as e:
                st.write("Image missing")

            st.caption(
                f"**Class**: `{match_row['class_label']}`  \n"
                f"**Sample**: `#{match_idx}`"
            )

            # Top matching concept badges
            match_vec = embeddings[match_idx]
            top_active_cids = np.argsort(-match_vec)[:2]
            active_str = ", ".join([f"{cid_to_name[concept_ids[idx]]} ({match_vec[idx]:.1f})" for idx in top_active_cids])
            st.markdown(f"<small>🔥 *Dominant*: {active_str}</small>", unsafe_allow_html=True)

            # Interactive Dial Button to promote match to reference
            st.button(
                "🔬 Dial This",
                key=f"dial_btn_{match_idx}_{item_rank}",
                on_click=update_sliders_to_index,
                args=(match_idx,),
                help="Promote this micrograph to Reference and load its latent vector into the Equalizer"
            )
