import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_viz1_clustermap_artifact():
    viz1_file = PROJECT_ROOT / "visualizations" / "viz1_concept_clustermap_dendrogram.png"
    assert viz1_file.exists(), "Viz 1 image must exist"
    assert viz1_file.stat().st_size > 500_000, "Viz 1 image should be high-resolution publication size"


def test_viz2_filmstrips_and_monotonicity():
    viz2_img = PROJECT_ROOT / "visualizations" / "viz2_latent_traversal_filmstrips.png"
    viz2_json = PROJECT_ROOT / "visualizations" / "viz2_traversal_paths.json"

    assert viz2_img.exists(), "Viz 2 filmstrip image must exist"
    assert viz2_json.exists(), "Viz 2 JSON trajectory metadata must exist"

    with open(viz2_json, "r", encoding="utf-8") as f:
        traversals = json.load(f)

    assert len(traversals) == 4, "Should have 4 traversals"
    for t in traversals:
        scores = t["target_scores"]
        assert len(scores) == 6, "Each traversal should have 6 steps"
        # Strict monotonicity check
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Scores along trajectory must be non-decreasing: {scores}"
        assert t["non_target_drift_var"] < 100.0, "Non-target drift variance must be constrained"


def test_viz4_gibbs_barycentric_simplex():
    viz4_img = PROJECT_ROOT / "visualizations" / "viz4_gibbs_morphological_simplex.png"
    assert viz4_img.exists(), "Viz 4 image must exist"

    embeddings = np.load(PROJECT_ROOT / "assume_embeddings_1600x30.npy")
    with open(PROJECT_ROOT / "concept_dictionary_k30.json", "r", encoding="utf-8") as f:
        concept_dict = json.load(f)
    concept_ids = [c["concept_id"] for c in concept_dict]

    from scripts.generate_viz4_gibbs_simplex import compute_barycentric_coords
    f_0d, f_1d, f_3d = compute_barycentric_coords(embeddings, concept_ids)

    # Check conservation of barycentric coordinates
    sums = f_0d + f_1d + f_3d
    np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="Barycentric coords must sum to 1.0")
    assert np.all(f_0d >= 0), "Fractions must be non-negative"
    assert np.all(f_1d >= 0), "Fractions must be non-negative"
    assert np.all(f_3d >= 0), "Fractions must be non-negative"


def test_demo1_streamlit_apptest():
    from streamlit.testing.v1 import AppTest
    app_path = PROJECT_ROOT / "demos" / "app_microstructure_equalizer.py"
    assert app_path.exists(), "App file must exist"

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, "Streamlit app should run without unhandled exceptions"

    # Verify curated continuous sliders (at least 10: 4 degradations + 6 morphologies)
    assert len(at.slider) >= 10, "Should have curated continuous sliders"

    # Verify buttons exist by finding their labels
    button_labels = [b.label for b in at.button]
    assert any("Random Image" in lbl for lbl in button_labels), "Must have Random Image button"
    assert any("Subtract Blur" in lbl for lbl in button_labels), "Must have Subtract Blur button"
    assert any("Remove Dust" in lbl for lbl in button_labels), "Must have Remove Dust button"
    assert any("Suppress Charge" in lbl for lbl in button_labels), "Must have Suppress Charge button"
    assert any("Clear Halos" in lbl for lbl in button_labels), "Must have Clear Halos button"
    assert any("Pristine Specimen" in lbl for lbl in button_labels), "Must have Pristine Specimen button"

    # Helper to click button by substring
    def click_btn(substring):
        for b in at.button:
            if substring in b.label:
                b.click().run()
                assert not at.exception
                return
        raise AssertionError(f"Button with label matching '{substring}' not found")

    # Test random button
    click_btn("Random Image")
    assert "selected_img_idx" in at.session_state

    # Test subtract blur button
    click_btn("Subtract Blur")
    assert at.session_state["slider_depth_of_field_focal_degradation"] == 0.0

    # Test remove dust button
    click_btn("Remove Dust")
    assert at.session_state["slider_discrete_particle_spatial_areal_density"] == 0.0

    # Test suppress charge button
    click_btn("Suppress Charge")
    assert at.session_state["slider_isolated_point_emitter_brightness_contrast"] == 0.0

    # Test clear halos button
    click_btn("Clear Halos")
    assert at.session_state["slider_macroscopic_dark_stain_cloudiness"] == 0.0

    # Test clean specimen button (all 4 degradations -> 0.0)
    click_btn("Pristine Specimen")
    assert at.session_state["slider_depth_of_field_focal_degradation"] == 0.0
    assert at.session_state["slider_discrete_particle_spatial_areal_density"] == 0.0
    assert at.session_state["slider_isolated_point_emitter_brightness_contrast"] == 0.0
    assert at.session_state["slider_macroscopic_dark_stain_cloudiness"] == 0.0

    # Test interactive dialing of a match (Dial This button)
    dial_buttons = [b for b in at.button if "Dial This" in b.label]
    assert len(dial_buttons) > 0, "Should have interactive 'Dial This' buttons on search results"
    dial_buttons[0].click().run()
    assert not at.exception
