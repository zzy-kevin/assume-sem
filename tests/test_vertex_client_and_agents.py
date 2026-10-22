"""Comprehensive test suite for ASSUME Milestone 3: Resilient Gemini Vertex API Harness.

Covers:
- ResilientVertexClient: exponential backoff with full jitter, HTTP 429 / 5xx handling,
  network exception recovery, max retries limit, non-retryable failure handling,
  clean JSON extraction (codeblocks and conversational wrappers), image encoding.
- ConceptRubric: atomic schema (name, definition, scale_1, scale_5, scale_10),
  snake_case ID derivation, dict interface, and robust parsing from alternative formats.
- ProposerAgent: single-property atomicity prompts, monotonic anchor enforcement,
  structured JSON decoding, and concept list generation.
- ScorerAgent: mandatory <= 10 concept chunking logic (e.g. 25 rubrics -> [10, 10, 5]),
  image scoring with [1, 10] integer clamping, (N, K) numpy matrix assembly.
- VisualMergerAgent: collision evaluation (|r| > 0.80), MergeDecision outcome handling
  for MERGE and KEEP_BOTH actions.
- Live integration smoke test: real Vertex AI Express API connectivity verification.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from PIL import Image

from assume.agents.merger import MergeDecision, VisualMergerAgent
from assume.agents.proposer import ConceptRubric, ProposerAgent
from assume.agents.scorer import ScorerAgent
from assume.agents.vertex_client import ResilientVertexClient


class TestResilientVertexClient(unittest.TestCase):
    """Unit tests for ResilientVertexClient backoff, retries, and utilities."""

    def setUp(self) -> None:
        self.api_key = "test_fake_api_key_12345"
        self.project_id = "1034989953225"
        self.client = ResilientVertexClient(
            api_key=self.api_key,
            project_id=self.project_id,
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            timeout=5.0,
        )

    def test_init_raises_without_credentials(self) -> None:
        """Instantiating without api_key/project_id and without env vars must raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ResilientVertexClient(api_key=None, project_id=None)

    def test_endpoint_and_headers_constructed(self) -> None:
        """Verify endpoint URL format and authentication header."""
        expected_url = (
            f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/global/publishers/google/models/{self.client.model}:generateContent"
        )
        self.assertEqual(self.client.endpoint, expected_url)
        self.assertEqual(self.client.headers["x-goog-api-key"], self.api_key)
        self.assertEqual(self.client.headers["Content-Type"], "application/json")

    def test_model_path_prefix_sanitization(self) -> None:
        """Client must strip any redundant publishers/google/models/ prefix from model name."""
        c = ResilientVertexClient(
            api_key="key",
            project_id="proj",
            model="publishers/google/models/gemini-2.5-flash",
        )
        self.assertEqual(c.model, "gemini-2.5-flash")
        expected_url = (
            "https://aiplatform.googleapis.com/v1/projects/proj"
            "/locations/global/publishers/google/models/gemini-2.5-flash:generateContent"
        )
        self.assertEqual(c.endpoint, expected_url)

    def test_backoff_full_jitter_bounds(self) -> None:
        """Backoff calculation must adhere to exponential progression and full jitter bounds."""
        for attempt in range(1, 5):
            ceiling = min(self.client.max_delay, self.client.base_delay * (2.0 ** (attempt - 1)))
            for _ in range(20):
                sleep_dur = self.client.compute_backoff(attempt)
                self.assertGreaterEqual(sleep_dur, 0.05)
                self.assertLessEqual(sleep_dur, ceiling)

    def test_backoff_honors_retry_after_header(self) -> None:
        """If Retry-After header is provided, backoff must respect header value."""
        sleep_dur = self.client.compute_backoff(attempt=1, retry_after_header="3")
        self.assertGreaterEqual(sleep_dur, 3.0)
        self.assertLessEqual(sleep_dur, 4.1)

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_retry_on_429_then_succeed(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """Client must retry on HTTP 429 and succeed when 200 is subsequently returned."""
        resp_429 = MagicMock(status_code=429, text="Rate limit exceeded", headers={})
        resp_200 = MagicMock(
            status_code=200,
            json=lambda: {
                "candidates": [{
                    "content": {"parts": [{"text": "{\"status\": \"ok\"}"}]}
                }]
            },
        )
        mock_post.side_effect = [resp_429, resp_429, resp_200]

        result = self.client.call({"test": "payload"})
        self.assertIn("candidates", result)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_retry_on_503_and_network_exception(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """Client must retry on 503 and network exceptions like TimeoutException."""
        resp_503 = MagicMock(status_code=503, text="Service Unavailable", headers={})
        resp_200 = MagicMock(
            status_code=200,
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "success"}]}}]},
        )
        mock_post.side_effect = [
            httpx.TimeoutException("Read timeout"),
            resp_503,
            resp_200,
        ]

        result = self.client.call({"test": "payload"})
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertIn("candidates", result)

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_max_retries_exceeded_raises_runtime_error(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """Exceeding max retries on persistent 429 must raise RuntimeError."""
        resp_429 = MagicMock(status_code=429, text="Resource Exhausted", headers={})
        mock_post.return_value = resp_429

        with self.assertRaises(RuntimeError) as ctx:
            self.client.call({"test": "payload"})

        self.assertIn("Max retries", str(ctx.exception))
        # 1 initial + 3 retries = 4 calls total
        self.assertEqual(mock_post.call_count, 4)

    @patch("httpx.Client.post")
    def test_non_retryable_http_error_fails_immediately(self, mock_post: MagicMock) -> None:
        """HTTP 400 or 403 must fail immediately without wasting retry attempts."""
        resp_400 = MagicMock(status_code=400, text="Invalid argument: bad JSON", headers={})
        mock_post.return_value = resp_400

        with self.assertRaises(RuntimeError) as ctx:
            self.client.call({"bad": "request"})

        self.assertIn("non-retryable error 400", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)

    @patch("time.sleep", return_value=None)
    @patch("httpx.Client.post")
    def test_body_level_resource_exhausted_retries(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """If response body contains RESOURCE_EXHAUSTED error payload, must retry."""
        resp_body_error = MagicMock(
            status_code=200,
            json=lambda: {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}},
        )
        resp_ok = MagicMock(
            status_code=200,
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "{\"key\": \"val\"}"}]}}]},
        )
        mock_post.side_effect = [resp_body_error, resp_ok]

        result = self.client.call({"test": "payload"})
        self.assertIn("candidates", result)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_extract_clean_json_variations(self) -> None:
        """Test extraction across markdown codeblocks, raw JSON, and text wrappers."""
        # 1. Fenced with ```json
        s1 = "```json\n{\"action\": \"MERGE\", \"score\": 8}\n```"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s1), {"action": "MERGE", "score": 8})

        # 2. Fenced without json tag
        s2 = "```\n[1, 2, 3, 4]\n```"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s2), [1, 2, 3, 4])

        # 3. Text with surrounding conversation
        s3 = "Here is the requested output:\n{\"result\": true}\nHope that helps!"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s3), {"result": True})

        # 4. Raw JSON
        s4 = "{\"scores\": {\"pore_density\": 7}}"
        self.assertEqual(ResilientVertexClient.extract_clean_json(s4), {"scores": {"pore_density": 7}})

        # 5. Invalid JSON raises ValueError
        with self.assertRaises(ValueError):
            ResilientVertexClient.extract_clean_json("Not a JSON string at all.")

    def test_extract_text_filters_thoughts(self) -> None:
        """Thinking parts labeled with thought=True must be filtered out."""
        raw_resp = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": True, "text": "Let me think about this image carefully..."},
                        {"text": "{\"decision\": \"MERGE\"}"},
                    ]
                }
            }]
        }
        extracted = ResilientVertexClient.extract_text_from_response(raw_resp)
        self.assertEqual(extracted, "{\"decision\": \"MERGE\"}")

    def test_encode_image_geometry(self) -> None:
        """Image encoding must produce valid base64 JPEG from PIL Image."""
        test_img = Image.new("RGB", (1024, 768), color=(200, 100, 50))
        b64_str = ResilientVertexClient.encode_image(test_img)
        self.assertIsInstance(b64_str, str)
        self.assertGreater(len(b64_str), 100)

        # Decode and verify dimensions match TARGET_IMAGE_SIZE (512, 288)
        decoded_bytes = io.BytesIO(base64.b64decode(b64_str))
        reconstructed = Image.open(decoded_bytes)
        self.assertEqual(reconstructed.size, (512, 288))

    def test_encode_image_no_double_crop_on_preprocessed(self) -> None:
        """Passing an already-preprocessed (512, 288) image to encode_image must not crop it again."""
        from assume.data.ingestion import preprocess_micrograph

        arr = np.zeros((768, 1024, 3), dtype=np.uint8)
        arr[:384, :, 0] = 255       # Top 50%: Red
        arr[384:576, :, 1] = 255    # 50%-75%: Green
        arr[576:, :, 2] = 255       # Bottom 25% (databar): Blue
        raw_img = Image.fromarray(arr, mode="RGB")

        preprocessed = preprocess_micrograph(raw_img)
        b64_from_raw = ResilientVertexClient.encode_image(raw_img)
        b64_from_prep = ResilientVertexClient.encode_image(preprocessed)

        self.assertEqual(b64_from_raw, b64_from_prep)

        # Also verify the fallback branch inside encode_image when preprocess_micrograph raises
        with patch("assume.data.ingestion.preprocess_micrograph", side_effect=RuntimeError("fallback")):
            b64_fallback_raw = ResilientVertexClient.encode_image(raw_img)
            b64_fallback_prep = ResilientVertexClient.encode_image(preprocessed)
            self.assertEqual(b64_from_raw, b64_fallback_raw)
            self.assertEqual(b64_from_raw, b64_fallback_prep)


class TestConceptRubric(unittest.TestCase):
    """Unit tests for ConceptRubric dataclass schema and flexibility."""

    def test_rubric_schema_and_id_derivation(self) -> None:
        """ConceptRubric must auto-generate clean snake_case concept_id from name and 0-5 scoring_guide."""
        r = ConceptRubric(
            name="Structural Tortuosity & Curvature",
            definition="Degree of curvature and winding of linear microstructural elements.",
            scale_1="Rigid, perfectly straight linear features with no discernible bending.",
            scale_3="Gentle wave-like curvature with moderate deflection.",
            scale_5="Heavily looped, coiled, and tortuous tangled morphology.",
        )
        self.assertEqual(r.concept_id, "structural_tortuosity_curvature")
        self.assertEqual(r.name, "Structural Tortuosity & Curvature")
        self.assertEqual(r["scale_0"], r.scale_0)
        self.assertEqual(r["scale_1"], r.scale_1)
        self.assertEqual(r["scale_3"], r.scale_3)
        self.assertEqual(r["scale_5"], r.scale_5)
        self.assertIn("0:", r.scoring_guide)
        self.assertIn("1:", r.scoring_guide)
        self.assertIn("3:", r.scoring_guide)
        self.assertIn("5:", r.scoring_guide)

    def test_from_dict_and_to_dict_symmetry(self) -> None:
        """Dict serialization and deserialization must preserve all attributes."""
        d = {
            "concept_id": "open_macroporosity",
            "name": "Open Macroporosity",
            "definition": "Visual fraction of dark open void space in 2D SEM projection.",
            "scale_0": "Absent or non-porous planar surface.",
            "scale_1": "Sparse isolated pores (<5% area).",
            "scale_3": "Moderate network of cellular voids occupying ~30-50% area.",
            "scale_5": "High void fraction (>75%) consisting of deep open sponge pores.",
            "scale_10": "High void fraction (>75%) consisting of deep open sponge pores.",
        }
        r = ConceptRubric.from_dict(d)
        self.assertEqual(r.concept_id, "open_macroporosity")
        self.assertEqual(r.definition, d["definition"])
        self.assertEqual(r.to_dict(), d)

    def test_from_dict_alternative_scoring_guide_format(self) -> None:
        """from_dict must parse anchors from combined scoring_guide string (both 0-5 and legacy 1-5-10)."""
        d = {
            "name": "Grain Boundary Angularity",
            "description": "Sharpness of crystalline facet vertices.",
            "scoring_guide": "1: Rounded smooth amorphous curves; 5: Mixed rounded and faceted boundaries; 10: Highly faceted sharp geometric vertices",
        }
        r = ConceptRubric.from_dict(d)
        self.assertEqual(r.name, "Grain Boundary Angularity")
        self.assertEqual(r.definition, "Sharpness of crystalline facet vertices.")
        self.assertEqual(r.scale_1, "Rounded smooth amorphous curves")
        self.assertEqual(r.scale_5, "Mixed rounded and faceted boundaries")
        self.assertEqual(r.scale_10, "Highly faceted sharp geometric vertices")

    def test_concept_rubric_to_dict_from_dict_roundtrip_equality(self) -> None:
        """ConceptRubric must satisfy r == ConceptRubric.from_dict(r.to_dict()) for both schemas."""
        # 1. New 0-5 schema rubric
        r_new = ConceptRubric(
            name="Void Interconnectivity",
            definition="Degree of continuous open void pathways.",
            scale_1="Sparse isolated voids",
            scale_3="Partially connected cellular voids",
            scale_5="Completely percolating void labyrinth",
        )
        d_new = r_new.to_dict()
        r_new_rec = ConceptRubric.from_dict(d_new)
        self.assertEqual(r_new, r_new_rec)

        # 2. Legacy 1-5-10 schema rubric
        r_legacy = ConceptRubric(
            name="Faceted Particle Angularity",
            definition="Degree of polygonal facet sharpness.",
            scale_1="Smooth rounded spheres",
            scale_5="Intermediate mixed rounded and faceted edges",
            scale_10="Sharp planar polyhedral crystal vertices",
        )
        d_legacy = r_legacy.to_dict()
        r_legacy_rec = ConceptRubric.from_dict(d_legacy)
        self.assertEqual(r_legacy, r_legacy_rec)


class TestProposerAgent(unittest.TestCase):
    """Unit tests for ProposerAgent synthesis and constraints."""

    @patch.object(ResilientVertexClient, "generate_content")
    def test_propose_returns_validated_rubrics(self, mock_gen: MagicMock) -> None:
        """ProposerAgent must parse JSON proposals into valid ConceptRubric list."""
        mock_proposals = json.dumps([
            {
                "name": "Interconnectivity Density",
                "definition": "Frequency of branch junctions and nodal connections.",
                "scale_1": "Sparse isolated branch junctions.",
                "scale_3": "Moderate branching with secondary nodes.",
                "scale_5": "Dense, continuous percolating network of interconnected nodes.",
            },
            {
                "name": "Surface Textural Granularity",
                "definition": "Density and relief of nanoscale surface particulates.",
                "scale_1": "Sparse trace surface nano-bumps.",
                "scale_3": "Moderate surface ripple and scattered nano-bumps.",
                "scale_5": "Heavily encrusted with dense granular nodules.",
            },
        ])
        mock_gen.return_value = mock_proposals

        client = ResilientVertexClient(api_key="mock_key", project_id="mock_pid")
        agent = ProposerAgent(client=client)

        dummy_img = Image.new("RGB", (512, 288), color=(100, 100, 100))
        rubrics = agent.propose(
            images=[dummy_img] * 6,
            existing_concepts=["porosity", "grain_size"],
            n_proposals=2,
        )

        self.assertEqual(len(rubrics), 2)
        self.assertIsInstance(rubrics[0], ConceptRubric)
        self.assertEqual(rubrics[0].name, "Interconnectivity Density")
        self.assertEqual(rubrics[0].concept_id, "interconnectivity_density")
        self.assertEqual(rubrics[1].name, "Surface Textural Granularity")

        # Verify thinking budget and temperature configuration passed to generate_content
        call_kwargs = mock_gen.call_args[1]
        self.assertEqual(call_kwargs.get("thinking_budget"), 1024)
        self.assertEqual(call_kwargs.get("max_output_tokens"), 8192)
        self.assertEqual(call_kwargs.get("temperature"), 0.3)

    @patch.object(ResilientVertexClient, "generate_content")
    def test_propose_collision_mode_formats_contrastive_pairs(self, mock_gen: MagicMock) -> None:
        """In collision mode, ProposerAgent must explicitly label colliding image pairs in the prompt."""
        mock_gen.return_value = json.dumps([
            {
                "name": "Pillar Sidewall Scalloping",
                "definition": "Periodic horizontal etching grooves along vertical pillar sidewalls.",
                "scale_1": "Faint shallow ripples on sidewalls.",
                "scale_3": "Distinct periodic scalloping grooves.",
                "scale_5": "Deeply notched corrugated sidewall trenches.",
            }
        ])
        client = ResilientVertexClient(api_key="mock_key", project_id="mock_pid")
        agent = ProposerAgent(client=client)
        dummy_img = Image.new("RGB", (512, 288), color=(100, 100, 100))

        agent.propose(
            images=[dummy_img] * 6,
            existing_concepts=["porosity"],
            n_proposals=1,
            batch_mode="collision",
            collision_pairs=[(10, 25), (33, 48), (50, 99)],
        )

        contents = mock_gen.call_args[1]["contents"]
        parts_text = " ".join(p.get("text", "") for p in contents[0]["parts"] if "text" in p)
        self.assertIn("Pair 1: Micrograph #1 (index 10) vs Micrograph #2 (index 25)", parts_text)
        self.assertIn("Pair 2: Micrograph #3 (index 33) vs Micrograph #4 (index 48)", parts_text)
        self.assertIn("Pair 3: Micrograph #5 (index 50) vs Micrograph #6 (index 99)", parts_text)

    @patch.object(ResilientVertexClient, "generate_content")
    def test_propose_collision_mode_partial_pairs_labels_only_genuine_pairs(self, mock_gen: MagicMock) -> None:
        """When fewer collision pairs exist than B//2, only genuine pairs are labeled as collision pairs."""
        mock_gen.return_value = json.dumps([
            {
                "name": "Facet Angularity",
                "definition": "Sharpness of crystalline facet boundaries.",
                "scale_1": "Low",
                "scale_3": "Medium",
                "scale_5": "High",
            }
        ])
        client = ResilientVertexClient(api_key="mock_key", project_id="mock_pid")
        agent = ProposerAgent(client=client)
        dummy_img = Image.new("RGB", (512, 288), color=(100, 100, 100))

        # 6 images in batch, but only 1 colliding pair found (images 0 and 1)
        agent.propose(
            images=[dummy_img] * 6,
            existing_concepts=[],
            n_proposals=1,
            batch_mode="collision",
            collision_pairs=[(42, 99)],
        )

        contents = mock_gen.call_args[1]["contents"]
        text_parts = [p.get("text", "") for p in contents[0]["parts"] if "text" in p]
        full_text = " ".join(text_parts)

        # Micrographs 1 and 2 should be labeled as Collision Pair #1
        self.assertTrue(any("Collision Pair #1 (A) - Micrograph #1" in t for t in text_parts))
        self.assertTrue(any("Collision Pair #1 (B) - Micrograph #2" in t for t in text_parts))

        # Micrographs 3-6 should be labeled as normal micrographs, NOT Collision Pair #2 or #3
        self.assertFalse(any("Collision Pair #2" in t for t in text_parts))
        self.assertFalse(any("Collision Pair #3" in t for t in text_parts))
        self.assertTrue(any("Micrograph #3:" in t for t in text_parts))
        self.assertTrue(any("Micrograph #4:" in t for t in text_parts))

        # Instructions must list Pair 1 and NOT fabricate Pair 2 or Pair 3
        self.assertIn("Pair 1: Micrograph #1 (index 42) vs Micrograph #2 (index 99)", full_text)
        self.assertNotIn("Pair 2:", full_text)
        self.assertNotIn("Pair 3:", full_text)

    def test_propose_raises_on_empty_images(self) -> None:
        """ProposerAgent must raise ValueError when no micrographs are passed."""
        agent = ProposerAgent(client=MagicMock())
        with self.assertRaises(ValueError):
            agent.propose(images=[])


class TestScorerAgent(unittest.TestCase):
    """Unit tests for ScorerAgent <= 10 concept chunking and integer [0, 5] scoring."""

    def setUp(self) -> None:
        self.client = MagicMock()
        self.agent = ScorerAgent(client=self.client, max_chunk_size=10)

    def test_chunking_logic_25_concepts(self) -> None:
        """25 concepts must partition into 3 chunks: sizes 10, 10, 5."""
        rubrics = [
            ConceptRubric(
                name=f"Concept {i}",
                definition=f"Definition {i}",
                scale_1=f"Scale 1 for {i}",
                scale_3=f"Scale 3 for {i}",
                scale_5=f"Scale 5 for {i}",
            )
            for i in range(25)
        ]
        chunks = ScorerAgent.chunk_rubrics(rubrics, chunk_size=10)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(c) for c in chunks], [10, 10, 5])

    def test_max_chunk_size_exceeded_raises_error(self) -> None:
        """Score chunk with > 10 rubrics must raise ValueError."""
        rubrics = [
            ConceptRubric(name=f"C{i}", definition="def", scale_1="1", scale_3="3", scale_5="5")
            for i in range(11)
        ]
        dummy_img = Image.new("RGB", (512, 288))
        with self.assertRaises(ValueError):
            self.agent.score_chunk(dummy_img, rubrics)

    def test_scorer_clamps_and_cleans_integer_scores(self) -> None:
        """ScorerAgent must clamp float/out-of-bounds scores to integer [0, 5]."""
        rubrics = [
            ConceptRubric(name="Porosity", definition="Void ratio", scale_1="1", scale_3="3", scale_5="5"),
            ConceptRubric(name="Tortuosity", definition="Curvature", scale_1="1", scale_3="3", scale_5="5"),
            ConceptRubric(name="Roughness", definition="Relief", scale_1="1", scale_3="3", scale_5="5"),
        ]

        # Model returns out-of-range float 12.5, negative -2, and valid 3.1
        self.client.generate_content.return_value = json.dumps({
            "scores": {
                "porosity": 12.5,   # should clamp to 5
                "tortuosity": -2,    # should clamp to 0
                "roughness": 3.1,    # should round to 3
            }
        })
        self.client.extract_clean_json.side_effect = ResilientVertexClient.extract_clean_json
        self.client.encode_image.return_value = "fake_b64"

        dummy_img = Image.new("RGB", (512, 288))
        scores = self.agent.score_chunk(dummy_img, rubrics)

        self.assertEqual(scores["porosity"], 5)
        self.assertEqual(scores["tortuosity"], 0)
        self.assertEqual(scores["roughness"], 3)

        # Verify thinking budget configuration is 0 and temperature is 0.0 for fast deterministic scoring
        call_kwargs = self.client.generate_content.call_args[1]
        self.assertEqual(call_kwargs.get("thinking_budget"), 0)
        self.assertEqual(call_kwargs.get("temperature"), 0.0)

    def test_score_images_returns_correct_shape_and_dtype(self) -> None:
        """score_images must return numpy array of shape (N, K) and dtype int64."""
        rubrics = [
            ConceptRubric(name=f"Concept_{i}", definition="def", scale_1="1", scale_5="5", scale_10="10")
            for i in range(12)  # 12 concepts -> 2 chunks (10, 2)
        ]
        images = [Image.new("RGB", (512, 288)) for _ in range(4)]

        # Mock score_single_image
        with patch.object(self.agent, "score_single_image") as mock_single:
            mock_single.return_value = {r.concept_id: 6 for r in rubrics}
            result_matrix = self.agent.score_images(images, rubrics, max_workers=2)

        self.assertEqual(result_matrix.shape, (4, 12))
        self.assertEqual(result_matrix.dtype, np.int64)
        self.assertTrue(np.all(result_matrix == 6))


class TestVisualMergerAgent(unittest.TestCase):
    """Unit tests for VisualMergerAgent collision adjudication."""

    def setUp(self) -> None:
        self.client = MagicMock()
        self.agent = VisualMergerAgent(client=self.client)

    def test_evaluate_collision_merge_decision(self) -> None:
        """Collinear redundant pair must parse MERGE decision and unified rubric."""
        rubric_a = ConceptRubric(
            name="Pore Area Fraction",
            definition="2D area occupied by visible voids.",
            scale_1="Solid matrix with no void area.",
            scale_5="Intermediate void coverage.",
            scale_10="Extensive void coverage across micrograph.",
        )
        rubric_b = ConceptRubric(
            name="Interconnected Void Density",
            definition="Perceived network of porous cellular open channels.",
            scale_1="Zero void channels.",
            scale_5="Moderate channel network.",
            scale_10="Continuous cellular sponge network.",
        )

        mock_response = json.dumps({
            "action": "MERGE",
            "rationale": "In 2D projection, void area and interconnected channel density capture identical visual signals.",
            "merged_rubric": {
                "name": "Open Macroporosity Fraction",
                "definition": "Visual fraction of open porous network in 2D SEM projection.",
                "scale_1": "Dense solid matrix with minimal void space (<5%).",
                "scale_5": "Porous network occupying 30-50% area.",
                "scale_10": "High macroporosity (>75%) sponge matrix.",
            },
        })
        self.client.generate_content.return_value = mock_response
        self.client.extract_clean_json.side_effect = ResilientVertexClient.extract_clean_json
        self.client.encode_image.return_value = "fake_b64"

        dummy_imgs = [Image.new("RGB", (512, 288)) for _ in range(2)]
        decision = self.agent.evaluate_collision(rubric_a, rubric_b, dummy_imgs, correlation_r=0.88)

        self.assertIsInstance(decision, MergeDecision)
        self.assertEqual(decision.action, "MERGE")
        self.assertIn("identical visual signals", decision.rationale)
        self.assertIsNotNone(decision.merged_rubric)
        self.assertEqual(decision.merged_rubric.name, "Open Macroporosity Fraction")
        self.assertEqual(decision["action"], "MERGE")

    def test_evaluate_collision_keep_both_decision(self) -> None:
        """Collinear but physically distinct pair must parse KEEP_BOTH decision."""
        rubric_a = ConceptRubric(
            name="Fiber Diameter",
            definition="Thickness of individual cylindrical filaments.",
            scale_1="Extremely fine sub-100nm filaments.",
            scale_5="Intermediate ~500nm filaments.",
            scale_10="Coarse multi-micron fibers.",
        )
        rubric_b = ConceptRubric(
            name="Fiber Alignment Degree",
            definition="Unidirectional orientation of filament axes.",
            scale_1="Randomly tangled non-woven felt.",
            scale_5="Semi-aligned directional orientation.",
            scale_10="Perfect parallel array alignment.",
        )

        mock_response = json.dumps({
            "action": "KEEP_BOTH",
            "rationale": "Diameter and alignment represent independent geometric axes that can vary independently.",
            "merged_rubric": None,
        })
        self.client.generate_content.return_value = mock_response
        self.client.extract_clean_json.side_effect = ResilientVertexClient.extract_clean_json
        self.client.encode_image.return_value = "fake_b64"

        dummy_imgs = [Image.new("RGB", (512, 288)) for _ in range(2)]
        decision = self.agent.evaluate_collision(rubric_a, rubric_b, dummy_imgs, correlation_r=0.82)

        self.assertEqual(decision.action, "KEEP_BOTH")
        self.assertIsNone(decision.merged_rubric)

    def test_merger_raises_on_empty_images(self) -> None:
        """VisualMergerAgent must raise ValueError if no micrographs are provided."""
        ra = ConceptRubric("A", "def", "1", "5", "10")
        rb = ConceptRubric("B", "def", "1", "5", "10")
        with self.assertRaises(ValueError):
            self.agent.evaluate_collision(ra, rb, images=[])


class TestLiveIntegrationSmokeTest(unittest.TestCase):
    """Smoke test against live Vertex AI Express API if credentials exist in environment."""

    def setUp(self) -> None:
        self.api_key = os.environ.get("VERTEX_API_KEY")
        self.project_id = os.environ.get("VERTEX_PROJECT_ID")
        if not self.api_key or not self.project_id:
            self.skipTest("VERTEX_API_KEY or VERTEX_PROJECT_ID not set; skipping live test.")

    def test_live_vertex_generate_content(self) -> None:
        """Verify real HTTP 200 connectivity to Gemini 3.8 Flash via Vertex AI Express."""
        client = ResilientVertexClient(
            api_key=self.api_key,
            project_id=self.project_id,
            max_retries=3,
            timeout=30.0,
        )
        try:
            text = client.generate_content(
                contents=[{"role": "user", "parts": [{"text": "Return JSON: {\"status\": \"ok\", \"ping\": \"pong\"}"}]}],
                system_instruction="You are a helpful assistant. Reply strictly in JSON.",
                thinking_budget=0,
                max_output_tokens=100,
                temperature=0.0,
                response_mime_type="application/json",
            )
        except RuntimeError as e:
            if "403" in str(e) or "PERMISSION_DENIED" in str(e):
                self.skipTest(f"Vertex API key lacked permission or expired: {e}")
            raise
        parsed = ResilientVertexClient.extract_clean_json(text)
        self.assertEqual(parsed.get("status"), "ok")
        self.assertEqual(parsed.get("ping"), "pong")

    def test_live_scorer_agent_smoke(self) -> None:
        """Verify ScorerAgent against live API with 1 preprocessed image and 2 rubrics."""
        client = ResilientVertexClient(
            api_key=self.api_key,
            project_id=self.project_id,
            timeout=45.0,
        )
        scorer = ScorerAgent(client=client)

        # Create a synthetic 512x288 micrograph
        img = Image.new("RGB", (512, 288), color=(128, 128, 128))
        rubrics = [
            ConceptRubric(
                name="Overall Porosity",
                definition="Fraction of dark open void area.",
                scale_1="Sparse void.",
                scale_3="Moderate void fraction.",
                scale_5="Extensive open sponge network.",
            ),
            ConceptRubric(
                name="Structural Tortuosity",
                definition="Degree of curvature and looping.",
                scale_1="Nearly straight linear elements.",
                scale_3="Gentle wave-like curves.",
                scale_5="Heavily coiled loops.",
            ),
        ]

        try:
            scores = scorer.score_chunk(img, rubrics)
        except RuntimeError as e:
            if "403" in str(e) or "PERMISSION_DENIED" in str(e):
                self.skipTest(f"Vertex API key lacked permission or expired: {e}")
            raise
        self.assertIn("overall_porosity", scores)
        self.assertIn("structural_tortuosity", scores)
        self.assertGreaterEqual(scores["overall_porosity"], 0)
        self.assertLessEqual(scores["overall_porosity"], 5)
        self.assertGreaterEqual(scores["structural_tortuosity"], 0)
        self.assertLessEqual(scores["structural_tortuosity"], 5)


if __name__ == "__main__":
    unittest.main()
