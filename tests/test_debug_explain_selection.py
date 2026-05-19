import os
import unittest
from pathlib import Path
from unittest.mock import patch


TEXT_PAYLOAD = {
    "document_id": "doc",
    "highlight_id": "text-1",
    "highlight_type": "text",
    "page_number": 3,
    "selected_text": "The selected passage describes hearing difficulty moments.",
    "text_available": True,
    "recommended_llm_mode": "text_context",
    "matched_block": {
        "block_id": "p3-b2",
        "block_type": "text",
        "markdown_content": "The selected passage describes hearing difficulty moments in conversation.",
    },
    "nearby_useful_context": [
        {"block_type": "text", "markdown_content": "Nearby useful context."},
    ],
    "crop_image_data_url": "",
}


AREA_PAYLOAD = {
    "document_id": "doc",
    "highlight_id": "area-1",
    "highlight_type": "area",
    "page_number": 3,
    "selected_text": "",
    "text_available": False,
    "reason": "area_selection",
    "recommended_llm_mode": "image_plus_context",
    "caption": "Figure 2. Hearing difficulty moments over time.",
    "selected_caption": {
        "block_id": "doc-p3-b2",
        "page_number": 3,
        "markdown_content": "Figure 2. Hearing difficulty moments over time.",
        "score": 0.91,
    },
    "caption_confidence": "high",
    "candidate_captions": [
        {
            "block_id": "doc-p3-b2",
            "page_number": 3,
            "markdown_content": "Figure 2. Hearing difficulty moments over time.",
            "horizontal_overlap": 0.92,
            "vertical_distance": 0.02,
            "relation": "below",
            "score": 0.91,
        }
    ],
    "nearby_useful_context": [
        {"block_type": "caption", "markdown_content": "Figure 2. Hearing difficulty moments over time."},
    ],
    "crop_image_data_url": "data:image/png;base64,QUJDRA==",
}


RETRIEVAL_CONTEXT = {
    "paper_profile": {
        "title": "Identifying Hearing Difficulty Moments in Conversational Audio",
        "one_sentence_summary": "This paper studies hearing difficulty moments in conversational audio.",
        "research_problem": "The paper asks when listeners experience communication breakdowns.",
        "method_summary": "Participants identified difficult moments in conversational recordings.",
        "dataset_or_materials": "Conversational audio recordings.",
        "main_findings": "Difficulty moments cluster around overlapping speech.",
        "key_terms": ["hearing", "difficulty", "conversation"],
        "section_map": [{"heading": "Results", "page_number": 3}],
    },
    "matched_block": {
        "block_id": "doc-p3-b1",
        "block_type": "text",
        "page_number": 3,
        "markdown_content": "The selected passage describes hearing difficulty moments in conversation.",
        "match_score": 0.92,
        "coordinate_overlap": 0.88,
    },
    "nearby_context": [
        {"block_id": "doc-p3-b2", "block_type": "caption", "page_number": 3, "markdown_content": "Figure 2. Hearing difficulty moments over time."},
        {"block_id": "doc-p3-hdr", "block_type": "text", "page_number": 3, "markdown_content": "Trends in Hearing 30(0)"},
        {"block_id": "doc-p3-chi", "block_type": "text", "page_number": 3, "markdown_content": "CHI \u201924, May 11\u201316, 2024, Honolulu, HI, USA"},
    ],
    "same_section_context": [
        {"block_id": "doc-p3-h", "block_type": "title", "page_number": 3, "markdown_content": "Results"},
    ],
    "related_blocks": [
        {"block_id": "doc-p1-b1", "block_type": "text", "page_number": 1, "markdown_content": "Abstract This paper studies hearing difficulty moments in conversational audio."},
    ],
    "global_rag_context": [
        {"block_id": "doc-p2-b1", "block_type": "text", "page_number": 2, "markdown_content": "Overlapping speech appears as a related difficulty cue.", "score": 0.77},
    ],
    "retrieval_strategy": "coordinate_plus_keyword",
    "retrieval_method": "keyword",
}


class DebugExplainSelectionTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {}, clear=True)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_debug_explain_selection_uses_mock_provider_for_text_payload(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        app.state.retrieve_context = lambda data: RETRIEVAL_CONTEXT
        response = app.test_request("POST", "/api/debug/explain-selection", TEXT_PAYLOAD)

        self.assertEqual(response["status"], 200)
        payload = response["json"]
        self.assertEqual(payload["provider"], "mock")
        self.assertEqual(payload["model"], "mock")
        self.assertEqual(payload["mode"], "text_context")
        self.assertFalse(payload["used_image"])
        self.assertTrue(payload["paper_profile_used"])
        self.assertEqual(payload["retrieved_block_count"], 5)
        self.assertIn(f"selected text length {len(TEXT_PAYLOAD['selected_text'])}", payload["answer"])
        self.assertIn("highlight_type: text", payload["prompt_preview"])
        self.assertIn("paper_profile:", payload["prompt_preview"])
        self.assertIn("related_blocks:", payload["prompt_preview"])
        self.assertIsNone(payload["error"])

    def test_debug_explain_selection_uses_mock_provider_for_area_payload(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        app.state.retrieve_context = lambda data: RETRIEVAL_CONTEXT
        response = app.test_request("POST", "/api/debug/explain-selection", AREA_PAYLOAD)

        self.assertEqual(response["status"], 200)
        payload = response["json"]
        self.assertEqual(payload["provider"], "mock")
        self.assertTrue(payload["used_image"])
        self.assertIn("selected area", payload["answer"])
        self.assertIn("crop image available true", payload["answer"])

    def test_missing_gemini_key_falls_back_to_mock_with_warning(self):
        from emotion_aware_assistant.llm.providers import explain_selection

        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_MODEL": "gemini-flash-latest"}, clear=True):
            result = explain_selection(TEXT_PAYLOAD)

        self.assertEqual(result["provider"], "mock")
        self.assertEqual(result["model"], "mock")
        self.assertEqual(result["mode"], "text_context")
        self.assertIn("GEMINI_API_KEY is missing", result["answer"])
        self.assertIsNone(result["error"])

    def test_explain_selection_prompt_includes_paper_profile_and_retrieved_blocks(self):
        from emotion_aware_assistant.llm.providers import explain_selection

        payload = {**TEXT_PAYLOAD, "retrieval_context": RETRIEVAL_CONTEXT}
        result = explain_selection(payload)

        self.assertEqual(result["provider"], "mock")
        self.assertTrue(result["paper_profile_used"])
        self.assertEqual(result["retrieved_block_count"], 5)
        self.assertIn("paper_profile:", result["prompt_preview"])
        self.assertIn("This paper studies hearing difficulty moments", result["prompt_preview"])
        self.assertIn("related_blocks:", result["prompt_preview"])
        self.assertIn("global_rag_context:", result["prompt_preview"])
        self.assertIn("useful_follow_up_question", result["prompt_preview"])
        self.assertNotIn("Trends in Hearing 30(0)", result["prompt_preview"])
        self.assertNotIn("CHI \u201924", result["prompt_preview"])

    def test_explain_selection_prompt_normalizes_selected_text_hyphenation(self):
        from emotion_aware_assistant.llm.providers import explain_selection

        payload = {
            **TEXT_PAYLOAD,
            "selected_text": "This writ-\n ing includes a \ufb01gure reference.",
            "retrieval_context": RETRIEVAL_CONTEXT,
        }
        result = explain_selection(payload)

        self.assertIn("This writing includes a figure reference.", result["prompt_preview"])
        self.assertNotIn("writ-", result["prompt_preview"])
        self.assertNotIn("\ufb01", result["prompt_preview"])

    def test_gemini_payload_strips_png_data_url_prefix_for_area_crop(self):
        from emotion_aware_assistant.llm.providers import build_gemini_request

        prompt, body, used_image = build_gemini_request(AREA_PAYLOAD)

        self.assertTrue(used_image)
        self.assertIn("caption: Figure 2", prompt)
        image_part = body["contents"][0]["parts"][1]["inline_data"]
        self.assertEqual(image_part["mime_type"], "image/png")
        self.assertEqual(image_part["data"], "QUJDRA==")
        self.assertNotIn("data:image/png;base64,", image_part["data"])

    def test_gemini_payload_includes_inline_data_for_area_crop(self):
        from emotion_aware_assistant.llm.providers import build_gemini_request

        _, body, _ = build_gemini_request(AREA_PAYLOAD)

        self.assertIn("inline_data", body["contents"][0]["parts"][1])

    def test_low_confidence_area_caption_prompt_makes_image_primary(self):
        from emotion_aware_assistant.llm.providers import build_gemini_request

        payload = {
            **AREA_PAYLOAD,
            "caption": "Figure 2. Earlier visual result.",
            "selected_caption": {},
            "caption_confidence": "low",
            "candidate_captions": [
                {
                    "block_id": "fig2",
                    "page_number": 4,
                    "markdown_content": "Figure 2. Earlier visual result.",
                    "horizontal_overlap": 0.42,
                    "vertical_distance": 0.02,
                    "relation": "below",
                    "score": 0.61,
                },
                {
                    "block_id": "fig3",
                    "page_number": 4,
                    "markdown_content": "Figure 3. Later visual result.",
                    "horizontal_overlap": 0.40,
                    "vertical_distance": 0.02,
                    "relation": "below",
                    "score": 0.60,
                },
            ],
        }

        prompt, _, used_image = build_gemini_request(payload)

        self.assertTrue(used_image)
        self.assertIn("The crop image is the primary source. Candidate captions may be imperfect.", prompt)
        self.assertIn("candidate_captions:", prompt)
        self.assertIn("Figure 3. Later visual result", prompt)

    def test_gemini_payload_omits_inline_data_for_text_only_highlight(self):
        from emotion_aware_assistant.llm.providers import build_gemini_request

        text_payload = {
            **TEXT_PAYLOAD,
            "crop_image_data_url": "data:image/png;base64,QUJDRA==",
        }
        prompt, body, used_image = build_gemini_request(text_payload)

        self.assertFalse(used_image)
        self.assertEqual(len(body["contents"][0]["parts"]), 1)
        self.assertIn("crop_image_attached: false", prompt)
        self.assertNotIn("inline_data", body["contents"][0]["parts"][0])

    def test_debug_endpoint_does_not_call_old_chat_route(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        with patch.object(app.state, "chat", side_effect=AssertionError("old chat must not be called")):
            response = app.test_request("POST", "/api/debug/explain-selection", TEXT_PAYLOAD)

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["json"]["provider"], "mock")

    def test_frontend_bundle_contains_explain_selection_ui_without_chat_or_key(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx").read_text(encoding="utf-8")

        for required in [
            "Explain Selection",
            "explainSelection",
            'postJson("/api/debug/explain-selection"',
            "explainStatus",
            "explainResult",
            "Provider",
            "Model",
            "Paper profile summary",
            "Retrieved related blocks",
            "paper_profile_summary",
            "retrieved_blocks",
            "Prompt preview",
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "/api/chat",
            "GEMINI_API_KEY",
            "X-goog-api-key",
            "AI" + "za",
        ]:
            self.assertNotIn(forbidden, source)

    def test_api_key_is_not_present_in_static_assets(self):
        paths = [
            "emotion_aware_assistant/web/static/pdf-workspace/pdf-test.js",
            "emotion_aware_assistant/web/static/pdf_test.html",
        ]
        for path in paths:
            content = Path(path).read_text(encoding="utf-8")
            self.assertNotIn("AI" + "za", content)
            self.assertNotIn("GEMINI_API_" + "KEY=", content)


if __name__ == "__main__":
    unittest.main()
