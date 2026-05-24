import json
import os
import tempfile
import threading
import time
import urllib.error
import unittest
import urllib.request
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from tests.test_pdf_debug_page import tiny_pdf_bytes


class LlmCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.runtime_dir = Path(self.temp_dir.name) / "runtime_uploads"
        import emotion_aware_assistant.core.llm_config as llm_config
        self.root_patch = patch.object(llm_config, "PROJECT_ROOT", self.root, create=True)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        from emotion_aware_assistant.web.server import create_web_app

        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.app = create_web_app(force_dummy_llm=True, load_local_env=False)
        self.app.state.upload_dir = self.runtime_dir.resolve()
        self.app.state.documents_dir = self.app.state.upload_dir / "documents"

    def upload_pdf(self):
        response = self.app.test_request(
            "POST",
            "/api/documents/upload",
            files={"file": ("paper.pdf", tiny_pdf_bytes())},
        )
        self.assertEqual(response["status"], 200, response)
        document_id = response["json"]["document_id"]
        self.wait_for_prepared(document_id)
        return document_id

    def wait_for_prepared(self, document_id: str):
        deadline = time.time() + 5
        while time.time() < deadline:
            detail = self.app.test_request("GET", f"/api/documents/{document_id}")["json"]
            status = (detail.get("prepare_status") or {}).get("status")
            if status in {"completed", "failed"}:
                return detail
            time.sleep(0.05)
        self.fail(f"Document {document_id} did not finish preparation.")

    def baseline_explain(self, document_id: str):
        return self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "highlight_id": "h-base",
                "highlight_type": "text",
                "page_number": 1,
                "selected_text": "The method retrieves paper context before answering.",
                "text_available": True,
                "recommended_llm_mode": "text_context",
                "matched_block": {"markdown_content": "The method retrieves paper context before answering."},
                "nearby_useful_context": [{"markdown_content": "Nearby paragraph about retrieval."}],
            },
        )

    def strategy_explain(self, document_id: str):
        strategy = {
            "strategy_id": "step_by_step_breakdown",
            "strategy_family": "step_by_step_breakdown",
            "title": "Break it into steps",
            "pedagogical_move": "Walk through the method one step at a time",
            "context_focus": "retrieval and answer generation",
            "why_recommended": "Clarification cue after the baseline explanation.",
            "prompt_instruction": "Explain the selected passage as ordered steps.",
            "expected_answer_shape": ["Main idea", "Steps", "Why it matters"],
        }
        return self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "highlight_id": "h-strategy",
                "highlight_type": "text",
                "page_number": 1,
                "selected_text": "The method retrieves paper context before answering.",
                "text_available": True,
                "recommended_llm_mode": "text_context",
                "matched_block": {"markdown_content": "The method retrieves paper context before answering."},
                "baseline_explanation": "The baseline explanation described retrieval before answering.",
                "default_task": "explain_current_selection_with_selected_strategy",
                "selected_strategy_id": "step_by_step_breakdown",
                "selected_strategy": strategy,
                "reaction_window_summary": {
                    "support_cue": "sustained_clarification",
                    "duration_sec": 10,
                    "avg_confidence": 0.73,
                },
            },
        )

    def strategy_candidates(self, document_id: str):
        reaction_summary = {
            "source_turn_id": "turn_base",
            "highlight_id": "h-plan",
            "duration_sec": 8.0,
            "dominant_state": "engagement",
            "secondary_state": "confusion",
            "avg_confidence": 0.85,
            "avg_distribution": {"boredom": 0.04, "confusion": 0.18, "engagement": 0.72, "frustration": 0.06},
            "trend": "stable",
            "support_cue": "deepening",
            "support_cue_label": "Deepening cue",
            "trigger_reason": "The baseline explanation was being read while the learning signal showed a deepening cue.",
        }
        matched_block = "The method retrieves paper context before answering."
        selected_text = "The method retrieves paper context before answering."
        return self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/strategy-candidates",
            {
                "highlight_id": "h-plan",
                "source_turn_id": "turn_base",
                "selection_type": "text",
                "page_number": 1,
                "selected_text": selected_text,
                "baseline_explanation": "The baseline explanation described retrieval before answering.",
                "reaction_window_summary": reaction_summary,
                "support_cue": "deepening",
                "paper_context": {
                    "matched_block": {"page_number": 1, "block_type": "paragraph", "markdown_content": matched_block},
                    "nearby_context": [
                        {"page_number": 1, "block_type": "paragraph", "markdown_content": matched_block},
                        {"page_number": 1, "block_type": "paragraph", "markdown_content": "Nearby paragraph about retrieval order."},
                        {"page_number": 2, "block_type": "paragraph", "markdown_content": "Nearby paragraph about answer generation."},
                    ],
                    "retrieved_chunks": [
                        {"page_number": 2, "block_type": "paragraph", "content": matched_block, "score": 0.95},
                        {"page_number": 3, "block_type": "paragraph", "content": "Retrieved RAG chunk about grounded answers.", "score": 0.81},
                    ],
                    "paper_profile": {"summary": "This paper studies retrieval-grounded paper reading support."},
                    "passage_type": "method",
                    "difficulty_hint": "multi_step_process",
                },
                "planner_input_summary": {
                    "recent_conversation_count": 2,
                    "passage_type": "method",
                    "difficulty_hint": "multi_step_process",
                },
                "recent_conversation": [
                    {
                        "role": "assistant",
                        "turn_id": "turn_base",
                        "turn_type": "baseline_explanation",
                        "content": "Baseline explanation.",
                        "context_used": {"retrieved_blocks": [{"content": "large duplicated block"}]},
                        "prompt_preview": "debug prompt preview",
                        "global_rag_context": [{"content": "duplicated global context"}],
                        "learning_state_snapshot": {"face_detection": {"actual_detector": "openface"}},
                        "crop_image_data_url": "data:image/png;base64,AAAA",
                        "api_key": "must-not-save",
                    },
                    {
                        "role": "assistant",
                        "turn_id": "turn_strategy",
                        "turn_type": "strategy_reexplanation",
                        "strategy_id": "deep_technical_explanation",
                        "strategy_family": "deep_technical_explanation",
                        "pedagogical_move": "Deepen the technical explanation",
                        "context_focus": "retrieval and answer generation",
                        "why_recommended": "The deepening cue supported more technical detail.",
                        "content": "Strategy explanation.",
                        "trigger_context": {"debug": True},
                        "planner_input_summary": {"debug": True},
                    },
                ],
                "trigger_context": {"triggered_by": "reaction_window"},
            },
        )

    def test_prompt_snapshot_is_saved_for_baseline_explanation(self):
        document_id = self.upload_pdf()
        response = self.baseline_explain(document_id)

        self.assertEqual(response["status"], 200, response)
        message = response["json"]["assistant_message"]
        snapshot_id = message["prompt_snapshot_id"]
        self.assertEqual(message["turn_type"], "baseline_explanation")

        listed = self.app.test_request("GET", "/api/llm-compare/prompt-snapshots?stage=rag_baseline")
        self.assertEqual(listed["status"], 200, listed)
        self.assertTrue(any(item["snapshot_id"] == snapshot_id for item in listed["json"]["prompt_snapshots"]))

        loaded = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{snapshot_id}")
        snapshot = loaded["json"]["snapshot"]
        serialized = json.dumps(snapshot)
        self.assertEqual(snapshot["stage"], "rag_baseline")
        self.assertEqual(snapshot["document_id"], document_id)
        self.assertEqual(snapshot["highlight_id"], "h-base")
        self.assertEqual(snapshot["messages"][0]["role"], "user")
        self.assertIn("selected_text:", snapshot["messages"][0]["content"])
        self.assertEqual(snapshot["full_prompt_text"], snapshot["prompt_text"])
        self.assertIn("selected_passage", snapshot["variables"])
        self.assertIn("paper_context", snapshot["variables"])
        self.assertTrue(snapshot["redaction"]["api_keys_removed"])
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("GEMINI_API_KEY", serialized)

    def test_prompt_snapshot_is_saved_for_strategy_explanation(self):
        document_id = self.upload_pdf()
        response = self.strategy_explain(document_id)

        self.assertEqual(response["status"], 200, response)
        message = response["json"]["assistant_message"]
        snapshot_id = message["prompt_snapshot_id"]
        self.assertEqual(message["turn_type"], "strategy_reexplanation")
        self.assertEqual(message["strategy_family"], "step_by_step_breakdown")

        loaded = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{snapshot_id}")
        snapshot = loaded["json"]["snapshot"]
        summary = snapshot["context_summary"]
        self.assertEqual(snapshot["stage"], "emotion_strategy")
        self.assertEqual(summary["support_cue"], "sustained_clarification")
        self.assertEqual(summary["selected_strategy_id"], "step_by_step_breakdown")
        self.assertEqual(summary["strategy_family"], "step_by_step_breakdown")
        self.assertEqual(summary["pedagogical_move"], "Walk through the method one step at a time")
        self.assertGreater(summary["baseline_explanation_length"], 0)
        self.assertIn("Selected pedagogical support strategy", snapshot["prompt_text"])
        self.assertEqual(snapshot["full_prompt_text"], snapshot["prompt_text"])
        self.assertIn("selected_strategy", snapshot["variables"])

    def test_strategy_planner_prompt_snapshot_is_saved_with_reaction_summary(self):
        document_id = self.upload_pdf()
        response = self.strategy_candidates(document_id)

        self.assertEqual(response["status"], 200, response)
        snapshot_id = response["json"]["prompt_snapshot_id"]
        listed = self.app.test_request("GET", "/api/llm-compare/prompt-snapshots?stage=strategy_planner")
        loaded = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{snapshot_id}")

        self.assertEqual(listed["status"], 200, listed)
        self.assertTrue(any(item["snapshot_id"] == snapshot_id for item in listed["json"]["prompt_snapshots"]))
        snapshot = loaded["json"]["snapshot"]
        summary = snapshot["context_summary"]
        serialized = json.dumps(snapshot)
        self.assertEqual(snapshot["stage"], "strategy_planner")
        self.assertEqual(snapshot["document_id"], document_id)
        self.assertEqual(snapshot["highlight_id"], "h-plan")
        self.assertEqual(snapshot["source_turn_id"], "turn_base")
        self.assertEqual(summary["support_cue"], "deepening")
        self.assertEqual(summary["dominant_state"], "engagement")
        self.assertEqual(summary["dominant_academic_state"], "engagement")
        self.assertEqual(summary["inferred_process_state"], "engaged_deepening")
        self.assertEqual(summary["recommended_strategy"], "deepen_or_extend")
        self.assertEqual(summary["secondary_state"], "confusion")
        self.assertEqual(summary["reaction_window_duration"], 8.0)
        self.assertEqual(summary["reaction_window_avg_confidence"], 0.85)
        self.assertEqual(
            summary["allowed_strategy_families"],
            self.app.state._allowed_strategy_families_for_support_cue("deepening"),
        )
        self.assertEqual(summary["baseline_explanation_length"], len("The baseline explanation described retrieval before answering."))
        self.assertEqual(snapshot["full_prompt_text"], snapshot["prompt_text"])
        self.assertIn("learning_state", snapshot["variables"])
        self.assertTrue(snapshot["redaction"]["api_keys_removed"])
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("GEMINI_API_KEY", serialized)

    def test_strategy_planner_snapshot_stores_rich_structured_context_without_debug_bloat(self):
        document_id = self.upload_pdf()
        response = self.strategy_candidates(document_id)
        snapshot_id = response["json"]["prompt_snapshot_id"]
        snapshot = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{snapshot_id}")["json"]["snapshot"]
        context = snapshot["strategy_planning_context"]
        serialized = json.dumps(snapshot)

        self.assertEqual(context["selected_evidence"]["selected_text"], "The method retrieves paper context before answering.")
        self.assertEqual(context["previous_explanation"]["baseline_explanation"], "The baseline explanation described retrieval before answering.")
        self.assertEqual(context["reaction_context"]["reaction_window_summary"]["support_cue"], "deepening")
        self.assertEqual(context["reaction_context"]["avg_distribution"]["engagement"], 0.72)
        self.assertEqual(context["academic_state_layer"]["dominant_academic_state"], "engagement")
        self.assertEqual(context["academic_state_layer"]["secondary_academic_state"], "confusion")
        self.assertEqual(context["academic_state_layer"]["academic_state_scores"]["engagement"], 0.72)
        self.assertEqual(context["support_cue_layer"]["support_cue"], "deepening")
        self.assertEqual(context["support_cue_layer"]["inferred_process_state"], "engaged_deepening")
        self.assertIn(context["support_cue_layer"]["recommended_strategy"], context["support_cue_layer"]["allowed_strategy_families"])
        self.assertNotEqual(context["support_cue_layer"]["recommended_strategy"], "deep_technical_explanation")
        self.assertEqual(
            context["strategy_constraints"]["allowed_strategy_families"],
            self.app.state._allowed_strategy_families_for_support_cue("deepening"),
        )
        self.assertIn("retrieves paper context", context["paper_context"]["matched_block"])
        self.assertEqual(len(context["paper_context"]["nearby_context"]), 2)
        self.assertEqual(len(context["paper_context"]["retrieved_rag_chunks"]), 1)
        self.assertEqual(context["paper_context"]["retrieved_rag_chunks"][0]["content"], "Retrieved RAG chunk about grounded answers.")
        self.assertEqual(snapshot["context_summary"]["rag_chunk_count"], 1)
        self.assertEqual(snapshot["context_summary"]["nearby_context_count"], 2)
        self.assertEqual(snapshot["context_summary"]["recent_conversation_count"], 2)
        self.assertIn("strategy_planning_context", snapshot["messages"][1]["content"])
        for forbidden in [
            "context_used",
            "prompt_preview",
            "global_rag_context",
            "learning_state_snapshot",
            "face_detection",
            "trigger_context",
            "planner_input_summary",
            "data:image",
            "must-not-save",
        ]:
            self.assertNotIn(forbidden, serialized)
        recent = context["recent_conversation"]
        self.assertEqual(recent[0]["role"], "assistant")
        self.assertEqual(recent[0]["turn_type"], "baseline_explanation")
        self.assertEqual(recent[0]["content"], "Baseline explanation.")
        self.assertEqual(recent[1]["strategy_family"], "deep_technical_explanation")
        self.assertEqual(recent[1]["pedagogical_move"], "Deepen the technical explanation")

    def test_snapshot_list_empty_message(self):
        response = self.app.test_request("GET", "/api/llm-compare/prompt-snapshots")

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["prompt_snapshots"], [])
        self.assertIn("No prompt snapshots found", response["json"]["message"])

    def test_run_comparison_uses_same_snapshot_messages_and_continues_on_failure(self):
        document_id = self.upload_pdf()
        explain = self.baseline_explain(document_id)
        snapshot_id = explain["json"]["assistant_message"]["prompt_snapshot_id"]
        snapshot = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{snapshot_id}")["json"]["snapshot"]

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Compared model output."}}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            if "fail/model" in request.data.decode("utf-8"):
                raise urllib.error.URLError("simulated failure")
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "router-secret",
                "OPENAI_API_KEY": "compatible-secret",
                "OPENAI_BASE_URL": "http://localhost:11434/v1",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "snapshot_id": snapshot_id,
                    "models": [
                        {"label": "OpenRouter", "provider": "openrouter", "model": "openai/gpt-5.2"},
                        {"label": "Local", "provider": "openai_compatible", "model": "local/model"},
                        {"label": "Failure", "provider": "openrouter", "model": "fail/model"},
                    ],
                },
            )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(len(payload["results"]), 3)
        self.assertTrue(payload["results"][0]["ok"])
        self.assertTrue(payload["results"][1]["ok"])
        self.assertFalse(payload["results"][2]["ok"])
        for request in requests[:2]:
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["messages"], snapshot["messages"])
        self.assertIn("Bearer router-secret", str(requests[0].headers))
        self.assertNotIn("router-secret", serialized)
        self.assertNotIn("compatible-secret", serialized)

    def test_provider_not_configured_returns_per_model_error(self):
        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]

        response = self.app.test_request(
            "POST",
            "/api/llm-compare/run",
            {
                "snapshot_id": snapshot_id,
                "models": [{"label": "OpenRouter", "provider": "openrouter", "model": "openai/gpt-5.2"}],
            },
        )

        result = response["json"]["results"][0]
        self.assertEqual(response["status"], 200, response)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "Provider is not configured.")

    def test_run_comparison_uses_configured_compare_model_slots_when_models_omitted(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        root = Path(self.temp_dir.name)
        secret = "or-compare-secret"
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Comparison slot output"}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch("urllib.request.urlopen", fake_urlopen):
            save = self.app.test_request("POST", "/api/settings/openrouter", {"api_key": secret})
            profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {
                    "id": "configured_compare",
                    "display_name": "Configured Compare",
                    "model_id": "openai/gpt-4o-mini",
                    "enabled": True,
                },
            )
            roles = self.app.test_request(
                "POST",
                "/api/settings/roles",
                {"compare_model_profile_ids": [profile["json"]["model_profile"]["id"]]},
            )
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {"snapshot_id": snapshot_id},
            )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(save["status"], 200, save)
        self.assertEqual(roles["status"], 200, roles)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(len(payload["results"]), 1)
        self.assertEqual(payload["results"][0]["label"], "Configured Compare")
        self.assertEqual(payload["results"][0]["model"], "openai/gpt-4o-mini")
        self.assertTrue(payload["results"][0]["ok"])
        self.assertEqual(json.loads(requests[0].data.decode("utf-8"))["model"], "openai/gpt-4o-mini")
        self.assertNotIn(secret, serialized)

    def test_run_compare_uses_only_selected_prompt_and_shared_generation_parameters(self):
        document_id = self.upload_pdf()
        first = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        second = self.strategy_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Selected prompt output."}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-secret"}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "prompt_id": first,
                    "generation_parameters": {
                        "temperature": 0.4,
                        "top_p": 0.8,
                        "top_k": "",
                        "max_tokens": 700,
                        "timeout": 17,
                    },
                    "models": [
                        {"profile_id": "m1", "label": "Model 1", "provider": "openrouter", "model": "openai/gpt-4o-mini", "enabled": True},
                        {"profile_id": "m2", "label": "Model 2", "provider": "openrouter", "model": "anthropic/claude-opus-4.7-fast", "enabled": True},
                    ],
                },
            )

        payload = response["json"]
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(payload["selected_prompt_id"], first)
        self.assertNotEqual(payload["selected_prompt_id"], second)
        self.assertEqual(len(payload["results"]), 2)
        selected_message = self.app.state.get_llm_prompt_snapshot(first)["snapshot"]["messages"][0]["content"]
        bodies = [json.loads(request.data.decode("utf-8")) for request in requests]
        self.assertEqual({body["messages"][0]["content"] for body in bodies}, {selected_message})
        for body in bodies:
            self.assertEqual(body["temperature"], 0.4)
            self.assertEqual(body["top_p"], 0.8)
            self.assertEqual(body["max_tokens"], 700)
            self.assertNotIn("top_k", body)
        self.assertEqual([result["generation_parameters"]["timeout"] for result in payload["results"]], [17, 17])

    def test_prompt_library_create_edit_duplicate_delete_and_categories(self):
        full_prompt = "Explain dropout in one paragraph.\n\nKeep this complete line for the compare request."
        created = self.app.test_request(
            "POST",
            "/api/llm-compare/prompts",
            {
                "title": "Explain dropout",
                "category": "Technical explanation",
                "subcategory": "draft",
                "description": "Custom prompt for manual comparison.",
                "full_prompt_text": full_prompt,
                "variables": ["selected_passage", "paper_context"],
                "context_fields": {"selected_passage": "Dropout randomly masks activations."},
            },
        )
        prompt_id = created["json"]["prompt"]["id"]
        updated = self.app.test_request(
            "PUT",
            f"/api/llm-compare/prompts/{prompt_id}",
            {"title": "Explain dropout clearly", "category": "Custom", "full_prompt_text": "Explain dropout clearly."},
        )
        duplicated = self.app.test_request("POST", f"/api/llm-compare/prompts/{prompt_id}/duplicate")
        listed = self.app.test_request("GET", "/api/llm-compare/prompts?search=dropout")
        deleted = self.app.test_request("DELETE", f"/api/llm-compare/prompts/{prompt_id}")
        listed_after = self.app.test_request("GET", "/api/llm-compare/prompts?search=dropout")

        self.assertEqual(created["status"], 200, created)
        self.assertEqual(updated["status"], 200, updated)
        self.assertEqual(duplicated["status"], 200, duplicated)
        self.assertEqual(deleted["status"], 200, deleted)
        self.assertEqual(created["json"]["prompt"]["category"], "Custom / Experimental")
        self.assertEqual(created["json"]["prompt"]["subcategory"], "draft")
        self.assertEqual(created["json"]["prompt"]["full_prompt_text"], full_prompt)
        self.assertEqual(created["json"]["prompt"]["variables"], ["selected_passage", "paper_context"])
        self.assertEqual(updated["json"]["prompt"]["category"], "Custom / Experimental")
        self.assertEqual(updated["json"]["prompt"]["full_prompt_text"], "Explain dropout clearly.")
        self.assertNotEqual(duplicated["json"]["prompt"]["id"], prompt_id)
        self.assertEqual(duplicated["json"]["prompt"]["full_prompt_text"], "Explain dropout clearly.")
        self.assertTrue(any(group["category"] == "Custom / Experimental" for group in listed["json"]["categories"]))
        self.assertFalse(any(prompt["id"] == prompt_id for prompt in listed_after["json"]["prompts"]))

    def test_prompt_taxonomy_reclassifies_snapshots_and_planner_subcategories(self):
        document_id = self.upload_pdf()
        base_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        strategy_id = self.strategy_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        planner_id = self.strategy_candidates(document_id)["json"]["prompt_snapshot_id"]

        listed = self.app.test_request("GET", "/api/llm-compare/prompts")
        prompts = {prompt["id"]: prompt for prompt in listed["json"]["prompts"]}
        categories = {group["category"]: group for group in listed["json"]["categories"]}

        self.assertEqual(listed["status"], 200, listed)
        self.assertEqual(prompts[base_id]["category"], "Base Prompt")
        self.assertEqual(prompts[strategy_id]["category"], "Strategy Response Prompt")
        self.assertEqual(prompts[planner_id]["category"], "Planner Prompt")
        self.assertEqual(prompts[planner_id]["subcategory"], "Engagement")
        self.assertIn("Base Prompt", categories)
        self.assertIn("Planner Prompt", categories)
        self.assertIn("Strategy Response Prompt", categories)
        self.assertEqual(categories["Planner Prompt"]["count"], 1)

    def test_run_compare_uses_full_prompt_text_not_preview_text(self):
        full_prompt = "Preview line only.\n\nFULL PROMPT UNIQUE MARKER should be sent to every model."
        created = self.app.test_request(
            "POST",
            "/api/llm-compare/prompts",
            {
                "title": "Full prompt check",
                "category": "Custom / Experimental",
                "description": "Preview line only.",
                "prompt_text": "Preview line only.",
                "full_prompt_text": full_prompt,
            },
        )
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Full prompt output."}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-secret"}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "prompt_id": created["json"]["prompt"]["id"],
                    "models": [{"profile_id": "m1", "label": "Model 1", "provider": "openrouter", "model": "openai/gpt-4o-mini", "enabled": True}],
                },
            )

        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["full_prompt_text"], full_prompt)
        self.assertEqual(body["messages"][0]["content"], full_prompt)
        self.assertIn("FULL PROMPT UNIQUE MARKER", body["messages"][0]["content"])
        self.assertNotEqual(body["messages"][0]["content"], "Preview line only.")

    def test_edited_snapshot_prompt_can_be_deleted_cleanly(self):
        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]

        updated = self.app.test_request(
            "PUT",
            f"/api/llm-compare/prompts/{snapshot_id}",
            {"title": "Edited snapshot prompt", "category": "Custom / Experimental", "full_prompt_text": "Use this edited prompt only."},
        )
        deleted = self.app.test_request("DELETE", f"/api/llm-compare/prompts/{snapshot_id}")
        listed = self.app.test_request("GET", "/api/llm-compare/prompts")

        self.assertEqual(updated["status"], 200, updated)
        self.assertEqual(updated["json"]["prompt"]["source"], "custom")
        self.assertEqual(deleted["status"], 200, deleted)
        self.assertFalse(any(prompt["id"] == snapshot_id for prompt in listed["json"]["prompts"]))

    def test_manual_scores_are_saved_to_evaluation_jsonl(self):
        record = {
            "run_id": "run-manual-1",
            "selected_prompt_id": "prompt-1",
            "selected_prompt_title": "Prompt One",
            "prompt_text": "Explain the selected passage.",
            "model_profile_id": "gemini-profile",
            "model_display_name": "Gemini 2.5 Flash",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "generation_parameters": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 700},
            "response_text": "A clear answer.",
            "latency_ms": 123,
            "status": "success",
            "manual_scores": {
                "clarity": 5,
                "pedagogical_quality": 4,
                "emotional_alignment": 4,
                "structure": 5,
                "cognitive_load": 4,
            },
            "evaluator_notes": "Best response.",
            "overall_preference": True,
            "rank": 1,
        }
        saved = self.app.test_request("POST", "/api/llm-compare/evaluations", record)
        listed = self.app.test_request("GET", "/api/llm-compare/evaluations")
        path = self.runtime_dir / "llm_evaluations" / "evaluations.jsonl"
        line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(listed["status"], 200, listed)
        self.assertEqual(line["manual_scores"]["clarity"], 5)
        self.assertEqual(line["model_id"], "gemini-2.5-flash")
        self.assertEqual(line["model_display_name"], "Gemini 2.5 Flash")
        self.assertEqual(line["selected_prompt_id"], "prompt-1")
        self.assertEqual(line["latency_ms"], 123)
        self.assertEqual(line["response_text"], "A clear answer.")
        self.assertEqual(listed["json"]["evaluations"][0]["rank"], 1)

    def test_run_comparison_uses_mixed_gemini_and_openrouter_profile_slots(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        root = Path(self.temp_dir.name)
        router_secret = "or-mixed-secret"
        gemini_secret = "AI" + "za" + "mixed-gemini-secret"
        stale_gemini_secret = "AI" + "za" + "stale-compare-env"
        requests = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            if "generativelanguage.googleapis.com" in request.full_url:
                return FakeResponse({"candidates": [{"content": {"parts": [{"text": "Gemini compare output"}]}}]})
            return FakeResponse({"choices": [{"message": {"content": "OpenRouter compare output"}, "finish_reason": "stop"}]})

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GEMINI_API_KEY": stale_gemini_secret}, clear=False), patch("urllib.request.urlopen", fake_urlopen):
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": router_secret})
            self.app.test_request("POST", "/api/settings/gemini", {"api_key": gemini_secret})
            router_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {
                    "id": "router_compare",
                    "provider": "openrouter",
                    "display_name": "Router Compare",
                    "model_id": "openai/gpt-4o-mini",
                    "enabled": True,
                },
            )["json"]["model_profile"]["id"]
            gemini_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {
                    "id": "gemini_compare",
                    "provider": "gemini",
                    "display_name": "Gemini Compare",
                    "model_id": "gemini-2.5-flash",
                    "enabled": True,
                },
            )["json"]["model_profile"]["id"]
            roles = self.app.test_request(
                "POST",
                "/api/settings/roles",
                {"compare_model_profile_ids": [router_profile, gemini_profile]},
            )
            response = self.app.test_request("POST", "/api/llm-compare/run", {"snapshot_id": snapshot_id})

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(roles["status"], 200, roles)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual([result["provider"] for result in payload["results"]], ["openrouter", "gemini"])
        self.assertEqual(payload["results"][0]["label"], "Router Compare")
        self.assertEqual(payload["results"][1]["label"], "Gemini Compare")
        self.assertEqual(payload["results"][0]["model"], "openai/gpt-4o-mini")
        self.assertEqual(payload["results"][1]["model"], "gemini-2.5-flash")
        self.assertTrue(all(result["ok"] for result in payload["results"]))
        self.assertEqual(json.loads(requests[0].data.decode("utf-8"))["model"], "openai/gpt-4o-mini")
        gemini_body = json.loads(requests[1].data.decode("utf-8"))
        gemini_headers = {key.lower(): value for key, value in requests[1].header_items()}
        self.assertIn("/models/gemini-2.5-flash:generateContent", requests[1].full_url)
        self.assertEqual(gemini_headers["x-goog-api-key"], gemini_secret)
        self.assertNotEqual(gemini_headers["x-goog-api-key"], stale_gemini_secret)
        self.assertEqual(gemini_body["contents"][0]["parts"][0]["text"], self.app.state._messages_to_prompt_text(self.app.state.get_llm_prompt_snapshot(snapshot_id)["snapshot"]["messages"]))
        self.assertIn("generationConfig", gemini_body)
        self.assertIn("temperature", gemini_body["generationConfig"])
        self.assertIn("topP", gemini_body["generationConfig"])
        self.assertIn("maxOutputTokens", gemini_body["generationConfig"])
        self.assertNotIn("messages", gemini_body)
        self.assertNotIn("top_p", gemini_body)
        self.assertNotIn("top_k", gemini_body)
        self.assertNotIn("max_tokens", gemini_body)
        self.assertNotIn("response_format", gemini_body)
        self.assertNotIn("tools", gemini_body)
        self.assertNotIn(router_secret, serialized)
        self.assertNotIn(gemini_secret, serialized)
        self.assertNotIn(stale_gemini_secret, serialized)

    def test_run_comparison_gemini_model_id_only_uses_direct_generate_content(self):
        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        saved_secret = "AI" + "za" + "saved-compare-gemini"
        stale_secret = "AI" + "za" + "stale-env-gemini"
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "Gemini direct output"}]}, "finishReason": "STOP"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(os.environ, {"GEMINI_API_KEY": stale_secret, "GEMINI_MODEL": "gemini-flash-latest"}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.test_request("POST", "/api/settings/gemini", {"api_key": saved_secret, "model": "gemini-2.5-flash"})
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "snapshot_id": snapshot_id,
                    "generation_parameters": {"temperature": 0.2, "top_p": 0.9, "top_k": "", "max_tokens": 700},
                    "models": [
                        {
                            "profile_id": "gemini_profile",
                            "display_name": "Gemini Flash",
                            "provider": "gemini",
                            "model_id": "gemini-2.5-flash",
                            "enabled": True,
                        }
                    ],
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        body = json.loads(request.data.decode("utf-8"))
        headers = {key.lower(): value for key, value in request.header_items()}
        result = response["json"]["results"][0]
        serialized = json.dumps(response["json"])
        self.assertTrue(result["ok"], result)
        self.assertIn("/models/gemini-2.5-flash:generateContent", request.full_url)
        self.assertEqual(headers["x-goog-api-key"], saved_secret)
        self.assertNotEqual(headers["x-goog-api-key"], stale_secret)
        self.assertIn("contents", body)
        self.assertIn("parts", body["contents"][0])
        self.assertNotIn("messages", body)
        self.assertNotIn("Authorization", dict(request.header_items()))
        self.assertNotIn("topK", body.get("generationConfig", {}))
        self.assertEqual(result["model_id"], "gemini-2.5-flash")
        self.assertEqual(result["model_source"], "model_profile")
        self.assertEqual(result["key_source"], "local_settings")
        self.assertNotIn(saved_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_run_comparison_gemini_failure_returns_safe_google_diagnostics(self):
        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        saved_secret = "AI" + "za" + "diagnostic-gemini"
        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            body = {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": "generationConfig.topK is not supported for this Gemini model.",
                }
            }
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, BytesIO(json.dumps(body).encode("utf-8")))

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AI" + "za" + "stale-diagnostic-gemini"}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.test_request("POST", "/api/settings/gemini", {"api_key": saved_secret, "model": "gemini-2.5-flash"})
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "snapshot_id": snapshot_id,
                    "generation_parameters": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 700},
                    "models": [
                        {
                            "profile_id": "gemini_profile",
                            "display_name": "Gemini Flash",
                            "provider": "gemini",
                            "model": "gemini-2.5-flash",
                            "model_id": "gemini-2.5-flash",
                            "enabled": True,
                        }
                    ],
                },
            )

        result = response["json"]["results"][0]
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertFalse(result["ok"])
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["model_id"], "gemini-2.5-flash")
        self.assertEqual(result["key_source"], "local_settings")
        self.assertEqual(result["model_source"], "model_profile")
        self.assertEqual(result["status_code"], 400)
        self.assertEqual(result["google_error_status"], "INVALID_ARGUMENT")
        self.assertIn("topK", result["google_error_message"])
        self.assertEqual(result["generation_parameters_sent"]["temperature"], 0.2)
        self.assertIn("prompt_chars", result)
        self.assertIn("estimated_prompt_tokens", result)
        self.assertEqual(len(requests), 1)
        self.assertNotIn(saved_secret, serialized)
        self.assertNotIn("stale-diagnostic-gemini", serialized)

    def test_quick_model_check_uses_compare_runner_path(self):
        saved_secret = "AI" + "za" + "quick-check-gemini"
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.test_request("POST", "/api/settings/gemini", {"api_key": saved_secret, "model": "gemini-2.5-flash"})
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/quick-check",
                {
                    "generation_parameters": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 16},
                    "models": [
                        {
                            "profile_id": "gemini_profile",
                            "display_name": "Gemini Flash",
                            "provider": "gemini",
                            "model_id": "gemini-2.5-flash",
                            "enabled": True,
                        }
                    ],
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["prompt_text"], "Reply with exactly: OK")
        self.assertEqual(response["json"]["results"][0]["output"], "OK")
        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "Reply with exactly: OK")
        self.assertNotIn("messages", body)

    def test_quick_model_check_uses_saved_openrouter_compare_profile(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        root = Path(self.temp_dir.name)
        saved_secret = "or-quick-check-openrouter"
        stale_secret = "or-stale-quick-check"
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": stale_secret,
                "OPENROUTER_MODEL": "openai/stale-compare-model",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": saved_secret})
            profile_id = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Quick Router", "model_id": "openai/quick-check-model", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"compare_model_profile_ids": [profile_id]})
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/quick-check",
                {"generation_parameters": {"temperature": 0.1, "top_p": 0.8, "top_k": "", "max_tokens": 16}},
            )

        body = json.loads(requests[0].data.decode("utf-8"))
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        result = response["json"]["results"][0]
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(body["model"], "openai/quick-check-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "Reply with exactly: OK"}])
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["top_p"], 0.8)
        self.assertEqual(body["max_tokens"], 16)
        self.assertNotIn("top_k", body)
        self.assertEqual(headers["authorization"], f"Bearer {saved_secret}")
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["model_id"], "openai/quick-check-model")
        self.assertEqual(result["model_source"], "model_profile")
        self.assertEqual(result["key_source"], "local_settings")
        self.assertEqual(result["status_code"], 200)
        self.assertTrue(result["ok"])
        self.assertNotIn(saved_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_openrouter_compare_failure_returns_safe_diagnostics(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        root = Path(self.temp_dir.name)
        saved_secret = "or-diagnostic-openrouter"
        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            body = {"error": {"message": f"Model unavailable for key {saved_secret}"}}
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(json.dumps(body).encode("utf-8")))

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": saved_secret})
            profile_id = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Broken Router", "model_id": "openai/broken-model", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"compare_model_profile_ids": [profile_id]})
            response = self.app.test_request("POST", "/api/llm-compare/run", {"snapshot_id": snapshot_id})

        result = response["json"]["results"][0]
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertFalse(result["ok"])
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["model_id"], "openai/broken-model")
        self.assertEqual(result["key_source"], "local_settings")
        self.assertEqual(result["model_source"], "model_profile")
        self.assertEqual(result["status_code"], 404)
        self.assertIn("[redacted]", result["provider_error_message"])
        self.assertEqual(result["generation_parameters_sent"]["temperature"], 0.2)
        self.assertIn("prompt_chars", result)
        self.assertIn("estimated_prompt_tokens", result)
        self.assertNotIn(saved_secret, serialized)

    def test_disabled_openrouter_compare_profile_is_not_used(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        root = Path(self.temp_dir.name)
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Enabled output"}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": "or-disabled-filter"})
            disabled = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Disabled Router", "model_id": "openai/disabled-model", "enabled": False},
            )["json"]["model_profile"]["id"]
            enabled = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Enabled Router", "model_id": "openai/enabled-model", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"compare_model_profile_ids": [disabled, enabled]})
            response = self.app.test_request("POST", "/api/llm-compare/run", {"snapshot_id": snapshot_id})

        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(response["json"]["results"]), 1)
        self.assertEqual(body["model"], "openai/enabled-model")
        self.assertEqual(response["json"]["results"][0]["model_id"], "openai/enabled-model")

    def test_run_comparison_uses_new_saved_openrouter_key_without_restart(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]
        root = Path(self.temp_dir.name)
        stale_secret = "or-stale-env-compare"
        old_secret = "or-old-settings-compare"
        new_secret = "or-new-settings-compare"
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Fresh settings output"}, "finish_reason": "stop"}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"OPENROUTER_API_KEY": stale_secret}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": old_secret})
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": new_secret})
            profile_id = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Fresh Router", "model_id": "openai/gpt-4o-mini", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"compare_model_profile_ids": [profile_id]})
            response = self.app.test_request("POST", "/api/llm-compare/run", {"snapshot_id": snapshot_id})

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(headers["authorization"], f"Bearer {new_secret}")
        self.assertNotEqual(headers["authorization"], f"Bearer {old_secret}")
        self.assertNotEqual(headers["authorization"], f"Bearer {stale_secret}")
        self.assertEqual(response["json"]["results"][0]["key_source"], "local_settings")
        self.assertNotIn(old_secret, serialized)
        self.assertNotIn(new_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_strategy_planner_comparison_runs_json_checks_and_save_keeps_parsed_json(self):
        document_id = self.upload_pdf()
        snapshot_id = self.strategy_candidates(document_id)["json"]["prompt_snapshot_id"]
        planner_output = {
            "candidates": [
                {
                    "strategy_id": "deep_technical_explanation",
                    "strategy_family": "deep_technical_explanation",
                    "pedagogical_move": "Deepen the technical explanation",
                    "context_focus": "retrieval and answer generation",
                    "title": "Deepen the technical explanation",
                    "short_description": "Add technical detail grounded in the passage.",
                    "why_recommended": "Do not say you are confused; use the deepening cue.",
                    "prompt_instruction": "Explain the technical mechanism.",
                    "expected_answer_shape": ["Mechanism", "Assumptions", "Implications"],
                    "recommended": True,
                    "recommended_score": 0.91,
                },
                {
                    "strategy_id": "critique_assumptions",
                    "strategy_family": "critique_assumptions",
                    "pedagogical_move": "Critique the core assumption",
                    "context_focus": "retrieval grounding",
                    "title": "Critique the core assumption",
                    "short_description": "Inspect the assumption behind retrieval grounding.",
                    "why_recommended": "It supports deeper reading.",
                    "prompt_instruction": "Critique the assumption carefully.",
                    "expected_answer_shape": ["Assumption", "Evidence", "Implication"],
                    "recommended": False,
                    "recommended_score": 0.72,
                },
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": f"```json\n{json.dumps(planner_output)}\n```", "finish_reason": "stop"}}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-secret"}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            response = self.app.test_request(
                "POST",
                "/api/llm-compare/run",
                {
                    "snapshot_id": snapshot_id,
                    "models": [{"label": "Planner", "provider": "openrouter", "model": "openai/gpt-5.2"}],
                },
            )

        result = response["json"]["results"][0]
        checks = result["auto_checks"]
        request_body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(response["status"], 200, response)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(request_body["max_tokens"], 2500)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(checks["finish_reason"], "stop")
        self.assertTrue(checks["json_valid"])
        self.assertTrue(checks["has_candidates"])
        self.assertEqual(checks["candidate_count"], 2)
        self.assertTrue(checks["exactly_one_recommended"])
        self.assertTrue(checks["required_fields_present"])
        self.assertTrue(checks["allowed_strategy_family"])
        self.assertIn("you are confused", checks["unsafe_affect_phrases_found"])
        self.assertTrue(checks["topic_title_warning"])

        saved = self.app.test_request(
            "POST",
            "/api/llm-compare/save",
            {
                "comparison_id": "planner-comparison",
                "snapshot_id": snapshot_id,
                "stage": "strategy_planner",
                "prompt_summary": {"allowed_strategy_families": ["deep_technical_explanation", "critique_assumptions"]},
                "models": [{"label": "Planner", "provider": "openrouter", "model": "openai/gpt-5.2"}],
                "results": [result],
                "manual_scores": {"Planner": {"json_validity": 5}},
            },
        )
        saved_result = saved["json"]["comparison"]["results"][0]
        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(saved_result["parsed_json"]["candidates"][0]["strategy_family"], "deep_technical_explanation")
        self.assertEqual(saved["json"]["comparison"]["allowed_strategy_families"], ["deep_technical_explanation", "critique_assumptions"])

    def test_comparison_save_list_get_omits_keys(self):
        document_id = self.upload_pdf()
        snapshot_id = self.baseline_explain(document_id)["json"]["assistant_message"]["prompt_snapshot_id"]

        saved = self.app.test_request(
            "POST",
            "/api/llm-compare/save",
            {
                "comparison_id": "comparison-test",
                "snapshot_id": snapshot_id,
                "stage": "rag_baseline",
                "models": [{"label": "OpenRouter", "provider": "openrouter", "model": "openai/gpt-5.2"}],
                "results": [{"label": "OpenRouter", "output": "Answer", "api_key": "must-not-save"}],
                "manual_scores": {"OpenRouter": {"grounding": 5}},
                "notes": "Useful comparison.",
            },
        )
        listed = self.app.test_request("GET", "/api/llm-compare/list")
        loaded = self.app.test_request("GET", "/api/llm-compare/comparison-test")

        path = self.runtime_dir / "llm_comparisons" / "comparison-test.json"
        text = path.read_text(encoding="utf-8")
        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(listed["json"]["comparisons"][0]["comparison_id"], "comparison-test")
        self.assertEqual(loaded["json"]["comparison"]["notes"], "Useful comparison.")
        self.assertNotIn("api_key", text)
        self.assertNotIn("must-not-save", text)

    def test_llm_compare_route_serves_static_page(self):
        import emotion_aware_assistant.web.server as server

        app = self.app

        class Handler(server.WebRequestHandler):
            web_app = app

        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address
            with urllib.request.urlopen(f"http://{host}:{port}/llm-compare", timeout=5) as response:
                body = response.read().decode("utf-8")
                status = response.status
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        self.assertEqual(status, 200)
        self.assertIn("Prompt-based LLM Compare", body)

    def test_llm_compare_page_source_has_required_controls_without_browser_key_storage(self):
        page = Path("emotion_aware_assistant/web/static/llm_compare.html")
        source = page.read_text(encoding="utf-8")

        for required in [
            "Manual LLM Evaluation Workspace",
            "Prompt Directory",
            "Total prompts",
            "Expand all",
            "Collapse all",
            "Base Prompt",
            "Planner Prompt",
            "Strategy Response Prompt",
            "Custom / Experimental",
            "Prompt Library",
            "Compare Run",
            "Results & Scoring",
            "Saved Evaluations",
            "Create prompt",
            "Duplicate prompt",
            "Edit prompt",
            "Delete prompt",
            "Run compare with this prompt",
            "Quick model check",
            "selected-prompt-panel",
            "Model selection",
            "Generation settings",
            "temperature controls randomness",
            "top_p controls nucleus sampling",
            "top_k restricts sampling to the top k tokens",
            "Clarity",
            "Pedagogical quality",
            "Emotional alignment",
            "Cognitive load",
            "Export current run as JSON",
            "Export saved evaluations as CSV",
            "/api/settings/llm",
            "/api/llm-compare/prompts",
            "/api/llm-compare/quick-check",
            "/api/llm-compare/evaluations",
        ]:
            self.assertIn(required, source)
        self.assertNotIn("OpenAI-compatible", source)
        self.assertNotIn("openai/gpt-5.2", source)
        self.assertNotIn("sessionStorage", source)
        self.assertNotIn("apiKey", source)
        self.assertNotIn("API key", source)

    def test_llm_compare_page_preserves_prompt_category_collapse_state(self):
        page = Path("emotion_aware_assistant/web/static/llm_compare.html")
        source = page.read_text(encoding="utf-8")

        for required in [
            "EXPANDED_CATEGORIES_KEY",
            "llmCompareExpandedCategories",
            "expandedCategories: loadExpandedCategories()",
            "saveExpandedCategories",
            "isCategoryExpanded",
            "setCategoryExpanded",
            "data-category-id",
            "promptLibrary.addEventListener(\"toggle\"",
            "renderPromptDirectory",
            "jumpToCategory",
            "expand-all-categories",
            "collapse-all-categories",
            "${isCategoryExpanded(group.id) ? \"open\" : \"\"}",
        ]:
            self.assertIn(required, source)
        self.assertNotIn("<details open>", source)


if __name__ == "__main__":
    unittest.main()
