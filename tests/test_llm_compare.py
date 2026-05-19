import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from tests.test_pdf_debug_page import tiny_pdf_bytes


class LlmCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.runtime_dir = Path(self.temp_dir.name) / "runtime_uploads"
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
        self.assertEqual(summary["secondary_state"], "confusion")
        self.assertEqual(summary["reaction_window_duration"], 8.0)
        self.assertEqual(summary["reaction_window_avg_confidence"], 0.85)
        self.assertEqual(
            summary["allowed_strategy_families"],
            self.app.state._allowed_strategy_families_for_support_cue("deepening"),
        )
        self.assertEqual(summary["baseline_explanation_length"], len("The baseline explanation described retrieval before answering."))
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
            "Snapshot selector",
            "stage-filter",
            "No prompt snapshots found.",
            "Run a new explanation in /pdf-chat first:",
            "Open /pdf-chat",
            "Prompt preview",
            "Model selection",
            "Run comparison",
            "Results",
            "Manual evaluation",
            "Strategy planner",
            "strategy_planner",
            "JSON validity",
            "pedagogical strategy quality",
            "context sensitivity",
            "usefulness for next explanation",
            "allowed strategy families",
            "JSON validity:",
            "grounding",
            "clarity",
            "strategy adherence",
            "safety wording",
            "usefulness",
            "Copy JSON",
            "Download JSON",
            "Download Markdown summary",
            "OpenRouter model availability, pricing, and rate limits may change",
            "openai/gpt-5.2",
            "maxTokensForStage",
            "3000",
        ]:
            self.assertIn(required, source)
        self.assertNotIn("localStorage", source)
        self.assertNotIn("sessionStorage", source)


if __name__ == "__main__":
    unittest.main()
