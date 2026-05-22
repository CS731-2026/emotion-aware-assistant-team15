import base64
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from tests.test_pdf_debug_page import tiny_pdf_bytes


class PdfChatBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.runtime_dir = self.root / "runtime_uploads"
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.llm_root_patch = patch.object(llm_config, "PROJECT_ROOT", self.root, create=True)
        self.state_root_patch = patch.object(state_module, "PROJECT_ROOT", self.root, create=True)
        self.llm_root_patch.start()
        self.state_root_patch.start()
        self.addCleanup(self.llm_root_patch.stop)
        self.addCleanup(self.state_root_patch.stop)
        from emotion_aware_assistant.web.server import create_web_app

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
        payload = response["json"]
        detail = self.wait_for_prepared(payload["document_id"])
        payload["meta"] = detail["meta"]
        payload["prepare_status"] = detail["prepare_status"]
        payload["retrieval_method"] = detail["meta"].get("retrieval_method")
        return payload

    def wait_for_prepared(self, document_id: str):
        deadline = time.time() + 5
        last = None
        while time.time() < deadline:
            last = self.app.test_request("GET", f"/api/documents/{document_id}")["json"]
            status = (last.get("prepare_status") or {}).get("status")
            if status in {"completed", "failed"}:
                return last
            time.sleep(0.05)
        self.fail(f"Document {document_id} did not finish preparation: {last}")

    def solid_image_data_url(self, size=(96, 72), color=(20, 80, 150)) -> str:
        from PIL import Image  # type: ignore

        image = Image.new("RGB", size, color)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def live_http_json_request(self, method: str, path: str, body: dict | None = None) -> dict:
        from emotion_aware_assistant.web.server import WebRequestHandler

        app = self.app

        class Handler(WebRequestHandler):
            web_app = app

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        try:
            raw_body = json.dumps(body or {}).encode("utf-8") if body is not None else b""
            headers = {"Content-Type": "application/json"} if body is not None else {}
            connection.request(method, path, body=raw_body, headers=headers)
            response = connection.getresponse()
            payload = response.read().decode("utf-8")
            return {
                "status": response.status,
                "json": json.loads(payload or "{}"),
            }
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_document_upload_creates_prepared_library_document(self):
        payload = self.upload_pdf()
        document_id = payload["document_id"]
        document_dir = self.app.state.documents_dir / document_id

        self.assertTrue((document_dir / "original.pdf").exists())
        self.assertTrue((document_dir / "meta.json").exists())
        self.assertTrue((document_dir / "parsed" / "document.md").exists())
        self.assertTrue((document_dir / "parsed" / "blocks_index.json").exists())
        self.assertTrue((document_dir / "rag" / "paper_profile.json").exists())
        self.assertTrue((document_dir / "rag" / "keyword_index.json").exists())
        self.assertTrue((document_dir / "rag" / "prepare_status.json").exists())
        self.assertIn(payload["retrieval_method"], {"embedding", "keyword", "unknown"})
        self.assertEqual(payload["meta"]["document_id"], document_id)
        self.assertEqual(payload["meta"]["file_name"], "paper.pdf")
        self.assertEqual(payload["meta"]["uploaded_from"], "pdf_chat")
        self.assertTrue(payload["meta"]["library_visible"])
        self.assertGreaterEqual(payload["meta"]["page_count"], 1)

        meta = json.loads((document_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["uploaded_from"], "pdf_chat")
        self.assertTrue(meta["library_visible"])

    def test_speech_transcribe_reports_unavailable_when_faster_whisper_is_missing(self):
        with patch.dict(sys.modules, {"faster_whisper": None}):
            response = self.app.test_request(
                "POST",
                "/api/speech/transcribe",
                files={"audio": ("question.webm", b"\x1a\x45\xdf\xa3webm-audio")},
            )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertFalse(payload["available"])
        self.assertEqual(payload["text"], "")
        self.assertEqual(payload["error"], "Speech transcription unavailable")
        self.assertIn("faster-whisper", payload["message"])

    def test_speech_transcribe_uses_faster_whisper_for_uploaded_audio(self):
        calls = []

        class FakeWhisperModel:
            def __init__(self, model_name, device="cpu", compute_type="int8"):
                calls.append(("init", model_name, device, compute_type))

            def transcribe(self, audio_path):
                calls.append(("transcribe", Path(audio_path).suffix, Path(audio_path).exists()))
                segment = types.SimpleNamespace(text=" What is the main claim? ")
                return [segment], {"language": "en"}

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}), patch.dict(os.environ, {"SPEECH_WHISPER_MODEL": "tiny"}, clear=False):
            response = self.app.test_request(
                "POST",
                "/api/speech/transcribe",
                files={"audio": ("question.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ")},
            )

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["text"], "What is the main claim?")
        self.assertTrue(response["json"]["available"])
        self.assertEqual(calls[0], ("init", "tiny", "cpu", "int8"))
        self.assertEqual(calls[1][0], "transcribe")
        self.assertEqual(calls[1][1], ".wav")
        self.assertTrue(calls[1][2])

    def test_speech_transcribe_rejects_non_audio_uploads(self):
        response = self.app.test_request(
            "POST",
            "/api/speech/transcribe",
            files={"audio": ("notes.txt", b"not audio")},
        )

        self.assertEqual(response["status"], 400)
        self.assertIn("audio/webm or wav", response["json"]["error"])

    def test_fresh_upload_creates_persistence_subdirectories_and_matching_meta(self):
        payload = self.upload_pdf()
        document_id = payload["document_id"]
        document_dir = self.app.state.documents_dir / document_id
        meta = json.loads((document_dir / "meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["document_id"], document_id)
        for relative in [
            "highlights",
            "highlights/crops",
            "threads",
            "prompt_snapshots",
            "logs",
        ]:
            self.assertTrue((document_dir / relative).is_dir(), relative)

    def test_archive_document_hides_from_library_without_deleting_files(self):
        payload = self.upload_pdf()
        document_id = payload["document_id"]
        document_dir = self.app.state.documents_dir / document_id
        original = document_dir / "original.pdf"
        rag_status = document_dir / "rag" / "prepare_status.json"

        response = self.app.test_request("POST", f"/api/documents/{document_id}/archive")
        listed = self.app.test_request("GET", "/api/documents")
        debug_listed = self.app.test_request("GET", "/api/documents?show_debug_docs=1")
        meta = json.loads((document_dir / "meta.json").read_text(encoding="utf-8"))

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["archived"])
        self.assertFalse(meta["library_visible"])
        self.assertIn("archived_at", meta)
        self.assertFalse(any(item["document_id"] == document_id for item in listed["json"]["documents"]))
        self.assertTrue(any(item["document_id"] == document_id for item in debug_listed["json"]["documents"]))
        self.assertTrue(original.exists())
        self.assertTrue(rag_status.exists())

        missing = self.app.test_request("POST", "/api/documents/missing-doc/archive")
        self.assertEqual(missing["status"], 404)

    def test_documents_list_is_library_only_by_default_and_can_show_debug_docs(self):
        hidden_id = "debug-hidden-doc"
        hidden_dir = self.app.state.documents_dir / hidden_id
        hidden_dir.mkdir(parents=True)
        (hidden_dir / "original.pdf").write_bytes(tiny_pdf_bytes())
        (hidden_dir / "meta.json").write_text(
            json.dumps({"document_id": hidden_id, "file_name": "debug.pdf", "title": "Debug PDF"}),
            encoding="utf-8",
        )
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        library = self.app.test_request("GET", "/api/documents")
        debug = self.app.test_request("GET", "/api/documents?show_debug_docs=1")

        self.assertEqual(library["status"], 200)
        library_ids = {item["document_id"] for item in library["json"]["documents"]}
        self.assertIn(document_id, library_ids)
        self.assertNotIn(hidden_id, library_ids)
        debug_ids = {item["document_id"] for item in debug["json"]["documents"]}
        self.assertIn(hidden_id, debug_ids)

    def test_upload_remains_visible_after_new_app_instance_reloads_file_metadata(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        from emotion_aware_assistant.web.server import create_web_app

        reloaded = create_web_app(force_dummy_llm=True)
        reloaded.state.upload_dir = self.runtime_dir.resolve()
        reloaded.state.documents_dir = reloaded.state.upload_dir / "documents"
        response = reloaded.test_request("GET", "/api/documents")

        self.assertEqual(response["status"], 200)
        self.assertIn(document_id, {item["document_id"] for item in response["json"]["documents"]})

    def test_prepare_status_exposes_progress_payload(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        detail = self.app.test_request("GET", f"/api/documents/{document_id}")["json"]
        status = detail["prepare_status"]

        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["stage"], "ready")
        self.assertEqual(status["stage_label"], "Ready")
        self.assertEqual(status["progress_percent"], 100)
        self.assertIn("elapsed_seconds", status)
        self.assertIn("estimated_remaining_seconds", status)
        self.assertIsInstance(status["steps"], list)
        self.assertTrue(all("status" in step for step in status["steps"]))

    def test_documents_list_infers_meta_and_counts_highlights_threads(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {
                "highlights": [
                    {"highlight_id": "h1", "type": "text", "page_number": 1},
                    {"id": "legacy-id", "type": "area", "page_number": 1},
                ]
            },
        )
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h1",
            {"messages": [{"role": "assistant", "content": "Saved answer."}]},
        )

        response = self.app.test_request("GET", "/api/documents")

        self.assertEqual(response["status"], 200)
        documents = response["json"]["documents"]
        current = next(item for item in documents if item["document_id"] == document_id)
        self.assertEqual(current["highlight_count"], 2)
        self.assertEqual(current["thread_count"], 1)
        self.assertEqual(current["title"] or current["file_name"], "paper.pdf")

    def test_document_detail_and_file_route(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        detail = self.app.test_request("GET", f"/api/documents/{document_id}")
        file_response = self.app.test_file_request(f"/api/documents/{document_id}/file")

        self.assertEqual(detail["status"], 200)
        self.assertTrue(detail["json"]["files"]["original_pdf"])
        self.assertTrue(detail["json"]["files"]["blocks_index"])
        self.assertEqual(file_response["status"], 200)
        self.assertEqual(file_response["content_type"], "application/pdf")
        self.assertTrue(file_response["body"].startswith(b"%PDF"))

    def test_open_updates_last_opened_and_page(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/open",
            {"last_page": 4},
        )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["json"]["meta"]["last_page"], 4)
        self.assertTrue(response["json"]["meta"]["last_opened_at"])

    def test_highlights_are_saved_with_stable_ids_and_reloaded(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        saved = self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {
                "highlights": [
                    {
                        "type": "text",
                        "page_number": 1,
                        "selected_text": "writ-\n ing with \ufb01gures",
                    }
                ]
            },
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")

        self.assertEqual(saved["status"], 200)
        self.assertEqual(loaded["status"], 200)
        highlight = loaded["json"]["highlights"][0]
        self.assertTrue(highlight["highlight_id"])
        self.assertEqual(highlight["id"], highlight["highlight_id"])
        self.assertEqual(highlight["document_id"], document_id)
        self.assertEqual(highlight["selected_text"], "writing with figures")

    def test_threads_are_saved_and_reloaded(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h1.json"
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "h1", "type": "text", "page_number": 1}]},
        )

        empty = self.app.test_request("GET", f"/api/documents/{document_id}/threads/h1")
        self.assertEqual(empty["status"], 200)
        self.assertEqual(empty["json"]["messages"], [])
        self.assertFalse(thread_path.exists())

        saved = self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h1",
            {
                "selection_snapshot": {"highlight_id": "h1", "highlight_type": "text"},
                "messages": [{"role": "user", "content": "Explain this."}],
            },
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/threads/h1")

        self.assertEqual(saved["status"], 200)
        self.assertEqual(loaded["json"]["messages"][0]["content"], "Explain this.")
        self.assertEqual(json.loads(thread_path.read_text(encoding="utf-8"))["messages"][0]["content"], "Explain this.")

        loaded_again = self.app.test_request("GET", f"/api/documents/{document_id}/threads/h1")
        self.assertEqual(loaded_again["status"], 200)
        self.assertEqual(loaded_again["json"]["messages"][0]["content"], "Explain this.")

    def test_get_thread_for_unknown_highlight_id_returns_404_not_empty_template(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "known-highlight", "type": "text", "page_number": 1}]},
        )

        valid_empty = self.app.test_request("GET", f"/api/documents/{document_id}/threads/known-highlight")
        invalid = self.app.test_request("GET", f"/api/documents/{document_id}/threads/wrong-generated-id")

        self.assertEqual(valid_empty["status"], 200, valid_empty)
        self.assertEqual(valid_empty["json"]["messages"], [])
        self.assertEqual(invalid["status"], 404, invalid)
        self.assertIn("Unknown highlight_id", invalid["json"]["error"])

    def test_empty_thread_put_does_not_overwrite_existing_non_empty_thread(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-non-empty.json"

        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-non-empty",
            {
                "selection_snapshot": {"highlight_id": "h-non-empty", "highlight_type": "text"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Persisted answer.",
                        "turn_id": "turn_saved",
                        "prompt_snapshot_id": "snap_saved",
                    }
                ],
            },
        )
        overwritten = self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-non-empty",
            {
                "selection_snapshot": {"highlight_id": "h-non-empty", "highlight_type": "text"},
                "messages": [],
            },
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/threads/h-non-empty")
        thread = json.loads(thread_path.read_text(encoding="utf-8"))

        self.assertEqual(overwritten["status"], 200, overwritten)
        self.assertTrue(overwritten["json"]["empty_overwrite_ignored"])
        self.assertEqual(loaded["json"]["messages"][0]["content"], "Persisted answer.")
        self.assertEqual(thread["messages"][0]["prompt_snapshot_id"], "snap_saved")

    def test_thread_turn_metadata_persists_multiple_reaction_windows(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        candidate_one = self.strategy_candidate("step_by_step_breakdown", "Break it into steps", recommended=True, score=0.9)
        candidate_two = self.strategy_candidate("critique_assumptions", "Critique the assumption", recommended=True, score=0.88)

        saved = self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-turn-meta",
            {
                "selection_snapshot": {"highlight_id": "h-turn-meta", "highlight_type": "text"},
                "messages": [
                    {"role": "assistant", "content": "Baseline answer.", "turn_id": "turn_base", "turn_type": "baseline_explanation"},
                    {"role": "assistant", "content": "Strategy answer.", "turn_id": "turn_strategy", "turn_type": "strategy_reexplanation"},
                ],
                "turn_metadata": {
                    "turn_base": {
                        "reaction_window_summary": {"source_turn_id": "turn_base", "support_cue": "sustained_clarification"},
                        "strategy_candidates": [candidate_one],
                        "support_cue": "sustained_clarification",
                        "planner_mode": "heuristic",
                    },
                    "turn_strategy": {
                        "reaction_window_summary": {"source_turn_id": "turn_strategy", "support_cue": "deepening"},
                        "strategy_candidates": [candidate_two],
                        "support_cue": "deepening",
                        "planner_mode": "llm",
                    },
                },
            },
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/threads/h-turn-meta")
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-turn-meta.json"
        disk_thread = json.loads(thread_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(loaded["json"]["turn_metadata"]["turn_base"]["strategy_candidates"][0]["strategy_id"], "step_by_step_breakdown")
        self.assertEqual(loaded["json"]["turn_metadata"]["turn_strategy"]["reaction_window_summary"]["support_cue"], "deepening")
        self.assertEqual(disk_thread["turn_metadata"]["turn_strategy"]["planner_mode"], "llm")

    def test_delete_highlight_removes_highlight_and_thread_without_deleting_document_files(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        document_dir = self.app.state.documents_dir / document_id
        original_pdf = document_dir / "original.pdf"
        parsed_index = document_dir / "parsed" / "blocks_index.json"
        rag_profile = document_dir / "rag" / "paper_profile.json"
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "h-delete", "type": "text", "page_number": 1}]},
        )
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-delete",
            {
                "selection_snapshot": {"highlight_id": "h-delete", "highlight_type": "text"},
                "messages": [{"role": "assistant", "content": "Saved answer."}],
            },
        )

        response = self.app.test_request("DELETE", f"/api/documents/{document_id}/highlights/h-delete")
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(loaded["json"]["highlights"], [])
        self.assertFalse((document_dir / "threads" / "h-delete.json").exists())
        self.assertTrue(original_pdf.exists())
        self.assertTrue(parsed_index.exists())
        self.assertTrue(rag_profile.exists())
        self.assertEqual(response["json"]["meta"]["highlight_count"], 0)
        self.assertEqual(response["json"]["meta"]["thread_count"], 0)

    def test_clear_conversation_keeps_highlight_but_clears_thread_messages_and_strategy_state(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "h-clear", "type": "text", "page_number": 1}]},
        )
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-clear",
            {
                "selection_snapshot": {"highlight_id": "h-clear", "highlight_type": "text"},
                "strategy_candidates": [self.strategy_candidate("structured_breakdown", "Structured breakdown")],
                "selected_strategy_id": "structured_breakdown",
                "selected_strategy": self.strategy_candidate("structured_breakdown", "Structured breakdown"),
                "reaction_window_summary": {"support_cue": "sustained_clarification"},
                "turn_metadata": {
                    "turn_keep": {
                        "reaction_window_summary": {"source_turn_id": "turn_keep", "support_cue": "sustained_clarification"},
                        "strategy_candidates": [self.strategy_candidate("structured_breakdown", "Structured breakdown")],
                    }
                },
                "messages": [{"role": "assistant", "content": "Saved answer.", "turn_id": "turn_keep"}],
            },
        )

        response = self.app.test_request("POST", f"/api/documents/{document_id}/threads/h-clear/clear")
        highlights = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")

        self.assertEqual(response["status"], 200, response)
        thread = response["json"]["thread"]
        self.assertEqual(thread["messages"], [])
        self.assertEqual(thread["strategy_candidates"], [])
        self.assertEqual(thread["selected_strategy_id"], "")
        self.assertEqual(thread["selected_strategy"], {})
        self.assertEqual(thread["reaction_window_summary"], {})
        self.assertEqual(thread["turn_metadata"], {})
        self.assertEqual(thread["selection_snapshot"]["highlight_id"], "h-clear")
        self.assertEqual(highlights["json"]["highlights"][0]["highlight_id"], "h-clear")
        self.assertEqual(response["json"]["meta"]["highlight_count"], 1)
        self.assertEqual(response["json"]["meta"]["thread_count"], 0)

    def test_delete_thread_turn_removes_only_matching_turn_messages(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-turn-delete",
            {
                "selection_snapshot": {"highlight_id": "h-turn-delete", "highlight_type": "text"},
                "messages": [
                    {"role": "assistant", "content": "Turn one.", "turn_id": "turn_one"},
                    {"role": "user", "content": "Question.", "turn_id": "turn_two"},
                    {"role": "assistant", "content": "Turn two.", "turn_id": "turn_two"},
                ],
                "turn_metadata": {
                    "turn_one": {"reaction_window_summary": {"source_turn_id": "turn_one"}},
                    "turn_two": {"reaction_window_summary": {"source_turn_id": "turn_two"}},
                },
            },
        )

        response = self.app.test_request(
            "DELETE",
            f"/api/documents/{document_id}/threads/h-turn-delete/turns/turn_two",
        )

        self.assertEqual(response["status"], 200, response)
        messages = response["json"]["thread"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Turn one.")
        self.assertEqual(messages[0]["turn_id"], "turn_one")
        self.assertIn("turn_one", response["json"]["thread"]["turn_metadata"])
        self.assertNotIn("turn_two", response["json"]["thread"]["turn_metadata"])

    def test_live_http_delete_highlight_reaches_route_handler_without_501(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        document_dir = self.app.state.documents_dir / document_id
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "h-live-delete", "type": "text", "page_number": 1}]},
        )
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-live-delete",
            {"messages": [{"role": "assistant", "content": "Saved answer."}]},
        )

        response = self.live_http_json_request("DELETE", f"/api/documents/{document_id}/highlights/h-live-delete", {})
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")

        self.assertNotEqual(response["status"], 501)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(loaded["json"]["highlights"], [])
        self.assertFalse((document_dir / "threads" / "h-live-delete.json").exists())

    def test_live_http_delete_turn_reaches_route_handler_without_501(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-live-turn-delete",
            {
                "messages": [
                    {"role": "assistant", "content": "Turn one.", "turn_id": "turn_one"},
                    {"role": "user", "content": "Question.", "turn_id": "turn_two"},
                    {"role": "assistant", "content": "Turn two.", "turn_id": "turn_two"},
                ],
            },
        )

        response = self.live_http_json_request(
            "DELETE",
            f"/api/documents/{document_id}/threads/h-live-turn-delete/turns/turn_two",
            {},
        )

        self.assertNotEqual(response["status"], 501)
        self.assertEqual(response["status"], 200, response)
        messages = response["json"]["thread"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["turn_id"], "turn_one")

    def test_live_http_delete_invalid_ids_return_json_errors_not_501(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        missing_highlight = self.live_http_json_request("DELETE", f"/api/documents/{document_id}/highlights/missing-highlight", {})
        missing_turn = self.live_http_json_request("DELETE", f"/api/documents/{document_id}/threads/missing-thread/turns/missing-turn", {})

        self.assertNotEqual(missing_highlight["status"], 501)
        self.assertNotEqual(missing_turn["status"], 501)
        self.assertEqual(missing_highlight["status"], 400)
        self.assertEqual(missing_turn["status"], 400)
        self.assertIn("error", missing_highlight["json"])
        self.assertIn("error", missing_turn["json"])

    def test_cleanup_routes_reject_unknown_ids_safely(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        missing_doc = self.app.test_request("DELETE", "/api/documents/missing-doc/highlights/h1")
        missing_highlight = self.app.test_request("DELETE", f"/api/documents/{document_id}/highlights/missing-highlight")
        missing_thread = self.app.test_request("DELETE", f"/api/documents/{document_id}/threads/missing-thread/turns/turn_one")

        self.assertEqual(missing_doc["status"], 404)
        self.assertEqual(missing_highlight["status"], 400)
        self.assertEqual(missing_thread["status"], 400)

    def test_explain_selection_reuses_debug_pipeline_and_persists_thread_without_old_chat(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        request = {
            "highlight_id": "h-text",
            "highlight_type": "text",
            "page_number": 1,
            "selected_text": "PDF debug test",
            "text_available": True,
            "recommended_llm_mode": "text_context",
            "matched_block": {"markdown_content": "PDF debug test"},
            "nearby_useful_context": [],
        }

        with patch.object(self.app.state, "chat", side_effect=AssertionError("old chat must not be called")):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/explain-selection",
                request,
            )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertEqual(payload["provider"], "mock")
        self.assertEqual(payload["response_style"], "chat_conversational")
        self.assertIn("mock explanation", payload["answer"])
        self.assertIn("context_used", payload)
        self.assertIn("natural conversational style", payload["prompt_preview"])
        self.assertNotIn("answer_format:", payload["prompt_preview"])
        self.assertNotIn("useful_follow_up_question", payload["prompt_preview"])

        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-text.json"
        thread = json.loads(thread_path.read_text(encoding="utf-8"))
        self.assertEqual(thread["messages"][-1]["role"], "assistant")
        self.assertIn("mock explanation", thread["messages"][-1]["content"])

    def test_pdf_chat_explain_uses_configured_default_llm_settings(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Configured model answer"}}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        root = Path(self.temp_dir.name)
        secret = "or-pdf-chat-secret"
        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            save = self.app.test_request("POST", "/api/settings/openrouter", {"api_key": secret})
            profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"display_name": "GPT via OpenRouter", "model_id": "openai/gpt-4o-mini", "enabled": True},
            )
            profile_id = profile["json"]["model_profile"]["id"]
            roles = self.app.test_request("POST", "/api/settings/roles", {"default_answer_model_profile_id": profile_id})
            upload = self.upload_pdf()
            response = self.app.test_request(
                "POST",
                f"/api/documents/{upload['document_id']}/explain-selection",
                {
                    "highlight_id": "h-configured",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF chat should use the configured answer model.",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                    "matched_block": {"markdown_content": "PDF chat should use the configured answer model."},
                },
            )

        payload = response["json"]
        request_body = json.loads(requests[0].data.decode("utf-8")) if requests else {}
        serialized = json.dumps(payload)
        self.assertEqual(save["status"], 200, save)
        self.assertEqual(roles["status"], 200, roles)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(payload["provider"], "openrouter")
        self.assertEqual(payload["model"], "openai/gpt-4o-mini")
        self.assertEqual(payload["answer"], "Configured model answer")
        self.assertEqual(request_body["model"], "openai/gpt-4o-mini")
        self.assertNotIn(secret, serialized)

    def test_pdf_chat_explain_can_use_configured_gemini_default_profile(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "Gemini configured answer"}]}}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        root = Path(self.temp_dir.name)
        secret = "AI" + "za" + "pdf-chat-gemini-secret"
        stale_secret = "AI" + "za" + "stale-pdf-chat-env"
        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GEMINI_API_KEY": stale_secret}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            save = self.app.test_request("POST", "/api/settings/gemini", {"api_key": secret, "model": "gemini-2.5-flash"})
            profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "gemini", "display_name": "Gemini 2.5 Flash", "model_id": "gemini-2.5-flash", "enabled": True},
            )
            profile_id = profile["json"]["model_profile"]["id"]
            roles = self.app.test_request("POST", "/api/settings/roles", {"default_answer_model_profile_id": profile_id})
            upload = self.upload_pdf()
            response = self.app.test_request(
                "POST",
                f"/api/documents/{upload['document_id']}/explain-selection",
                {
                    "highlight_id": "h-gemini-configured",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF chat should use the configured Gemini answer model.",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                    "matched_block": {"markdown_content": "PDF chat should use the configured Gemini answer model."},
                },
            )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(save["status"], 200, save)
        self.assertEqual(roles["status"], 200, roles)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(payload["provider"], "gemini")
        self.assertEqual(payload["model"], "gemini-2.5-flash")
        self.assertEqual(payload["key_source"], "local_settings")
        self.assertEqual(payload["answer"], "Gemini configured answer")
        self.assertIn("/models/gemini-2.5-flash:generateContent", requests[0].full_url)
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        self.assertEqual(headers["x-goog-api-key"], secret)
        self.assertNotEqual(headers["x-goog-api-key"], stale_secret)
        self.assertNotIn(secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_pdf_chat_uses_changed_default_role_and_key_without_restart(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Fresh answer model output"}, "finish_reason": "stop"}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        root = Path(self.temp_dir.name)
        old_secret = "or-old-pdf-chat"
        new_secret = "or-new-pdf-chat"
        stale_secret = "or-stale-env-pdf-chat"
        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": stale_secret,
                "LLM_PROVIDER": "openrouter",
                "LLM_MODEL": "openai/stale-env-model",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": old_secret})
            old_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Old Router", "model_id": "openai/old-model", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"default_answer_model_profile_id": old_profile})
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": new_secret})
            new_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "New Router", "model_id": "openai/new-model", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request("POST", "/api/settings/roles", {"default_answer_model_profile_id": new_profile})
            upload = self.upload_pdf()
            response = self.app.test_request(
                "POST",
                f"/api/documents/{upload['document_id']}/explain-selection",
                {
                    "highlight_id": "h-runtime-role-change",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF chat should use the new runtime answer model.",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                    "matched_block": {"markdown_content": "PDF chat should use the new runtime answer model."},
                },
            )

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        body = json.loads(requests[0].data.decode("utf-8"))
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["provider"], "openrouter")
        self.assertEqual(response["json"]["model"], "openai/new-model")
        self.assertEqual(body["model"], "openai/new-model")
        self.assertEqual(headers["authorization"], f"Bearer {new_secret}")
        self.assertNotIn(old_secret, serialized)
        self.assertNotIn(new_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_strategy_planner_uses_configured_strategy_profile_and_falls_back_to_default(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        root = Path(self.temp_dir.name)
        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": "or-strategy-secret"})
            default = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"display_name": "Default Model", "model_id": "openai/gpt-4o-mini", "enabled": True},
            )["json"]["model_profile"]["id"]
            strategy = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"display_name": "Strategy Model", "model_id": "anthropic/claude-opus-4.7-fast", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request(
                "POST",
                "/api/settings/roles",
                {"default_answer_model_profile_id": default, "strategy_model_profile_id": strategy},
            )
            strategy_role = llm_config.role_config_from_env("strategy_planner_model")
            self.app.test_request("DELETE", f"/api/settings/models/{strategy}")
            fallback_role = llm_config.role_config_from_env("strategy_planner_model")

        self.assertEqual(strategy_role["provider"], "openrouter")
        self.assertEqual(strategy_role["model"], "anthropic/claude-opus-4.7-fast")
        self.assertEqual(fallback_role["provider"], "openrouter")
        self.assertEqual(fallback_role["model"], "openai/gpt-4o-mini")

    def test_strategy_response_generation_uses_configured_openrouter_strategy_profile(self):
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Strategy model answer"}, "finish_reason": "stop"}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        root = Path(self.temp_dir.name)
        secret = "or-strategy-response-secret"
        stale_secret = "or-stale-strategy-response"
        selected_strategy = self.strategy_candidate("concrete_example", "Use a concrete example")
        with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": stale_secret,
                "LLM_PROVIDER": "openrouter",
                "LLM_MODEL": "openai/stale-default",
                "STRATEGY_PLANNER_MODEL": "openai/stale-strategy",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            self.app.state.upload_dir = (root / "runtime_uploads").resolve()
            self.app.state.documents_dir = self.app.state.upload_dir / "documents"
            self.app.test_request("POST", "/api/settings/openrouter", {"api_key": secret})
            default_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Default Answer", "model_id": "openai/default-answer", "enabled": True},
            )["json"]["model_profile"]["id"]
            strategy_profile = self.app.test_request(
                "POST",
                "/api/settings/models",
                {"provider": "openrouter", "display_name": "Strategy Answer", "model_id": "anthropic/strategy-answer", "enabled": True},
            )["json"]["model_profile"]["id"]
            self.app.test_request(
                "POST",
                "/api/settings/roles",
                {"default_answer_model_profile_id": default_profile, "strategy_model_profile_id": strategy_profile},
            )
            upload = self.upload_pdf()
            response = self.app.test_request(
                "POST",
                f"/api/documents/{upload['document_id']}/explain-selection",
                {
                    "highlight_id": "h-strategy-openrouter",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "Strategy responses should use the configured strategy model.",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                    "matched_block": {"markdown_content": "Strategy responses should use the configured strategy model."},
                    "default_task": "explain_current_selection_with_selected_strategy",
                    "selected_strategy_id": selected_strategy["strategy_id"],
                    "selected_strategy": selected_strategy,
                },
            )

        body = json.loads(requests[0].data.decode("utf-8"))
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        serialized = json.dumps(response["json"])
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(body["model"], "anthropic/strategy-answer")
        self.assertEqual(headers["authorization"], f"Bearer {secret}")
        self.assertEqual(response["json"]["provider"], "openrouter")
        self.assertEqual(response["json"]["model"], "anthropic/strategy-answer")
        self.assertEqual(response["json"]["model_source"], "role_profile")
        self.assertEqual(response["json"]["key_source"], "local_settings")
        self.assertNotIn(secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_area_crop_is_saved_under_document_highlight_crops(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {
                "highlights": [
                    {
                        "highlight_id": "h-area",
                        "type": "area",
                        "page_number": 1,
                        "viewport_rects": [{"x1": 10, "y1": 20, "x2": 110, "y2": 120, "pageNumber": 1}],
                        "normalized_rects": [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "pageNumber": 1}],
                        "parser_rects_1000": [{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "pageNumber": 1}],
                        "caption": "Figure 1. Example area.",
                        "caption_confidence": "high",
                        "matched_block": {"block_id": "b1", "markdown_content": "Figure 1. Example area."},
                        "nearby_context": [{"block_id": "b2", "markdown_content": "Nearby area context."}],
                    }
                ]
            },
        )
        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/highlights/h-area/crop",
            {"crop_image_data_url": "data:image/png;base64,QUJDRA=="},
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")
        crop_response = self.app.test_file_request(f"/api/documents/{document_id}/highlights/h-area/crop")

        self.assertEqual(response["status"], 200)
        crop_path = self.app.state.documents_dir / document_id / "highlights" / "crops" / "h-area.png"
        self.assertTrue(crop_path.exists())
        self.assertEqual(crop_path.read_bytes(), b"ABCD")
        area = loaded["json"]["highlights"][0]
        self.assertEqual(area["highlight_id"], "h-area")
        self.assertEqual(area["type"], "area")
        self.assertEqual(area["crop_path"], "highlights/crops/h-area.png")
        self.assertEqual(area["crop_image_path"], "highlights/crops/h-area.png")
        self.assertEqual(area["crop_url"], f"/api/documents/{document_id}/highlights/h-area/crop")
        self.assertEqual(area["caption"], "Figure 1. Example area.")
        self.assertEqual(crop_response["status"], 200)
        self.assertEqual(crop_response["content_type"], "image/png")
        self.assertEqual(crop_response["body"], b"ABCD")

    def test_area_highlight_thread_reloads_by_stable_highlight_id(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {
                "highlights": [
                    {
                        "highlight_id": "area-stable",
                        "type": "area",
                        "page_number": 1,
                        "crop_image_path": "highlights/crops/area-stable.png",
                        "caption": "Figure 2. Stable area.",
                    }
                ]
            },
        )
        saved = self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/area-stable",
            {
                "selection_snapshot": {"highlight_id": "area-stable", "highlight_type": "area"},
                "messages": [{"role": "assistant", "content": "Saved area explanation."}],
            },
        )
        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/threads/area-stable")

        self.assertEqual(saved["status"], 200)
        self.assertEqual(loaded["status"], 200)
        self.assertEqual(loaded["json"]["highlight_id"], "area-stable")
        self.assertEqual(loaded["json"]["selection_snapshot"]["highlight_type"], "area")
        self.assertEqual(loaded["json"]["messages"][0]["content"], "Saved area explanation.")

    def test_loading_area_highlights_migrates_existing_crop_file_metadata(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        document_dir = self.app.state.documents_dir / document_id
        crop_dir = document_dir / "highlights" / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / "legacy-area.png").write_bytes(b"ABCD")
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/highlights",
            {"highlights": [{"highlight_id": "legacy-area", "type": "area", "page_number": 1}]},
        )
        highlights_path = document_dir / "highlights" / "highlights.json"
        raw_before = json.loads(highlights_path.read_text(encoding="utf-8"))
        raw_before["highlights"][0].pop("crop_path", None)
        raw_before["highlights"][0].pop("crop_image_path", None)
        raw_before["highlights"][0].pop("crop_url", None)
        highlights_path.write_text(json.dumps(raw_before, indent=2), encoding="utf-8")

        loaded = self.app.test_request("GET", f"/api/documents/{document_id}/highlights")
        raw_after = json.loads(highlights_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["json"]["highlights"][0]["crop_path"], "highlights/crops/legacy-area.png")
        self.assertEqual(raw_after["highlights"][0]["crop_path"], "highlights/crops/legacy-area.png")
        self.assertEqual(raw_after["highlights"][0]["crop_url"], f"/api/documents/{document_id}/highlights/legacy-area/crop")

    def test_follow_up_persists_user_and_assistant_messages_with_thread_history(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h1",
            {
                "selection_snapshot": {
                    "highlight_id": "h1",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF debug test",
                    "recommended_llm_mode": "text_context",
                },
                "messages": [{"role": "assistant", "content": "First answer."}],
            },
        )

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/threads/h1/follow-up",
            {"question": "Why does that matter?"},
        )

        self.assertEqual(response["status"], 200, response)
        messages = response["json"]["thread"]["messages"]
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIn("Why does that matter?", response["json"]["prompt_preview"])

    def test_normal_text_follow_up_ignores_saved_selected_strategy_but_keeps_context_and_learning_cue(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        selected_strategy = self.strategy_candidate("structured_breakdown", "Structured breakdown")
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-normal-text",
            {
                "selection_snapshot": {
                    "highlight_id": "h-normal-text",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF debug test",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                    "matched_block": {"markdown_content": "PDF debug test context"},
                    "global_rag_context": [{"text": "Retrieved RAG context for the selected passage."}],
                },
                "selected_strategy": selected_strategy,
                "selected_strategy_id": selected_strategy["strategy_id"],
                "messages": [{"role": "assistant", "content": "First answer.", "turn_id": "turn-base"}],
            },
        )

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/threads/h-normal-text/follow-up",
            {
                "question": "How does this connect to the method?",
                "input_source": "text",
                "learning_state": {
                    "academic_state": "confusion",
                    "confidence": 0.72,
                    "distribution": {"confusion": 0.72, "engagement": 0.18},
                    "trend": "rising",
                },
            },
        )

        self.assertEqual(response["status"], 200, response)
        prompt = response["json"]["prompt_preview"]
        assistant = response["json"]["thread"]["messages"][-1]
        self.assertEqual(assistant["turn_type"], "follow_up")
        self.assertEqual(assistant["input_source"], "text")
        self.assertEqual(assistant["response_mode"], "normal_followup")
        self.assertIsNone(assistant["selected_strategy_id"])
        self.assertFalse(assistant.get("strategy_id"))
        self.assertFalse(assistant.get("strategy_title"))
        self.assertNotIn("Selected pedagogical support strategy", prompt)
        self.assertIn('follow_up_question: "How does this connect to the method?"', prompt)
        self.assertIn("selected_text:", prompt)
        self.assertIn("PDF debug test", prompt)
        self.assertIn("Soft learning-state style cue", prompt)
        self.assertIn("confusion", prompt)

    def test_speech_follow_up_uses_normal_followup_mode_not_strategy_mode(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        selected_strategy = self.strategy_candidate("concrete_example", "Concrete example")
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-speech",
            {
                "selection_snapshot": {
                    "highlight_id": "h-speech",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "Speech follow-up should stay grounded in this passage.",
                    "text_available": True,
                    "recommended_llm_mode": "text_context",
                },
                "selected_strategy": selected_strategy,
                "selected_strategy_id": selected_strategy["strategy_id"],
                "messages": [{"role": "assistant", "content": "First answer.", "turn_id": "turn-base"}],
            },
        )

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/threads/h-speech/follow-up",
            {
                "question": "Can you restate it more simply?",
                "input_source": "speech",
                "selected_strategy": selected_strategy,
                "selected_strategy_id": selected_strategy["strategy_id"],
            },
        )

        self.assertEqual(response["status"], 200, response)
        assistant = response["json"]["thread"]["messages"][-1]
        self.assertEqual(assistant["input_source"], "speech")
        self.assertEqual(assistant["response_mode"], "normal_followup")
        self.assertIsNone(assistant["selected_strategy_id"])
        self.assertFalse(assistant.get("strategy_status"))
        self.assertNotIn("Selected pedagogical support strategy", response["json"]["prompt_preview"])

    def test_reading_session_start_creates_simulated_learning_state_stream(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")

        self.assertEqual(started["status"], 200, started)
        session_id = started["json"]["session_id"]
        session_dir = self.app.state.upload_dir / "sessions" / session_id
        learning_dir = session_dir / "learning_state"
        self.assertTrue((learning_dir / "current_state.json").exists())
        self.assertTrue((learning_dir / "state_stream.jsonl").exists())
        self.assertTrue((learning_dir / "simulator_config.json").exists())
        state = started["json"]["learning_state"]
        self.assertEqual(state["source"], "simulated_camera")
        self.assertEqual(state["model_output_type"], "academic_state_model")
        self.assertFalse(state["raw_facial_emotion_available"])
        self.assertIsNone(state["raw_facial_emotion"])
        self.assertIn(state["academic_state"], {"boredom", "confusion", "engagement", "frustration"})
        self.assertEqual(set(state["distribution"]), {"boredom", "confusion", "engagement", "frustration"})

        current = self.app.test_request("GET", f"/api/reading-sessions/{session_id}/learning-state/current")

        self.assertEqual(current["status"], 200)
        self.assertEqual(current["json"]["session_id"], session_id)
        stream_lines = (learning_dir / "state_stream.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(stream_lines), 2)

    def test_reading_session_events_are_logged_and_can_recover_signal_after_answer(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")
        session_id = started["json"]["session_id"]

        event = self.app.test_request(
            "POST",
            f"/api/reading-sessions/{session_id}/events",
            {"event_type": "answer_generated", "document_id": document_id, "highlight_id": "h1"},
        )
        current = self.app.test_request("GET", f"/api/reading-sessions/{session_id}/learning-state/current")

        self.assertEqual(event["status"], 200)
        self.assertEqual(event["json"]["event"]["event_type"], "answer_generated")
        event_log = self.app.state.upload_dir / "sessions" / session_id / "events.jsonl"
        self.assertIn("answer_generated", event_log.read_text(encoding="utf-8"))
        self.assertEqual(current["status"], 200)
        self.assertEqual(current["json"]["learning_state"]["academic_state"], "engagement")

    def test_emotion_model_status_endpoint_reports_safe_academic_state_schema(self):
        response = self.app.test_request("GET", "/api/emotion/model/status")

        self.assertEqual(response["status"], 200, response)
        status = response["json"]
        self.assertIn("model_loaded", status)
        self.assertEqual(status["model_output_type"], "academic_state")
        self.assertEqual(status["architecture"], "convnext_tiny.fb_in22k_ft_in1k")
        self.assertEqual(status["classes"], ["boredom", "confusion", "engagement", "frustration"])
        self.assertFalse(status["raw_emotion_available"])
        self.assertNotIn("/home/rli/下载/best", json.dumps(status, ensure_ascii=False))

    def test_session_frame_endpoint_updates_learning_state_without_logging_frames(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")
        session_id = started["json"]["session_id"]

        class FakeAdapter:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "checkpoint_path": "models/emotion_model/best_model.pt",
                    "raw_emotion_available": False,
                    "loading_error": None,
                    "device": "cpu",
                }

            def predict(self, image):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "raw_emotion_available": False,
                    "raw_emotion": None,
                    "academic_state": "confusion",
                    "confidence": 0.81,
                    "state_distribution": {
                        "boredom": 0.04,
                        "confusion": 0.81,
                        "engagement": 0.08,
                        "frustration": 0.07,
                    },
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "device": "cpu",
                }

        self.app.state._emotion_adapter = FakeAdapter()
        missing_academic_checkpoint = Path(self.temp_dir.name) / "missing-academic.pt"
        with patch.dict(
            os.environ,
            {"EMOTION_MODEL_MODE": "academic_state", "EMOTION_CHECKPOINT_PATH": str(missing_academic_checkpoint)},
        ):
            response = self.app.test_request(
                "POST",
                f"/api/reading-sessions/{session_id}/emotion/frame",
                {
                    "document_id": document_id,
                    "image": "data:image/png;base64,QUJDRA==",
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["ok"])
        self.assertIn("emotion_pipeline", response["json"])
        state = response["json"]["learning_state"]
        self.assertEqual(state["source"], "webcam_model")
        self.assertEqual(state["model_output_type"], "academic_state")
        self.assertFalse(state["raw_facial_emotion_available"])
        self.assertIsNone(state["raw_facial_emotion"])
        self.assertEqual(state["academic_state"], "confusion")
        self.assertAlmostEqual(state["confidence"], 0.81)
        self.assertEqual(set(state["distribution"]), {"boredom", "confusion", "engagement", "frustration"})
        self.assertTrue(any("YOLO" in warning and "center crop fallback" in warning for warning in state["warnings"]))
        self.assertEqual(response["json"]["face_detection"]["detector"], "center_crop")
        self.assertEqual(response["json"]["face_detection"]["requested_detector"], "auto")
        self.assertEqual(response["json"]["face_detection"]["actual_detector"], "center_crop")
        self.assertFalse(response["json"]["face_detection"]["face_found"])
        self.assertTrue(response["json"]["face_detection"]["fallback_used"])

        learning_dir = self.app.state.upload_dir / "sessions" / session_id / "learning_state"
        current = json.loads((learning_dir / "current_state.json").read_text(encoding="utf-8"))
        self.assertEqual(current["source"], "webcam_model")
        stream_text = (learning_dir / "state_stream.jsonl").read_text(encoding="utf-8")
        event_text = (self.app.state.upload_dir / "sessions" / session_id / "events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("QUJDRA", stream_text)
        self.assertNotIn("QUJDRA", event_text)
        self.assertNotIn("data:image", event_text)

    def test_session_frame_endpoint_uses_shared_openface_crop_and_raw_pipeline_contract(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")
        session_id = started["json"]["session_id"]

        class FakeOpenFaceBox:
            x = 20
            y = 18
            w = 30
            h = 34
            confidence = 0.98
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.98,
                "landmarks": [[20.0, 18.0], [50.0, 18.0], [50.0, 52.0], [20.0, 52.0]],
                "landmark_count": 4,
                "bbox": [20, 18, 30, 34],
                "warning": None,
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "checkpoint_path": "models/emotion_model/best_model.pt",
                    "raw_emotion_available": False,
                    "loading_error": None,
                }

            def predict(self, image):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "academic_state": "engagement",
                    "confidence": 0.76,
                    "state_distribution": {
                        "boredom": 0.05,
                        "confusion": 0.07,
                        "engagement": 0.76,
                        "frustration": 0.12,
                    },
                }

        class FakePipeline:
            received_size = None

            def status(self, fallback_status=None):
                return {
                    "model_loaded": True,
                    "model_output_type": "raw_emotion",
                    "raw_detection_available": True,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "architecture": "convnextv2_pico.fcmae_ft_in1k",
                    "checkpoint_path": "models/emotion_model/raw_8class_best.pt",
                    "mapper_available": True,
                    "buffer_size": 10,
                }

            def predict(self, image, fallback_prediction=None, fallback_status=None):
                self.received_size = image.size
                return {
                    "model_output_type": "raw_emotion",
                    "checkpoint_path": "models/emotion_model/raw_8class_best.pt",
                    "architecture": "convnextv2_pico.fcmae_ft_in1k",
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "raw_detection_available": True,
                    "raw_detection": {
                        "label": "fear",
                        "confidence": 0.62,
                        "probabilities": {
                            "anger": 0.02,
                            "contempt": 0.03,
                            "disgust": 0.01,
                            "fear": 0.62,
                            "happy": 0.04,
                            "neutral": 0.10,
                            "sad": 0.05,
                            "surprise": 0.13,
                        },
                    },
                    "mapped_academic_state": {
                        "state": "confusion",
                        "scores": {
                            "frustration": 0.08,
                            "confusion": 0.75,
                            "boredom": 0.03,
                            "engagement": 0.14,
                        },
                        "mapping_rule": "fear + surprise -> confusion",
                    },
                    "smoothed_state": {"state": "confusion", "buffer": ["confusion"], "buffer_size": 10},
                    "response_strategy": "Clarify the key concept first.",
                }

        pipeline = FakePipeline()
        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()
        self.app.state._emotion_pipeline = pipeline

        response = self.app.test_request(
            "POST",
            f"/api/reading-sessions/{session_id}/emotion/frame",
            {
                "document_id": document_id,
                "image": self.solid_image_data_url(),
            },
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertTrue(payload["ok"])
        self.assertEqual(pipeline.received_size, (224, 224))
        self.assertEqual(payload["emotion_pipeline"]["model_output_type"], "raw_emotion")
        self.assertEqual(payload["emotion_pipeline"]["raw_detection"]["label"], "fear")
        face = payload["face_detection"]
        self.assertEqual(face["actual_detector"], "openface")
        self.assertTrue(face["face_found"])
        self.assertEqual(face["landmark_count"], 4)
        self.assertEqual(face["crop_strategy"], "openface_landmark_bbox")
        self.assertEqual(face["crop_mode"], "square_face_context")
        self.assertFalse(face["fallback_used"])
        state = payload["learning_state"]
        self.assertEqual(state["model_output_type"], "raw_emotion")
        self.assertTrue(state["raw_facial_emotion_available"])
        self.assertEqual(state["raw_facial_emotion"], "fear")
        self.assertEqual(state["academic_state"], "confusion")
        self.assertEqual(state["distribution"]["confusion"], 0.75)
        package = state["learning_signal_package"]
        self.assertEqual(package["active_source"], "raw_8class_process_aware")
        self.assertEqual(package["raw_emotion_evidence"]["raw_emotion_label"], "fear")
        self.assertEqual(package["academic_state_mapping"]["mapped_academic_state"], "confusion")
        self.assertFalse((package.get("diagnostic_only_direct_4class") or {}).get("used_for_strategy", True))
        serialized = json.dumps(payload)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("crop_preview_data_url", serialized)
        self.assertNotIn("model_input_preview_data_url", serialized)

        event_text = (self.app.state.upload_dir / "sessions" / session_id / "events.jsonl").read_text(encoding="utf-8")
        self.assertIn('"model_output_type": "raw_emotion"', event_text)
        self.assertIn('"raw_detection_available": true', event_text)
        self.assertIn('"learning_signal_package"', event_text)
        self.assertIn('"detector": "openface"', event_text)
        self.assertIn('"crop_strategy": "openface_landmark_bbox"', event_text)
        self.assertNotIn("data:image", event_text)

    def test_reaction_window_summary_maps_mixed_confusion_boredom_to_support_cue(self):
        samples = [
            {
                "timestamp": "2026-05-17T10:00:00Z",
                "academic_state": "confusion",
                "confidence": 0.72,
                "distribution": {"boredom": 0.22, "confusion": 0.62, "engagement": 0.10, "frustration": 0.06},
                "trend": "rising",
                "face_detection": {"detector": "center_crop", "fallback_used": True},
            },
            {
                "timestamp": "2026-05-17T10:00:04Z",
                "academic_state": "boredom",
                "confidence": 0.64,
                "distribution": {"boredom": 0.42, "confusion": 0.38, "engagement": 0.12, "frustration": 0.08},
                "trend": "stable",
                "face_detection": {"detector": "center_crop", "fallback_used": True},
            },
            {
                "timestamp": "2026-05-17T10:00:08Z",
                "academic_state": "confusion",
                "confidence": 0.70,
                "distribution": {"boredom": 0.30, "confusion": 0.52, "engagement": 0.10, "frustration": 0.08},
                "trend": "stable",
                "face_detection": {"detector": "center_crop", "fallback_used": True},
            },
        ]

        summary = self.app.state.summarize_reaction_window(
            samples=samples,
            source_turn_id="turn_base",
            highlight_id="h1",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:10Z",
        )

        self.assertEqual(summary["dominant_state"], "confusion")
        self.assertEqual(summary["secondary_state"], "boredom")
        self.assertEqual(summary["support_cue"], "clarify_and_reengage")
        self.assertEqual(summary["support_cue_label"], "Clarify and re-engage cue")
        self.assertEqual(summary["face_detection_summary"]["mode"], "center_crop")
        self.assertIn("baseline explanation", summary["trigger_reason"])

    def test_reaction_window_summary_maps_mixed_confusion_frustration_to_gentle_clarification(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "confusion",
                    "confidence": 0.68,
                    "distribution": {"boredom": 0.06, "confusion": 0.46, "engagement": 0.10, "frustration": 0.38},
                    "trend": "rising",
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.63,
                    "distribution": {"boredom": 0.06, "confusion": 0.40, "engagement": 0.11, "frustration": 0.43},
                    "trend": "stable",
                },
            ],
            source_turn_id="turn_base",
            highlight_id="h1",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:10Z",
        )

        self.assertEqual(summary["support_cue"], "gentle_clarification")
        self.assertEqual(summary["support_cue_label"], "Gentle clarification cue")

    def test_reaction_window_summary_maps_engagement_to_deepening_and_flat_to_neutral(self):
        deepening = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "engagement",
                    "confidence": 0.80,
                    "distribution": {"boredom": 0.04, "confusion": 0.08, "engagement": 0.74, "frustration": 0.14},
                    "trend": "stable",
                }
            ],
            source_turn_id="turn_base",
            highlight_id="h1",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:10Z",
        )
        flat = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "confusion",
                    "confidence": 0.44,
                    "distribution": {"boredom": 0.26, "confusion": 0.29, "engagement": 0.23, "frustration": 0.22},
                    "trend": "stable",
                }
            ],
            source_turn_id="turn_base",
            highlight_id="h2",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:10Z",
        )

        self.assertEqual(deepening["support_cue"], "deepening")
        self.assertEqual(flat["support_cue"], "neutral_or_uncertain")
        self.assertEqual(flat["support_cue_label"], "Possible ways to continue")

    def test_reaction_window_summary_keeps_frustration_cue_when_confidence_is_low(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "frustration",
                    "confidence": 0.406,
                    "distribution": {"boredom": 0.0, "confusion": 0.005, "engagement": 0.206, "frustration": 0.789},
                    "trend": "rising",
                }
            ],
            source_turn_id="turn_adaptive",
            highlight_id="h-low-confidence",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:08Z",
            source_turn_type="strategy_reexplanation",
        )

        self.assertEqual(summary["dominant_state"], "frustration")
        self.assertEqual(summary["support_cue"], "difficulty_support")
        self.assertEqual(summary["confidence_handling"], "low_confidence")
        self.assertIn("low-confidence signal", summary["academic_state_evidence_text"])
        self.assertIn("previous adaptive explanation", summary["trigger_reason"])
        self.assertIn("frustration 79%", summary["trigger_reason"])
        self.assertNotIn("learning signal was mixed", summary["trigger_reason"])

    def test_reaction_window_summary_tracks_reading_phases_and_resolved_confusion(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "confusion",
                    "confidence": 0.80,
                    "distribution": {"boredom": 0.04, "confusion": 0.72, "engagement": 0.16, "frustration": 0.08},
                    "trend": "stable",
                },
                {
                    "academic_state": "confusion",
                    "confidence": 0.76,
                    "distribution": {"boredom": 0.05, "confusion": 0.66, "engagement": 0.20, "frustration": 0.09},
                    "trend": "stable",
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.74,
                    "distribution": {"boredom": 0.04, "confusion": 0.38, "engagement": 0.49, "frustration": 0.09},
                    "trend": "rising",
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.82,
                    "distribution": {"boredom": 0.03, "confusion": 0.14, "engagement": 0.76, "frustration": 0.07},
                    "trend": "rising",
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.84,
                    "distribution": {"boredom": 0.02, "confusion": 0.10, "engagement": 0.82, "frustration": 0.06},
                    "trend": "rising",
                },
            ],
            source_turn_id="turn_base",
            highlight_id="h-reading",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:18Z",
            end_reason="near_bottom_reached",
            read_progress_estimate=0.9,
        )

        self.assertEqual(summary["end_reason"], "near_bottom_reached")
        self.assertEqual(summary["read_progress_estimate"], 0.9)
        self.assertEqual(summary["sample_count"], 5)
        self.assertIn("early_distribution", summary)
        self.assertIn("middle_distribution", summary)
        self.assertIn("late_distribution", summary)
        self.assertEqual(summary["early_dominant_state"], "confusion")
        self.assertEqual(summary["late_dominant_state"], "engagement")
        self.assertEqual(summary["state_transition"], "confusion_to_engagement")
        self.assertEqual(summary["transition_label"], "early confusion -> late engagement")
        self.assertEqual(summary["support_cue"], "deepening")
        self.assertIn("early confusion -> late engagement", summary["academic_state_evidence_text"])

    def test_reaction_window_summary_persistent_frustration_uses_difficulty_support(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "frustration",
                    "confidence": 0.74,
                    "distribution": {"boredom": 0.02, "confusion": 0.08, "engagement": 0.16, "frustration": 0.74},
                    "trend": "rising",
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.78,
                    "distribution": {"boredom": 0.03, "confusion": 0.07, "engagement": 0.14, "frustration": 0.76},
                    "trend": "rising",
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.82,
                    "distribution": {"boredom": 0.02, "confusion": 0.06, "engagement": 0.10, "frustration": 0.82},
                    "trend": "rising",
                },
            ],
            source_turn_id="turn_adaptive",
            highlight_id="h-reading",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:21Z",
            source_turn_type="strategy_reexplanation",
            end_reason="near_bottom_reached",
            read_progress_estimate=0.95,
        )

        self.assertEqual(summary["state_transition"], "persistent_frustration")
        self.assertEqual(summary["transition_label"], "persistent frustration")
        self.assertEqual(summary["dominant_state"], "frustration")
        self.assertEqual(summary["support_cue"], "difficulty_support")
        self.assertEqual(summary["support_cue_label"], "Reduce cognitive load cue")
        self.assertIn("frustration", summary["academic_state_evidence_text"])
        self.assertIn("reduce cognitive load", summary["reason_text"])

    def test_reaction_window_falling_frustration_with_close_engagement_marks_resolving_difficulty(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "frustration",
                    "confidence": 0.72,
                    "distribution": {"boredom": 0.01, "confusion": 0.01, "engagement": 0.26, "frustration": 0.72},
                    "trend": "falling",
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.54,
                    "distribution": {"boredom": 0.01, "confusion": 0.01, "engagement": 0.44, "frustration": 0.54},
                    "trend": "falling",
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.62,
                    "distribution": {"boredom": 0.01, "confusion": 0.01, "engagement": 0.62, "frustration": 0.36},
                    "trend": "rising",
                },
            ],
            source_turn_id="turn_base",
            highlight_id="h-resolving",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:20Z",
            source_turn_type="baseline_explanation",
            end_reason="near_bottom_reached",
            read_progress_estimate=0.9,
        )

        self.assertEqual(summary["dominant_state"], "frustration")
        self.assertEqual(summary["secondary_state"], "engagement")
        self.assertEqual(summary["trend"], "falling")
        self.assertEqual(summary["state_transition"], "resolving_difficulty")
        self.assertEqual(summary["transition_label"], "resolving_difficulty")
        self.assertEqual(summary["support_cue"], "deepening")
        self.assertIn("resolving_difficulty", summary["academic_state_evidence_text"])
        self.assertIn("difficulty signal was falling", summary["reason_text"])
        self.assertIn("engagement was close behind", summary["reason_text"])
        self.assertIn("cautious deepening strategy", summary["reason_text"])
        self.assertNotIn("answer could be extended", summary["reason_text"])

    def test_resolved_confusion_reaction_window_recommends_deepening_not_clarification(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "confusion",
                    "confidence": 0.78,
                    "distribution": {"boredom": 0.04, "confusion": 0.70, "engagement": 0.18, "frustration": 0.08},
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.82,
                    "distribution": {"boredom": 0.03, "confusion": 0.12, "engagement": 0.78, "frustration": 0.07},
                },
                {
                    "academic_state": "engagement",
                    "confidence": 0.84,
                    "distribution": {"boredom": 0.03, "confusion": 0.09, "engagement": 0.82, "frustration": 0.06},
                },
            ],
            source_turn_id="turn_base",
            highlight_id="h-resolved",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:16Z",
            end_reason="near_bottom_reached",
            read_progress_estimate=0.9,
        )

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-resolved",
                    "source_turn_id": "turn_base",
                    "selection_type": "text",
                    "selected_text": "The method retrieves related chunks before answering.",
                    "baseline_explanation": "The baseline explanation described retrieval before answering.",
                    "reaction_window_summary": summary,
                    "support_cue": summary["support_cue"],
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertEqual(payload["support_cue"], "deepening")
        self.assertEqual(payload["learning_signal_package"]["inferred_process_state"], "engaged_deepening")
        self.assertIn(payload["candidates"][0]["strategy_family"], self.app.state._allowed_strategy_families_for_support_cue("deepening"))
        self.assertNotIn(payload["candidates"][0]["strategy_family"], {"define_key_terms", "step_by_step_breakdown"})

    def test_persistent_frustration_reaction_window_recommends_reduce_load_family(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "frustration",
                    "confidence": 0.78,
                    "distribution": {"boredom": 0.02, "confusion": 0.06, "engagement": 0.14, "frustration": 0.78},
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.82,
                    "distribution": {"boredom": 0.02, "confusion": 0.05, "engagement": 0.11, "frustration": 0.82},
                },
                {
                    "academic_state": "frustration",
                    "confidence": 0.84,
                    "distribution": {"boredom": 0.01, "confusion": 0.05, "engagement": 0.10, "frustration": 0.84},
                },
            ],
            source_turn_id="turn_adaptive",
            highlight_id="h-persistent",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:20Z",
            source_turn_type="strategy_reexplanation",
            end_reason="near_bottom_reached",
            read_progress_estimate=0.9,
        )

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-persistent",
                    "source_turn_id": "turn_adaptive",
                    "source_turn_type": "strategy_reexplanation",
                    "selection_type": "text",
                    "selected_text": "The method retrieves related chunks before answering.",
                    "baseline_explanation": "The adaptive explanation walked through the passage.",
                    "reaction_window_summary": summary,
                    "support_cue": summary["support_cue"],
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertEqual(payload["support_cue"], "difficulty_support")
        self.assertEqual(payload["learning_signal_package"]["inferred_process_state"], "continued_difficulty")
        self.assertIn(payload["candidates"][0]["strategy_family"], self.app.state._allowed_strategy_families_for_support_cue("difficulty_support"))

    def test_reaction_window_max_timeout_uses_conservative_end_reason(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "engagement",
                    "confidence": 0.78,
                    "distribution": {"boredom": 0.03, "confusion": 0.07, "engagement": 0.82, "frustration": 0.08},
                }
            ],
            source_turn_id="turn_base",
            highlight_id="h-timeout",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:01:00Z",
            end_reason="max_timeout",
            read_progress_estimate=0.45,
        )

        self.assertEqual(summary["end_reason"], "max_timeout")
        self.assertEqual(summary["read_progress_estimate"], 0.45)
        self.assertEqual(summary["support_cue"], "neutral_or_uncertain")
        self.assertEqual(summary["support_cue_label"], "Possible ways to continue")

    def test_followup_interrupted_reaction_window_marks_early_reliability_note(self):
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "engagement",
                    "confidence": 0.60,
                    "distribution": {"boredom": 0.06, "confusion": 0.08, "engagement": 0.60, "frustration": 0.26},
                }
            ],
            source_turn_id="turn_base",
            highlight_id="h-followup-early",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:00:04Z",
            end_reason="followup_submitted",
            read_progress_estimate=0.35,
        )

        self.assertEqual(summary["end_reason"], "followup_submitted")
        self.assertIn("reading window ended early", " ".join(summary["reliability_notes"]))

    def test_max_timeout_low_sample_reaction_window_still_generates_strategy_candidates(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        summary = self.app.state.summarize_reaction_window(
            samples=[
                {
                    "academic_state": "confusion",
                    "confidence": 0.42,
                    "distribution": {"boredom": 0.20, "confusion": 0.42, "engagement": 0.25, "frustration": 0.13},
                }
            ],
            source_turn_id="turn_timeout",
            highlight_id="h-timeout-low-sample",
            window_start="2026-05-17T10:00:00Z",
            window_end="2026-05-17T10:01:00Z",
            end_reason="max_timeout",
            read_progress_estimate=0.4,
        )

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-timeout-low-sample",
                    "source_turn_id": "turn_timeout",
                    "selection_type": "text",
                    "selected_text": "The method retrieves related chunks before answering.",
                    "baseline_explanation": "The baseline explanation described retrieval before answering.",
                    "reaction_window_summary": summary,
                    "support_cue": summary["support_cue"],
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["candidates"])
        self.assertEqual(response["json"]["reaction_window_summary"]["end_reason"], "max_timeout")
        self.assertIn("low sample count", " ".join(response["json"]["reaction_window_summary"]["reliability_notes"]))
        self.assertEqual(response["json"]["support_cue"], "neutral_or_uncertain")

    def test_support_cue_maps_to_allowed_strategy_families(self):
        expected = {
            "sustained_clarification": {"step_by_step_breakdown", "define_key_terms", "concrete_example", "input_process_output_map", "mechanism_walkthrough", "formula_intuition"},
            "difficulty_support": {"concrete_example", "example_based_explanation", "analogy_or_reframe", "connect_to_paper_argument", "simplest_version_first", "one_small_next_step", "reduce_information_density", "key_takeaway_first"},
            "reduce_load": {"concrete_example", "example_based_explanation", "analogy_or_reframe", "connect_to_paper_argument", "simplest_version_first", "one_small_next_step", "reduce_information_density", "key_takeaway_first"},
            "re_engagement": {"why_it_matters", "one_sentence_takeaway", "make_it_relevant", "compare_with_familiar_method", "quick_quiz"},
            "deepening": {"deep_technical_explanation", "critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"},
            "clarify_and_reengage": {"concise_explanation", "concrete_example", "why_it_matters", "step_by_step_breakdown", "compare_with_familiar_method"},
            "gentle_clarification": {"simplest_version_first", "one_small_next_step", "define_key_terms", "analogy_or_reframe", "concrete_example"},
            "neutral_or_uncertain": {"concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"},
        }

        for support_cue, families in expected.items():
            self.assertEqual(set(self.app.state._allowed_strategy_families_for_support_cue(support_cue)), families)

    def test_reaction_window_strategy_candidates_require_baseline_explanation(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/strategy-candidates",
            {
                "highlight_id": "h-reaction",
                "selection_type": "text",
                "selected_text": "The method retrieves related chunks before answering.",
                "trigger_context": {"triggered_by": "reaction_window"},
            },
        )

        self.assertEqual(response["status"], 400)
        self.assertIn("baseline_explanation", response["json"]["error"])

    def test_strategy_candidates_use_reaction_window_summary(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        reaction_summary = {
            "source_turn_id": "turn_base",
            "highlight_id": "h-reaction",
            "duration_sec": 12.4,
            "dominant_state": "confusion",
            "secondary_state": "boredom",
            "avg_confidence": 0.72,
            "max_confidence": 0.84,
            "avg_distribution": {"boredom": 0.22, "confusion": 0.58, "engagement": 0.12, "frustration": 0.08},
            "trend": "rising",
            "stability": "stable",
            "support_cue": "clarify_and_reengage",
            "support_cue_label": "Clarify and re-engage cue",
            "trigger_reason": "The baseline explanation was being read while the learning signal showed a sustained clarification cue.",
            "face_detection_summary": {"mode": "center_crop", "fallback_used": True},
        }

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/strategy-candidates",
            {
                "highlight_id": "h-reaction",
                "source_turn_id": "turn_base",
                "selection_type": "text",
                "selected_text": "The method retrieves related chunks before answering.",
                "baseline_explanation": "This passage says the assistant retrieves related evidence first.",
                "reaction_window_summary": reaction_summary,
                "support_cue": "clarify_and_reengage",
                "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                "trigger_context": {"triggered_by": "reaction_window", "trigger_reason": reaction_summary["trigger_reason"]},
            },
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertEqual(payload["planner_prompt_version"], "reaction_strategy_planner_v2")
        self.assertEqual(payload["support_cue"], "clarify_and_reengage")
        self.assertEqual(sum(1 for candidate in payload["candidates"] if candidate["recommended"]), 1)
        self.assertEqual(
            payload["planner_input_summary"]["allowed_strategy_families"],
            self.app.state._allowed_strategy_families_for_support_cue("clarify_and_reengage"),
        )
        for candidate in payload["candidates"]:
            self.assertIn(candidate["strategy_family"], self.app.state._allowed_strategy_families_for_support_cue("clarify_and_reengage"))
            self.assertTrue(candidate["pedagogical_move"])
            self.assertTrue(candidate["context_focus"])
        self.assertTrue(any("Why this appeared" not in candidate["why_recommended"] for candidate in payload["candidates"]))
        self.assertIn("baseline explanation", payload["candidates"][0]["why_recommended"])
        self.assertIn("academic_state_evidence_text", payload["candidates"][0])
        self.assertIn("reason_text", payload["candidates"][0])
        self.assertIn("confusion 58%", payload["candidates"][0]["academic_state_evidence_text"])

    def test_reaction_window_overrides_stale_single_frame_learning_signal_package(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        stale_package = {
            "active_source": "raw_8class_process_aware",
            "academic_state_scores": {"boredom": 0.25, "confusion": 0.25, "engagement": 0.25, "frustration": 0.25},
            "dominant_academic_state": "uncertain",
            "secondary_academic_state": None,
            "support_cue": "uncertain",
            "inferred_process_state": "baseline_ready",
            "strategy_state": "uncertain",
            "recommended_strategy": "baseline_explanation",
            "strategy_reason": "single frame fallback",
            "prompt_guidance": [],
            "reaction_window_summary": {
                "trend": "single_frame_only",
                "evidence_count": 0,
                "support_cue": "neutral_or_uncertain",
                "avg_mapped_scores": {"boredom": 0.25, "confusion": 0.25, "engagement": 0.25, "frustration": 0.25},
            },
            "learning_process_context": {"question_intent": "general_followup", "followup_count_for_highlight": 0},
            "diagnostic_only_direct_4class": {"used_for_strategy": False},
        }
        reaction_summary = {
            "source_turn_id": "turn_base",
            "highlight_id": "h-reaction",
            "duration_sec": 8.0,
            "dominant_state": "engagement",
            "secondary_state": "confusion",
            "avg_confidence": 0.84,
            "avg_distribution": {"boredom": 0.04, "confusion": 0.18, "engagement": 0.72, "frustration": 0.06},
            "trend": "stable",
            "support_cue": "deepening",
            "support_cue_label": "Deepening cue",
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-reaction",
                    "source_turn_id": "turn_base",
                    "selection_type": "text",
                    "selected_text": "The method retrieves related chunks before answering.",
                    "baseline_explanation": "The baseline explanation described retrieval before answering.",
                    "reaction_window_summary": reaction_summary,
                    "support_cue": "deepening",
                    "learning_state": {
                        "academic_state": "engagement",
                        "confidence": 0.70,
                        "learning_signal_package": stale_package,
                    },
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        package = response["json"]["learning_signal_package"]
        package_reaction = package["reaction_window_summary"]
        self.assertEqual(package_reaction["trend"], "stable")
        self.assertGreater(package_reaction["evidence_count"], 0)
        self.assertEqual(package["academic_state_scores"]["engagement"], 0.72)
        self.assertEqual(package["dominant_academic_state"], "engagement")
        self.assertEqual(package["secondary_academic_state"], "confusion")
        self.assertEqual(package["support_cue"], "deepening")
        self.assertEqual(package["inferred_process_state"], "engaged_deepening")
        self.assertNotEqual(package["recommended_strategy"], "baseline_explanation")
        self.assertEqual(response["json"]["planner_input_summary"]["support_cue"], "deepening")

    def test_previous_strategy_reexplanation_is_extracted_and_not_repeated_when_alternatives_exist(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        reaction_summary = {
            "source_turn_id": "turn_base",
            "highlight_id": "h-repeat",
            "duration_sec": 8.0,
            "dominant_state": "engagement",
            "secondary_state": "confusion",
            "avg_confidence": 0.82,
            "avg_distribution": {"boredom": 0.04, "confusion": 0.18, "engagement": 0.72, "frustration": 0.06},
            "trend": "stable",
            "support_cue": "deepening",
            "support_cue_label": "Deepening cue",
        }
        planner_payload = {
            "state_interpretation": {
                "support_need": "deepening",
                "confidence_handling": "Use signal as a support cue only.",
                "context_reasoning": "Engaged follow-up after a baseline explanation.",
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": [
                self.strategy_candidate("deep_technical_explanation", "Deep technical explanation", recommended=True, score=0.95),
                self.strategy_candidate("critique_assumptions", "Critique assumptions", recommended=False, score=0.80),
                self.strategy_candidate("connect_to_related_work", "Connect to related work", recommended=False, score=0.75),
            ],
            "warnings": [],
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value=planner_payload, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-repeat",
                    "source_turn_id": "turn_base",
                    "selection_type": "text",
                    "selected_text": "The discussion connects the method to related work.",
                    "baseline_explanation": "The baseline explanation covered the main discussion point.",
                    "reaction_window_summary": reaction_summary,
                    "support_cue": "deepening",
                    "paper_context": {"passage_type": "discussion", "difficulty_hint": "dense_theory"},
                    "recent_conversation": [
                        {"role": "assistant", "turn_type": "baseline_explanation", "content": "Baseline answer."},
                        {
                            "role": "assistant",
                            "turn_type": "strategy_reexplanation",
                            "content": "Prior adaptive explanation.",
                            "strategy_id": "deep_technical_explanation",
                            "strategy_family": "deep_technical_explanation",
                        },
                    ],
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        candidates = response["json"]["candidates"]
        self.assertNotEqual(candidates[0]["strategy_family"], "deep_technical_explanation")
        self.assertIn(
            candidates[0]["strategy_family"],
            {"critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"},
        )
        snapshot = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{response['json']['prompt_snapshot_id']}")["json"]["snapshot"]
        constraints = snapshot["strategy_planning_context"]["strategy_constraints"]
        self.assertEqual(constraints["previous_strategy_family"], "deep_technical_explanation")
        self.assertEqual(constraints["previous_strategy_id"], "deep_technical_explanation")

    def test_previous_strategy_reexplanation_uses_applied_strategy_family_metadata(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        reaction_summary = {
            "source_turn_id": "turn_adaptive",
            "highlight_id": "h-applied-prev",
            "duration_sec": 8.0,
            "dominant_state": "engagement",
            "secondary_state": "confusion",
            "avg_confidence": 0.82,
            "avg_distribution": {"boredom": 0.04, "confusion": 0.18, "engagement": 0.72, "frustration": 0.06},
            "trend": "stable",
            "support_cue": "deepening",
            "support_cue_label": "Deepening cue",
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-applied-prev",
                    "source_turn_id": "turn_adaptive",
                    "selection_type": "text",
                    "selected_text": "The discussion connects the method to related work.",
                    "baseline_explanation": "Prior adaptive explanation.",
                    "reaction_window_summary": reaction_summary,
                    "support_cue": "deepening",
                    "paper_context": {"passage_type": "discussion", "difficulty_hint": "dense_theory"},
                    "recent_conversation": [
                        {
                            "role": "assistant",
                            "turn_type": "strategy_reexplanation",
                            "content": "Prior adaptive explanation.",
                            "applied_strategy_id": "deep_technical_explanation",
                            "applied_strategy_family": "deep_technical_explanation",
                            "strategy_status": "applied",
                        },
                    ],
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertNotEqual(response["json"]["candidates"][0]["strategy_family"], "deep_technical_explanation")
        snapshot = self.app.test_request("GET", f"/api/llm-compare/prompt-snapshots/{response['json']['prompt_snapshot_id']}")["json"]["snapshot"]
        constraints = snapshot["strategy_planning_context"]["strategy_constraints"]
        self.assertEqual(constraints["previous_strategy_family"], "deep_technical_explanation")
        self.assertEqual(constraints["previous_strategy_id"], "deep_technical_explanation")

    def test_difficulty_support_escalates_after_recent_step_by_step_move(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        reaction_summary = {
            "source_turn_id": "turn_after_step",
            "highlight_id": "h-escalate",
            "duration_sec": 8.0,
            "dominant_state": "frustration",
            "secondary_state": "engagement",
            "avg_confidence": 0.82,
            "avg_distribution": {"boredom": 0.0, "confusion": 0.02, "engagement": 0.16, "frustration": 0.82},
            "trend": "rising",
            "support_cue": "difficulty_support",
            "support_cue_label": "Reduce cognitive load cue",
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-escalate",
                    "source_turn_id": "turn_after_step",
                    "selection_type": "text",
                    "selected_text": "The method retrieves evidence, ranks it, and generates an answer.",
                    "baseline_explanation": "Previous adaptive explanation walked through the method step by step.",
                    "reaction_window_summary": reaction_summary,
                    "support_cue": "difficulty_support",
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "recent_conversation": [
                        {
                            "role": "assistant",
                            "turn_type": "strategy_reexplanation",
                            "content": "Prior adaptive explanation.",
                            "applied_strategy_id": "step_by_step_breakdown",
                            "applied_strategy_family": "step_by_step_breakdown",
                            "applied_pedagogical_move": "Walk through the passage step by step",
                            "applied_title": "Walk through the passage step by step",
                            "strategy_status": "applied",
                        }
                    ],
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        first = response["json"]["candidates"][0]
        self.assertNotEqual(first["normalized_pedagogical_move_key"], "step_by_step")
        self.assertIn(first["normalized_pedagogical_move_key"], {"worked_example", "analogy", "connect_to_paper_argument", "concise_summary"})
        self.assertNotEqual(first["title"].lower(), "walk through the passage step by step")

    def test_candidate_normalization_demotes_near_identical_previous_title(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        planner_payload = {
            "state_interpretation": {
                "support_need": "clarification",
                "confidence_handling": "Use signal as a support cue only.",
                "context_reasoning": "Still needs support.",
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": [
                self.strategy_candidate("step_by_step_breakdown", "Walk through the passage step by step", recommended=True, score=0.99),
                self.strategy_candidate("concrete_example", "Use a concrete example", recommended=False, score=0.70),
                self.strategy_candidate("define_key_terms", "Define the key terms", recommended=False, score=0.68),
            ],
            "warnings": [],
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value=planner_payload, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-title-repeat",
                    "source_turn_id": "turn_title_repeat",
                    "selection_type": "text",
                    "selected_text": "The passage describes a multi-step method.",
                    "baseline_explanation": "Previous adaptive explanation.",
                    "reaction_window_summary": {
                        "source_turn_id": "turn_title_repeat",
                        "highlight_id": "h-title-repeat",
                        "duration_sec": 8.0,
                        "dominant_state": "confusion",
                        "secondary_state": "engagement",
                        "avg_confidence": 0.72,
                        "avg_distribution": {"boredom": 0.02, "confusion": 0.68, "engagement": 0.20, "frustration": 0.10},
                        "trend": "rising",
                        "support_cue": "clarification",
                    },
                    "support_cue": "clarification",
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "recent_conversation": [
                        {
                            "role": "assistant",
                            "turn_type": "strategy_reexplanation",
                            "applied_strategy_id": "step_by_step_breakdown",
                            "applied_strategy_family": "step_by_step_breakdown",
                            "applied_pedagogical_move": "Walk through the passage step by step",
                            "applied_title": "Walk through the passage step by step",
                            "strategy_status": "applied",
                        }
                    ],
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertNotEqual(response["json"]["candidates"][0]["title"].lower(), "walk through the passage step by step")
        self.assertNotEqual(response["json"]["candidates"][0]["normalized_pedagogical_move_key"], "step_by_step")

    def test_strategy_candidates_prefer_process_aware_package_over_direct_state(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        package = {
            "active_source": "raw_8class_process_aware",
            "inferred_process_state": "continued_difficulty",
            "strategy_state": "confusion",
            "recommended_strategy": "worked_example",
            "strategy_reason": "recent learning signal suggested continued difficulty after the previous explanation",
            "prompt_guidance": ["Use a concrete worked example before returning to the paper."],
            "academic_state_mapping": {
                "mapped_academic_state": "confusion",
                "mapped_academic_scores": {"boredom": 0.03, "confusion": 0.74, "engagement": 0.14, "frustration": 0.09},
            },
            "reaction_window_summary": {
                "support_cue": "sustained_clarification",
                "dominant_mapped_states": [{"state": "confusion", "probability": 0.74}],
                "avg_mapped_scores": {"boredom": 0.03, "confusion": 0.74, "engagement": 0.14, "frustration": 0.09},
                "evidence_count": 5,
            },
            "learning_process_context": {
                "highlight_id": "h-process",
                "followup_count_for_highlight": 2,
                "same_highlight_repeated_questions": True,
                "current_user_question": "I still do not understand. Can you give an example?",
                "question_intent": "still_confused",
                "previous_strategy": "simplify_and_define_terms",
                "recent_strategies": ["baseline_explanation", "simplify_and_define_terms"],
            },
            "diagnostic_only_direct_4class": {"state": "engagement", "confidence": 0.95, "used_for_strategy": False},
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-process",
                    "selection_type": "text",
                    "selected_text": "The method retrieves paper evidence before answering.",
                    "learning_state": {"academic_state": "engagement", "confidence": 0.95},
                    "learning_signal_package": package,
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                    "user_question": "I still do not understand. Can you give an example?",
                },
            )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertEqual(payload["learning_signal_package"]["recommended_strategy"], "worked_example")
        self.assertEqual(payload["state_interpretation"]["support_need"], "clarification")
        self.assertEqual(payload["candidates"][0]["strategy_family"], "concrete_example")
        self.assertIn("continued difficulty", payload["candidates"][0]["why_recommended"])
        self.assertIn("confusion 74%", payload["candidates"][0]["academic_state_evidence_text"])

    def test_strategy_candidates_endpoint_returns_contextual_cards_and_logs_metadata(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")
        session_id = started["json"]["session_id"]
        learning_state = {
            "academic_state": "confusion",
            "confidence": 0.81,
            "distribution": {"boredom": 0.04, "confusion": 0.81, "engagement": 0.08, "frustration": 0.07},
            "trend": "rising",
            "duration_sec": 14.2,
            "intensity": 0.81,
        }

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/strategy-candidates",
            {
                "session_id": session_id,
                "highlight_id": "h-method",
                "selection_type": "text",
                "page_number": 1,
                "selected_text": "The system embeds parsed passages and retrieves related chunks.",
                "learning_state": learning_state,
                "paper_context": {
                    "passage_type": "method",
                    "difficulty_hint": "multi_step_process",
                    "matched_block": {"markdown_content": "The system embeds parsed passages."},
                    "nearby_context": "The retrieved chunks support the explanation.",
                },
                "trigger_context": {
                    "triggered_by": "learning_state_monitor",
                    "trigger_reason": "confusion confidence remained above threshold",
                    "duration_sec": 14.2,
                    "trend": "rising",
                    "intensity": 0.81,
                },
            },
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertIn(payload["planner_mode"], {"llm", "heuristic"})
        self.assertEqual(len(payload["candidates"]), 3)
        self.assertIn(payload["state_interpretation"]["support_need"], {"clarification", "neutral"})
        titles = [candidate["title"] for candidate in payload["candidates"]]
        self.assertTrue(any("input" in title.lower() or "step" in title.lower() for title in titles))
        self.assertNotIn("you are confused", json.dumps(payload).lower())
        doc_log = self.app.state.documents_dir / document_id / "logs" / "interactions.jsonl"
        session_log = self.app.state.upload_dir / "sessions" / session_id / "events.jsonl"
        self.assertIn("strategy_candidates_shown", doc_log.read_text(encoding="utf-8"))
        self.assertIn("strategy_candidates_shown", session_log.read_text(encoding="utf-8"))
        self.assertNotIn("GEMINI_API_KEY", doc_log.read_text(encoding="utf-8"))

    def test_strategy_candidates_use_heuristic_fallback_when_llm_planner_fails(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value={"not": "valid"}, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-formula",
                    "selection_type": "text",
                    "page_number": 1,
                    "selected_text": "Let z = W x + b and apply softmax.",
                    "learning_state": {
                        "academic_state": "confusion",
                        "confidence": 0.72,
                        "distribution": {"boredom": 0.05, "confusion": 0.72, "engagement": 0.16, "frustration": 0.07},
                        "trend": "stable",
                        "duration_sec": 10,
                    },
                    "paper_context": {"passage_type": "formula", "difficulty_hint": "formula"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertEqual(response["json"]["planner_mode"], "heuristic")
        self.assertEqual(len(response["json"]["candidates"]), 3)
        self.assertTrue(any(candidate["strategy_id"] == "formula_intuition" for candidate in response["json"]["candidates"]))
        for candidate in response["json"]["candidates"]:
            self.assertIn("strategy_family", candidate)
            self.assertIn("pedagogical_move", candidate)
            self.assertIn("context_focus", candidate)

    def test_strategy_candidate_normalization_constrains_llm_output_to_allowed_families(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        planner_payload = {
            "state_interpretation": {
                "support_need": "deepening",
                "confidence_handling": "Use signal as a cue only.",
                "context_reasoning": "Conceptual passage.",
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": [
                {
                    "strategy_id": "topic_title",
                    "strategy_family": "invented_topic_family",
                    "title": "The tension between AI and authenticity",
                    "short_description": "Discuss the topic.",
                    "why_recommended": "Deepening cue.",
                    "prompt_instruction": "Discuss the topic.",
                    "expected_answer_shape": ["Topic"],
                    "recommended": True,
                    "recommended_score": 0.90,
                },
                self.strategy_candidate("critique_assumptions", "Critique assumptions", recommended=False, score=0.80),
                self.strategy_candidate("limitations_and_implications", "Limitations", recommended=False, score=0.70),
            ],
            "warnings": [],
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value=planner_payload, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-concept",
                    "selection_type": "text",
                    "selected_text": "AI co-authorship complicates the authenticity of personal journaling.",
                    "baseline_explanation": "The passage frames authenticity as a conceptual tension.",
                    "source_turn_id": "turn_base",
                    "reaction_window_summary": {
                        "source_turn_id": "turn_base",
                        "support_cue": "deepening",
                        "support_cue_label": "Deepening cue",
                        "duration_sec": 10,
                        "avg_confidence": 0.82,
                        "avg_distribution": {"boredom": 0.04, "confusion": 0.08, "engagement": 0.78, "frustration": 0.10},
                    },
                    "support_cue": "deepening",
                    "paper_context": {"passage_type": "discussion", "difficulty_hint": "dense_theory"},
                    "trigger_context": {"triggered_by": "reaction_window"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        allowed = set(self.app.state._allowed_strategy_families_for_support_cue("deepening"))
        candidates = response["json"]["candidates"]
        self.assertEqual(sum(1 for candidate in candidates if candidate["recommended"]), 1)
        for candidate in candidates:
            self.assertIn(candidate["strategy_family"], allowed)
            self.assertEqual(candidate["strategy_id"], candidate["strategy_family"])
            self.assertTrue(candidate["pedagogical_move"])
            self.assertTrue(candidate["context_focus"])

    def test_strategy_candidate_normalization_keeps_only_highest_recommended(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        planner_payload = {
            "state_interpretation": {
                "support_need": "clarification",
                "confidence_handling": "Use signal as a cue only.",
                "context_reasoning": "Dense method passage.",
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": [
                self.strategy_candidate("low", "Lower score", recommended=True, score=0.61),
                self.strategy_candidate("high", "Higher score", recommended=True, score=0.91),
                self.strategy_candidate("mid", "Middle score", recommended=False, score=0.72),
            ],
            "warnings": [],
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value=planner_payload, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-method",
                    "selection_type": "text",
                    "selected_text": "The method compares guided and unguided authoring.",
                    "learning_state": {"academic_state": "confusion", "confidence": 0.8, "trend": "stable", "duration_sec": 12},
                    "paper_context": {"passage_type": "method", "difficulty_hint": "multi_step_process"},
                },
            )

        self.assertEqual(response["status"], 200, response)
        candidates = response["json"]["candidates"]
        self.assertEqual(sum(1 for candidate in candidates if candidate["recommended"]), 1)
        self.assertEqual(candidates[0]["strategy_id"], "high")
        self.assertTrue(candidates[0]["recommended"])
        self.assertFalse(next(candidate for candidate in candidates if candidate["strategy_id"] == "low")["recommended"])

    def test_strategy_candidate_normalization_recommends_highest_when_none_marked(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        planner_payload = {
            "state_interpretation": {
                "support_need": "neutral",
                "confidence_handling": "Neutral.",
                "context_reasoning": "Unknown passage.",
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": [
                self.strategy_candidate("first", "First", recommended=False, score=0.5),
                self.strategy_candidate("best", "Best", recommended=False, score=0.83),
                self.strategy_candidate("missing", "Missing score", recommended=False, score=None),
            ],
            "warnings": [],
        }

        with patch.object(self.app.state, "_call_strategy_planner_llm", return_value=planner_payload, create=True):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/strategy-candidates",
                {
                    "highlight_id": "h-neutral",
                    "selection_type": "text",
                    "selected_text": "A passage.",
                    "learning_state": {"academic_state": "confusion", "confidence": 0.8, "trend": "stable", "duration_sec": 12},
                    "paper_context": {"passage_type": "unknown", "difficulty_hint": "unknown"},
                },
            )

        candidates = response["json"]["candidates"]
        self.assertEqual(sum(1 for candidate in candidates if candidate["recommended"]), 1)
        self.assertEqual(candidates[0]["strategy_id"], "best")
        self.assertIsInstance(next(candidate for candidate in candidates if candidate["strategy_id"] == "missing")["recommended_score"], float)

    def test_selected_strategy_and_learning_state_snapshot_persist_into_thread(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        started = self.app.test_request("POST", f"/api/documents/{document_id}/reading-session/start")
        session_id = started["json"]["session_id"]
        selected_strategy = {
            "strategy_id": "input_process_output_map",
            "title": "Map this method as input -> process -> output",
            "short_description": "Break down the selected method.",
            "why_recommended": "This may help with a dense method passage.",
            "prompt_instruction": "Explain the passage as Input, Process, Output, and one small example.",
            "expected_answer_shape": ["Core intuition", "Input", "Process", "Output", "Mini example"],
            "recommended": True,
            "recommended_score": 0.88,
        }
        learning_state = {
            "academic_state": "confusion",
            "confidence": 0.81,
            "distribution": {"boredom": 0.04, "confusion": 0.81, "engagement": 0.08, "frustration": 0.07},
            "trend": "rising",
            "duration_sec": 14.2,
            "intensity": 0.81,
        }

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "session_id": session_id,
                "highlight_id": "h-strategy",
                "highlight_type": "text",
                "page_number": 1,
                "selected_text": "The model retrieves chunks before generating an answer.",
                "text_available": True,
                "recommended_llm_mode": "text_context",
                "matched_block": {"markdown_content": "The model retrieves chunks before generating an answer."},
                "learning_state": learning_state,
                "strategy_candidates": [selected_strategy],
                "selected_strategy_id": selected_strategy["strategy_id"],
                "selected_strategy": selected_strategy,
                "trigger_context": {"triggered_by": "learning_state_monitor", "trigger_reason": "sustained support cue"},
            },
        )
        follow_up = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/threads/h-strategy/follow-up",
            {
                "question": "Can you keep that same structure?",
                "selected_strategy": selected_strategy,
                "selected_strategy_id": selected_strategy["strategy_id"],
                "learning_state": learning_state,
                "trigger_context": {"triggered_by": "follow_up"},
            },
        )

        self.assertEqual(response["status"], 200, response)
        self.assertIn("Selected pedagogical support strategy", response["json"]["prompt_preview"])
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-strategy.json"
        thread = json.loads(thread_path.read_text(encoding="utf-8"))
        self.assertEqual(thread["session_id"], session_id)
        self.assertEqual(thread["learning_state_snapshot"]["academic_state"], "confusion")
        self.assertEqual(thread["selected_strategy_id"], "input_process_output_map")
        self.assertEqual(thread["selected_strategy"]["title"], selected_strategy["title"])
        self.assertEqual(thread["selected_strategy"]["strategy_family"], "input_process_output_map")
        self.assertEqual(thread["selected_strategy"]["pedagogical_move"], "Map the method as input, process, and output")
        self.assertTrue(thread["selected_strategy"]["context_focus"])
        self.assertEqual(thread["strategy_candidates"][0]["strategy_id"], "input_process_output_map")
        self.assertEqual(thread["trigger_context"]["trigger_reason"], "sustained support cue")
        self.assertEqual(follow_up["status"], 200, follow_up)
        follow_up_assistant = follow_up["json"]["thread"]["messages"][-1]
        self.assertEqual(follow_up_assistant["response_mode"], "normal_followup")
        self.assertIsNone(follow_up_assistant["selected_strategy_id"])
        self.assertNotIn("Selected pedagogical support strategy", follow_up["json"]["prompt_preview"])

    def test_baseline_explain_without_selected_strategy_persists_baseline_turn(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "highlight_id": "h-baseline",
                "highlight_type": "text",
                "page_number": 1,
                "selected_text": "The assistant retrieves evidence before answering.",
                "text_available": True,
                "recommended_llm_mode": "text_context",
                "matched_block": {"markdown_content": "The assistant retrieves evidence before answering."},
                "user_question": None,
                "default_task": "baseline_explain_current_selection",
                "selected_strategy_id": "",
                "selected_strategy": None,
            },
        )

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["ok"])
        self.assertIn("answer", response["json"])
        self.assertGreater(len(response["json"]["answer"]), 0)
        self.assertIn("assistant_message", response["json"])
        self.assertEqual(response["json"]["assistant_message"]["content"], response["json"]["answer"])
        self.assertEqual(response["json"]["assistant_message"]["turn_type"], "baseline_explanation")
        self.assertIn("thread", response["json"])
        self.assertNotIn("Selected pedagogical support strategy", response["json"]["prompt_preview"])
        self.assertEqual(response["json"]["thread"]["messages"][-1]["role"], "assistant")
        self.assertGreater(len(response["json"]["thread"]["messages"][-1]["content"]), 0)
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-baseline.json"
        thread = json.loads(thread_path.read_text(encoding="utf-8"))
        message = thread["messages"][-1]
        self.assertEqual(message["role"], "assistant")
        self.assertGreater(len(message["content"]), 0)
        self.assertEqual(message["turn_type"], "baseline_explanation")
        self.assertTrue(message["turn_id"].startswith("turn_"))
        self.assertTrue(message["prompt_snapshot_id"].startswith("snap_"))
        self.assertFalse(message.get("strategy_id"))
        self.assertFalse(thread.get("selected_strategy_id"))
        self.assertNotIn("prompt_text", json.dumps(thread))
        snapshot_path = self.app.state.documents_dir / document_id / "prompt_snapshots" / f"{message['prompt_snapshot_id']}.json"
        self.assertTrue(snapshot_path.exists())

    def test_prompt_snapshot_failure_does_not_prevent_thread_save(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        provider_result = {
            "provider": "mock",
            "model": "mock",
            "mode": "text_context",
            "recommended_llm_mode": "text_context",
            "response_style": "chat_conversational",
            "used_image": False,
            "prompt_preview": "safe preview",
            "answer": "Answer survives snapshot failure.",
            "error": None,
        }

        with patch.object(self.app.state, "_save_prompt_snapshot_for_payload", side_effect=OSError("snapshot write failed")), patch.object(self.app.state, "explain_debug_selection", return_value=provider_result):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/explain-selection",
                {
                    "highlight_id": "h-snapshot-failure",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "Snapshot failure persistence test.",
                    "recommended_llm_mode": "text_context",
                    "default_task": "baseline_explain_current_selection",
                },
            )

        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-snapshot-failure.json"
        thread = json.loads(thread_path.read_text(encoding="utf-8"))
        message = thread["messages"][-1]
        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["ok"])
        self.assertEqual(response["json"]["answer"], "Answer survives snapshot failure.")
        self.assertEqual(message["content"], "Answer survives snapshot failure.")
        self.assertFalse(message.get("prompt_snapshot_id"))
        self.assertIn("prompt_snapshot_error", message)
        self.assertIn("prompt snapshot could not be saved", response["json"]["warnings"][0])

    def test_baseline_explain_copies_provider_answer_to_stable_response_contract(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        provider_result = {
            "provider": "mock",
            "model": "mock",
            "mode": "text_context",
            "recommended_llm_mode": "text_context",
            "response_style": "chat_conversational",
            "used_image": False,
            "prompt_preview": "safe preview",
            "answer": "Provider supplied answer.",
            "error": None,
        }
        with patch.object(self.app.state, "explain_debug_selection", return_value=provider_result):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/explain-selection",
                {
                    "highlight_id": "h-provider-answer",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "Provider answer test.",
                    "recommended_llm_mode": "text_context",
                    "default_task": "baseline_explain_current_selection",
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["ok"])
        self.assertEqual(response["json"]["answer"], "Provider supplied answer.")
        self.assertEqual(response["json"]["assistant_message"]["content"], "Provider supplied answer.")
        self.assertEqual(response["json"]["thread"]["messages"][-1]["content"], "Provider supplied answer.")

    def test_baseline_explain_provider_failure_returns_clear_error_without_empty_message(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        provider_result = {
            "provider": "gemini",
            "model": "gemini-test",
            "mode": "text_context",
            "recommended_llm_mode": "text_context",
            "response_style": "chat_conversational",
            "used_image": False,
            "prompt_preview": "safe preview",
            "answer": "",
            "error": "Gemini request failed: test provider error",
        }
        with patch.object(self.app.state, "explain_debug_selection", return_value=provider_result):
            response = self.app.test_request(
                "POST",
                f"/api/documents/{document_id}/explain-selection",
                {
                    "highlight_id": "h-provider-failure",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "Provider failure test.",
                    "recommended_llm_mode": "text_context",
                    "default_task": "baseline_explain_current_selection",
                },
            )

        self.assertEqual(response["status"], 200, response)
        self.assertFalse(response["json"]["ok"])
        self.assertEqual(response["json"]["answer"], "")
        self.assertIn("Gemini request failed", response["json"]["error"])
        self.assertIsNone(response["json"]["assistant_message"])
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-provider-failure.json"
        self.assertFalse(thread_path.exists())

    def test_strategy_explain_without_user_question_adds_task_and_message_metadata(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        selected_strategy = {
            "strategy_id": "concrete_example",
            "strategy_family": "concrete_example",
            "title": "Use a concrete example for this method passage",
            "pedagogical_move": "Use a concrete example",
            "context_focus": "Anchor the retrieval step in a small paper-reading scenario.",
            "short_description": "Anchor the explanation in a small example.",
            "why_recommended": "This may help with a dense method passage.",
            "prompt_instruction": "Explain using one small example, then connect it back to the paper.",
            "expected_answer_shape": ["Plain-language idea", "Mini example", "Back to the paper"],
            "recommended": True,
            "recommended_score": 0.82,
        }

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "highlight_id": "h-no-question",
                "highlight_type": "text",
                "page_number": 1,
                "selected_text": "The system retrieves relevant chunks before generating an answer.",
                "text_available": True,
                "recommended_llm_mode": "text_context",
                "matched_block": {"markdown_content": "The system retrieves relevant chunks before generating an answer."},
                "user_question": None,
                "default_task": "explain_current_selection_with_selected_strategy",
                "selected_strategy_id": selected_strategy["strategy_id"],
                "selected_strategy": selected_strategy,
                "strategy_candidates": [selected_strategy],
                "learning_state": {
                    "academic_state": "confusion",
                    "confidence": 0.76,
                    "distribution": {"boredom": 0.06, "confusion": 0.76, "engagement": 0.11, "frustration": 0.07},
                    "trend": "stable",
                    "duration_sec": 12,
                },
                "source_turn_id": "turn_base",
                "reaction_window_summary": {
                    "source_turn_id": "turn_base",
                    "highlight_id": "h-no-question",
                    "duration_sec": 12,
                    "support_cue": "sustained_clarification",
                    "support_cue_label": "Sustained clarification cue",
                    "trigger_reason": "The baseline explanation was being read while the learning signal showed a sustained clarification cue.",
                    "avg_confidence": 0.72,
                    "trend": "rising",
                    "avg_distribution": {"boredom": 0.06, "confusion": 0.76, "engagement": 0.11, "frustration": 0.07},
                    "face_detection_summary": {"mode": "center_crop"},
                },
                "trigger_context": {"triggered_by": "reaction_window", "trigger_reason": "sustained support cue"},
            },
        )

        self.assertEqual(response["status"], 200, response)
        self.assertIn("Task:", response["json"]["prompt_preview"])
        self.assertIn("Explain the selected paper passage using the selected pedagogical support strategy.", response["json"]["prompt_preview"])
        thread_path = self.app.state.documents_dir / document_id / "threads" / "h-no-question.json"
        thread = json.loads(thread_path.read_text(encoding="utf-8"))
        message = thread["messages"][-1]
        self.assertEqual(thread["selected_strategy_id"], "concrete_example")
        self.assertEqual(message["strategy_id"], "concrete_example")
        self.assertEqual(message["strategy_title"], selected_strategy["title"])
        self.assertEqual(message["learning_state_snapshot"]["academic_state"], "confusion")
        self.assertEqual(message["trigger_context"]["trigger_reason"], "sustained support cue")
        self.assertTrue(message["turn_id"].startswith("turn_"))
        self.assertEqual(message["strategy_short_description"], selected_strategy["short_description"])
        self.assertEqual(message["turn_type"], "strategy_reexplanation")
        self.assertEqual(message["strategy_status"], "applied")
        self.assertEqual(message["applied_strategy_id"], "concrete_example")
        self.assertEqual(message["applied_strategy_family"], "concrete_example")
        self.assertEqual(message["applied_pedagogical_move"], selected_strategy["pedagogical_move"])
        self.assertEqual(message["applied_context_focus"], selected_strategy["context_focus"])
        self.assertEqual(message["source_reaction_turn_id"], "turn_base")
        self.assertEqual(message["source_turn_id"], "turn_base")
        self.assertEqual(message["reaction_window_summary"]["support_cue"], "sustained_clarification")
        self.assertIn("confusion 76%", message["academic_state_evidence_text"])
        self.assertIn("baseline explanation", message["strategy_reason"])
        self.assertIn("clarification", message["strategy_reason"])
        self.assertNotEqual(message["strategy_reason"], selected_strategy["why_recommended"])
        self.assertEqual(message["planner_prompt_version"], "reaction_strategy_planner_v2")
        self.assertEqual(message["face_detection_summary"]["mode"], "center_crop")
        self.assertEqual(thread["turn_metadata"]["turn_base"]["strategy_status"], "applied")
        self.assertEqual(thread["turn_metadata"]["turn_base"]["applied_strategy_id"], "concrete_example")
        self.assertEqual(thread["turn_metadata"]["turn_base"]["applied_strategy_family"], "concrete_example")

    def test_follow_up_user_and_assistant_messages_share_turn_id(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]
        self.app.test_request(
            "PUT",
            f"/api/documents/{document_id}/threads/h-turn",
            {
                "selection_snapshot": {
                    "highlight_id": "h-turn",
                    "highlight_type": "text",
                    "page_number": 1,
                    "selected_text": "PDF debug test",
                    "recommended_llm_mode": "text_context",
                },
                "selected_strategy": self.strategy_candidate("structured_breakdown", "Structured breakdown"),
                "selected_strategy_id": "structured_breakdown",
                "messages": [{"role": "assistant", "content": "First answer."}],
            },
        )

        response = self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/threads/h-turn/follow-up",
            {"question": "Can you explain the contrast?"},
        )

        self.assertEqual(response["status"], 200, response)
        user_message, assistant_message = response["json"]["thread"]["messages"][-2:]
        self.assertEqual(user_message["role"], "user")
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertTrue(user_message["turn_id"].startswith("turn_"))
        self.assertEqual(user_message["turn_id"], assistant_message["turn_id"])
        self.assertEqual(assistant_message["input_source"], "text")
        self.assertEqual(assistant_message["response_mode"], "normal_followup")
        self.assertIsNone(assistant_message["selected_strategy_id"])
        self.assertFalse(assistant_message.get("strategy_id"))

    def test_interaction_logs_do_not_store_api_keys_or_base64_crop_payload(self):
        upload = self.upload_pdf()
        document_id = upload["document_id"]

        self.app.test_request(
            "POST",
            f"/api/documents/{document_id}/explain-selection",
            {
                "highlight_id": "h-area",
                "highlight_type": "area",
                "page_number": 1,
                "crop_image_data_url": "data:image/png;base64,QUJDRA==",
                "recommended_llm_mode": "image_plus_context",
            },
        )

        log_path = self.app.state.documents_dir / document_id / "logs" / "interactions.jsonl"
        log_text = log_path.read_text(encoding="utf-8")
        self.assertIn("explain_selection", log_text)
        self.assertNotIn("GEMINI_API_KEY", log_text)
        self.assertNotIn("data:image/png;base64", log_text)
        self.assertNotIn("QUJDRA", log_text)

    @staticmethod
    def strategy_candidate(strategy_id: str, title: str, recommended: bool = False, score: float | None = 0.7) -> dict:
        candidate = {
            "strategy_id": strategy_id,
            "title": title,
            "short_description": f"{title} short description.",
            "why_recommended": f"{title} reason.",
            "prompt_instruction": f"{title} instruction.",
            "expected_answer_shape": ["Idea", "Example"],
            "recommended": recommended,
        }
        if score is not None:
            candidate["recommended_score"] = score
        return candidate


if __name__ == "__main__":
    unittest.main()
