from __future__ import annotations

from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from .state import WebState


RouteHandler = Callable[[dict[str, Any]], dict[str, Any]]


class WebRoutes:
    def __init__(self, state: WebState):
        self.state = state

    def handle(self, method: str, path: str, data: dict[str, Any] | None = None, files: dict[str, Any] | None = None):
        data = data or {}
        method = method.upper()
        path, query = self._split_query(path)
        if path == "/api/documents" and method == "GET":
            include_hidden = self._truthy_query(query, "show_debug_docs")
            return self.state.list_library_documents(include_hidden=include_hidden)
        if path == "/api/documents/upload" and method == "POST":
            if not files or "file" not in files:
                raise ValueError("PDF file upload is required.")
            filename, content = files["file"]
            return self.state.upload_library_document(filename, content)
        document_route = self._document_route(path)
        if document_route:
            document_id, tail = document_route
            if method == "GET" and tail == "":
                return self.state.library_document_detail(document_id)
            if method == "POST" and tail == "open":
                return self.state.open_library_document(document_id, data)
            if method == "POST" and tail == "archive":
                return self.state.archive_library_document(document_id)
            if method == "POST" and tail == "reading-session/start":
                return self.state.start_reading_session(document_id)
            if method == "GET" and tail == "highlights":
                return self.state.get_library_highlights(document_id)
            if method == "PUT" and tail == "highlights":
                return self.state.save_library_highlights(document_id, data)
            if method == "POST" and tail == "explain-selection":
                return self.state.explain_library_selection(document_id, data)
            if method == "POST" and tail == "strategy-candidates":
                return self.state.strategy_candidates(document_id, data)
            parts = tail.split("/") if tail else []
            if len(parts) == 2 and parts[0] == "highlights" and method == "DELETE":
                return self.state.delete_library_highlight(document_id, parts[1])
            if len(parts) == 3 and parts[0] == "highlights" and parts[2] == "crop" and method == "POST":
                return self.state.save_library_crop(document_id, parts[1], data)
            if len(parts) == 2 and parts[0] == "threads":
                if method == "GET":
                    return self.state.get_library_thread(document_id, parts[1])
                if method == "PUT":
                    return self.state.save_library_thread(document_id, parts[1], data)
            if len(parts) == 3 and parts[0] == "threads" and parts[2] == "follow-up" and method == "POST":
                return self.state.follow_up_library_thread(document_id, parts[1], data)
            if len(parts) == 3 and parts[0] == "threads" and parts[2] == "clear" and method == "POST":
                return self.state.clear_library_thread(document_id, parts[1])
            if len(parts) == 4 and parts[0] == "threads" and parts[2] == "turns" and method == "DELETE":
                return self.state.delete_library_thread_turn(document_id, parts[1], parts[3])
        session_route = self._reading_session_route(path)
        if session_route:
            session_id, tail = session_route
            if method == "GET" and tail == "learning-state/current":
                return self.state.current_learning_state(session_id)
            if method == "POST" and tail == "emotion/frame":
                return self.state.session_frame_emotion(session_id, data)
            if method == "POST" and tail == "events":
                return self.state.record_reading_session_event(session_id, data)
        if method == "GET" and path == "/api/status":
            return self.state.status()
        if method == "GET" and path == "/api/local-config/status":
            return self.state.local_config_status()
        if method == "GET" and path == "/api/local-config/llm/status":
            return self.state.local_llm_status()
        if method == "POST" and path == "/api/local-config/llm/provider":
            return self.state.save_local_llm_provider(data)
        if method == "POST" and path == "/api/local-config/llm/roles":
            return self.state.save_local_llm_roles(data)
        if method == "GET" and path == "/api/local-config/llm/comparison-models":
            return self.state.local_llm_comparison_models()
        if method == "PUT" and path == "/api/local-config/llm/comparison-models":
            return self.state.save_local_llm_comparison_models(data)
        if method == "POST" and path == "/api/local-config/llm/test":
            return self.state.test_local_llm_config(data)
        if method == "POST" and path == "/api/local-config/gemini":
            return self.state.save_local_gemini_config(data)
        if method == "POST" and path == "/api/local-config/face-detector":
            return self.state.save_local_face_detector_config(data)
        if method == "POST" and path == "/api/local-config/face-crop":
            return self.state.save_local_face_crop_config(data)
        if method == "POST" and path == "/api/local-config/openface":
            return self.state.save_local_openface_config(data)
        if method == "POST" and path == "/api/local-config/emotion-checkpoint":
            return self.state.save_local_emotion_checkpoint_config(data)
        if method == "POST" and path == "/api/local-config/test-gemini":
            return self.state.test_local_gemini_config()
        if method == "GET" and path == "/api/llm-compare/prompt-snapshots":
            return self.state.list_llm_prompt_snapshots(self._first_query_values(query))
        if method == "GET" and path.startswith("/api/llm-compare/prompt-snapshots/"):
            snapshot_id = unquote(path.removeprefix("/api/llm-compare/prompt-snapshots/").strip("/"))
            return self.state.get_llm_prompt_snapshot(snapshot_id)
        if method == "POST" and path == "/api/llm-compare/run":
            return self.state.run_llm_comparison(data)
        if method == "POST" and path == "/api/llm-compare/save":
            return self.state.save_llm_comparison(data)
        if method == "GET" and path == "/api/llm-compare/list":
            return self.state.list_llm_comparisons()
        if method == "GET" and path.startswith("/api/llm-compare/"):
            comparison_id = unquote(path.removeprefix("/api/llm-compare/").strip("/"))
            return self.state.get_llm_comparison(comparison_id)
        if method == "GET" and path == "/api/emotion/model/status":
            return self.state.emotion_model_status()
        if method == "GET" and path == "/api/camera-debug/status":
            return self.state.camera_debug_status()
        if method == "POST" and path == "/api/camera-debug/analyze-frame":
            return self.state.camera_debug_analyze_frame(data)
        if method == "POST" and path == "/api/camera-debug/reaction-summary":
            return self.state.camera_debug_reaction_summary(data)
        if method == "POST" and path == "/api/document/load-sample":
            return self.state.load_sample()
        if method == "POST" and path == "/api/document/upload":
            if files and "file" in files:
                filename, content = files["file"]
                return self.state.upload_document(filename, content)
            filename = str(data.get("filename") or "uploaded.txt")
            content = data.get("content", "")
            if isinstance(content, str):
                content = content.encode("utf-8")
            return self.state.upload_document(filename, content)
        if method == "GET" and path.startswith("/api/document/page/"):
            page_number = int(path.rsplit("/", 1)[1])
            return self.state.get_page(page_number)
        if method == "POST" and path == "/api/document/context":
            return self.state.build_context(data)
        if method == "POST" and path == "/api/document/highlight":
            return self.state.add_highlight(data)
        if method == "GET" and path.startswith("/api/document/highlights/"):
            document_id = path.rsplit("/", 1)[1]
            return self.state.get_highlights(document_id)
        if method == "POST" and path == "/api/debug/parse":
            return self.state.parse_debug_document()
        if method == "POST" and path == "/api/debug/explain-selection":
            return self.state.explain_debug_selection(data)
        if method == "POST" and path == "/api/document/parse":
            document_id = str(data.get("document_id") or self.state.current_document_id or "")
            return self.state.start_parse_job(document_id)
        if method == "GET" and path.startswith("/api/document/parse-status/"):
            document_id = path.rsplit("/", 1)[1]
            return self.state.parse_status(document_id)
        if method == "POST" and path == "/api/document/match-blocks":
            return self.state.match_parsed_blocks(data)
        if method == "POST" and path == "/api/document/retrieve-context":
            return self.state.retrieve_context(data)
        if method == "POST" and path == "/api/emotion/manual":
            return self.state.manual_emotion(data)
        if method == "POST" and path == "/api/emotion/frame":
            return self.state.frame_emotion(data)
        if method == "GET" and path == "/api/emotion/state":
            return self.state.emotion_state()
        if method == "POST" and path == "/api/chat":
            return self.state.chat(data)
        if method == "POST" and path == "/api/speech/transcribe":
            return self.state.speech_transcribe()
        raise KeyError(f"No route for {method} {path}")

    @staticmethod
    def _document_route(path: str) -> tuple[str, str] | None:
        prefix = "/api/documents/"
        if not path.startswith(prefix):
            return None
        remainder = path.removeprefix(prefix).strip("/")
        if not remainder:
            return None
        parts = remainder.split("/", 1)
        document_id = unquote(parts[0])
        tail = unquote(parts[1]) if len(parts) > 1 else ""
        return document_id, tail

    @staticmethod
    def _split_query(path: str) -> tuple[str, dict[str, list[str]]]:
        if "?" not in path:
            return path, {}
        clean_path, query_string = path.split("?", 1)
        return clean_path, parse_qs(query_string)

    @staticmethod
    def _reading_session_route(path: str) -> tuple[str, str] | None:
        prefix = "/api/reading-sessions/"
        if not path.startswith(prefix):
            return None
        remainder = path.removeprefix(prefix).strip("/")
        if not remainder:
            return None
        parts = remainder.split("/", 1)
        session_id = unquote(parts[0])
        tail = unquote(parts[1]) if len(parts) > 1 else ""
        return session_id, tail

    @staticmethod
    def _truthy_query(query: dict[str, list[str]], key: str) -> bool:
        values = query.get(key) or []
        return any(str(value).lower() in {"1", "true", "yes", "on"} for value in values)

    @staticmethod
    def _first_query_values(query: dict[str, list[str]]) -> dict[str, str]:
        return {
            key: str(values[0])
            for key, values in query.items()
            if values
        }
