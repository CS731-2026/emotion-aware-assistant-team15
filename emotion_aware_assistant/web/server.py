from __future__ import annotations

import json
import mimetypes
import errno
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from emotion_aware_assistant.core.config import load_config, load_project_local_env

from .routes import WebRoutes
from .state import WebState


STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAM_ABORT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
FILE_CHUNK_SIZE = 64 * 1024


class RangeNotSatisfiable(ValueError):
    pass


class WebApp:
    def __init__(self, state: WebState):
        self.state = state
        self.routes = WebRoutes(state)

    def test_request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        return self.dispatch(method, path, json_body or {}, files)

    def test_file_request(self, path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        headers = headers or {}
        prefix = "/api/document/file/"
        library_prefix = "/api/documents/"
        try:
            if path == "/api/debug/pdf":
                file_path = debug_pdf_path()
            elif path.startswith(prefix):
                document_id = unquote(path.removeprefix(prefix))
                file_path = self.state.document_file_path(document_id)
            elif path.startswith(library_prefix) and path.endswith("/file"):
                document_id = unquote(path.removeprefix(library_prefix).removesuffix("/file").strip("/"))
                file_path = self.state.document_file_path(document_id)
            elif path.startswith(library_prefix) and path.endswith("/crop") and "/highlights/" in path:
                tail = path.removeprefix(library_prefix)
                document_id, _, highlight_tail = tail.partition("/highlights/")
                highlight_id = unquote(highlight_tail.removesuffix("/crop").strip("/"))
                file_path = self.state.library_crop_path(unquote(document_id.strip("/")), highlight_id)
            else:
                return {"status": 404, "body": b"", "content_type": "text/plain", "headers": {}}
            response = file_response_metadata(file_path, headers.get("Range"))
            with file_path.open("rb") as file:
                file.seek(response["start"])
                body = file.read(response["length"])
            return {
                "status": int(response["status"]),
                "body": body,
                "content_type": response["headers"]["Content-Type"],
                "headers": response["headers"],
            }
        except RangeNotSatisfiable:
            size = file_path.stat().st_size if "file_path" in locals() and file_path.exists() else 0
            return {
                "status": int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE),
                "body": b"",
                "content_type": "text/plain",
                "headers": {"Accept-Ranges": "bytes", "Content-Range": f"bytes */{size}"},
            }
        except Exception as exc:
            return {
                "status": 404,
                "body": str(exc).encode("utf-8"),
                "content_type": "text/plain",
                "headers": {},
            }

    def dispatch(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> dict[str, Any]:
        try:
            payload = self.routes.handle(method, path, json_body or {}, files)
            return {"status": 200, "json": payload}
        except KeyError as exc:
            return {"status": 404, "json": {"error": str(exc)}}
        except Exception as exc:
            return {"status": 400, "json": {"error": str(exc)}}


def create_web_app(config_path: str = "config.yaml", force_dummy_llm: bool = False, load_local_env: bool = False) -> WebApp:
    if load_local_env:
        load_project_local_env()
    return WebApp(WebState(load_config(config_path), force_dummy_llm=force_dummy_llm))


def run_web_server(host: str = "127.0.0.1", port: int = 8000, config_path: str = "config.yaml") -> None:
    env_result = load_project_local_env()
    if env_result.get("present"):
        print("Loaded local environment configuration from .env.local")
    app = create_web_app(config_path=config_path, load_local_env=False)

    class Handler(WebRequestHandler):
        web_app = app

    server = _bind_server(host, port, Handler)
    actual_host, actual_port = server.server_address
    print(f"Web app running at http://{actual_host}:{actual_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _bind_server(host: str, port: int, handler_cls) -> ThreadingHTTPServer:
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), handler_cls)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if candidate == port:
                print(f"Port {port} is already in use; trying the next available port.")
    raise OSError(f"No available port found in range {port}-{port + 19}.")


class WebRequestHandler(BaseHTTPRequestHandler):
    web_app: WebApp

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/settings", "/settings/", "/local-settings", "/local-settings/"):
            if not self._is_local_client():
                self._send_json({"status": 403, "json": {"error": "Local settings are only available from localhost."}})
                return
            self._send_static("/local_settings.html")
            return
        if path in ("/camera-debug", "/camera-debug/", "/emotion-debug", "/emotion-debug/"):
            if not self._is_local_client():
                self._send_json({"status": 403, "json": {"error": "Camera debug is only available from localhost."}})
                return
            self._send_static("/camera_debug.html")
            return
        if path in ("/llm-compare", "/llm-compare/"):
            if not self._is_local_client():
                self._send_json({"status": 403, "json": {"error": "LLM compare is only available from localhost."}})
                return
            self._send_static("/llm_compare.html")
            return
        if path == "/api/debug/pdf":
            self._send_debug_pdf()
            return
        if path.startswith("/api/document/file/"):
            self._send_document_file(path)
            return
        if path.startswith("/api/documents/") and path.endswith("/crop") and "/highlights/" in path:
            self._send_document_library_crop(path)
            return
        if path.startswith("/api/documents/") and path.endswith("/file"):
            self._send_document_library_file(path)
            return
        if path.startswith("/api/"):
            if (path.startswith("/api/local-config") or path.startswith("/api/camera-debug") or path.startswith("/api/llm-compare")) and not self._is_local_client():
                self._send_json({"status": 403, "json": {"error": "This local-development API is only available from localhost."}})
                return
            route_path = path + (f"?{parsed.query}" if parsed.query else "")
            self._send_json(self.web_app.dispatch("GET", route_path))
            return
        if path in ("/pdf-test", "/pdf-test/", "/static/pdf_test.html"):
            self._send_static("/pdf_test.html")
            return
        if path in ("/pdf-chat", "/pdf-chat/", "/static/pdf_chat.html"):
            self._send_static("/pdf_chat.html")
            return
        self._send_static(path)

    def do_POST(self) -> None:
        self._handle_json_or_multipart("POST")

    def do_PUT(self) -> None:
        self._handle_json_or_multipart("PUT")

    def do_DELETE(self) -> None:
        self._handle_json_or_multipart("DELETE")

    def _handle_json_or_multipart(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/"):
            self._send_json({"status": 404, "json": {"error": "Not found"}})
            return
        if (path.startswith("/api/local-config") or path.startswith("/api/camera-debug") or path.startswith("/api/llm-compare")) and not self._is_local_client():
            self._send_json({"status": 403, "json": {"error": "This local-development API is only available from localhost."}})
            return
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        if content_type.startswith("multipart/form-data"):
            data, files = self._parse_multipart(content_type, body)
            self._send_json(self.web_app.dispatch(method, path, data, files))
            return
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError as exc:
            self._send_json({"status": 400, "json": {"error": f"Invalid JSON: {exc}"}})
            return
        self._send_json(self.web_app.dispatch(method, path, data))

    def log_message(self, format: str, *args) -> None:
        return

    def _is_local_client(self) -> bool:
        host = self.client_address[0] if self.client_address else ""
        return host in {"127.0.0.1", "::1", "localhost"} or host.startswith("127.")

    def _send_json(self, response: dict[str, Any]) -> None:
        status = int(response.get("status", 200))
        body = json.dumps(response.get("json", {}), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        if path in ("", "/"):
            target = STATIC_DIR / "index.html"
        else:
            relative = Path(unquote(path.lstrip("/")))
            target = STATIC_DIR / relative
        try:
            target = target.resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                raise FileNotFoundError
            if not target.exists() or not target.is_file():
                raise FileNotFoundError
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return
        except FileNotFoundError:
            body = b"Not found"
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return

    def _send_document_file(self, path: str) -> None:
        prefix = "/api/document/file/"
        try:
            document_id = unquote(path.removeprefix(prefix))
            target = self.web_app.state.document_file_path(document_id)
            self._send_file(target)
        except STREAM_ABORT_ERRORS:
            return
        except Exception as exc:
            body = str(exc).encode("utf-8")
            try:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return

    def _send_document_library_file(self, path: str) -> None:
        prefix = "/api/documents/"
        try:
            document_id = unquote(path.removeprefix(prefix).removesuffix("/file").strip("/"))
            target = self.web_app.state.document_file_path(document_id)
            self._send_file(target)
        except STREAM_ABORT_ERRORS:
            return
        except Exception as exc:
            body = str(exc).encode("utf-8")
            try:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return

    def _send_document_library_crop(self, path: str) -> None:
        prefix = "/api/documents/"
        try:
            tail = path.removeprefix(prefix)
            document_id, _, highlight_tail = tail.partition("/highlights/")
            highlight_id = unquote(highlight_tail.removesuffix("/crop").strip("/"))
            target = self.web_app.state.library_crop_path(unquote(document_id.strip("/")), highlight_id)
            self._send_file(target)
        except STREAM_ABORT_ERRORS:
            return
        except Exception as exc:
            body = str(exc).encode("utf-8")
            try:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return

    def _send_debug_pdf(self) -> None:
        try:
            self._send_file(debug_pdf_path())
        except STREAM_ABORT_ERRORS:
            return
        except Exception as exc:
            body = str(exc).encode("utf-8")
            try:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except STREAM_ABORT_ERRORS:
                return

    def _send_file(self, target: Path) -> None:
        try:
            response = file_response_metadata(target, self.headers.get("Range"))
        except RangeNotSatisfiable:
            size = target.stat().st_size if target.exists() else 0
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(response["status"])
        for key, value in response["headers"].items():
            self.send_header(key, value)
        self.end_headers()
        remaining = response["length"]
        with target.open("rb") as file:
            file.seek(response["start"])
            while remaining > 0:
                chunk = file.read(min(FILE_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
        match = re.search(r"boundary=(.+)", content_type)
        if not match:
            return {}, {}
        boundary = ("--" + match.group(1).strip().strip('"')).encode("utf-8")
        data: dict[str, Any] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for part in body.split(boundary):
            part = part.strip()
            if not part or part == b"--":
                continue
            if b"\r\n\r\n" not in part:
                continue
            raw_headers, content = part.split(b"\r\n\r\n", 1)
            content = content.rstrip(b"\r\n-")
            headers = raw_headers.decode("utf-8", errors="replace")
            name_match = re.search(r'name="([^"]+)"', headers)
            if not name_match:
                continue
            name = name_match.group(1)
            filename_match = re.search(r'filename="([^"]*)"', headers)
            if filename_match:
                files[name] = (filename_match.group(1), content)
            else:
                data[name] = content.decode("utf-8", errors="replace")
        return data, files


def debug_pdf_path() -> Path:
    configured = os.environ.get("PDF_DEBUG_PATH")
    target = Path(configured).expanduser().resolve() if configured else Path("runtime_uploads/debug/test.pdf").resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(
            "Debug PDF not found. Set PDF_DEBUG_PATH or copy a PDF to runtime_uploads/debug/test.pdf."
        )
    if target.suffix.lower() != ".pdf":
        raise ValueError("PDF_DEBUG_PATH must point to a PDF file.")
    return target


def file_response_metadata(target: Path, range_header: str | None = None) -> dict[str, Any]:
    size = target.stat().st_size
    start = 0
    end = max(size - 1, 0)
    status = HTTPStatus.OK
    headers = {
        "Content-Type": mimetypes.guess_type(str(target))[0] or "application/pdf",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
    }
    if range_header:
        start, end = parse_range_header(range_header, size)
        status = HTTPStatus.PARTIAL_CONTENT
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = max(0, end - start + 1) if size else 0
    headers["Content-Length"] = str(length)
    return {"status": status, "headers": headers, "start": start, "end": end, "length": length}


def parse_range_header(range_header: str, size: int) -> tuple[int, int]:
    if size <= 0:
        raise RangeNotSatisfiable("Cannot range an empty file.")
    value = range_header.strip()
    if not value.startswith("bytes="):
        raise RangeNotSatisfiable("Only byte ranges are supported.")
    spec = value.removeprefix("bytes=").strip()
    if "," in spec or "-" not in spec:
        raise RangeNotSatisfiable("Only a single byte range is supported.")
    start_text, end_text = spec.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError
            start = max(size - suffix_length, 0)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as exc:
        raise RangeNotSatisfiable("Invalid byte range.") from exc
    if start < 0 or end < start or start >= size:
        raise RangeNotSatisfiable("Byte range is outside the file.")
    return start, min(end, size - 1)
