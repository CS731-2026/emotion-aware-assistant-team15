import base64
import importlib.util
import json
import unittest
from pathlib import Path


SELECTED = (
    "The proposed method receives a selected passage, surrounding paper context, "
    "and a compact learning-state signal."
)


def pymupdf_imports() -> bool:
    try:
        import fitz  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


class PdfReaderUpgradeTests(unittest.TestCase):
    def test_frontend_declares_react_pdf_workspace_spike_without_removing_fallback_reader(self):
        root = Path("emotion_aware_assistant/web/static")
        workspace = Path("emotion_aware_assistant/web/pdf_workspace/src/main.jsx")
        bundle = root / "pdf-workspace" / "pdf-workspace.js"
        package_json = Path("package.json")
        html = (root / "index.html").read_text(encoding="utf-8")
        js = (root / "app.js").read_text(encoding="utf-8")
        workspace_source = workspace.read_text(encoding="utf-8")
        package_data = json.loads(package_json.read_text(encoding="utf-8"))

        for element_id in ["pdfWorkspaceRoot", "pdfViewer", "paperText", "highlightSelectionBtn"]:
            self.assertIn(element_id, html)
        self.assertIn("Highlight Selection", html)
        self.assertIn("Explain Selected", html)
        self.assertIn("PdfWorkspaceIsland.mount", js)
        self.assertIn("showPdfReader", js)
        self.assertIn("legacyReaderControls", html)
        self.assertIn("legacySelectionPanel", html)
        self.assertIn("setLegacyReaderVisible(false)", js)
        self.assertIn("react-pdf-highlighter-plus", package_data["dependencies"])
        self.assertIn("PdfHighlighter", workspace_source)
        self.assertIn("TextHighlight", workspace_source)
        self.assertIn("AreaHighlight", workspace_source)
        self.assertIn("areaSelectionMode", workspace_source)
        self.assertIn("screenshot_fallback", workspace_source)
        self.assertIn("pdf-explanation-rail", workspace_source)
        self.assertIn("GlobalWorkerOptions.workerSrc = pdfWorkerUrl", workspace_source)
        self.assertIn("workerSrc={pdfWorkerUrl}", workspace_source)
        self.assertIn('base: "/pdf-workspace/"', Path("vite.pdf-workspace.config.mjs").read_text(encoding="utf-8"))
        self.assertTrue(bundle.exists(), "Run `npm run build:pdf-workspace` after changing the React PDF workspace.")

    def test_sample_and_txt_upload_keep_working_with_document_metadata(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        sample = app.test_request("POST", "/api/document/load-sample")
        self.assertEqual(sample["status"], 200)
        self.assertTrue(sample["json"]["document_id"])
        self.assertEqual(sample["json"]["type"], "txt")
        self.assertIsNone(sample["json"]["pdf_url"])

        upload = app.test_request(
            "POST",
            "/api/document/upload",
            files={"file": ("reader_upload.txt", b"Title: Reader Upload\n\nMethod\nSelection context still works.")},
        )
        self.assertEqual(upload["status"], 200)
        self.assertTrue(upload["json"]["document_id"])
        self.assertEqual(upload["json"]["type"], "txt")
        self.assertIn("Reader Upload", upload["json"]["current_page_text"])

    def test_pdf_upload_still_renders_when_pymupdf_is_missing(self):
        if pymupdf_imports():
            self.skipTest("PyMuPDF is installed in this environment.")

        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        response = app.test_request(
            "POST",
            "/api/document/upload",
            files={"file": ("paper.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")},
        )
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["json"]["type"], "pdf")
        self.assertTrue(response["json"]["document_id"])
        self.assertTrue(response["json"]["pdf_url"].startswith("/api/document/file/"))
        self.assertEqual(response["json"]["metadata"]["text_extraction_status"], "missing_pymupdf")
        self.assertIn("PDF text extraction unavailable", response["json"]["current_page_text"])

        served = app.test_file_request(response["json"]["pdf_url"])
        self.assertEqual(served["status"], 200)
        self.assertEqual(served["content_type"], "application/pdf")

    def test_pdf_upload_extracts_pages_and_serves_file_when_pymupdf_is_available(self):
        if not pymupdf_imports():
            self.skipTest("PyMuPDF is not installed in this environment.")

        import fitz  # type: ignore
        from emotion_aware_assistant.web.server import create_web_app

        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PDF Reader Test\n\nThe highlighted passage comes from a PDF page.")
        pdf_bytes = pdf.tobytes()
        pdf.close()

        app = create_web_app(force_dummy_llm=True)
        upload = app.test_request(
            "POST",
            "/api/document/upload",
            files={"file": ("reader.pdf", pdf_bytes)},
        )
        self.assertEqual(upload["status"], 200)
        self.assertEqual(upload["json"]["type"], "pdf")
        self.assertEqual(upload["json"]["page_count"], 1)
        self.assertTrue(upload["json"]["pdf_url"].startswith("/api/document/file/"))
        self.assertIn("highlighted passage", upload["json"]["current_page_text"])

        served = app.test_file_request(upload["json"]["pdf_url"])
        self.assertEqual(served["status"], 200)
        self.assertEqual(served["content_type"], "application/pdf")
        self.assertTrue(served["body"].startswith(b"%PDF"))

    def test_highlight_stores_selected_text_context_and_chat_log_metadata(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        loaded = app.test_request("POST", "/api/document/load-sample")
        self.assertEqual(loaded["status"], 200)
        document_id = loaded["json"]["document_id"]

        highlight = app.test_request(
            "POST",
            "/api/document/highlight",
            {
                "document_id": document_id,
                "page_number": 1,
                "selected_text": SELECTED,
                "rects": [{"left": 10, "top": 20, "width": 180, "height": 18}],
                "color": "yellow",
            },
        )
        self.assertEqual(highlight["status"], 200)
        highlight_id = highlight["json"]["highlight_id"]
        self.assertEqual(highlight["json"]["context"]["selected_text"], SELECTED)
        self.assertEqual(highlight["json"]["context"]["document_id"], document_id)
        self.assertEqual(highlight["json"]["highlight"]["rects"][0]["width"], 180.0)

        stored = app.test_request("GET", f"/api/document/highlights/{document_id}")
        self.assertEqual(stored["status"], 200)
        self.assertEqual(stored["json"]["highlights"][0]["highlight_id"], highlight_id)

        chat = app.test_request(
            "POST",
            "/api/chat",
            {
                "document_id": document_id,
                "highlight_id": highlight_id,
                "selected_text": SELECTED,
                "page_number": 1,
                "user_question": "Can you explain the highlighted passage?",
                "model_alias": "dummy",
            },
        )
        self.assertEqual(chat["status"], 200)
        self.assertEqual(chat["json"]["highlight_id"], highlight_id)
        self.assertEqual(chat["json"]["context_debug"]["selected_text"], SELECTED)
        self.assertIn(SELECTED, chat["json"]["prompt_preview"])

        log_record = json.loads(app.state.session.logger.path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(log_record["document_id"], document_id)
        self.assertEqual(log_record["document_type"], "txt")
        self.assertEqual(log_record["highlight_id"], highlight_id)
        self.assertEqual(log_record["selected_text_preview"], SELECTED[:240])
        self.assertEqual(log_record["page_number"], 1)

    def test_spike_highlight_metadata_supports_scaled_area_and_screenshot_fallback(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        loaded = app.test_request("POST", "/api/document/load-sample")
        self.assertEqual(loaded["status"], 200)
        document_id = loaded["json"]["document_id"]
        image_data = "data:image/png;base64," + base64.b64encode(b"fake-png-bytes").decode("ascii")

        area = app.test_request(
            "POST",
            "/api/document/highlight",
            {
                "document_id": document_id,
                "page_number": 1,
                "highlight_type": "area",
                "selected_text": "",
                "rects": [{"left": 420, "top": 60, "width": 160, "height": 140}],
                "scaled_rects": [],
                "position": {
                    "boundingRect": {"x1": 0.68, "y1": 0.08, "x2": 0.94, "y2": 0.26, "width": 0.26, "height": 0.18, "pageNumber": 1},
                    "rects": [],
                },
                "cropped_image": image_data,
                "column_side": "right",
                "user_question": "Explain this table.",
            },
        )
        self.assertEqual(area["status"], 200)
        area_highlight = area["json"]["highlight"]
        self.assertEqual(area_highlight["highlight_type"], "area")
        self.assertEqual(area_highlight["column_side"], "right")
        self.assertLess(area_highlight["text_confidence"], 0.5)
        self.assertTrue(area_highlight["cropped_image_path"])
        self.assertTrue(Path(area_highlight["cropped_image_path"]).exists())
        self.assertEqual(area_highlight["scaled_rects"][0]["x1"], 0.68)
        self.assertEqual(area_highlight["explanation_thread"]["rail_side"], "left")

        garbled = app.test_request(
            "POST",
            "/api/document/highlight",
            {
                "document_id": document_id,
                "page_number": 1,
                "highlight_type": "text",
                "selected_text": "m e t h o d \ufffd \ufffd 9 9 ;;",
                "rects": [{"left": 40, "top": 80, "width": 220, "height": 24}],
                "scaled_rects": [{"x1": 0.06, "y1": 0.10, "x2": 0.38, "y2": 0.14, "width": 0.32, "height": 0.04, "pageNumber": 1}],
                "cropped_image": image_data,
                "column_side": "left",
            },
        )
        self.assertEqual(garbled["status"], 200)
        fallback_highlight = garbled["json"]["highlight"]
        self.assertEqual(fallback_highlight["highlight_type"], "screenshot_fallback")
        self.assertLess(fallback_highlight["text_confidence"], 0.5)
        self.assertEqual(fallback_highlight["explanation_thread"]["rail_side"], "right")
        self.assertTrue(fallback_highlight["cropped_image_path"])


if __name__ == "__main__":
    unittest.main()
