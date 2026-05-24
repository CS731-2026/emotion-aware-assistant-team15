import os
import tempfile
import unittest
from pathlib import Path


def tiny_pdf_bytes() -> bytes:
    stream = b"BT /F1 18 Tf 72 720 Td (PDF debug test) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    return pdf


class PdfDebugPageTests(unittest.TestCase):
    def test_pdf_test_page_is_pdf_only_and_uses_callback_safe_loader_props(self):
        html_path = Path("emotion_aware_assistant/web/static/pdf_test.html")
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        style_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.css")
        self.assertTrue(html_path.exists(), "/pdf-test should have its own static HTML page.")
        self.assertTrue(source_path.exists(), "PDF debug page should use a scoped React entry.")
        self.assertTrue(style_path.exists(), "/pdf-test should keep its outer layout CSS isolated.")

        html = html_path.read_text(encoding="utf-8")
        source = source_path.read_text(encoding="utf-8")
        styles = style_path.read_text(encoding="utf-8")
        self.assertIn('id="pdf-test-root"', html)
        self.assertIn('/pdf-workspace/pdf-test.js', html)
        for forbidden in [
            "Load Sample",
            "Selected Passage",
            "Chat",
            "Camera",
            "Emotion",
            "Context Preview",
            "Highlight Selection",
        ]:
            self.assertNotIn(forbidden, html)
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, styles)
        self.assertIn('import "pdfjs-dist/web/pdf_viewer.css";', source)
        self.assertIn('import "react-pdf-highlighter-plus/style/style.css";', source)
        self.assertLess(
            source.index('import "pdfjs-dist/web/pdf_viewer.css";'),
            source.index('import "react-pdf-highlighter-plus/style/style.css";'),
        )
        self.assertIn('import "./pdf_test.css";', source)
        self.assertIn("beforeLoad={(progress) =>", source)
        self.assertIn("errorMessage={(error) =>", source)
        self.assertIn("onError={(error) =>", source)

        for required in [
            "html,",
            "body,",
            "#pdf-test-root",
            ".pdf-test-app",
            ".pdf-test-toolbar",
            ".pdf-test-viewer",
            "overflow: hidden;",
            "overflow: auto;",
            "position: relative;",
        ]:
            self.assertIn(required, styles)
        for forbidden_selector in [
            ".textLayer",
            ".textLayer span",
            ".page",
            ".pdfViewer",
            "canvas",
            "section",
        ]:
            self.assertNotIn(forbidden_selector, styles)

    def test_pdf_test_page_uses_minimal_highlighter_until_runtime_crash_is_resolved(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        source = source_path.read_text(encoding="utf-8")

        for required in [
            "PdfLoader",
            "PdfHighlighter",
            "TextHighlight",
            "AreaHighlight",
            "useHighlightContainerContext",
            'const [pdfUrl, setPdfUrl] = useState("/api/debug/pdf");',
            "document={pdfUrl}",
            "onPdfDocumentLoaded(pdfDocument)",
            "const [highlights, setHighlights] = useState([]);",
            "highlights={highlights}",
            "enableAreaSelection={(event) => areaMode || event.altKey}",
            "utilsRef={(utils) => {",
            "function HighlightContainer()",
        ]:
            self.assertIn(required, source)
        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "pdf-workspace:highlight-created",
            "selectionTip=",
            "pdfScaleValue=",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_highlighting_is_debug_only_without_persisting_or_chat(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        style_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.css")
        source = source_path.read_text(encoding="utf-8")
        styles = style_path.read_text(encoding="utf-8")

        for required in [
            "const [highlights, setHighlights] = useState([]);",
            "const [pendingSelection, setPendingSelection] = useState(null);",
            "const pendingSelectionRef = useRef(null);",
            "const highlighterUtilsRef = useRef(null);",
            "function handleSelection(selection)",
            'if (!["text", "area"].includes(selection.type))',
            "function handleHighlightSelection()",
            "pendingSelectionRef.current = selection;",
            "setPendingSelection(selection);",
            "makeGhostHighlight()",
            "crypto.randomUUID()",
            "setHighlights((currentHighlights) =>",
            "onSelection={handleSelection}",
            "utilsRef={(utils) => {",
            "highlighterUtilsRef.current = utils;",
            "className=\"pdf-test-highlight-button\"",
            "Highlight",
        ]:
            self.assertIn(required, source)
        self.assertIn(".pdf-test-highlight-button", styles)

        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "localStorage",
            "sessionStorage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_has_upload_and_preparation_status_without_old_ui(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        source = source_path.read_text(encoding="utf-8")

        for required in [
            "Upload PDF",
            "handleUploadPdf",
            'fetch("/api/document/upload"',
            "currentDocument",
            "prepareStatus",
            "document_id",
            "file name",
            "page count",
            "parsed blocks count",
            "paper profile status",
            "keyword index status",
            "embedding/File Search index status",
            "Local coordinate context",
            "Global RAG context",
            "retrieval_method",
            "global_rag_context",
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "/api/chat",
            "Camera",
            "Emotion",
            "Load Sample",
            "Selected Passage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_displays_captured_selection_debug_without_backend_calls(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        style_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.css")
        source = source_path.read_text(encoding="utf-8")
        styles = style_path.read_text(encoding="utf-8")

        for required in [
            "const [selectionDebug, setSelectionDebug] = useState",
            "const [lastHighlightDebug, setLastHighlightDebug] = useState",
            "const selectionDebugRef = useRef",
            "window.getSelection()?.toString() || \"\"",
            "const libraryText = selection?.content?.text || \"\";",
            "libraryText,",
            "selectionKeys: Object.keys(selection || {})",
            "console.log(\"[pdf-test] selection\"",
            "console.log(\"[pdf-test] ghost highlight\"",
            "ghost?.content?.text",
            "ghost?.content?.image",
            "normalizedText",
            "textLength",
            "suspicious",
            "pageNumberFromPosition",
            "boundingRect",
            "rects",
            "function PdfDebugPanel",
            "Browser selection text",
            "Library selection text",
            "Final highlight text",
            "Copy Captured Text",
            "Crop preview",
            "navigator.clipboard.writeText",
            "className=\"pdf-test-debug-panel\"",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-test-content",
            ".pdf-test-debug-panel",
            ".pdf-test-debug-panel pre",
            ".pdf-test-debug-grid",
            ".pdf-test-copy-button",
        ]:
            self.assertIn(required_style, styles)

        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "localStorage",
            "sessionStorage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_calls_coordinate_markdown_matcher_after_highlight(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        source = source_path.read_text(encoding="utf-8")

        for required in [
            'const DEBUG_DOCUMENT_ID = "debug-pdf";',
            "const [matchDebug, setMatchDebug] = useState",
            "const [matchStatus, setMatchStatus] = useState",
            "const parsePromiseRef = useRef(null);",
            "matchHighlightToBlocks(highlight, nextHighlightDebug);",
            "async function ensureDebugParse()",
            'postJson("/api/debug/parse", {})',
            "async function matchHighlightToBlocks(highlight, highlightDebug)",
            'postJson("/api/document/match-blocks",',
            "document_id: currentDocumentId",
            "highlight_id: highlightDebug.id",
            "page_number: highlightDebug.pageNumber",
            "selected_text: highlightDebug.normalizedText",
            "viewport_rects: viewportRectsFromPosition(highlight.position)",
            "normalized_rects: normalizedRectsFromPosition(highlight.position)",
            "parser_rects_1000: parserRects1000FromPosition(highlight.position)",
            "position: highlight.position",
            "setMatchDebug({",
            "matchedBlocks: matchResult.matched_blocks || []",
            "previousBlocks: matchResult.previous_blocks || []",
            "nextBlocks: matchResult.next_blocks || []",
            "Matched Markdown Blocks",
            "Matched block id",
            "Block type",
            "Coordinate overlap",
            "Text bonus",
            "Match score",
            "Markdown content",
            "Previous block",
            "Next block",
            "function MatchBlockList",
            "function postJson",
            "fetch(url",
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "pdf-workspace:highlight-created",
            "localStorage",
            "sessionStorage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_area_selection_debug_crop_and_coordinate_match(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        style_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.css")
        source = source_path.read_text(encoding="utf-8")
        styles = style_path.read_text(encoding="utf-8")

        for required in [
            "const [areaMode, setAreaMode] = useState(false);",
            "selectionCropImage",
            "cropImage",
            'pendingSelection.type === "area"',
            "Highlight Area",
            "Area Select",
            "setAreaMode((value) => !value)",
            "enableAreaSelection={(event) => areaMode || event.altKey}",
            "areaSelectionMode={areaMode}",
            "className={`pdf-test-area-button${areaMode ? \" active\" : \"\"}`}",
            "className=\"pdf-test-crop-preview\"",
            "src={lastHighlightDebug.cropImage}",
            "viewport_rects: viewportRectsFromPosition(highlight.position)",
            "normalized_rects: normalizedRectsFromPosition(highlight.position)",
            "selected_text: highlightDebug.normalizedText",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-test-area-button",
            ".pdf-test-area-button.active",
            ".pdf-test-crop-preview",
        ]:
            self.assertIn(required_style, styles)

        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "pdf-workspace:highlight-created",
            "localStorage",
            "sessionStorage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_test_shows_llm_input_preview_without_calling_llm(self):
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx")
        source = source_path.read_text(encoding="utf-8")

        for required in [
            "const llmInputPreview = buildLlmInputPreview(lastHighlightDebug, matchDebug);",
            "LLM Input Preview",
            "Cleaned Prompt Preview",
            "Recommended LLM mode",
            "function buildLlmInputPreview(lastHighlightDebug, matchDebug)",
            "function recommendLlmMode(lastHighlightDebug, matchDebug, matchedBlock)",
            "function buildCleanedPromptPreview(llmInputPreview)",
            "function selectCaptionForHighlight(lastHighlightDebug, matchDebug, matchedBlock)",
            "function shouldShowCaptionFields(llmInputPreview)",
            "const showCaptionFields = shouldShowCaptionFields(llmInputPreview);",
            "selected_caption",
            "caption_confidence",
            "candidate_captions",
            "Candidate captions",
            "function usefulContextBlocks(matchDebug)",
            "function findCaptionBlock(blocks)",
            "function isLowValueContextBlock(block)",
            "function normalizePdfText(text)",
            "highlight_type: lastHighlightDebug.type",
            "page_number: lastHighlightDebug.pageNumber",
            "selected_text: normalizedSelectedText",
            "text_available: hasText",
            "reason: textUnavailableReason(lastHighlightDebug, hasText)",
            '"area_selection"',
            "viewport_rects: matchDebug.viewportRects",
            "normalized_rects: matchDebug.normalizedRects",
            "parser_rects_1000: matchDebug.parserRects1000",
            "crop_image_available: Boolean(lastHighlightDebug.cropImage)",
            "crop_image_data_url: lastHighlightDebug.cropImage || \"\"",
            "crop_image_data_url_length: lastHighlightDebug.cropImage?.length || 0",
            "caption: selectedCaption?.markdown_content || \"\"",
            "candidate_captions: showCaptionDebug ? matchDebug.candidateCaptions || [] : []",
            "nearby_useful_context: contextBlocks.map(summarizeBlock)",
            "matched_block_id",
            "matched_block_type",
            "match_score",
            "coordinate_overlap",
            "text_bonus",
            "matched_block: summarizeBlock(matchedBlock)",
            "previous_block",
            "next_block",
            "recommended_llm_mode",
            "<details className=\"pdf-test-raw-debug\">",
            "<summary>Raw LLM debug payload</summary>",
            '"text_context"',
            '"table_context"',
            '"formula_context"',
            '"image_multimodal"',
            '"image_plus_context"',
            '"fallback_image_only"',
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "/api/chat",
            "/api/document/highlight",
            "pdf-workspace:highlight-created",
            "localStorage",
            "sessionStorage",
        ]:
            self.assertNotIn(forbidden, source)

    def test_debug_pdf_endpoint_supports_full_and_range_requests(self):
        from emotion_aware_assistant.web.server import create_web_app

        pdf_bytes = tiny_pdf_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "debug.pdf"
            pdf_path.write_bytes(pdf_bytes)
            previous = os.environ.get("PDF_DEBUG_PATH")
            os.environ["PDF_DEBUG_PATH"] = str(pdf_path)
            try:
                app = create_web_app(force_dummy_llm=True)
                full = app.test_file_request("/api/debug/pdf")
                self.assertEqual(full["status"], 200)
                self.assertEqual(full["headers"]["Accept-Ranges"], "bytes")
                self.assertEqual(full["headers"]["Content-Length"], str(len(pdf_bytes)))
                self.assertEqual(full["content_type"], "application/pdf")
                self.assertTrue(full["body"].startswith(b"%PDF"))

                partial = app.test_file_request("/api/debug/pdf", headers={"Range": "bytes=0-3"})
                self.assertEqual(partial["status"], 206)
                self.assertEqual(partial["headers"]["Accept-Ranges"], "bytes")
                self.assertEqual(partial["headers"]["Content-Range"], f"bytes 0-3/{len(pdf_bytes)}")
                self.assertEqual(partial["headers"]["Content-Length"], "4")
                self.assertEqual(partial["body"], b"%PDF")
            finally:
                if previous is None:
                    os.environ.pop("PDF_DEBUG_PATH", None)
                else:
                    os.environ["PDF_DEBUG_PATH"] = previous


if __name__ == "__main__":
    unittest.main()
