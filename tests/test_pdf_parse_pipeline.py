import os
import tempfile
import unittest
from pathlib import Path

from tests.test_pdf_debug_page import tiny_pdf_bytes


class PdfParsePipelineTests(unittest.TestCase):
    def test_block_overlap_prefers_same_page_and_returns_neighbors(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p1-title",
                "page_number": 1,
                "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.90, "y2": 0.12},
                "block_type": "title",
                "markdown_content": "# Title",
                "reading_order_index": 0,
            },
            {
                "block_id": "p1-body",
                "page_number": 1,
                "bbox": {"x1": 0.10, "y1": 0.20, "x2": 0.82, "y2": 0.34},
                "block_type": "text",
                "markdown_content": "The matching paragraph.",
                "reading_order_index": 1,
            },
            {
                "block_id": "p1-next",
                "page_number": 1,
                "bbox": {"x1": 0.10, "y1": 0.36, "x2": 0.82, "y2": 0.50},
                "block_type": "text",
                "markdown_content": "The next paragraph.",
                "reading_order_index": 2,
            },
            {
                "block_id": "p2-body",
                "page_number": 2,
                "bbox": {"x1": 0.10, "y1": 0.20, "x2": 0.82, "y2": 0.34},
                "block_type": "text",
                "markdown_content": "Wrong page.",
                "reading_order_index": 3,
            },
        ]
        result = match_blocks_for_rects(
            blocks,
            page_number=1,
            rects=[{"x1": 0.12, "y1": 0.22, "x2": 0.75, "y2": 0.32, "pageNumber": 1}],
        )

        self.assertEqual(result["matched_blocks"][0]["block_id"], "p1-body")
        self.assertEqual(result["previous_blocks"][0]["block_id"], "p1-title")
        self.assertEqual(result["next_blocks"][0]["block_id"], "p1-next")
        self.assertGreater(result["matched_blocks"][0]["coordinate_overlap"], 0.5)
        self.assertLessEqual(result["matched_blocks"][0]["match_score"], 1.0)

    def test_selected_text_does_not_match_blocks_without_coordinate_overlap(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p1-body",
                "page_number": 1,
                "bbox": {"x1": 0.10, "y1": 0.20, "x2": 0.82, "y2": 0.34},
                "block_type": "text",
                "markdown_content": "The matching paragraph text appears here.",
                "reading_order_index": 1,
            },
        ]

        result = match_blocks_for_rects(
            blocks,
            page_number=1,
            rects=[{"x1": 0.10, "y1": 0.70, "x2": 0.82, "y2": 0.80, "pageNumber": 1}],
            selected_text="matching paragraph text",
        )

        self.assertEqual(result["matched_blocks"], [])
        self.assertTrue(result["fallback_required"])

    def test_no_overlap_on_page_three_returns_same_page_neighbors_only(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p1-title",
                "page_number": 1,
                "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.90, "y2": 0.12},
                "block_type": "title",
                "markdown_content": "Research Article",
                "reading_order_index": 0,
            },
            {
                "block_id": "p3-before",
                "page_number": 3,
                "bbox": {"x1": 0.10, "y1": 0.10, "x2": 0.82, "y2": 0.18},
                "block_type": "text",
                "markdown_content": "Earlier page three text.",
                "reading_order_index": 10,
            },
            {
                "block_id": "p3-after",
                "page_number": 3,
                "bbox": {"x1": 0.10, "y1": 0.55, "x2": 0.82, "y2": 0.66},
                "block_type": "text",
                "markdown_content": "Later page three text.",
                "reading_order_index": 11,
            },
        ]

        result = match_blocks_for_rects(
            blocks,
            page_number=3,
            rects=[{"x1": 0.12, "y1": 0.32, "x2": 0.72, "y2": 0.42, "pageNumber": 3}],
        )

        self.assertEqual(result["matched_blocks"], [])
        self.assertEqual([block["block_id"] for block in result["previous_blocks"]], ["p3-before"])
        self.assertEqual([block["block_id"] for block in result["next_blocks"]], ["p3-after"])
        self.assertTrue(all(block["page_number"] == 3 for block in result["previous_blocks"]))
        self.assertTrue(all(block["page_number"] == 3 for block in result["next_blocks"]))
        self.assertTrue(result["fallback_required"])

    def test_area_bounding_rect_is_normalized_for_page_three_matching(self):
        from emotion_aware_assistant.web.server import create_web_app

        with tempfile.TemporaryDirectory() as temp_dir:
            blocks_path = Path(temp_dir) / "blocks_index.json"
            blocks_path.write_text(
                """
                {
                  "blocks": [
                    {
                      "block_id": "debug-pdf-p1-b0",
                      "page_number": 1,
                      "page_idx": 0,
                      "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.90, "y2": 0.12},
                      "block_type": "title",
                      "markdown_content": "Research Article",
                      "reading_order_index": 0
                    },
                    {
                      "block_id": "debug-pdf-p3-b0",
                      "page_idx": 2,
                      "bbox": {"x1": 0.10, "y1": 0.20, "x2": 0.50, "y2": 0.30},
                      "block_type": "image",
                      "markdown_content": "Page three figure.",
                      "reading_order_index": 20
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )

            app = create_web_app(force_dummy_llm=True)
            app.state.documents["debug-pdf"] = {
                "document_id": "debug-pdf",
                "type": "pdf",
                "path": Path(temp_dir) / "original.pdf",
                "parse_status": {"blocks_index_path": str(blocks_path)},
            }

            matched = app.test_request(
                "POST",
                "/api/document/match-blocks",
                {
                    "document_id": "debug-pdf",
                    "highlight_id": "area-p3",
                    "page_number": 3,
                    "selected_text": "",
                    "position": {
                        "boundingRect": {
                            "x1": 100,
                            "y1": 200,
                            "x2": 500,
                            "y2": 300,
                            "width": 1000,
                            "height": 1000,
                            "pageNumber": 3,
                        },
                        "rects": [],
                    },
                },
            )

            self.assertEqual(matched["status"], 200)
            self.assertEqual(matched["json"]["page_number"], 3)
            self.assertEqual(matched["json"]["matched_blocks"][0]["block_id"], "debug-pdf-p3-b0")
            self.assertEqual(matched["json"]["matched_blocks"][0]["page_number"], 3)
            self.assertEqual(matched["json"]["viewport_rects"][0]["x1"], 100.0)
            self.assertAlmostEqual(matched["json"]["normalized_rects"][0]["x1"], 0.1)
            self.assertEqual(matched["json"]["parser_rects_1000"][0]["x1"], 100.0)

    def test_debug_parse_and_match_returns_markdown_block(self):
        from emotion_aware_assistant.web.server import create_web_app

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "debug.pdf"
            pdf_path.write_bytes(tiny_pdf_bytes())
            previous = os.environ.get("PDF_DEBUG_PATH")
            os.environ["PDF_DEBUG_PATH"] = str(pdf_path)
            try:
                app = create_web_app(force_dummy_llm=True)
                parsed = app.test_request("POST", "/api/debug/parse")
                self.assertEqual(parsed["status"], 200)
                self.assertEqual(parsed["json"]["document_id"], "debug-pdf")
                self.assertTrue(parsed["json"]["parsed"]["blocks_index_path"].endswith("blocks_index.json"))

                matched = app.test_request(
                    "POST",
                    "/api/document/match-blocks",
                    {
                        "document_id": "debug-pdf",
                        "page_number": 1,
                        "scaled_rects": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "pageNumber": 1}],
                    },
                )
                self.assertEqual(matched["status"], 200)
                self.assertIn("PDF debug test", matched["json"]["matched_blocks"][0]["markdown_content"])
            finally:
                if previous is None:
                    os.environ.pop("PDF_DEBUG_PATH", None)
                else:
                    os.environ["PDF_DEBUG_PATH"] = previous

    def test_pdf_upload_copies_original_and_starts_parse_job(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        upload = app.test_request(
            "POST",
            "/api/document/upload",
            files={"file": ("parse-me.pdf", tiny_pdf_bytes())},
        )
        self.assertEqual(upload["status"], 200)
        document_id = upload["json"]["document_id"]
        record = app.state.documents[document_id]
        self.assertTrue(str(record["path"]).endswith(f"runtime_uploads/documents/{document_id}/original.pdf"))
        self.assertEqual(record["parse_status"]["status"], "completed")
        self.assertTrue(Path(record["parse_status"]["parsed_dir"]).exists())
        self.assertTrue(Path(record["parse_status"]["blocks_index_path"]).exists())
        self.assertTrue(Path(record["parse_status"]["paper_profile_path"]).exists())
        self.assertTrue(Path(record["parse_status"]["rag_prepare_status_path"]).exists())
        self.assertTrue(Path(record["parse_status"]["embeddings_path"]).exists())
        self.assertIn("embedding_index_status", record["parse_status"])
        self.assertIn("parse_status", upload["json"])

    def test_document_response_prefers_prepared_pdf_page_count(self):
        from emotion_aware_assistant.paper.document import Document, Page
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        app.state.current_document_id = "doc-pages"
        app.state.current_document_type = "pdf"
        app.state.session.document = Document(
            title="Prepared PDF",
            source_path=Path("runtime_uploads/documents/doc-pages/original.pdf"),
            pages=[Page(page_number=1, text="fallback page")],
            metadata={"format": "pdf"},
        )
        app.state.documents["doc-pages"] = {
            "document_id": "doc-pages",
            "type": "pdf",
            "path": Path("runtime_uploads/documents/doc-pages/original.pdf"),
            "parse_status": {"page_count": 4},
        }

        response = app.state._document_response("fallback page")

        self.assertEqual(response["page_count"], 4)

    def test_match_score_fields_are_capped_and_named(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p1-body",
                "page_number": 1,
                "bbox": {"x1": 0.10, "y1": 0.20, "x2": 0.82, "y2": 0.34},
                "block_type": "text",
                "markdown_content": "matching paragraph text with repeated matching paragraph text",
                "reading_order_index": 1,
            },
        ]

        result = match_blocks_for_rects(
            blocks,
            page_number=1,
            rects=[{"x1": 0.10, "y1": 0.20, "x2": 0.82, "y2": 0.34, "pageNumber": 1}],
            selected_text="matching paragraph text",
        )

        block = result["matched_blocks"][0]
        self.assertIn("coordinate_overlap", block)
        self.assertIn("text_bonus", block)
        self.assertIn("match_score", block)
        self.assertLessEqual(block["match_score"], 1.0)
        self.assertLessEqual(block["overlap_score"], 1.0)

    def test_area_caption_candidates_prefer_same_region_caption_below(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p4-fig2-caption",
                "page_number": 4,
                "bbox": {"x1": 0.08, "y1": 0.40, "x2": 0.46, "y2": 0.46},
                "block_type": "caption",
                "markdown_content": "Figure 2. Earlier visual result.",
                "reading_order_index": 10,
            },
            {
                "block_id": "p4-fig3-caption",
                "page_number": 4,
                "bbox": {"x1": 0.54, "y1": 0.40, "x2": 0.92, "y2": 0.46},
                "block_type": "caption",
                "markdown_content": "Figure 3. Later visual result.",
                "reading_order_index": 11,
            },
        ]

        fig2 = match_blocks_for_rects(
            blocks,
            page_number=4,
            rects=[{"x1": 0.08, "y1": 0.12, "x2": 0.46, "y2": 0.38, "pageNumber": 4}],
        )
        fig3 = match_blocks_for_rects(
            blocks,
            page_number=4,
            rects=[{"x1": 0.54, "y1": 0.12, "x2": 0.92, "y2": 0.38, "pageNumber": 4}],
        )

        self.assertEqual(fig2["selected_caption"]["block_id"], "p4-fig2-caption")
        self.assertEqual(fig2["caption_confidence"], "high")
        self.assertEqual(fig2["candidate_captions"][0]["relation"], "below")
        self.assertEqual(fig3["selected_caption"]["block_id"], "p4-fig3-caption")
        self.assertEqual(fig3["caption_confidence"], "high")

    def test_area_caption_candidates_keep_low_confidence_between_two_figures(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p4-fig2-caption",
                "page_number": 4,
                "bbox": {"x1": 0.08, "y1": 0.40, "x2": 0.52, "y2": 0.46},
                "block_type": "caption",
                "markdown_content": "Figure 2. Earlier visual result.",
                "reading_order_index": 10,
            },
            {
                "block_id": "p4-fig3-caption",
                "page_number": 4,
                "bbox": {"x1": 0.48, "y1": 0.40, "x2": 0.92, "y2": 0.46},
                "block_type": "caption",
                "markdown_content": "Figure 3. Later visual result.",
                "reading_order_index": 11,
            },
        ]

        result = match_blocks_for_rects(
            blocks,
            page_number=4,
            rects=[{"x1": 0.43, "y1": 0.12, "x2": 0.57, "y2": 0.38, "pageNumber": 4}],
        )

        self.assertIn(result["caption_confidence"], {"low", "medium"})
        self.assertLessEqual(result["selected_caption"].get("score", 0.0), 0.72)
        self.assertEqual(len(result["candidate_captions"]), 2)

    def test_table_area_can_use_nearest_caption_above(self):
        from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

        blocks = [
            {
                "block_id": "p5-table1-caption",
                "page_number": 5,
                "bbox": {"x1": 0.12, "y1": 0.12, "x2": 0.88, "y2": 0.16},
                "block_type": "caption",
                "markdown_content": "Table 1. Participant demographics.",
                "reading_order_index": 20,
            },
            {
                "block_id": "p5-fig4-caption",
                "page_number": 5,
                "bbox": {"x1": 0.12, "y1": 0.72, "x2": 0.88, "y2": 0.78},
                "block_type": "caption",
                "markdown_content": "Figure 4. A different visual.",
                "reading_order_index": 21,
            },
        ]

        result = match_blocks_for_rects(
            blocks,
            page_number=5,
            rects=[{"x1": 0.12, "y1": 0.18, "x2": 0.88, "y2": 0.54, "pageNumber": 5}],
        )

        self.assertEqual(result["selected_caption"]["block_id"], "p5-table1-caption")
        self.assertEqual(result["selected_caption"]["relation"], "above")
        self.assertIn(result["caption_confidence"], {"medium", "high"})


if __name__ == "__main__":
    unittest.main()
