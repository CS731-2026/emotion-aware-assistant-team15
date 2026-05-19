import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def sample_blocks():
    return [
        {
            "block_id": "doc-p1-b0",
            "page_number": 1,
            "page_idx": 0,
            "bbox": {"x1": 0.05, "y1": 0.04, "x2": 0.90, "y2": 0.10},
            "block_type": "title",
            "markdown_content": "Identifying Hearing Difficulty Moments in Conversational Audio",
            "reading_order_index": 0,
        },
        {
            "block_id": "doc-p1-b1",
            "page_number": 1,
            "page_idx": 0,
            "bbox": {"x1": 0.05, "y1": 0.12, "x2": 0.90, "y2": 0.24},
            "block_type": "text",
            "markdown_content": (
                "Abstract This paper studies hearing difficulty moments in conversational audio "
                "and asks how listeners experience communication breakdowns."
            ),
            "reading_order_index": 1,
        },
        {
            "block_id": "doc-p2-b0",
            "page_number": 2,
            "page_idx": 1,
            "bbox": {"x1": 0.05, "y1": 0.06, "x2": 0.40, "y2": 0.10},
            "block_type": "text",
            "markdown_content": "Collins, et al.",
            "reading_order_index": 2,
        },
        {
            "block_id": "doc-p2-b1",
            "page_number": 2,
            "page_idx": 1,
            "bbox": {"x1": 0.05, "y1": 0.16, "x2": 0.90, "y2": 0.26},
            "block_type": "title",
            "markdown_content": "Method",
            "reading_order_index": 3,
        },
        {
            "block_id": "doc-p2-b2",
            "page_number": 2,
            "page_idx": 1,
            "bbox": {"x1": 0.10, "y1": 0.30, "x2": 0.80, "y2": 0.40},
            "block_type": "text",
            "markdown_content": (
                "Participants listened to conversational audio and identified moments of "
                "difficulty during the interaction."
            ),
            "reading_order_index": 4,
        },
        {
            "block_id": "doc-p2-b3",
            "page_number": 2,
            "page_idx": 1,
            "bbox": {"x1": 0.10, "y1": 0.42, "x2": 0.80, "y2": 0.50},
            "block_type": "caption",
            "markdown_content": "Figure 2. Hearing difficulty moments over time.",
            "reading_order_index": 5,
        },
        {
            "block_id": "doc-p3-b0",
            "page_number": 3,
            "page_idx": 2,
            "bbox": {"x1": 0.05, "y1": 0.16, "x2": 0.90, "y2": 0.24},
            "block_type": "title",
            "markdown_content": "Results",
            "reading_order_index": 6,
        },
        {
            "block_id": "doc-p3-b1",
            "page_number": 3,
            "page_idx": 2,
            "bbox": {"x1": 0.10, "y1": 0.30, "x2": 0.80, "y2": 0.40},
            "block_type": "text",
            "markdown_content": (
                "The results show that hearing difficulty moments cluster around overlapping "
                "speech and fast turn transitions."
            ),
            "reading_order_index": 7,
        },
    ]


class PdfRagTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {}, clear=True)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_text_normalization_removes_pdf_hyphenation_ligatures_and_headers(self):
        from emotion_aware_assistant.paper.paper_rag import (
            is_low_value_context_block,
            normalize_pdf_text,
        )

        cleaned = normalize_pdf_text("con- versational emer-\n ging ﬁgure ﬂow")

        self.assertEqual(cleaned, "conversational emerging figure flow")
        for text in [
            "Trends in Hearing 30(0)",
            "CHI \u201924, May 11\u201316, 2024, Honolulu, HI, USA",
            "12",
            "Collins, et al.",
            "Smith et al.",
        ]:
            self.assertTrue(is_low_value_context_block({"block_type": "text", "markdown_content": text}))
        self.assertFalse(
            is_low_value_context_block(
                {"block_type": "caption", "markdown_content": "Figure 2. Hearing difficulty moments over time."}
            )
        )

    def test_paper_profile_rule_based_fallback_writes_expected_rag_files(self):
        from emotion_aware_assistant.paper.paper_rag import prepare_paper_memory

        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir) / "documents" / "doc"
            document_dir.mkdir(parents=True)
            status = prepare_paper_memory("doc", document_dir, sample_blocks())

            self.assertEqual(status["status"], "completed")
            for relative in [
                "rag/paper_profile.json",
                "rag/section_map.json",
                "rag/keyword_index.json",
                "rag/prepare_status.json",
            ]:
                self.assertTrue((document_dir / relative).exists(), relative)
            profile = json.loads((document_dir / "rag" / "paper_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(profile["title"], "Identifying Hearing Difficulty Moments in Conversational Audio")
            self.assertIn("hearing", profile["key_terms"])
            self.assertGreaterEqual(len(profile["section_map"]), 2)

    def test_retrieve_context_returns_matched_nearby_same_section_and_related_blocks(self):
        from emotion_aware_assistant.paper.paper_rag import prepare_paper_memory, retrieve_context

        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir) / "documents" / "doc"
            parsed_dir = document_dir / "parsed"
            parsed_dir.mkdir(parents=True)
            blocks = sample_blocks()
            (parsed_dir / "blocks_index.json").write_text(
                json.dumps({"document_id": "doc", "blocks": blocks}),
                encoding="utf-8",
            )
            prepare_paper_memory("doc", document_dir, blocks)
            result = retrieve_context(
                document_id="doc",
                document_dir=document_dir,
                blocks=blocks,
                highlight_payload={
                    "page_number": 2,
                    "selected_text": "moments of difficulty during the interaction",
                    "normalized_rects": [{"x1": 0.12, "y1": 0.32, "x2": 0.70, "y2": 0.38, "pageNumber": 2}],
                },
            )

            self.assertEqual(result["retrieval_strategy"], "coordinate_plus_keyword")
            self.assertEqual(result["matched_block"]["block_id"], "doc-p2-b2")
            self.assertTrue(any(block["block_id"] == "doc-p2-b3" for block in result["nearby_context"]))
            self.assertTrue(any(block["block_id"] == "doc-p2-b1" for block in result["same_section_context"]))
            self.assertGreaterEqual(len(result["related_blocks"]), 1)
            self.assertIn("one_sentence_summary", result["paper_profile"])

    def test_retrieve_context_endpoint_prepares_memory_and_uses_document_blocks(self):
        from emotion_aware_assistant.web.server import create_web_app

        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_web_app(force_dummy_llm=True)
            app.state.documents_dir = Path(temp_dir) / "documents"
            document_dir = app.state.documents_dir / "doc"
            parsed_dir = document_dir / "parsed"
            parsed_dir.mkdir(parents=True)
            blocks_path = parsed_dir / "blocks_index.json"
            blocks_path.write_text(json.dumps({"document_id": "doc", "blocks": sample_blocks()}), encoding="utf-8")
            app.state.documents["doc"] = {
                "document_id": "doc",
                "type": "pdf",
                "path": document_dir / "original.pdf",
                "parse_status": {"blocks_index_path": str(blocks_path), "parsed_dir": str(parsed_dir)},
            }

            response = app.test_request(
                "POST",
                "/api/document/retrieve-context",
                {
                    "document_id": "doc",
                    "highlight_payload": {
                        "page_number": 2,
                        "selected_text": "moments of difficulty",
                        "normalized_rects": [{"x1": 0.12, "y1": 0.32, "x2": 0.70, "y2": 0.38, "pageNumber": 2}],
                    },
                },
            )

            self.assertEqual(response["status"], 200)
            self.assertEqual(response["json"]["matched_block"]["block_id"], "doc-p2-b2")
            self.assertTrue((document_dir / "rag" / "paper_profile.json").exists())

    def test_prepare_memory_writes_embedding_ready_status_and_keyword_fallback(self):
        from emotion_aware_assistant.paper.paper_rag import prepare_paper_memory, retrieve_global_context

        with tempfile.TemporaryDirectory() as temp_dir:
            document_dir = Path(temp_dir) / "documents" / "doc"
            document_dir.mkdir(parents=True)
            status = prepare_paper_memory("doc", document_dir, sample_blocks())

            embeddings_path = document_dir / "rag" / "embeddings.json"
            self.assertTrue(embeddings_path.exists())
            self.assertEqual(status["embedding_index_status"], "unavailable")
            self.assertEqual(status["embedding_provider"], "gemini")

            result = retrieve_global_context(
                document_id="doc",
                document_dir=document_dir,
                query="overlapping speech difficulty moments",
                top_k=3,
            )

            self.assertEqual(result["retrieval_method"], "keyword")
            self.assertGreaterEqual(len(result["related_blocks"]), 1)
            self.assertIn("score", result["related_blocks"][0])


if __name__ == "__main__":
    unittest.main()
