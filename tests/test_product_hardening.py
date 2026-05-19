import json
import unittest


SELECTED = (
    "The proposed method receives a selected passage, surrounding paper context, "
    "and a compact learning-state signal."
)


class ProductHardeningTests(unittest.TestCase):
    def test_context_prompt_policy_emotion_history_and_logs_are_product_depth(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        self.assertEqual(app.test_request("POST", "/api/document/load-sample")["status"], 200)

        context = app.test_request(
            "POST",
            "/api/document/context",
            {
                "selected_text": SELECTED,
                "page_number": 1,
                "user_question": "Can you explain this method?",
            },
        )
        self.assertEqual(context["status"], 200)
        body = context["json"]
        self.assertEqual(body["selected_text"], SELECTED)
        self.assertIn("difficulty_hint", body)
        self.assertIn("passage_analysis", body)
        self.assertIn("retrieval_debug", body)
        self.assertGreater(len(body["retrieval_debug"]["ranked_chunks"]), 0)
        self.assertNotEqual(body["passage_analysis"]["passage_type"], "general")
        self.assertIn("selected_text", body["retrieval_debug"]["included_sources"])

        answers = {}
        strategies = {}
        for state in ["confusion", "frustration", "boredom", "engagement"]:
            manual = app.test_request("POST", "/api/emotion/manual", {"emotion": state})
            self.assertEqual(manual["status"], 200)
            self.assertGreaterEqual(len(manual["json"]["history"]), 1)
            chat = app.test_request(
                "POST",
                "/api/chat",
                {
                    "selected_text": SELECTED,
                    "page_number": 1,
                    "user_question": "Can you explain this method?",
                    "model_alias": "dummy",
                },
            )
            self.assertEqual(chat["status"], 200)
            payload = chat["json"]
            self.assertIn(SELECTED, payload["prompt_preview"])
            self.assertEqual(payload["context_debug"]["selected_text"], SELECTED)
            self.assertEqual(payload["passage_type"], body["passage_type"])
            self.assertIn("response_policy", payload)
            self.assertIn("ideal_response_shape", payload["response_policy"])
            answers[state] = payload["answer"]
            strategies[state] = payload["strategy"]

        self.assertEqual(len(set(answers.values())), 4)
        self.assertEqual(len(set(strategies.values())), 4)

        emotion_state = app.test_request("GET", "/api/emotion/state")
        self.assertEqual(emotion_state["status"], 200)
        self.assertIn("probabilities", emotion_state["json"])
        self.assertIn("history", emotion_state["json"])
        self.assertIn("dominant_state", emotion_state["json"])

        log_lines = app.state.session.logger.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(log_lines), 4)
        last_record = json.loads(log_lines[-1])
        self.assertIn("passage_type", last_record)
        self.assertIn("followup_action", last_record)
        self.assertGreater(last_record["selected_passage_length"], 0)

    def test_passage_analysis_distinguishes_dataset_and_comparison(self):
        from emotion_aware_assistant.paper.passage_analyzer import analyze_passage

        dataset = analyze_passage("We evaluate on RAF-DB using accuracy, macro F1, and a held-out test split.")
        comparison = analyze_passage("Compared with the baseline, our approach trades latency for better robustness.")

        self.assertEqual(dataset.passage_type, "dataset/evaluation")
        self.assertEqual(comparison.passage_type, "comparison/related work")
        self.assertTrue(dataset.detected_keywords)
        self.assertTrue(comparison.suggested_explanation_mode)


if __name__ == "__main__":
    unittest.main()
