import unittest


class WebApiWorkflowTests(unittest.TestCase):
    def test_web_backend_loads_sample_sets_states_chats_and_logs(self):
        from emotion_aware_assistant.web.server import create_web_app

        app = create_web_app(force_dummy_llm=True)
        status = app.test_request("GET", "/api/status")
        self.assertEqual(status["status"], 200)
        self.assertIn("models", status["json"])

        loaded = app.test_request("POST", "/api/document/load-sample")
        self.assertEqual(loaded["status"], 200)
        self.assertGreater(len(loaded["json"]["current_page_text"]), 100)

        context = app.test_request(
            "POST",
            "/api/document/context",
            {
                "selected_text": "The proposed method receives a selected passage and retrieves relevant chunks.",
                "page_number": 1,
                "user_question": "Can you explain this method?",
            },
        )
        self.assertEqual(context["status"], 200)
        self.assertIn("retrieved_chunks", context["json"])

        answers = {}
        for override in ["confusion", "frustration", "boredom", "engagement"]:
            emotion = app.test_request("POST", "/api/emotion/manual", {"emotion": override})
            self.assertEqual(emotion["status"], 200)
            self.assertEqual(emotion["json"]["state"], override)
            chat = app.test_request(
                "POST",
                "/api/chat",
                {
                    "selected_text": context["json"]["selected_text"],
                    "page_number": 1,
                    "user_question": "Can you explain this method?",
                    "model_alias": "dummy",
                    "followup_action": None,
                },
            )
            self.assertEqual(chat["status"], 200)
            self.assertIn("answer", chat["json"])
            self.assertEqual(chat["json"]["learning_state"]["state"], override)
            answers[override] = chat["json"]["answer"]

        self.assertEqual(len(set(answers.values())), 4)
        self.assertTrue(app.state.session.logger.path.exists())
        self.assertGreater(app.state.session.logger.path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
