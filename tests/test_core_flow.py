import unittest


class AdaptiveCoreFlowTests(unittest.TestCase):
    def test_manual_emotion_changes_learning_state_policy_and_answer_style(self):
        from emotion_aware_assistant.app import AssistantSession
        from emotion_aware_assistant.core.config import load_config
        from emotion_aware_assistant.core.types import PaperContext

        config = load_config("config.yaml")
        session = AssistantSession(config=config, force_dummy_llm=True)
        context = PaperContext(
            document_title="Sample",
            page_number=1,
            selected_text="The proposed method optimizes a latent objective with iterative updates.",
            surrounding_text="This section introduces the method and discusses why each update exists.",
            retrieved_chunks=[
                "The method alternates between estimating hidden variables and updating parameters."
            ],
            passage_type="method/process/mechanism",
        )

        expectations = {
            "fear": ("confusion", "step_by_step_clarification", "Step-by-step"),
            "angry": ("frustration", "supportive_simplification", "simplest version"),
            "contempt": ("boredom", "concise_reengagement", "Quick check"),
            "happy": ("engagement", "deeper_academic_expansion", "Technical read"),
        }
        answers = {}

        for emotion, (state, strategy, phrase) in expectations.items():
            snapshot = session.set_manual_emotion(emotion)
            self.assertEqual(snapshot.state, state)
            self.assertEqual(snapshot.strategy, strategy)
            response = session.ask(
                "Can you explain this method?",
                paper_context=context,
                model_alias="dummy",
            )
            self.assertIn(phrase, response.text)
            answers[state] = response.text

        self.assertEqual(len(set(answers.values())), 4)


if __name__ == "__main__":
    unittest.main()
