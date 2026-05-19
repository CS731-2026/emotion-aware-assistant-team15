import json
import tempfile
import unittest
from pathlib import Path


class TeammateEmotionAdapterTests(unittest.TestCase):
    def write_academic_metadata(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_output_type": "academic_state",
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "framework": "timm",
                    "num_classes": 4,
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "class_to_idx": {
                        "boredom": 0,
                        "confusion": 1,
                        "engagement": 2,
                        "frustration": 3,
                    },
                    "input_size": 224,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "checkpoint_key": "model_state_dict",
                }
            ),
            encoding="utf-8",
        )

    def test_status_reports_academic_state_checkpoint_without_raw_emotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.write_academic_metadata(model_dir)

            from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

            adapter = TeammateEmotionAdapter(model_dir=model_dir)
            status = adapter.status()

            self.assertFalse(status["model_loaded"])
            self.assertEqual(status["model_output_type"], "academic_state")
            self.assertEqual(status["architecture"], "convnext_tiny.fb_in22k_ft_in1k")
            self.assertEqual(status["classes"], ["boredom", "confusion", "engagement", "frustration"])
            self.assertFalse(status["raw_emotion_available"])
            self.assertIn("best_model.pt", status["loading_error"])

    def test_academic_prediction_payload_does_not_fabricate_raw_emotion(self):
        from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

        payload = TeammateEmotionAdapter.academic_prediction_payload(
            probabilities={
                "boredom": 0.04,
                "confusion": 0.81,
                "engagement": 0.08,
                "frustration": 0.07,
            },
            architecture="convnext_tiny.fb_in22k_ft_in1k",
            classes=["boredom", "confusion", "engagement", "frustration"],
            device="cpu",
        )

        self.assertTrue(payload["model_loaded"])
        self.assertEqual(payload["model_output_type"], "academic_state")
        self.assertFalse(payload["raw_emotion_available"])
        self.assertIsNone(payload["raw_emotion"])
        self.assertEqual(payload["academic_state"], "confusion")
        self.assertAlmostEqual(payload["confidence"], 0.81)
        self.assertEqual(set(payload["state_distribution"]), {"boredom", "confusion", "engagement", "frustration"})


if __name__ == "__main__":
    unittest.main()
