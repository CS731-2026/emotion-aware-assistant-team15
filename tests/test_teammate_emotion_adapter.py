import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_raw_prediction_payload_aggregates_anger_and_angry_aliases(self):
        from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

        payload = TeammateEmotionAdapter.raw_emotion_prediction_payload(
            probabilities={
                "anger": 0.22,
                "angry": 0.08,
                "sad": 0.10,
                "disgust": 0.05,
                "fear": 0.04,
                "surprise": 0.03,
                "contempt": 0.02,
                "happy": 0.16,
                "neutral": 0.30,
            },
            architecture="convnext_tiny.fb_in22k_ft_in1k",
            classes=["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
            device="cpu",
        )

        self.assertEqual(payload["model_output_type"], "raw_emotion")
        self.assertEqual(payload["raw_emotion"], "anger")
        self.assertAlmostEqual(payload["raw_distribution"]["anger"], 0.30)
        self.assertAlmostEqual(payload["state_distribution"]["frustration"], 0.45)
        self.assertAlmostEqual(payload["state_distribution"]["engagement"], 0.46)
        self.assertNotIn("angry", payload["raw_distribution"])

    def test_load_uses_checkpoint_class_to_idx_as_academic_label_order(self):
        from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

        checkpoint = {
            "arch": "convnext_tiny.fb_in22k_ft_in1k",
            "num_classes": 4,
            "class_to_idx": {"frustration": 0, "engagement": 1, "confusion": 2, "boredom": 3},
            "model_state_dict": {"head.weight": object()},
        }
        created_models = []

        class FakeModel:
            def load_state_dict(self, state_dict, strict=True):
                return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                return self

            def eval(self):
                return self

        fake_timm = types.SimpleNamespace(
            create_model=lambda architecture, pretrained, num_classes: created_models.append(
                {"architecture": architecture, "num_classes": num_classes}
            )
            or FakeModel()
        )
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            load=lambda path, map_location=None: checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"timm": fake_timm, "torch": fake_torch}):
            model_dir = Path(tmp)
            (model_dir / "best_model.pt").write_bytes(b"placeholder")
            adapter = TeammateEmotionAdapter(model_dir=model_dir)
            status = adapter.load()

        self.assertTrue(status["model_loaded"])
        self.assertEqual(status["classes"], ["frustration", "engagement", "confusion", "boredom"])
        self.assertEqual(status["academic_label_order_source"], "checkpoint_metadata")
        self.assertEqual(created_models[0]["num_classes"], 4)


if __name__ == "__main__":
    unittest.main()
