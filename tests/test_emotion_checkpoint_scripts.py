import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class EmotionCheckpointScriptTests(unittest.TestCase):
    def test_inspect_emotion_checkpoint_reports_raw_mode(self):
        try:
            import torch  # type: ignore
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        from scripts.inspect_emotion_checkpoint import inspect_checkpoint_file

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "raw.pt"
            torch.save(
                {
                    "arch": "convnextv2_pico.fcmae_ft_in1k",
                    "num_classes": 8,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "model_state_dict": {"head.weight": 1},
                },
                checkpoint,
            )

            info = inspect_checkpoint_file(checkpoint)

        self.assertEqual(info["checkpoint_path"], str(checkpoint))
        self.assertEqual(info["arch"], "convnextv2_pico.fcmae_ft_in1k")
        self.assertEqual(info["num_classes"], 8)
        self.assertEqual(info["detected_model_mode"], "raw_emotion")
        self.assertTrue(info["model_state_dict_present"])
        self.assertIn("head.weight", info["sample_keys"])

    def test_configure_emotion_checkpoint_preserves_env_and_sets_raw_path(self):
        try:
            import torch  # type: ignore
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        from scripts.configure_emotion_checkpoint import configure_emotion_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "raw.pt"
            torch.save(
                {
                    "arch": "convnextv2_pico.fcmae_ft_in1k",
                    "num_classes": 8,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "model_state_dict": {},
                },
                checkpoint,
            )
            (root / ".env.local").write_text("UNRELATED=value\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                result = configure_emotion_checkpoint(root, checkpoint, mode="auto", quiet=True)

            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")

        self.assertTrue(result["saved"])
        self.assertEqual(result["detected_model_mode"], "raw_emotion")
        self.assertIn("UNRELATED=value", env_text)
        self.assertIn(f"EMOTION_CHECKPOINT_PATH={checkpoint}", env_text)
        self.assertIn("EMOTION_MODEL_MODE=auto", env_text)
        self.assertIn(f"RAW_EMOTION_CHECKPOINT_PATH={checkpoint}", env_text)
        self.assertIn(".env.local", gitignore_text)
