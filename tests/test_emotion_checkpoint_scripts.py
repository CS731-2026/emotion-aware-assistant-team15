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
                    "epoch": 17,
                    "val_acc": 0.8421,
                    "model_state_dict": {"head.weight": 1},
                },
                checkpoint,
            )

            info = inspect_checkpoint_file(checkpoint)

        self.assertEqual(info["checkpoint_path"], str(checkpoint))
        self.assertEqual(info["arch"], "convnextv2_pico.fcmae_ft_in1k")
        self.assertEqual(info["num_classes"], 8)
        self.assertEqual(info["detected_model_mode"], "raw_emotion")
        self.assertEqual(info["output_type"], "raw_emotion")
        self.assertEqual(info["epoch"], 17)
        self.assertAlmostEqual(info["val_acc"], 0.8421)
        self.assertTrue(info["model_state_dict_present"])
        self.assertIn("head.weight", info["sample_keys"])

    def test_inspect_emotion_checkpoint_reports_class_to_idx_head_shape_and_preprocessing(self):
        try:
            import torch  # type: ignore
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        from scripts.inspect_emotion_checkpoint import inspect_checkpoint_file

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "academic.pt"
            torch.save(
                {
                    "arch": "convnext_tiny.fb_in22k_ft_in1k",
                    "num_classes": 4,
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "class_to_idx": {"boredom": 0, "confusion": 1, "engagement": 2, "frustration": 3},
                    "model_state_dict": {
                        "head.fc.weight": torch.zeros((4, 768)),
                        "head.fc.bias": torch.zeros((4,)),
                    },
                },
                checkpoint,
            )

            info = inspect_checkpoint_file(checkpoint)

        self.assertEqual(info["output_type"], "academic_state")
        self.assertEqual(info["class_to_idx"]["frustration"], 3)
        self.assertEqual(info["head_fc_weight_shape"], [4, 768])
        self.assertEqual(info["classifier_head_shapes"]["head.fc.bias"], [4])
        self.assertEqual(info["metadata_source"], "checkpoint_metadata")
        self.assertEqual(info["class_order_source"], "checkpoint_class_to_idx")
        self.assertEqual(info["preprocessing_summary"]["current_runtime"]["mean"], [0.485, 0.456, 0.406])
        self.assertEqual(info["preprocessing_summary"]["checkpoint_metadata"], "not stored")

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

    def test_configure_raw_mode_uses_raw_checkpoint_path(self):
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
                    "arch": "convnext_tiny.fb_in22k_ft_in1k",
                    "num_classes": 8,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "model_state_dict": {},
                },
                checkpoint,
            )

            with patch.dict(os.environ, {}, clear=True):
                result = configure_emotion_checkpoint(root, checkpoint, mode="raw_emotion", quiet=True)

            env_text = (root / ".env.local").read_text(encoding="utf-8")

        self.assertEqual(result["mode"], "raw_emotion")
        self.assertEqual(result["raw_emotion_checkpoint_path"], str(checkpoint))
        self.assertIn("EMOTION_MODEL_MODE=raw_emotion", env_text)
        self.assertIn(f"RAW_EMOTION_CHECKPOINT_PATH={checkpoint}", env_text)

    def test_compare_emotion_models_script_hashes_crop_and_formats_preprocessing(self):
        from PIL import Image  # type: ignore

        from scripts.compare_emotion_models_on_crop import image_file_sha256, preprocessing_summary

        with tempfile.TemporaryDirectory() as temp_dir:
            crop = Path(temp_dir) / "crop.png"
            Image.new("RGB", (8, 6), (10, 20, 30)).save(crop)

            digest = image_file_sha256(crop)
            summary = preprocessing_summary({"input_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(summary["input_size"], 224)
        self.assertEqual(summary["mean"], [0.485, 0.456, 0.406])

    def test_audit_mapping_includes_required_and_archived_legacy_rules(self):
        from scripts.audit_emotion_model_alignment import (
            map_raw_label_to_academic,
            map_raw_label_to_legacy_archived_academic,
        )

        self.assertEqual(map_raw_label_to_academic("sad"), "frustration")
        self.assertEqual(map_raw_label_to_academic("anger"), "frustration")
        self.assertEqual(map_raw_label_to_academic("angry"), "frustration")
        self.assertEqual(map_raw_label_to_academic("contempt"), "boredom")
        self.assertEqual(map_raw_label_to_legacy_archived_academic("sad"), "boredom")
        self.assertEqual(map_raw_label_to_legacy_archived_academic("contempt"), "frustration")

    def test_audit_metrics_detect_prediction_distribution_and_label_permutation(self):
        from scripts.audit_emotion_model_alignment import (
            accuracy,
            confusion_matrix,
            detect_best_label_permutation,
            prediction_distribution,
        )

        labels = ["boredom", "confusion", "engagement", "frustration"]
        ground_truth = ["frustration", "frustration", "boredom", "boredom", "confusion", "engagement"]
        direct_predictions = ["boredom", "boredom", "frustration", "frustration", "confusion", "engagement"]

        self.assertAlmostEqual(accuracy(direct_predictions, ground_truth), 2 / 6)
        self.assertEqual(prediction_distribution(direct_predictions), {"boredom": 2, "confusion": 1, "engagement": 1, "frustration": 2})
        matrix = confusion_matrix(ground_truth, direct_predictions, labels)
        self.assertEqual(matrix["frustration"]["boredom"], 2)
        self.assertEqual(matrix["boredom"]["frustration"], 2)
        permutation = detect_best_label_permutation(direct_predictions, ground_truth, labels)
        self.assertTrue(permutation["suspected_permutation"])
        self.assertAlmostEqual(permutation["best_accuracy"], 1.0)
        self.assertEqual(permutation["best_mapping"]["boredom"], "frustration")

    def test_audit_collects_8class_folder_dataset_samples(self):
        from PIL import Image  # type: ignore

        from scripts.audit_emotion_model_alignment import collect_dataset_root_samples

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sad").mkdir()
            (root / "happy").mkdir()
            Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "sad" / "a.png")
            Image.new("RGB", (4, 4), (4, 5, 6)).save(root / "happy" / "b.jpg")

            samples = collect_dataset_root_samples(root)

        self.assertEqual([(sample.raw_label, sample.path.name) for sample in samples], [("happy", "b.jpg"), ("sad", "a.png")])

    def test_audit_scans_4class_dataset_for_sad_like_source_paths(self):
        from PIL import Image  # type: ignore

        from scripts.audit_emotion_model_alignment import scan_4class_dataset_for_raw_source_labels

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for label in ("boredom", "frustration"):
                (root / label).mkdir()
            Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "boredom" / "affectnet_sad_001.png")
            Image.new("RGB", (4, 4), (4, 5, 6)).save(root / "frustration" / "rafdb_sadness_002.jpg")
            Image.new("RGB", (4, 4), (7, 8, 9)).save(root / "frustration" / "anger_003.jpg")

            scan = scan_4class_dataset_for_raw_source_labels(root)

        self.assertEqual(scan["class_counts"]["boredom"], 1)
        self.assertEqual(scan["class_counts"]["frustration"], 2)
        self.assertEqual(scan["raw_label_mentions_by_academic_class"]["boredom"]["sad"], 1)
        self.assertEqual(scan["raw_label_mentions_by_academic_class"]["frustration"]["sad"], 1)
        self.assertEqual(scan["raw_label_mentions_by_academic_class"]["frustration"]["anger"], 1)
        self.assertTrue(scan["sad_like_source_paths_under_boredom"])
        self.assertTrue(scan["sad_like_source_paths_under_frustration"])

    def test_compare_script_adds_ground_truth_raw_label_fields(self):
        from scripts.compare_emotion_models_on_crop import add_ground_truth_fields

        result = {
            "mapped_from_8class_state": "frustration",
            "direct_4class_state": "boredom",
        }

        enriched = add_ground_truth_fields(result, "sad")

        self.assertEqual(enriched["ground_truth_raw_label"], "sad")
        self.assertEqual(enriched["ground_truth_mapped_label"], "frustration")
        self.assertFalse(enriched["direct_4class_matches_ground_truth"])
        self.assertTrue(enriched["raw8_mapped_matches_ground_truth"])
        self.assertFalse(enriched["direct_4class_agrees_with_raw8_mapped"])
