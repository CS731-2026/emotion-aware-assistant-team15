import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class OpenFaceScriptTests(unittest.TestCase):
    def test_diagnose_openface_reports_missing_binary_without_crashing(self):
        from scripts.diagnose_openface import diagnose_openface

        result = diagnose_openface(candidate_paths=[], include_path=False)

        self.assertFalse(result["found"])
        self.assertIsNone(result["binary_path"])
        self.assertIn("FeatureExtraction binary was not found", result["warning"])

    def test_diagnose_openface_finds_fake_executable(self):
        from scripts.diagnose_openface import diagnose_openface

        with tempfile.TemporaryDirectory() as temp_dir:
            binary = Path(temp_dir) / "FeatureExtraction"
            binary.write_text("#!/bin/sh\nprintf 'FeatureExtraction help\\n'\n", encoding="utf-8")
            os.chmod(binary, 0o755)

            result = diagnose_openface(candidate_paths=[binary], include_path=False)

        self.assertTrue(result["found"])
        self.assertEqual(result["binary_path"], str(binary))
        self.assertTrue(result["executable"])
        self.assertTrue(result["can_run"])
        self.assertIn("FeatureExtraction help", "\n".join(result["help_output_first_lines"]))

    def test_build_environment_diagnostic_reports_core_tool_fields(self):
        from scripts.diagnose_openface import diagnose_build_environment

        result = diagnose_build_environment()

        self.assertIn("platform", result)
        self.assertIn("python", result)
        self.assertIn("cmake", result["tools"])
        self.assertIn("git", result["tools"])
        self.assertIn("g++", result["tools"])
        self.assertIn("suggested_commands", result)

    def test_test_openface_feature_extraction_handles_missing_binary_safely(self):
        from scripts.test_openface_feature_extraction import run_binary_check

        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "FeatureExtraction"
            result = run_binary_check(missing)

        self.assertFalse(result["ok"])
        self.assertIn("not found", result["warning"].lower())

    def test_test_openface_parser_parses_fake_csv_bbox(self):
        from emotion_aware_assistant.emotion.face_detector import parse_openface_csv

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "frame.csv"
            headers = ["success", "confidence"]
            headers.extend(f"x_{index}" for index in range(68))
            headers.extend(f"y_{index}" for index in range(68))
            values = ["1", "0.91"]
            values.extend(str(5 + index) for index in range(68))
            values.extend(str(15 + index * 2) for index in range(68))
            csv_path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")

            parsed = parse_openface_csv(csv_path)

        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["landmark_count"], 68)
        self.assertEqual(parsed["bbox"], [5, 15, 67, 134])

    def test_diagnostics_do_not_print_api_keys(self):
        import scripts.diagnose_openface as diagnose_openface

        secret = "AI" + "za" + "diagnose-secret"
        output = io.StringIO()
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=False):
            with contextlib.redirect_stdout(output):
                code = diagnose_openface.main(["--json", "--no-path"])

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertNotIn(secret, text)
        self.assertNotIn("GEMINI_API_KEY", text)
        json.loads(text)


if __name__ == "__main__":
    unittest.main()
