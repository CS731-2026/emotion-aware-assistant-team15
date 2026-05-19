import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
import urllib.request
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from emotion_aware_assistant.core.types import FaceBox


class CameraDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        from emotion_aware_assistant.web.server import create_web_app

        self.app = create_web_app(force_dummy_llm=True)
        self.app.state.upload_dir = Path(self.temp_dir.name) / "runtime_uploads"
        self.app.state.documents_dir = self.app.state.upload_dir / "documents"

    def test_camera_debug_status_reports_safe_academic_state_mode(self):
        response = self.app.test_request("GET", "/api/camera-debug/status")

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertIn("model_status", payload)
        self.assertIn("face_detector_status", payload)
        self.assertEqual(payload["current_mode"], "academic_state")
        self.assertFalse(payload["raw_emotion_available"])
        self.assertIn("academic-state", payload["mode_explanation"]["current_mode"].lower())
        self.assertNotIn("GEMINI_API_KEY", serialized)
        self.assertNotIn("AI" + "za", serialized)

    def test_raw_emotion_checkpoint_classes_are_detected_as_raw_mode(self):
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import inspect_checkpoint_metadata

        info = inspect_checkpoint_metadata(
            {
                "arch": "convnextv2_pico.fcmae_ft_in1k",
                "num_classes": 8,
                "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                "model_state_dict": {},
            }
        )

        self.assertEqual(info["model_output_type"], "raw_emotion")
        self.assertTrue(info["raw_detection_available"])
        self.assertEqual(info["classes"], ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"])
        self.assertEqual(info["architecture"], "convnextv2_pico.fcmae_ft_in1k")

    def test_academic_checkpoint_classes_are_detected_as_academic_mode(self):
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import inspect_checkpoint_metadata

        info = inspect_checkpoint_metadata(
            {
                "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                "num_classes": 4,
                "class_to_idx": {"boredom": 0, "confusion": 1, "engagement": 2, "frustration": 3},
                "model_state_dict": {},
            }
        )

        self.assertEqual(info["model_output_type"], "academic_state")
        self.assertFalse(info["raw_detection_available"])
        self.assertEqual(info["classes"], ["boredom", "confusion", "engagement", "frustration"])

    def test_raw_emotion_probabilities_map_to_academic_states(self):
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import EmotionMapper

        mapper = EmotionMapper()

        self.assertEqual(mapper.map_probs_to_state({"fear": 0.30, "surprise": 0.45})[0], "confusion")
        self.assertEqual(mapper.map_probs_to_state({"sad": 0.20, "anger": 0.20, "disgust": 0.25})[0], "frustration")
        self.assertEqual(mapper.map_probs_to_state({"contempt": 0.60, "neutral": 0.20})[0], "boredom")
        state, scores = mapper.map_probs_to_state({"happy": 0.40, "neutral": 0.35})
        self.assertEqual(state, "engagement")
        self.assertAlmostEqual(scores["engagement"], 0.75)
        self.assertEqual(mapper.mapping_rule_for_state("confusion"), "fear + surprise -> confusion")

    def test_teammate_emotion_buffer_returns_majority_stable_state(self):
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import EmotionBuffer

        buffer = EmotionBuffer(maxlen=3)

        self.assertEqual(buffer.push("confusion"), "confusion")
        self.assertEqual(buffer.push("engagement"), "confusion")
        self.assertEqual(buffer.push("confusion"), "confusion")
        self.assertEqual(buffer.values(), ["confusion", "engagement", "confusion"])
        self.assertEqual(buffer.buffer_size, 3)

    def test_raw_emotion_pipeline_missing_checkpoint_returns_safe_error(self):
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import CombinedEmotionPipeline

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"RAW_EMOTION_CHECKPOINT_PATH": str(Path(temp_dir) / "missing.pt")},
            clear=True,
        ):
            pipeline = CombinedEmotionPipeline()
            result = pipeline.predict(None)

        self.assertEqual(result["model_output_type"], "unknown")
        self.assertFalse(result["raw_detection_available"])
        self.assertIsNone(result["raw_detection"])
        self.assertIn("missing", result["error"].lower())

    def test_checkpoint_selection_prefers_valid_raw_checkpoint_in_auto_mode(self):
        try:
            import torch  # type: ignore
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        import emotion_aware_assistant.emotion.raw_emotion_pipeline as raw_pipeline
        from emotion_aware_assistant.emotion.raw_emotion_pipeline import select_emotion_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw_8class_best.pt"
            academic_path = root / "best_model.pt"
            torch.save(
                {
                    "arch": "convnextv2_pico.fcmae_ft_in1k",
                    "num_classes": 8,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "model_state_dict": {},
                },
                raw_path,
            )
            torch.save(
                {
                    "arch": "convnext_tiny.fb_in22k_ft_in1k",
                    "num_classes": 4,
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "model_state_dict": {},
                },
                academic_path,
            )

            with patch.object(raw_pipeline, "PROJECT_ROOT", root), patch.dict(
                os.environ,
                {
                    "EMOTION_MODEL_MODE": "auto",
                    "RAW_EMOTION_CHECKPOINT_PATH": str(raw_path),
                    "EMOTION_CHECKPOINT_PATH": str(academic_path),
                },
                clear=True,
            ):
                selection = select_emotion_checkpoint()

        self.assertEqual(selection["checkpoint_path"], str(raw_path))
        self.assertEqual(selection["model_output_type"], "raw_emotion")
        self.assertEqual(selection["classes"], ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"])

    def test_camera_debug_analyze_frame_rejects_missing_frame_safely(self):
        response = self.app.test_request("POST", "/api/camera-debug/analyze-frame", {})

        self.assertEqual(response["status"], 400)
        self.assertIn("frame image is required", response["json"]["error"])

    def test_camera_debug_analyze_frame_returns_face_metadata_and_academic_prediction(self):
        class FakeAdapter:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "checkpoint_path": "models/emotion_model/best_model.pt",
                    "raw_emotion_available": False,
                    "loading_error": None,
                    "device": "cpu",
                }

            def predict(self, image):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "raw_emotion_available": False,
                    "raw_emotion": None,
                    "academic_state": "engagement",
                    "confidence": 0.78,
                    "state_distribution": {
                        "boredom": 0.08,
                        "confusion": 0.07,
                        "engagement": 0.78,
                        "frustration": 0.07,
                    },
                }

        self.app.state._emotion_adapter = FakeAdapter()
        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": "data:image/png;base64,QUJDRA=="},
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertTrue(payload["ok"])
        self.assertIn("emotion_pipeline", payload)
        self.assertEqual(payload["emotion_pipeline"]["model_output_type"], "academic_state")
        self.assertFalse(payload["emotion_pipeline"]["raw_detection_available"])
        self.assertIsNone(payload["emotion_pipeline"]["raw_detection"])
        self.assertEqual(payload["emotion_pipeline"]["mapped_academic_state"]["mapping_rule"], "bypassed: checkpoint directly predicts academic states")
        self.assertEqual(payload["emotion_pipeline"]["smoothed_state"]["state"], "engagement")
        self.assertRegex(payload["frame_id"], r"^frame_[0-9a-f]+$")
        self.assertEqual(payload["analyzed_frame_size"], [224, 224])
        self.assertTrue(payload["analyzed_frame_preview_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertIn(payload["face_detection"]["requested_detector"], {"auto", "yolo"})
        self.assertIn(payload["face_detection"]["actual_detector"], {"opencv_haar", "center_crop"})
        self.assertEqual(payload["face_detection"]["detector"], payload["face_detection"]["actual_detector"])
        self.assertTrue(payload["face_detection"]["fallback_used"])
        self.assertIn("expanded_bbox", payload["face_detection"])
        self.assertIn("crop_bbox_used", payload["face_detection"])
        self.assertIn("crop_strategy", payload["face_detection"])
        self.assertEqual(payload["prediction"]["model_output_type"], "academic_state")
        self.assertFalse(payload["prediction"]["raw_emotion_available"])
        self.assertEqual(payload["prediction"]["academic_state"], "engagement")
        self.assertEqual(set(payload["prediction"]["state_distribution"]), {"boredom", "confusion", "engagement", "frustration"})
        self.assertEqual(payload["model_input_size"], [224, 224])
        self.assertTrue(payload["crop_preview_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(payload["model_input_preview_data_url"].startswith("data:image/jpeg;base64,"))

        if self.app.state.upload_dir.exists():
            runtime_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in self.app.state.upload_dir.rglob("*")
                if path.is_file()
            )
            self.assertNotIn("QUJDRA", runtime_text)
            self.assertNotIn("data:image", runtime_text)

    def test_analyze_frame_raw_mode_includes_raw_mapped_and_smoothed_pipeline(self):
        class FakePipeline:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "raw_emotion",
                    "raw_detection_available": True,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "architecture": "convnextv2_pico.fcmae_ft_in1k",
                    "checkpoint_path": "models/emotion_model/raw.pt",
                    "mapper_available": True,
                    "buffer_size": 10,
                }

            def predict(self, image, fallback_prediction=None, fallback_status=None):
                self.input_size = getattr(image, "size", None)
                return {
                    "model_output_type": "raw_emotion",
                    "checkpoint_path": "models/emotion_model/raw.pt",
                    "architecture": "convnextv2_pico.fcmae_ft_in1k",
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "raw_detection_available": True,
                    "raw_detection": {
                        "label": "fear",
                        "confidence": 0.62,
                        "probabilities": {
                            "anger": 0.02,
                            "contempt": 0.03,
                            "disgust": 0.01,
                            "fear": 0.62,
                            "happy": 0.04,
                            "neutral": 0.10,
                            "sad": 0.05,
                            "surprise": 0.13,
                        },
                    },
                    "mapped_academic_state": {
                        "state": "confusion",
                        "scores": {"frustration": 0.08, "confusion": 0.75, "boredom": 0.03, "engagement": 0.14},
                        "mapping_rule": "fear + surprise -> confusion",
                    },
                    "smoothed_state": {"state": "confusion", "buffer": ["confusion"], "buffer_size": 10},
                    "response_strategy": "Clarify the key concept first.",
                }

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        pipeline = FakePipeline()
        self.app.state._emotion_pipeline = pipeline
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": self._solid_image_data_url((80, 80), (180, 20, 20))},
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]["emotion_pipeline"]
        self.assertEqual(pipeline.input_size, (224, 224))
        self.assertEqual(payload["model_output_type"], "raw_emotion")
        self.assertTrue(payload["raw_detection_available"])
        self.assertEqual(payload["raw_detection"]["label"], "fear")
        self.assertEqual(payload["mapped_academic_state"]["state"], "confusion")
        self.assertEqual(payload["smoothed_state"]["state"], "confusion")
        self.assertIn("Clarify", payload["response_strategy"])

    def test_camera_debug_analyze_frame_reports_original_and_expanded_yolo_bboxes(self):
        class FakeDetector:
            requested_detector = "yolo"
            actual_detector = "yolo"
            yolo_loaded = True
            yolo_model_path = "models/face_detector/yolov8n-face.pt"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FaceBox(52, 26, 44, 32, 0.91, "yolo")]

        class FakeAdapter:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
                    "classes": ["boredom", "confusion", "engagement", "frustration"],
                    "checkpoint_path": "models/emotion_model/best_model.pt",
                    "raw_emotion_available": False,
                    "loading_error": None,
                    "device": "cpu",
                }

            def predict(self, image):
                return {
                    "model_loaded": True,
                    "model_output_type": "academic_state",
                    "raw_emotion_available": False,
                    "academic_state": "engagement",
                    "confidence": 0.8,
                    "state_distribution": {"boredom": 0.05, "confusion": 0.08, "engagement": 0.8, "frustration": 0.07},
                }

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": "data:image/png;base64,QUJDRA=="},
        )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        self.assertEqual(face["requested_detector"], "yolo")
        self.assertEqual(face["actual_detector"], "yolo")
        self.assertEqual(face["bbox"], [52, 26, 44, 32])
        self.assertIn("expanded_bbox", face)
        self.assertIn("crop_bbox_used", face)
        self.assertIn("crop_margin", face)
        self.assertIn("crop_strategy", face)
        self.assertGreater(face["expanded_bbox"][2], face["bbox"][2])
        self.assertGreater(face["expanded_bbox"][3], face["bbox"][3])
        self.assertEqual(face["expanded_bbox"][2], face["expanded_bbox"][3])
        self.assertEqual(face["crop_bbox_used"], face["expanded_bbox"])
        self.assertFalse(face["fallback_used"])

    def test_camera_debug_reaction_summary_maps_support_cue_to_strategy_families(self):
        response = self.app.test_request(
            "POST",
            "/api/camera-debug/reaction-summary",
            {
                "samples": [
                    {
                        "timestamp": "2026-05-17T00:00:00Z",
                        "academic_state": "confusion",
                        "confidence": 0.72,
                        "distribution": {"boredom": 0.28, "confusion": 0.52, "engagement": 0.10, "frustration": 0.10},
                        "trend": "rising",
                    },
                    {
                        "timestamp": "2026-05-17T00:00:08Z",
                        "academic_state": "confusion",
                        "confidence": 0.76,
                        "distribution": {"boredom": 0.27, "confusion": 0.55, "engagement": 0.08, "frustration": 0.10},
                        "trend": "stable",
                    },
                ],
                "source_turn_id": "debug_turn",
                "highlight_id": "debug_highlight",
            },
        )

        self.assertEqual(response["status"], 200, response)
        summary = response["json"]["reaction_window_summary"]
        self.assertEqual(summary["support_cue"], "clarify_and_reengage")
        self.assertEqual(response["json"]["support_cue"], "clarify_and_reengage")
        self.assertIn("concrete_example", response["json"]["allowed_strategy_families"])
        self.assertIn("step_by_step_breakdown", response["json"]["allowed_strategy_families"])

    def test_camera_debug_page_source_contains_required_panels_and_safe_wording(self):
        page = Path("emotion_aware_assistant/web/static/camera_debug.html")
        source = page.read_text(encoding="utf-8")

        self.assertIn("Live camera preview", source)
        self.assertIn("Last analyzed frame", source)
        self.assertIn("OpenFace raw info", source)
        self.assertIn("Model prediction", source)
        self.assertIn("Raw Detection and Mapped Academic State", source)
        self.assertIn("Crop and model input", source)
        self.assertIn("Current Mode B", source)
        self.assertIn("/api/camera-debug/analyze-frame", source)
        self.assertIn("Actual detector", source)
        self.assertIn("Landmark bbox", source)
        self.assertIn("Crop bbox used", source)
        self.assertIn("Crop strategy", source)
        self.assertIn("crop-mode-select", source)
        self.assertIn('<option value="landmark_tight">landmark_tight</option>', source)
        self.assertIn('<option value="face_context">face_context</option>', source)
        self.assertIn('<option value="square_face_context" selected>square_face_context</option>', source)
        self.assertIn("crop-scale-slider", source)
        self.assertIn("crop-y-bias-slider", source)
        self.assertIn("crop-top-extra-slider", source)
        self.assertIn("crop-bottom-extra-slider", source)
        self.assertIn("crop-make-square-checkbox", source)
        self.assertIn("OpenFace confidence", source)
        self.assertIn("Landmark count", source)
        self.assertIn("drawAnalyzedOverlay", source)
        self.assertIn("analyzed_frame_preview_data_url", source)
        self.assertIn("analyzed_frame_size", source)
        self.assertIn("renderAnalyzedFrameSvg", source)
        self.assertIn('setAttribute("viewBox"', source)
        self.assertIn('setAttribute("href", imageHref)', source)
        self.assertIn("annotated_frame_preview_data_url", source)
        self.assertIn("overlay-mode-select", source)
        self.assertIn("Backend annotated image", source)
        self.assertIn("coordinate-debug", source)
        self.assertIn("missingContractWarnings", source)
        self.assertIn("Missing analyzed_frame_preview_data_url in analyze-frame response.", source)
        self.assertIn("Missing analyzed_frame_size in analyze-frame response.", source)
        self.assertIn("Missing face_detection.landmark_bbox in analyze-frame response.", source)
        self.assertIn("currentFrameWarnings", source)
        self.assertIn("data-overlay-source=\"analyzed-frame\"", source)
        self.assertIn("Crop preview unavailable.", source)
        self.assertIn("224×224 model input preview", source)
        self.assertIn("224x224 model input preview", source)
        self.assertIn("renderEmotionPipeline", source)
        self.assertIn("raw-probability-bars", source)
        self.assertIn("mapped-state-bars", source)
        self.assertIn("Raw detection unavailable for this checkpoint.", source)
        self.assertIn("bypassed: checkpoint directly predicts academic states", source)
        self.assertIn("sad + anger + disgust -> frustration", source)
        self.assertIn("fear + surprise -> confusion", source)
        self.assertIn("contempt -> boredom", source)
        self.assertIn("happy + neutral -> engagement", source)
        self.assertIn("smoothed-state", source)
        self.assertIn("buffer-contents", source)
        self.assertIn("emotion_pipeline", source)
        self.assertIn("emotion-checkpoint-path", source)
        self.assertIn("/api/local-config/emotion-checkpoint", source)
        self.assertIn("Set RAW_EMOTION_CHECKPOINT_PATH or EMOTION_CHECKPOINT_PATH to an 8-class checkpoint.", source)
        self.assertNotIn("Reaction Window & Strategy Mapping", source)
        self.assertNotIn("YOLO setup", source)
        self.assertNotIn("YOLO loaded", source)
        self.assertNotIn("Allowed strategy families", source)
        self.assertNotIn("neutral, happy, angry", source)
        self.assertNotIn("8-class labels active", source)
        self.assertNotIn("localStorage", source)
        self.assertNotIn("sessionStorage", source)

    def test_camera_debug_status_includes_emotion_pipeline_status(self):
        class FakePipeline:
            def status(self):
                return {
                    "model_loaded": True,
                    "model_output_type": "raw_emotion",
                    "raw_detection_available": True,
                    "classes": ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
                    "architecture": "convnextv2_pico.fcmae_ft_in1k",
                    "checkpoint_path": "models/emotion_model/raw.pt",
                    "mapper_available": True,
                    "buffer_size": 10,
                }

        self.app.state._emotion_pipeline = FakePipeline()

        response = self.app.test_request("GET", "/api/camera-debug/status")

        self.assertEqual(response["status"], 200, response)
        status = response["json"]["emotion_pipeline_status"]
        self.assertEqual(status["model_output_type"], "raw_emotion")
        self.assertTrue(status["raw_detection_available"])
        self.assertEqual(status["classes"][0], "anger")
        self.assertTrue(status["mapper_available"])
        self.assertEqual(status["buffer_size"], 10)

    def test_local_config_emotion_checkpoint_endpoint_writes_env_local_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "raw_emotion.pt"
            checkpoint.write_bytes(b"not a real checkpoint")
            (root / ".env.local").write_text("UNRELATED=value\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                sentinel = object()
                app.state._emotion_pipeline = sentinel
                response = app.test_request(
                    "POST",
                    "/api/local-config/emotion-checkpoint",
                    {"EMOTION_CHECKPOINT_PATH": str(checkpoint)},
                )
                process_path = os.environ.get("EMOTION_CHECKPOINT_PATH")
                pipeline_cache = app.state._emotion_pipeline

            env_text = (root / ".env.local").read_text(encoding="utf-8")
            serialized = json.dumps(response["json"])
            self.assertEqual(response["status"], 200, response)
            self.assertTrue(response["json"]["saved"])
            self.assertEqual(process_path, str(checkpoint))
            self.assertIsNot(pipeline_cache, sentinel)
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn(f"EMOTION_CHECKPOINT_PATH={checkpoint}", env_text)
            self.assertIn(".env.local", (root / ".gitignore").read_text(encoding="utf-8"))
            self.assertNotIn("data:image", serialized)

    def test_camera_debug_source_draws_landmarks_on_analyzed_frame_not_live_video(self):
        source = Path("emotion_aware_assistant/web/static/camera_debug.html").read_text(encoding="utf-8")

        self.assertIn('id="analyzed-frame-preview"', source)
        self.assertIn('id="analyzed-overlay"', source)
        self.assertIn("drawAnalyzedOverlay", source)
        self.assertIn("analysis.analyzed_frame_size", source)
        self.assertNotIn("const scaleX = renderedWidth / frameWidth", source)
        self.assertNotIn("getBoundingClientRect", source)
        self.assertNotIn("clientWidth", source)
        self.assertIn("do not draw landmarks without analyzed_frame_preview_data_url", source)
        self.assertNotIn('id="landmark-layer"', source)
        self.assertNotIn('id="expanded-bbox-overlay"', source)

    def test_pdf_test_source_remains_unchanged(self):
        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "--",
                "emotion_aware_assistant/web/pdf_workspace/src/pdf_test.jsx",
                "emotion_aware_assistant/web/pdf_workspace/src/pdf_test.css",
                "emotion_aware_assistant/web/static/pdf_test.html",
            ],
            check=False,
        )

        self.assertEqual(result.returncode, 0, "/pdf-test source files should remain unchanged.")

    def test_camera_debug_route_serves_static_page(self):
        import emotion_aware_assistant.web.server as server

        app = self.app

        class Handler(server.WebRequestHandler):
            web_app = app

        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address
            with urllib.request.urlopen(f"http://{host}:{port}/camera-debug", timeout=5) as response:
                body = response.read().decode("utf-8")
                status = response.status
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        self.assertEqual(status, 200)
        self.assertIn("Camera Debug Workspace", body)

    def test_face_detector_reports_yolo_requested_missing_weights_and_fallback(self):
        from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status

        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing-yolo.pt"
            with patch.dict(os.environ, {"FACE_DETECTOR": "yolo", "YOLO_FACE_MODEL_PATH": str(missing)}):
                detector = create_face_detector()
                status = detector_status(detector)

        self.assertEqual(status["requested_detector"], "yolo")
        self.assertEqual(status["configured_detector"], "yolo")
        self.assertFalse(status["yolo_loaded"])
        self.assertFalse(status["yolo_weight_exists"])
        self.assertEqual(status["yolo_model_path"], str(missing))
        self.assertIn("not found", status["warning"])
        self.assertIn(status["actual_detector"], {"opencv_haar", "center_crop"})
        self.assertTrue(status["fallback_used"])

    def test_face_detector_defaults_to_auto_not_unknown(self):
        from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status

        with patch.dict(os.environ, {}, clear=True):
            status = detector_status(create_face_detector())

        self.assertEqual(status["requested_detector"], "auto")
        self.assertEqual(status["configured_detector"], "auto")
        self.assertNotEqual(status["actual_detector"], "unknown")
        self.assertIn(status["actual_detector"], {"yolo", "opencv_haar", "center_crop"})

    def test_face_detector_openface_requested_missing_binary_reports_safe_warning(self):
        from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status

        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "FeatureExtraction"
            with patch.dict(
                os.environ,
                {"FACE_DETECTOR": "openface", "OPENFACE_FEATURE_EXTRACTION_BIN": str(missing)},
                clear=True,
            ):
                status = detector_status(create_face_detector())

        self.assertEqual(status["requested_detector"], "openface")
        self.assertEqual(status["configured_detector"], "openface")
        self.assertTrue(status["fallback_used"])
        self.assertIn(status["actual_detector"], {"yolo", "opencv_haar", "center_crop"})
        self.assertIn("openface", status)
        self.assertFalse(status["openface"]["available"])
        self.assertFalse(status["openface"]["binary_exists"])
        self.assertIsNone(status["openface"]["binary_path"])
        self.assertIn("OpenFace FeatureExtraction binary was not found", status["openface"]["warning"])
        self.assertIn("OpenFace unavailable or failed", status["warning"])

    def test_face_detector_reports_missing_ultralytics_dependency(self):
        from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status

        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "yolov8n-face.pt"
            weights.write_bytes(b"placeholder")
            fake_modules = {"ultralytics": None}
            with (
                patch.dict(os.environ, {"FACE_DETECTOR": "yolo", "YOLO_FACE_MODEL_PATH": str(weights)}),
                patch.dict(sys.modules, fake_modules),
            ):
                detector = create_face_detector()
                status = detector_status(detector)

        self.assertEqual(status["requested_detector"], "yolo")
        self.assertFalse(status["yolo_loaded"])
        self.assertTrue(status["yolo_weight_exists"])
        self.assertFalse(status["ultralytics_available"])
        self.assertIn("ultralytics is not installed", status["warning"])
        self.assertTrue(status["fallback_used"])

    def test_face_detector_uses_yolo_when_weights_and_backend_are_available(self):
        from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status

        loaded_paths = []

        class FakeYOLO:
            def __init__(self, path):
                loaded_paths.append(path)

            def __call__(self, frame_bgr, verbose=False):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "yolov8n-face.pt"
            weights.write_bytes(b"placeholder")
            fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
            with (
                patch.dict(os.environ, {"FACE_DETECTOR": "yolo", "YOLO_FACE_MODEL_PATH": str(weights)}),
                patch.dict(sys.modules, {"ultralytics": fake_module}),
            ):
                detector = create_face_detector()
                status = detector_status(detector)

        self.assertEqual(loaded_paths, [str(weights)])
        self.assertEqual(status["requested_detector"], "yolo")
        self.assertEqual(status["actual_detector"], "yolo")
        self.assertTrue(status["yolo_loaded"])
        self.assertTrue(status["yolo_weight_exists"])
        self.assertFalse(status["fallback_used"])

    def test_expand_face_bbox_expands_and_shifts_downward_with_parameters(self):
        from emotion_aware_assistant.emotion.face_detector import expand_face_bbox

        result = expand_face_bbox(
            [100, 50, 50, 50],
            image_width=300,
            image_height=300,
            scale=2.0,
            y_bias=0.20,
            bottom_extra=0.30,
            make_square=True,
        )

        self.assertEqual(result["original_bbox"], [100, 50, 50, 50])
        self.assertEqual(result["crop_parameters"], {"scale": 2.0, "y_bias": 0.2, "top_extra": 0.0, "bottom_extra": 0.3, "make_square": True})
        self.assertGreater(result["expanded_bbox"][2], 50)
        self.assertGreater(result["expanded_bbox"][3], 50)
        self.assertEqual(result["expanded_bbox"][2], result["expanded_bbox"][3])
        self.assertGreater(result["crop_bbox_used"][1], 25)
        self.assertEqual(result["crop_bbox_used"], result["expanded_bbox"])

    def test_expand_face_bbox_clamps_to_image_bounds(self):
        from emotion_aware_assistant.emotion.face_detector import expand_face_bbox

        result = expand_face_bbox(
            [4, 2, 24, 18],
            image_width=64,
            image_height=48,
            scale=2.4,
            y_bias=0.25,
            bottom_extra=0.35,
            make_square=True,
        )

        crop = result["crop_bbox_used"]
        self.assertGreaterEqual(crop[0], 0)
        self.assertGreaterEqual(crop[1], 0)
        self.assertLessEqual(crop[0] + crop[2], 64)
        self.assertLessEqual(crop[1] + crop[3], 48)
        self.assertEqual(crop[2], crop[3])

    def test_openface_parser_reads_success_confidence_landmarks_pose_and_aus(self):
        from emotion_aware_assistant.emotion.face_detector import parse_openface_csv

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "frame.csv"
            headers = ["frame", "success", "confidence"]
            headers.extend(f"x_{index}" for index in range(68))
            headers.extend(f"y_{index}" for index in range(68))
            headers.extend(["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz", "AU01_r", "AU02_c"])
            values = ["1", "1", "0.98"]
            values.extend(str(10 + index) for index in range(68))
            values.extend(str(20 + index) for index in range(68))
            values.extend(["1.0", "2.0", "3.0", "0.1", "0.2", "0.3", "1.5", "1"])
            csv_path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")

            parsed = parse_openface_csv(csv_path)

        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["confidence"], 0.98)
        self.assertEqual(parsed["landmark_count"], 68)
        self.assertEqual(parsed["landmarks"][0], [10.0, 20.0])
        self.assertEqual(parsed["landmarks"][-1], [77.0, 87.0])
        self.assertEqual(parsed["bbox"], [10, 20, 67, 67])
        self.assertEqual(parsed["pose"]["pose_Ry"], 0.2)
        self.assertTrue(parsed["head_pose_available"])
        self.assertTrue(parsed["aus_available"])
        self.assertEqual(parsed["aus"]["AU01_r"], 1.5)
        self.assertEqual(parsed["aus_summary"]["count"], 2)

    def test_openface_subprocess_uses_argument_list_without_shell(self):
        from emotion_aware_assistant.emotion.face_detector import run_openface_feature_extraction

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "FeatureExtraction"
            image = root / "input.jpg"
            out_dir = root / "out"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            image.write_bytes(b"image")

            with patch("emotion_aware_assistant.emotion.face_detector.subprocess.run") as run:
                run.return_value = types.SimpleNamespace(returncode=0, stdout="", stderr="")
                run_openface_feature_extraction(binary, image, out_dir, timeout=5)

        self.assertTrue(run.called)
        args, kwargs = run.call_args
        self.assertIsInstance(args[0], list)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(args[0][0], str(binary))
        self.assertIn("-f", args[0])
        self.assertIn("-out_dir", args[0])

    def test_center_crop_fallback_returns_crop_bbox_used(self):
        class FakeDetector:
            requested_detector = "center_crop"
            actual_detector = "center_crop"

            @property
            def is_available(self):
                return False

            def detect(self, frame_bgr):
                return []

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": "data:image/png;base64,QUJDRA=="},
        )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        self.assertEqual(face["actual_detector"], "center_crop")
        self.assertEqual(face["crop_bbox_used"], face["expanded_bbox"])
        self.assertEqual(face["crop_strategy"], "center_crop")

    def test_analyze_frame_includes_configured_crop_parameters_from_environment(self):
        class FakeDetector:
            requested_detector = "auto"
            actual_detector = "opencv_haar"

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FaceBox(20, 20, 20, 20, 0.8, "opencv_haar")]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        with patch.dict(
            os.environ,
            {
                "FACE_CROP_SCALE": "2.1",
                "FACE_CROP_Y_BIAS": "0.24",
                "FACE_CROP_BOTTOM_EXTRA": "0.34",
                "FACE_CROP_MAKE_SQUARE": "false",
            },
        ):
            response = self.app.test_request(
                "POST",
                "/api/camera-debug/analyze-frame",
                {"image": self._solid_image_data_url((80, 80), (180, 20, 20))},
            )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        self.assertEqual(face["crop_parameters"], {"scale": 2.1, "y_bias": 0.24, "top_extra": 0.22, "bottom_extra": 0.34, "make_square": False})
        self.assertEqual(face["crop_strategy"], "expanded_rect_opencv_haar_fallback")

    def test_analyze_frame_applies_request_crop_settings_and_model_preview_uses_crop_bbox_used(self):
        class FakeDetector:
            requested_detector = "auto"
            actual_detector = "opencv_haar"

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FaceBox(20, 20, 20, 20, 0.8, "opencv_haar")]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()
        image_data = self._crop_marker_image_data_url()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {
                "image": image_data,
                "crop_settings": {
                    "FACE_CROP_SCALE": "1.0",
                    "FACE_CROP_Y_BIAS": "0.50",
                    "FACE_CROP_TOP_EXTRA": "0.0",
                    "FACE_CROP_BOTTOM_EXTRA": "0.0",
                    "FACE_CROP_MAKE_SQUARE": "true",
                },
            },
        )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        self.assertEqual(face["bbox"], [20, 20, 20, 20])
        self.assertEqual(face["crop_bbox_used"], [20, 30, 20, 20])
        self.assertEqual(face["expanded_bbox"], [20, 30, 20, 20])
        self.assertEqual(face["crop_parameters"], {"scale": 1.0, "y_bias": 0.5, "top_extra": 0.0, "bottom_extra": 0.0, "make_square": True})
        preview = self._decode_data_url_image(response["json"]["model_input_preview_data_url"])
        self.assertEqual(preview.size, (224, 224))
        center_pixel = preview.getpixel((112, 112))
        self.assertGreater(center_pixel[1], 150)
        self.assertLess(center_pixel[0], 120)
        self.assertLess(center_pixel[2], 120)

    def test_analyze_frame_returns_analyzed_frame_preview_and_openface_landmark_bbox(self):
        class FakeOpenFaceBox:
            x = 10
            y = 12
            w = 30
            h = 34
            confidence = 0.98
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.98,
                "landmarks": [[10.0, 12.0], [40.0, 12.0], [40.0, 46.0], [10.0, 46.0]],
                "landmark_count": 4,
                "bbox": [10, 12, 30, 34],
                "head_pose_available": False,
                "aus_available": False,
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": self._solid_image_data_url((96, 72), (20, 80, 150))},
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        face = payload["face_detection"]
        self.assertEqual(payload["analyzed_frame_size"], [96, 72])
        self.assertTrue(payload["analyzed_frame_preview_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(face["landmarks"], [[10.0, 12.0], [40.0, 12.0], [40.0, 46.0], [10.0, 46.0]])
        self.assertEqual(face["landmark_count"], 4)
        self.assertEqual(face["landmark_bbox"], [10, 12, 30, 34])
        self.assertEqual(face["bbox"], face["landmark_bbox"])
        self.assertEqual(face["crop_strategy"], "openface_landmark_bbox")
        self.assertEqual(face["crop_mode"], "square_face_context")
        self.assertEqual(face["frame_size"], [96, 72])
        self.assertFalse(face["mirrored"])
        self.assertEqual(face["warnings"], [])

    def test_analyze_frame_openface_success_uses_stable_contract_without_yolo_frame_warning(self):
        class FakeOpenFaceBox:
            x = 20
            y = 20
            w = 20
            h = 20
            confidence = 0.98
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.98,
                "landmarks": [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]],
                "landmark_count": 4,
                "bbox": [20, 20, 20, 20],
                "warning": None,
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = "YOLO is unavailable, so the system is using calibrated OpenCV Haar crop fallback."

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": self._solid_image_data_url((80, 80), (180, 20, 20))},
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        face = payload["face_detection"]
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["analyzed_frame_preview_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(payload["analyzed_frame_size"], [80, 80])
        self.assertEqual(face["actual_detector"], "openface")
        self.assertTrue(face["face_found"])
        self.assertIsNone(face["warning"])
        self.assertEqual(face["warnings"], [])
        self.assertNotIn("YOLO", json.dumps(payload["warnings"]))
        self.assertNotIn("YOLO", json.dumps(face))
        self.assertEqual(face["bbox"], face["landmark_bbox"])
        self.assertEqual(face["crop_strategy"], "openface_landmark_bbox")
        self.assertTrue(payload["crop_preview_data_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(self._decode_data_url_image(payload["model_input_preview_data_url"]).size, (224, 224))

    def test_analyze_frame_include_debug_previews_returns_backend_annotated_frame(self):
        class FakeOpenFaceBox:
            x = 12
            y = 10
            w = 32
            h = 28
            confidence = 0.97
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.97,
                "landmarks": [[12.0, 10.0], [44.0, 10.0], [44.0, 38.0], [12.0, 38.0]],
                "landmark_count": 4,
                "bbox": [12, 10, 32, 28],
                "warning": None,
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()
        self.app.state.upload_dir.mkdir(parents=True, exist_ok=True)
        before_files = sorted(path.relative_to(self.app.state.upload_dir) for path in self.app.state.upload_dir.rglob("*") if path.is_file())

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {
                "image": self._solid_image_data_url((96, 72), (20, 80, 150)),
                "include_debug_previews": True,
            },
        )

        self.assertEqual(response["status"], 200, response)
        payload = response["json"]
        self.assertTrue(payload["annotated_frame_preview_data_url"].startswith("data:image/jpeg;base64,"))
        annotated = self._decode_data_url_image(payload["annotated_frame_preview_data_url"])
        self.assertEqual(annotated.size, (96, 72))
        corner = annotated.getpixel((2, 2))
        self.assertLess(abs(corner[0] - 20), 35)
        self.assertLess(abs(corner[1] - 80), 35)
        self.assertLess(abs(corner[2] - 150), 35)
        after_files = sorted(path.relative_to(self.app.state.upload_dir) for path in self.app.state.upload_dir.rglob("*") if path.is_file())
        self.assertEqual(after_files, before_files)

    def test_openface_square_context_crop_is_derived_from_landmarks_and_clamped(self):
        class FakeOpenFaceBox:
            x = -8
            y = 4
            w = 28
            h = 30
            confidence = 0.9
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.9,
                "landmarks": [[-8.0, 4.0], [20.0, 4.0], [20.0, 34.0], [-8.0, 34.0]],
                "landmark_count": 4,
                "bbox": [-8, 4, 28, 30],
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {
                "image": self._solid_image_data_url((64, 48), (80, 80, 80)),
                "crop_settings": {"crop_mode": "square_face_context"},
            },
        )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        crop = face["crop_bbox_used"]
        self.assertEqual(face["landmark_bbox"], [-8, 4, 28, 30])
        self.assertGreater(crop[2], 0)
        self.assertGreater(crop[3], 0)
        self.assertEqual(crop[2], crop[3])
        self.assertGreaterEqual(crop[0], 0)
        self.assertGreaterEqual(crop[1], 0)
        self.assertLessEqual(crop[0] + crop[2], 64)
        self.assertLessEqual(crop[1] + crop[3], 48)
        self.assertTrue(any("Landmark coordinates outside frame bounds" in item for item in response["json"]["warnings"]))

    def test_analyze_frame_openface_metadata_and_crop_use_landmark_bbox(self):
        class FakeOpenFaceBox:
            x = 20
            y = 20
            w = 20
            h = 20
            confidence = 0.98
            source = "openface"
            openface = {
                "success": True,
                "confidence": 0.98,
                "landmarks": [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]],
                "landmark_count": 4,
                "bbox": [20, 20, 20, 20],
                "pose": {"pose_Rx": 0.1},
                "aus": {"AU01_r": 1.0},
                "aus_summary": {"count": 1, "active_count": 0},
                "head_pose_available": True,
                "aus_available": True,
            }

        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "openface"
            warning = None

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FakeOpenFaceBox()]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {
                "image": self._solid_image_data_url((80, 80), (180, 20, 20)),
                "crop_settings": {
                    "FACE_CROP_SCALE": "2.0",
                    "FACE_CROP_Y_BIAS": "0.2",
                    "FACE_CROP_BOTTOM_EXTRA": "0.3",
                    "FACE_CROP_MAKE_SQUARE": "true",
                },
            },
        )

        self.assertEqual(response["status"], 200, response)
        serialized = json.dumps(response["json"])
        face = response["json"]["face_detection"]
        self.assertEqual(face["requested_detector"], "openface")
        self.assertEqual(face["actual_detector"], "openface")
        self.assertEqual(face["bbox"], [20, 20, 20, 20])
        self.assertGreater(face["expanded_bbox"][2], 20)
        self.assertEqual(face["crop_bbox_used"], face["expanded_bbox"])
        self.assertEqual(face["crop_strategy"], "openface_landmark_bbox")
        self.assertEqual(face["crop_mode"], "square_face_context")
        self.assertFalse(face["fallback_used"])
        self.assertIsNone(face["warning"])
        self.assertEqual(face["warnings"], [])
        self.assertTrue(face["landmarks_available"])
        self.assertEqual(face["landmark_count"], 4)
        self.assertEqual(face["landmarks"], [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]])
        self.assertEqual(face["landmark_bbox"], [20, 20, 20, 20])
        self.assertTrue(face["head_pose_available"])
        self.assertTrue(face["aus_available"])
        self.assertEqual(face["openface"]["landmarks"][0], [20.0, 20.0])
        self.assertNotIn("raw_csv", serialized)
        self.assertNotIn("data:image/png;base64", serialized)

    def test_openface_failure_falls_back_safely_in_analyze_frame(self):
        class FakeDetector:
            requested_detector = "openface"
            actual_detector = "opencv_haar"
            warning = "OpenFace unavailable or failed; using OpenCV Haar fallback."
            last_primary_empty = True

            @property
            def is_available(self):
                return True

            def detect(self, frame_bgr):
                return [FaceBox(20, 20, 20, 20, 0.7, "opencv_haar")]

        class FakeAdapter:
            def status(self):
                return {"model_loaded": False, "loading_error": "not needed"}

        self.app.state._face_detector = FakeDetector()
        self.app.state._emotion_adapter = FakeAdapter()

        response = self.app.test_request(
            "POST",
            "/api/camera-debug/analyze-frame",
            {"image": self._solid_image_data_url((80, 80), (180, 20, 20))},
        )

        self.assertEqual(response["status"], 200, response)
        face = response["json"]["face_detection"]
        self.assertEqual(face["requested_detector"], "openface")
        self.assertEqual(face["actual_detector"], "opencv_haar")
        self.assertTrue(face["fallback_used"])
        self.assertIn("OpenFace unavailable or failed", face["warning"])
        self.assertEqual(face["crop_strategy"], "expanded_square_opencv_haar_fallback")

    def test_local_config_face_detector_endpoint_updates_env_local_and_runtime_state(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\n", encoding="utf-8")
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state._face_detector = object()
                response = app.test_request(
                    "POST",
                    "/api/local-config/face-detector",
                    {
                        "FACE_DETECTOR": "yolo",
                        "YOLO_FACE_MODEL_PATH": "models/face_detector/yolov8n-face.pt",
                    },
                )
                detector_env = os.environ.get("FACE_DETECTOR")
                yolo_path_env = os.environ.get("YOLO_FACE_MODEL_PATH")
                detector_cache = app.state._face_detector

            env_text = (root / ".env.local").read_text(encoding="utf-8")
            self.assertEqual(response["status"], 200, response)
            self.assertEqual(response["json"]["face_detector_status"]["requested_detector"], "yolo")
            self.assertEqual(detector_env, "yolo")
            self.assertEqual(yolo_path_env, "models/face_detector/yolov8n-face.pt")
            self.assertIsNone(detector_cache)
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn("FACE_DETECTOR=yolo", env_text)
            self.assertIn("YOLO_FACE_MODEL_PATH=models/face_detector/yolov8n-face.pt", env_text)

    def test_find_face_detector_weights_finds_candidates_without_installing(self):
        from scripts.find_face_detector_weights import find_weight_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "nested" / "yolov8n-face.pt"
            candidate.parent.mkdir()
            candidate.write_bytes(b"weights")

            results = find_weight_candidates([root])

        self.assertEqual([item["path"] for item in results], [str(candidate)])
        self.assertEqual(results[0]["size_bytes"], 7)

    def test_emotion_debug_alias_serves_static_page(self):
        import emotion_aware_assistant.web.server as server

        app = self.app

        class Handler(server.WebRequestHandler):
            web_app = app

        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address
            with urllib.request.urlopen(f"http://{host}:{port}/emotion-debug", timeout=5) as response:
                body = response.read().decode("utf-8")
                status = response.status
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        self.assertEqual(status, 200)
        self.assertIn("Camera Debug Workspace", body)

    @staticmethod
    def _solid_image_data_url(size, color) -> str:
        from PIL import Image  # type: ignore

        image = Image.new("RGB", size, color)
        output = BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _crop_marker_image_data_url() -> str:
        from PIL import Image, ImageDraw  # type: ignore

        image = Image.new("RGB", (80, 80), (180, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 30, 39, 49), fill=(20, 210, 40))
        output = BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _decode_data_url_image(data_url: str):
        from PIL import Image  # type: ignore

        payload = data_url.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


if __name__ == "__main__":
    unittest.main()
