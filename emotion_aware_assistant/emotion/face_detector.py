from __future__ import annotations

import csv
import math
import os
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from typing import Protocol

from emotion_aware_assistant.core.types import FaceBox

YOLO_CANDIDATE_PATHS = (
    "models/face_detector/yolov8n-face.pt",
    "models/face_detector/yolo-face.pt",
    "models/face_detector/best.pt",
    "models/face_detector/best.onnx",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENFACE_CANDIDATE_PATHS = (
    "FeatureExtraction",
    "FeatureExtraction.exe",
    "build/bin/FeatureExtraction",
    "OpenFace/build/bin/FeatureExtraction",
    "external/OpenFace/build/bin/FeatureExtraction",
    "external/OpenFace/build/bin/FeatureExtraction.exe",
    "/home/rli/OpenFace/build/bin/FeatureExtraction",
    "/home/rli/PycharmProjects/OpenFace/build/bin/FeatureExtraction",
    "/home/rli/CS731-Work/OpenFace/build/bin/FeatureExtraction",
    "/home/rli/CS731-Work/external/OpenFace/build/bin/FeatureExtraction",
    "/usr/local/bin/FeatureExtraction",
    "/opt/OpenFace/build/bin/FeatureExtraction",
)

DEFAULT_FACE_CROP_PARAMETERS = {
    "scale": 1.35,
    "y_bias": -0.04,
    "top_extra": 0.22,
    "bottom_extra": 0.12,
    "make_square": True,
}


class FaceDetector(Protocol):
    def detect(self, frame_bgr) -> list[FaceBox]:
        ...

    @property
    def is_available(self) -> bool:
        ...


class OpenCVHaarFaceDetector:
    def __init__(self):
        self._cv2 = None
        self._cascade = None
        self._available = False
        try:
            import cv2  # type: ignore

            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                self._cv2 = cv2
                self._cascade = cascade
                self._available = True
        except Exception:
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, frame_bgr) -> list[FaceBox]:
        if not self._available:
            return []
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [FaceBox(int(x), int(y), int(w), int(h), 0.7, "opencv_haar") for x, y, w, h in boxes]


class OpenFaceFaceBox:
    def __init__(self, bbox: list[int], confidence: float, metadata: dict[str, Any]):
        self.x, self.y, self.w, self.h = [int(value) for value in bbox]
        self.confidence = float(confidence)
        self.source = "openface"
        self.openface = metadata


class OpenFaceFeatureExtractionDetector:
    def __init__(self, feature_extraction_bin: str | Path | None = None):
        self.requested_detector = "openface"
        self.configured_detector = "openface"
        self.status = discover_openface(feature_extraction_bin, configured=True)
        self.binary_path = Path(self.status["binary_path"]) if self.status.get("binary_path") else None
        self.warning: str | None = self.status.get("warning")
        self.last_openface: dict[str, Any] | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.status.get("available"))

    def detect(self, frame_bgr) -> list[OpenFaceFaceBox]:
        if not self.is_available or self.binary_path is None:
            self.warning = "OpenFace FeatureExtraction binary was not found."
            return []
        try:
            with tempfile.TemporaryDirectory(prefix="openface-frame-") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "frame.jpg"
                output_dir = temp_path / "openface"
                _write_frame_for_openface(frame_bgr, input_path)
                result = run_openface_feature_extraction(self.binary_path, input_path, output_dir)
                if result.returncode != 0:
                    self.warning = "OpenFace unavailable or failed; using fallback detector."
                    return []
                csv_path = _first_openface_csv(output_dir)
                if csv_path is None:
                    self.warning = "OpenFace did not produce landmark output; using fallback detector."
                    return []
                parsed = parse_openface_csv(csv_path)
        except subprocess.TimeoutExpired:
            self.warning = "OpenFace timed out; using fallback detector."
            return []
        except Exception:
            self.warning = "OpenFace unavailable or failed; using fallback detector."
            return []
        self.last_openface = parsed
        if not parsed.get("success") or not parsed.get("bbox"):
            self.warning = "OpenFace did not find a face; using fallback detector."
            return []
        self.warning = None
        return [OpenFaceFaceBox(list(parsed["bbox"]), float(parsed.get("confidence") or 0.0), parsed)]


class YoloV8FaceDetector:
    def __init__(self, weights_path: str | Path | None = None):
        self.requested_detector = "yolo"
        self.candidate_paths = _candidate_yolo_paths(weights_path)
        self.weights_path = next((path for path in self.candidate_paths if path.exists()), self.candidate_paths[0])
        self.weight_exists = self.weights_path.exists()
        self.ultralytics_available = _module_available("ultralytics")
        self.onnxruntime_available = _module_available("onnxruntime")
        self.model = None
        self.backend = ""
        self._available = False
        self.warning: str | None = None
        if not self.weight_exists:
            self.warning = f"YOLO requested but YOLO_FACE_MODEL_PATH was not found. Checked: {', '.join(str(path) for path in self.candidate_paths)}"
            return
        if self.weights_path.suffix.lower() == ".onnx":
            self._load_onnx()
            return
        self._load_ultralytics()

    def _load_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(str(self.weights_path))
            self.backend = "ultralytics"
            self._available = True
        except ModuleNotFoundError:
            self._available = False
            self.warning = "YOLO requested but ultralytics is not installed."
        except Exception as exc:
            self._available = False
            message = str(exc)
            if "ultralytics" in message.lower() and "not" in message.lower():
                self.warning = "YOLO requested but ultralytics is not installed."
            else:
                self.warning = f"YOLO face detector failed to load safely: {type(exc).__name__}: {message}"

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore

            self.model = ort.InferenceSession(str(self.weights_path), providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
            self._available = True
        except ModuleNotFoundError:
            self._available = False
            self.warning = "YOLO requested but onnxruntime is not installed."
        except Exception as exc:
            self._available = False
            self.warning = f"YOLO ONNX face detector failed to load safely: {type(exc).__name__}: {exc}"

    @property
    def is_available(self) -> bool:
        return self._available

    def detect(self, frame_bgr) -> list[FaceBox]:
        if not self._available or self.model is None:
            return []
        if self.backend == "onnxruntime":
            self.warning = "YOLO ONNX model is loaded, but ONNX face postprocessing is not configured; using fallback."
            return []
        results = self.model(frame_bgr, verbose=False)
        boxes: list[FaceBox] = []
        for result in results:
            for box in getattr(result, "boxes", []):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.5
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                boxes.append(FaceBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1), conf, "yolo"))
        return boxes


class FaceDetectorChain:
    def __init__(self, requested_detector: str, primary: Any, fallback: Any):
        self.requested_detector = requested_detector
        self.configured_detector = requested_detector
        self.primary = primary
        self.fallback = fallback
        self.last_primary_empty = False

    @property
    def is_available(self) -> bool:
        return bool(getattr(self.primary, "is_available", False) or getattr(self.fallback, "is_available", False))

    @property
    def actual_detector(self) -> str:
        if getattr(self.primary, "is_available", False):
            return _detector_mode(self.primary)
        if getattr(self.fallback, "is_available", False):
            return _detector_mode(self.fallback)
        return "center_crop"

    @property
    def warning(self) -> str | None:
        if self.actual_detector == self.requested_detector:
            return getattr(self.primary, "warning", None) or getattr(self.fallback, "warning", None)
        if self.requested_detector == "openface":
            fallback_name = self.actual_detector.replace("_", " ")
            if self.actual_detector == "opencv_haar":
                fallback_name = "OpenCV Haar"
            if self.actual_detector == "center_crop":
                fallback_name = "center crop"
            return f"OpenFace unavailable or failed; using {fallback_name} fallback."
        if self.requested_detector in {"yolo", "auto"}:
            primary_warning = getattr(self.primary, "warning", None)
            if not primary_warning:
                primary_warning = "YOLO requested but unavailable." if self.requested_detector == "yolo" else "YOLO auto-detection unavailable."
            elif self.requested_detector == "auto":
                primary_warning = primary_warning.replace("YOLO requested but", "YOLO auto-detection unavailable because")
            if primary_warning and primary_warning[-1] not in ".!?":
                primary_warning = f"{primary_warning}."
            if self.actual_detector == "opencv_haar":
                return f"{primary_warning} Using calibrated OpenCV Haar crop fallback."
            fallback_name = self.actual_detector.replace("_", " ")
            return f"{primary_warning} Using {fallback_name} fallback."
        return getattr(self.primary, "warning", None) or getattr(self.fallback, "warning", None)

    def detect(self, frame_bgr) -> list[FaceBox]:
        self.last_primary_empty = False
        if getattr(self.primary, "is_available", False):
            boxes = self.primary.detect(frame_bgr)
            if boxes:
                return boxes
            self.last_primary_empty = True
        if getattr(self.fallback, "is_available", False):
            return self.fallback.detect(frame_bgr)
        return []


def create_face_detector(preferred: str | None = None, yolo_weights: str | Path | None = None):
    preferred = _normalize_requested_detector(preferred or os.environ.get("FACE_DETECTOR") or "auto")
    yolo_weights = yolo_weights or os.environ.get("YOLO_FACE_MODEL_PATH")
    openface_bin = os.environ.get("OPENFACE_FEATURE_EXTRACTION_BIN")
    if preferred in {"center", "center_crop", "none"}:
        return FaceDetectorChain(
            "center_crop",
            CenterCropFaceDetector(),
            CenterCropFaceDetector(),
        )
    if preferred == "auto" and openface_bin and discover_openface(openface_bin, configured=True).get("available"):
        return FaceDetectorChain(
            "openface",
            OpenFaceFeatureExtractionDetector(openface_bin),
            FaceDetectorChain("yolo", YoloV8FaceDetector(yolo_weights), _fallback_detector()),
        )
    if preferred == "openface":
        return FaceDetectorChain(
            "openface",
            OpenFaceFeatureExtractionDetector(openface_bin),
            FaceDetectorChain("yolo", YoloV8FaceDetector(yolo_weights), _fallback_detector()),
        )
    if preferred in {"auto", "yolo"}:
        return FaceDetectorChain(preferred, YoloV8FaceDetector(yolo_weights), _fallback_detector())
    if preferred in {"opencv_haar", "haar"}:
        return FaceDetectorChain("opencv_haar", OpenCVHaarFaceDetector(), CenterCropFaceDetector())
    return FaceDetectorChain("auto", YoloV8FaceDetector(yolo_weights), _fallback_detector())


class CenterCropFaceDetector:
    def __init__(self, warning: str = "No face detector available; using center crop fallback."):
        self.warning = warning

    @property
    def is_available(self) -> bool:
        return False

    def detect(self, frame_bgr) -> list[FaceBox]:
        return []


def detector_status(detector: Any) -> dict[str, Any]:
    if isinstance(detector, FaceDetectorChain):
        primary_status = detector_status(detector.primary)
        fallback_status = detector_status(detector.fallback)
        requested = detector.requested_detector
        configured = getattr(detector, "configured_detector", requested) or "auto"
        actual = detector.actual_detector
        yolo_status = primary_status if requested in {"auto", "yolo"} else fallback_status
        openface_info = primary_status.get("openface") or fallback_status.get("openface") or discover_openface(configured=requested == "openface")
        warning = detector.warning
        yolo_loaded = bool(yolo_status.get("loaded"))
        fallback_used = bool(requested in {"auto", "yolo"} and actual != "yolo") or bool(actual != requested and requested != "auto")
        return {
            "mode": actual,
            "requested_detector": requested,
            "configured_detector": configured,
            "actual_detector": actual,
            "loaded": actual != "center_crop",
            "fallback": actual if fallback_used else None,
            "fallback_used": fallback_used,
            "warning": warning,
            "fallback_reason": warning or "",
            "yolo_loaded": yolo_loaded,
            "yolo_model_path": str(yolo_status.get("yolo_model_path") or os.environ.get("YOLO_FACE_MODEL_PATH") or YOLO_CANDIDATE_PATHS[0]),
            "yolo_candidate_paths": yolo_status.get("yolo_candidate_paths") or [str(Path(path)) for path in YOLO_CANDIDATE_PATHS],
            "yolo_backend": yolo_status.get("yolo_backend") or "",
            "yolo_weight_exists": bool(yolo_status.get("yolo_weight_exists")),
            "ultralytics_available": bool(yolo_status.get("ultralytics_available")),
            "onnxruntime_available": bool(yolo_status.get("onnxruntime_available")),
            "opencv_haar_loaded": bool(fallback_status.get("loaded")) if requested in {"auto", "yolo"} else bool(primary_status.get("loaded")),
            "openface": openface_info,
        }
    if isinstance(detector, OpenFaceFeatureExtractionDetector):
        actual = "openface" if detector.is_available else "center_crop"
        return {
            "mode": actual,
            "requested_detector": "openface",
            "configured_detector": "openface",
            "actual_detector": actual,
            "loaded": detector.is_available,
            "fallback": None if detector.is_available else "center_crop",
            "fallback_used": not detector.is_available,
            "warning": detector.warning,
            "fallback_reason": detector.warning or "",
            "yolo_loaded": False,
            "yolo_weight_exists": False,
            "ultralytics_available": _module_available("ultralytics"),
            "onnxruntime_available": _module_available("onnxruntime"),
            "openface": detector.status,
        }
    if isinstance(detector, YoloV8FaceDetector):
        return {
            "mode": "yolo",
            "requested_detector": "yolo",
            "configured_detector": "yolo",
            "actual_detector": "yolo" if detector.is_available else "center_crop",
            "loaded": detector.is_available,
            "fallback": None if detector.is_available else "center_crop",
            "fallback_used": not detector.is_available,
            "warning": detector.warning,
            "fallback_reason": detector.warning or "",
            "yolo_loaded": detector.is_available,
            "yolo_model_path": str(detector.weights_path),
            "yolo_candidate_paths": [str(path) for path in detector.candidate_paths],
            "yolo_backend": detector.backend,
            "yolo_weight_exists": detector.weight_exists,
            "ultralytics_available": detector.ultralytics_available,
            "onnxruntime_available": detector.onnxruntime_available,
        }
    if isinstance(detector, OpenCVHaarFaceDetector):
        actual = "opencv_haar" if detector.is_available else "center_crop"
        return {
            "mode": actual,
            "requested_detector": "opencv_haar",
            "configured_detector": "opencv_haar",
            "actual_detector": actual,
            "loaded": detector.is_available,
            "fallback": None if detector.is_available else "center_crop",
            "fallback_used": not detector.is_available,
            "warning": None if detector.is_available else "OpenCV Haar detector unavailable; using center crop fallback.",
            "fallback_reason": "" if detector.is_available else "OpenCV Haar detector unavailable; using center crop fallback.",
            "yolo_loaded": False,
            "yolo_weight_exists": False,
            "ultralytics_available": _module_available("ultralytics"),
            "onnxruntime_available": _module_available("onnxruntime"),
            "openface": discover_openface(configured=False),
        }
    requested = str(getattr(detector, "requested_detector", "auto") or "auto")
    configured = str(getattr(detector, "configured_detector", requested) or requested)
    actual = str(getattr(detector, "actual_detector", getattr(detector, "mode", "center_crop")) or "center_crop")
    if requested != "center_crop" or actual != "center_crop":
        fallback_used = actual != requested and requested != "auto"
        return {
            "mode": actual,
            "requested_detector": requested,
            "configured_detector": configured,
            "actual_detector": actual,
            "loaded": bool(getattr(detector, "is_available", False)),
            "fallback": actual if fallback_used else None,
            "fallback_used": fallback_used,
            "warning": getattr(detector, "warning", None),
            "fallback_reason": getattr(detector, "warning", None) or "",
            "yolo_loaded": bool(getattr(detector, "yolo_loaded", False)),
            "yolo_model_path": str(getattr(detector, "yolo_model_path", os.environ.get("YOLO_FACE_MODEL_PATH") or YOLO_CANDIDATE_PATHS[0])),
            "yolo_weight_exists": Path(str(getattr(detector, "yolo_model_path", os.environ.get("YOLO_FACE_MODEL_PATH") or YOLO_CANDIDATE_PATHS[0]))).exists(),
            "ultralytics_available": _module_available("ultralytics"),
            "onnxruntime_available": _module_available("onnxruntime"),
            "openface": discover_openface(configured=requested == "openface"),
        }
    return {
        "mode": "center_crop",
        "requested_detector": "center_crop",
        "configured_detector": "center_crop",
        "actual_detector": "center_crop",
        "loaded": False,
        "fallback": "center_crop",
        "fallback_used": True,
        "warning": getattr(detector, "warning", "Face detector unavailable; using center crop fallback."),
        "fallback_reason": getattr(detector, "warning", "Face detector unavailable; using center crop fallback."),
        "yolo_loaded": False,
        "yolo_weight_exists": False,
        "ultralytics_available": _module_available("ultralytics"),
        "onnxruntime_available": _module_available("onnxruntime"),
        "openface": discover_openface(configured=False),
    }


class FaceCropResult(dict):
    """Structured crop metadata with legacy integer bbox indexing support."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return super().__getitem__("crop_bbox_used")[key]
        return super().__getitem__(key)


def expand_face_bbox(
    bbox_or_x: Any,
    *args: Any,
    image_width: int | None = None,
    image_height: int | None = None,
    scale: float = DEFAULT_FACE_CROP_PARAMETERS["scale"],
    y_bias: float = DEFAULT_FACE_CROP_PARAMETERS["y_bias"],
    top_extra: float = 0.0,
    bottom_extra: float = DEFAULT_FACE_CROP_PARAMETERS["bottom_extra"],
    make_square: bool = DEFAULT_FACE_CROP_PARAMETERS["make_square"],
    margin: float | None = None,
    margin_ratio: float | None = None,
    bottom_bias: float | None = None,
) -> FaceCropResult:
    if isinstance(bbox_or_x, (list, tuple)):
        x, y, w, h = [int(value) for value in bbox_or_x[:4]]
    else:
        if len(args) < 3:
            raise TypeError("expand_face_bbox requires bbox [x, y, w, h] or x, y, w, h")
        x = int(bbox_or_x)
        y = int(args[0])
        w = int(args[1])
        h = int(args[2])
        if len(args) >= 5:
            image_width = int(args[3])
            image_height = int(args[4])
    if image_width is None or image_height is None:
        raise TypeError("image_width and image_height are required")
    if margin is not None or margin_ratio is not None:
        margin_value = margin if margin_ratio is None else margin_ratio
        scale = 1.0 + 2.0 * max(0.0, float(margin_value or 0.0))
    if bottom_bias is not None:
        y_bias = float(bottom_bias)
    scale_value = max(1.0, float(scale))
    y_bias_value = float(y_bias)
    top_extra_value = max(0.0, float(top_extra))
    bottom_extra_value = max(0.0, float(bottom_extra))
    make_square_value = bool(make_square)
    original_bbox = [int(x), int(y), int(w), int(h)]
    crop_parameters = {
        "scale": round(scale_value, 4),
        "y_bias": round(y_bias_value, 4),
        "top_extra": round(top_extra_value, 4),
        "bottom_extra": round(bottom_extra_value, 4),
        "make_square": make_square_value,
    }
    strategy = "expanded_square" if make_square_value else "expanded_rect"
    if image_width <= 0 or image_height <= 0 or w <= 0 or h <= 0:
        crop_bbox = [0, 0, max(0, int(image_width)), max(0, int(image_height))]
        return FaceCropResult(
            {
                "original_bbox": original_bbox,
                "expanded_bbox": crop_bbox,
                "crop_bbox_used": crop_bbox,
                "crop_strategy": strategy,
                "crop_parameters": crop_parameters,
            }
        )
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0 + float(h) * y_bias_value + float(h) * (bottom_extra_value - top_extra_value) / 2.0
    expanded_w = int(round(float(w) * scale_value))
    expanded_h = int(round(float(h) * scale_value + float(h) * top_extra_value + float(h) * bottom_extra_value))
    if make_square_value:
        side = max(w, h, expanded_w, expanded_h)
        side = min(side, image_width, image_height)
        left = int(round(center_x - side / 2.0))
        top = int(round(center_y - side / 2.0))
        left = max(0, min(left, image_width - side))
        top = max(0, min(top, image_height - side))
        crop_bbox = [int(left), int(top), int(side), int(side)]
        return FaceCropResult(
            {
                "original_bbox": original_bbox,
                "expanded_bbox": crop_bbox,
                "crop_bbox_used": crop_bbox,
                "crop_strategy": strategy,
                "crop_parameters": crop_parameters,
            }
        )
    expanded_w = min(max(w, expanded_w), image_width)
    expanded_h = min(max(h, expanded_h), image_height)
    left = int(round(center_x - expanded_w / 2.0))
    top = int(round(center_y - expanded_h / 2.0))
    left = max(0, min(left, image_width - expanded_w))
    top = max(0, min(top, image_height - expanded_h))
    crop_bbox = [int(left), int(top), int(expanded_w), int(expanded_h)]
    return FaceCropResult(
        {
            "original_bbox": original_bbox,
            "expanded_bbox": crop_bbox,
            "crop_bbox_used": crop_bbox,
            "crop_strategy": strategy,
            "crop_parameters": crop_parameters,
        }
    )


def discover_openface(feature_extraction_bin: str | Path | None = None, configured: bool | None = None) -> dict[str, Any]:
    env_path = os.environ.get("OPENFACE_FEATURE_EXTRACTION_BIN")
    requested_detector = _normalize_requested_detector(os.environ.get("FACE_DETECTOR") or "auto")
    configured_path = feature_extraction_bin or env_path
    configured_value = bool(configured_path or requested_detector == "openface") if configured is None else bool(configured)
    candidates = _candidate_openface_paths(configured_path)
    found = next((path for path in candidates if path.exists() and path.is_file()), None)
    binary_exists = bool(found)
    available = bool(found and os.access(found, os.X_OK))
    warning = None
    if not found:
        warning = "OpenFace FeatureExtraction binary was not found."
    elif not available:
        warning = "OpenFace FeatureExtraction binary is not executable."
    return {
        "configured": configured_value,
        "binary_path": str(found) if found and available else None,
        "binary_exists": binary_exists,
        "available": available,
        "version_check": "",
        "warning": warning,
    }


def parse_openface_csv(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if any(str(value or "").strip() for value in row.values())]
    if not rows:
        return _openface_parse_failure("OpenFace CSV did not contain a result row.")
    row = rows[0]
    success = _truthy_openface_success(row.get("success", "1"))
    confidence = round(_safe_float(row.get("confidence"), 0.0), 4)
    landmarks = _openface_landmarks_from_row(row)
    bbox = _landmark_bbox(landmarks)
    pose = {
        key.strip(): round(_safe_float(value), 4)
        for key, value in row.items()
        if key and key.strip().startswith("pose_") and _safe_float(value, None) is not None
    }
    aus = {
        key.strip(): round(_safe_float(value), 4)
        for key, value in row.items()
        if key and key.strip().startswith("AU") and (key.strip().endswith("_r") or key.strip().endswith("_c")) and _safe_float(value, None) is not None
    }
    if not success:
        return {
            "success": False,
            "confidence": confidence,
            "landmarks": landmarks,
            "landmark_count": len(landmarks),
            "bbox": bbox,
            "pose": pose,
            "aus": aus,
            "aus_summary": _aus_summary(aus),
            "head_pose_available": bool(pose),
            "aus_available": bool(aus),
            "warning": "OpenFace reported success=0.",
        }
    if bbox is None:
        return _openface_parse_failure("OpenFace landmarks were not found.", confidence=confidence, pose=pose, aus=aus)
    return {
        "success": True,
        "confidence": confidence,
        "landmarks": landmarks,
        "landmark_count": len(landmarks),
        "bbox": bbox,
        "pose": pose,
        "aus": aus,
        "aus_summary": _aus_summary(aus),
        "head_pose_available": bool(pose),
        "aus_available": bool(aus),
        "warning": None,
    }


def run_openface_feature_extraction(
    binary_path: str | Path,
    input_path: str | Path,
    output_dir: str | Path,
    timeout: float = 8.0,
) -> subprocess.CompletedProcess[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = [str(binary_path), "-f", str(input_path), "-out_dir", str(output)]
    attempts = [base + ["-aus", "-pose", "-2Dfp"], base]
    last_result: subprocess.CompletedProcess[str] | None = None
    for command in attempts:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        last_result = result
        if result.returncode == 0:
            return result
    return last_result if last_result is not None else subprocess.CompletedProcess(base, 1, "", "")


def _candidate_yolo_paths(configured_path: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if configured_path:
        paths.append(Path(configured_path).expanduser())
    paths.extend(Path(path) for path in YOLO_CANDIDATE_PATHS)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return deduped


def _candidate_openface_paths(configured_path: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    if configured_path:
        return [Path(configured_path).expanduser()]
    path_candidate = shutil.which("FeatureExtraction")
    if path_candidate:
        paths.append(Path(path_candidate))
    for candidate in OPENFACE_CANDIDATE_PATHS:
        path = Path(candidate).expanduser()
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return deduped


def _write_frame_for_openface(frame_bgr: Any, input_path: Path) -> None:
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        arr = np.asarray(frame_bgr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3][:, :, ::-1]
            Image.fromarray(rgb.astype("uint8")).save(input_path, format="JPEG", quality=92)
            return
        if arr.ndim == 2:
            Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB").save(input_path, format="JPEG", quality=92)
            return
    except Exception:
        pass
    raise ValueError("OpenFace input frame could not be written.")


def _first_openface_csv(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def _openface_landmarks_from_row(row: dict[str, Any]) -> list[list[float]]:
    landmarks: list[list[float]] = []
    for index in range(68):
        x = _safe_float(row.get(f"x_{index}"), None)
        y = _safe_float(row.get(f"y_{index}"), None)
        if x is None or y is None:
            continue
        landmarks.append([round(x, 4), round(y, 4)])
    return landmarks


def _landmark_bbox(landmarks: list[list[float]]) -> list[int] | None:
    if not landmarks:
        return None
    xs = [point[0] for point in landmarks if len(point) >= 2 and math.isfinite(point[0])]
    ys = [point[1] for point in landmarks if len(point) >= 2 and math.isfinite(point[1])]
    if not xs or not ys:
        return None
    min_x = math.floor(min(xs))
    min_y = math.floor(min(ys))
    width = max(1, math.ceil(max(xs) - min(xs)))
    height = max(1, math.ceil(max(ys) - min(ys)))
    return [int(min_x), int(min_y), int(width), int(height)]


def _aus_summary(aus: dict[str, float]) -> dict[str, Any]:
    intensities = [value for key, value in aus.items() if key.endswith("_r")]
    classifications = [value for key, value in aus.items() if key.endswith("_c")]
    return {
        "count": len(aus),
        "intensity_count": len(intensities),
        "classification_count": len(classifications),
        "active_count": sum(1 for value in classifications if value >= 0.5),
        "max_intensity": round(max(intensities), 4) if intensities else 0.0,
    }


def _openface_parse_failure(
    warning: str,
    confidence: float = 0.0,
    pose: dict[str, float] | None = None,
    aus: dict[str, float] | None = None,
) -> dict[str, Any]:
    aus = aus or {}
    pose = pose or {}
    return {
        "success": False,
        "confidence": round(float(confidence), 4),
        "landmarks": [],
        "landmark_count": 0,
        "bbox": None,
        "pose": pose,
        "aus": aus,
        "aus_summary": _aus_summary(aus),
        "head_pose_available": bool(pose),
        "aus_available": bool(aus),
        "warning": warning,
    }


def _truthy_openface_success(value: Any) -> bool:
    try:
        return float(str(value).strip()) > 0
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes", "on"}


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fallback_detector() -> Any:
    haar = OpenCVHaarFaceDetector()
    if haar.is_available:
        return haar
    return CenterCropFaceDetector("OpenCV Haar detector unavailable; using center crop fallback.")


def _detector_mode(detector: Any) -> str:
    if isinstance(detector, FaceDetectorChain):
        return detector.actual_detector
    if isinstance(detector, OpenFaceFeatureExtractionDetector):
        return "openface" if detector.is_available else "center_crop"
    if isinstance(detector, YoloV8FaceDetector):
        return "yolo" if detector.is_available else "center_crop"
    if isinstance(detector, OpenCVHaarFaceDetector):
        return "opencv_haar" if detector.is_available else "center_crop"
    return str(getattr(detector, "actual_detector", getattr(detector, "mode", "center_crop")) or "center_crop")


def _normalize_requested_detector(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "haar": "opencv_haar",
        "opencv": "opencv_haar",
        "opencv-haar": "opencv_haar",
        "center": "center_crop",
        "none": "center_crop",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"auto", "openface", "yolo", "opencv_haar", "center_crop"} else "auto"


def _module_available(name: str) -> bool:
    sentinel = object()
    module = sys.modules.get(name, sentinel)
    if module is None:
        return False
    if module is not sentinel:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
