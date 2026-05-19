from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from emotion_aware_assistant.app import AssistantSession
from emotion_aware_assistant.core import llm_config
from emotion_aware_assistant.core.config import LOCAL_ENV_FILE, PROJECT_ROOT, load_config, parse_env_file
from emotion_aware_assistant.core.types import EmotionPrediction, PaperContext
from emotion_aware_assistant.emotion.labels import ALLOWED_EMOTIONS
from emotion_aware_assistant.emotion.face_detector import create_face_detector, detector_status, expand_face_bbox
from emotion_aware_assistant.emotion.raw_emotion_pipeline import CombinedEmotionPipeline
from emotion_aware_assistant.emotion.state_mapper import map_prediction_to_learning_state
from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter
from emotion_aware_assistant.llm.response_policy import get_response_policy
from emotion_aware_assistant.llm.providers import DEFAULT_GEMINI_MODEL, GEMINI_ENDPOINT_TEMPLATE, build_prompt_messages, explain_selection
from emotion_aware_assistant.paper.document import Document, Page
from emotion_aware_assistant.llm.model_registry import configured_models
from emotion_aware_assistant.paper.passage_analyzer import analyze_passage, surrounding_text
from emotion_aware_assistant.paper.pdf_parse_pipeline import (
    load_blocks,
    match_blocks_for_rects,
    parse_pdf_to_blocks,
)
from emotion_aware_assistant.paper.paper_rag import retrieve_context as retrieve_paper_context
from emotion_aware_assistant.paper.paper_rag import normalize_pdf_text
from emotion_aware_assistant.paper.text_chunker import chunk_document
from scripts.configure_api_key import DEFAULTS as GEMINI_CONFIG_DEFAULTS
from scripts.configure_api_key import _ensure_gitignore_entry, _replace_or_append, configure_gemini_key
from scripts.create_sample_data import create_sample_data


class WebState:
    """Holds shared backend state for one local demo server."""

    ACADEMIC_STATES = ("boredom", "confusion", "engagement", "frustration")
    STRATEGY_FAMILY = {
        "confusion": ("step_by_step_breakdown", "concrete_example", "define_key_terms", "input_process_output_map", "formula_intuition", "mechanism_walkthrough"),
        "frustration": ("simplest_version_first", "analogy_or_reframe", "key_takeaway_first", "one_small_next_step", "reduce_information_density"),
        "boredom": ("one_sentence_takeaway", "why_it_matters", "quick_quiz", "compare_with_familiar_method", "make_it_relevant"),
        "engagement": ("deep_technical_explanation", "critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"),
        "neutral": ("concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"),
    }
    SUPPORT_CUE_STRATEGY_FAMILIES = {
        "sustained_clarification": ("step_by_step_breakdown", "define_key_terms", "concrete_example", "input_process_output_map", "mechanism_walkthrough", "formula_intuition"),
        "reduce_load": ("simplest_version_first", "one_small_next_step", "analogy_or_reframe", "reduce_information_density", "key_takeaway_first"),
        "re_engagement": ("why_it_matters", "one_sentence_takeaway", "make_it_relevant", "compare_with_familiar_method", "quick_quiz"),
        "deepening": ("deep_technical_explanation", "critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"),
        "clarify_and_reengage": ("concise_explanation", "concrete_example", "why_it_matters", "step_by_step_breakdown", "compare_with_familiar_method"),
        "gentle_clarification": ("simplest_version_first", "one_small_next_step", "define_key_terms", "analogy_or_reframe", "concrete_example"),
        "neutral_or_uncertain": ("concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"),
        "neutral": ("concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"),
    }
    PREPARE_STEPS = [
        {"id": "upload", "stage": "uploading_pdf", "label": "Uploading PDF", "start": 0, "end": 15},
        {"id": "extract", "stage": "extracting_text", "label": "Extracting text and layout", "start": 15, "end": 40},
        {"id": "profile", "stage": "building_profile", "label": "Building paper profile", "start": 40, "end": 55},
        {"id": "keyword", "stage": "building_keyword_index", "label": "Building keyword index", "start": 55, "end": 70},
        {"id": "embedding", "stage": "building_embedding_index", "label": "Building embedding index", "start": 70, "end": 95},
        {"id": "ready", "stage": "ready", "label": "Ready", "start": 95, "end": 100},
    ]
    FACE_CROP_DEFAULTS = {
        "crop_mode": "square_face_context",
        "crop_scale": 1.35,
        "crop_y_bias": -0.04,
        "crop_top_extra": 0.22,
        "crop_bottom_extra": 0.12,
        "crop_make_square": True,
    }
    REQUIRED_DOCUMENT_SUBDIRS = ("highlights", "highlights/crops", "threads", "prompt_snapshots", "logs")
    OPENFACE_CROP_MODE_DEFAULTS = {
        "landmark_tight": {
            "scale": 1.10,
            "y_bias": 0.0,
            "top_extra": 0.04,
            "bottom_extra": 0.04,
            "make_square": False,
        },
        "face_context": {
            "scale": 1.25,
            "y_bias": -0.06,
            "top_extra": 0.30,
            "bottom_extra": 0.14,
            "make_square": False,
        },
        "square_face_context": {
            "scale": 1.35,
            "y_bias": -0.04,
            "top_extra": 0.22,
            "bottom_extra": 0.12,
            "make_square": True,
        },
    }

    def __init__(self, config: dict[str, Any] | None = None, force_dummy_llm: bool = False):
        self.config = config or load_config("config.yaml")
        self.session = AssistantSession(self.config, force_dummy_llm=force_dummy_llm)
        self.last_prompt_preview = ""
        self.last_request_summary: dict[str, Any] = {}
        self.last_frame_status = "No frame processed yet."
        self.upload_dir = Path("runtime_uploads").resolve()
        self.documents_dir = self.upload_dir / "documents"
        self.documents: dict[str, dict[str, Any]] = {}
        self.current_document_id: str | None = None
        self.current_document_type: str | None = None
        self.highlights_by_document: dict[str, list[dict[str, Any]]] = {}
        self.last_highlight_id: str | None = None
        self._emotion_adapter: Any | None = None
        self._emotion_pipeline: Any | None = None
        self._face_detector: Any | None = None

    def status(self) -> dict[str, Any]:
        document = self.session.document
        emotion_model = self.emotion_model_status()
        llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        strategy_provider = os.environ.get("STRATEGY_PLANNER_PROVIDER", "").strip().lower()
        gemini_key_configured = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        return {
            "app": {"name": "Emotion-Aware Academic Assistant", "mode": "web"},
            "models": configured_models(self.config),
            "current_document": {
                "document_id": self.current_document_id,
                "type": self.current_document_type,
                "title": document.title,
                "page_count": document.page_count,
                "current_page": self.session.current_page_number,
            }
            if document
            else None,
            "emotion_model_status": emotion_model,
            "face_detector_status": self._face_detector_status(),
            "llm_available": self.session.llm.is_available,
            "llm_client": self.session.llm.name,
            "llm_provider": llm_provider or "not configured",
            "strategy_planner_provider": strategy_provider or "not configured",
            "llm_provider_configured": llm_provider == "gemini" and gemini_key_configured,
            "strategy_planner_provider_configured": strategy_provider == "gemini" and gemini_key_configured,
            "gemini_api_key_configured": gemini_key_configured,
            "api_key_status": "configured" if self.session.llm.name == "openrouter" else "not configured; dummy active",
            "dependency_status": self._dependency_status(),
            "log_path": str(self.session.logger.path),
            "last_prompt_preview": self.last_prompt_preview,
            "last_request_summary": self.last_request_summary,
        }

    def local_config_status(self) -> dict[str, Any]:
        env_path = PROJECT_ROOT / LOCAL_ENV_FILE
        values = parse_env_file(env_path)
        api_key = os.environ.get("GEMINI_API_KEY") or values.get("GEMINI_API_KEY") or ""
        llm_provider = os.environ.get("LLM_PROVIDER") or values.get("LLM_PROVIDER") or "not configured"
        strategy_provider = (
            os.environ.get("STRATEGY_PLANNER_PROVIDER")
            or values.get("STRATEGY_PLANNER_PROVIDER")
            or "not configured"
        )
        gemini_model = os.environ.get("GEMINI_MODEL") or values.get("GEMINI_MODEL") or GEMINI_CONFIG_DEFAULTS["GEMINI_MODEL"]
        gemini_embedding_model = (
            os.environ.get("GEMINI_EMBEDDING_MODEL")
            or values.get("GEMINI_EMBEDDING_MODEL")
            or GEMINI_CONFIG_DEFAULTS["GEMINI_EMBEDDING_MODEL"]
        )
        crop_settings = self._safe_face_crop_settings(file_values=values)
        return {
            "env_local_present": env_path.exists(),
            "gemini_api_key_configured": bool(api_key.strip()),
            "llm_provider": llm_provider,
            "strategy_planner_provider": strategy_provider,
            "gemini_model": gemini_model,
            "gemini_embedding_model": gemini_embedding_model,
            "masked_key": self._mask_api_key(api_key),
            **crop_settings,
        }

    def save_local_gemini_config(self, data: dict[str, Any]) -> dict[str, Any]:
        api_key = str(data.get("gemini_api_key") or "").strip()
        if not api_key:
            raise ValueError("Gemini API key is required.")
        llm_provider = str(data.get("llm_provider") or GEMINI_CONFIG_DEFAULTS["LLM_PROVIDER"]).strip()
        gemini_model = str(data.get("gemini_model") or GEMINI_CONFIG_DEFAULTS["GEMINI_MODEL"]).strip()
        gemini_embedding_model = str(
            data.get("gemini_embedding_model") or GEMINI_CONFIG_DEFAULTS["GEMINI_EMBEDDING_MODEL"]
        ).strip()
        strategy_provider = str(
            data.get("strategy_planner_provider") or GEMINI_CONFIG_DEFAULTS["STRATEGY_PLANNER_PROVIDER"]
        ).strip()
        configure_gemini_key(
            PROJECT_ROOT,
            api_key,
            llm_provider=llm_provider,
            gemini_model=gemini_model,
            gemini_embedding_model=gemini_embedding_model,
            strategy_planner_provider=strategy_provider,
            quiet=True,
        )
        os.environ["LLM_PROVIDER"] = llm_provider or GEMINI_CONFIG_DEFAULTS["LLM_PROVIDER"]
        os.environ["GEMINI_MODEL"] = gemini_model or GEMINI_CONFIG_DEFAULTS["GEMINI_MODEL"]
        os.environ["GEMINI_EMBEDDING_MODEL"] = gemini_embedding_model or GEMINI_CONFIG_DEFAULTS["GEMINI_EMBEDDING_MODEL"]
        os.environ["STRATEGY_PLANNER_PROVIDER"] = strategy_provider or GEMINI_CONFIG_DEFAULTS["STRATEGY_PLANNER_PROVIDER"]
        os.environ["GEMINI_API_KEY"] = api_key
        status = self.local_config_status()
        status.update({"saved": True, "restart_required": False})
        return status

    def save_local_face_detector_config(self, data: dict[str, Any]) -> dict[str, Any]:
        detector = self._normalize_face_detector_config_value(data.get("FACE_DETECTOR") or data.get("face_detector") or "auto")
        yolo_model_path = str(
            data.get("YOLO_FACE_MODEL_PATH")
            or data.get("yolo_face_model_path")
            or "models/face_detector/yolov8n-face.pt"
        ).strip()
        if not yolo_model_path:
            yolo_model_path = "models/face_detector/yolov8n-face.pt"
        env_path = PROJECT_ROOT / LOCAL_ENV_FILE
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        for key, value in {
            "FACE_DETECTOR": detector,
            "YOLO_FACE_MODEL_PATH": yolo_model_path,
        }.items():
            lines, _ = _replace_or_append(lines, key, value)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        _ensure_gitignore_entry(PROJECT_ROOT)
        os.environ["FACE_DETECTOR"] = detector
        os.environ["YOLO_FACE_MODEL_PATH"] = yolo_model_path
        self._face_detector = None
        status = detector_status(create_face_detector())
        self._face_detector = None
        return {
            "saved": True,
            "restart_required": False,
            "env_local_present": env_path.exists(),
            "face_detector_status": status,
        }

    def save_local_face_crop_config(self, data: dict[str, Any]) -> dict[str, Any]:
        settings = self._safe_face_crop_settings(overrides=data, file_values={})
        env_path = PROJECT_ROOT / LOCAL_ENV_FILE
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updates = {
            "FACE_CROP_MODE": str(settings["crop_mode"]),
            "FACE_CROP_SCALE": str(settings["crop_scale"]),
            "FACE_CROP_Y_BIAS": str(settings["crop_y_bias"]),
            "FACE_CROP_TOP_EXTRA": str(settings["crop_top_extra"]),
            "FACE_CROP_BOTTOM_EXTRA": str(settings["crop_bottom_extra"]),
            "FACE_CROP_MAKE_SQUARE": "true" if settings["crop_make_square"] else "false",
        }
        for key, value in updates.items():
            lines, _ = _replace_or_append(lines, key, value)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        _ensure_gitignore_entry(PROJECT_ROOT)
        os.environ.update(updates)
        return {
            "saved": True,
            "restart_required": False,
            "env_local_present": env_path.exists(),
            **settings,
        }

    def save_local_openface_config(self, data: dict[str, Any]) -> dict[str, Any]:
        detector = self._normalize_face_detector_config_value(data.get("FACE_DETECTOR") or data.get("face_detector") or "openface")
        if detector != "openface":
            detector = "openface"
        openface_bin = str(
            data.get("OPENFACE_FEATURE_EXTRACTION_BIN")
            or data.get("openface_feature_extraction_bin")
            or ""
        ).strip()
        env_path = PROJECT_ROOT / LOCAL_ENV_FILE
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updates = {
            "FACE_DETECTOR": detector,
            "OPENFACE_FEATURE_EXTRACTION_BIN": openface_bin,
        }
        for key, value in updates.items():
            lines, _ = _replace_or_append(lines, key, value)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        _ensure_gitignore_entry(PROJECT_ROOT)
        os.environ.update(updates)
        self._face_detector = None
        status = detector_status(create_face_detector())
        self._face_detector = None
        return {
            "saved": True,
            "restart_required": False,
            "env_local_present": env_path.exists(),
            "face_detector_status": status,
        }

    def save_local_emotion_checkpoint_config(self, data: dict[str, Any]) -> dict[str, Any]:
        checkpoint_path = str(
            data.get("EMOTION_CHECKPOINT_PATH")
            or data.get("RAW_EMOTION_CHECKPOINT_PATH")
            or data.get("emotion_checkpoint_path")
            or ""
        ).strip()
        if not checkpoint_path:
            raise ValueError("Emotion checkpoint path is required.")
        env_path = PROJECT_ROOT / LOCAL_ENV_FILE
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        mode = str(data.get("EMOTION_MODEL_MODE") or data.get("emotion_model_mode") or "auto").strip().lower()
        mode = mode if mode in {"auto", "raw_emotion", "academic_state"} else "auto"
        updates = {"EMOTION_CHECKPOINT_PATH": checkpoint_path, "EMOTION_MODEL_MODE": mode}
        if data.get("RAW_EMOTION_CHECKPOINT_PATH"):
            updates["RAW_EMOTION_CHECKPOINT_PATH"] = str(data.get("RAW_EMOTION_CHECKPOINT_PATH")).strip()
        for key, value in updates.items():
            lines, _ = _replace_or_append(lines, key, value)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass
        _ensure_gitignore_entry(PROJECT_ROOT)
        os.environ.update(updates)
        self._emotion_pipeline = None
        return {
            "saved": True,
            "restart_required": False,
            "env_local_present": env_path.exists(),
            "emotion_checkpoint_path": checkpoint_path,
            "emotion_pipeline_status": self._get_emotion_pipeline().status(),
        }

    def test_local_gemini_config(self) -> dict[str, Any]:
        status = self.local_config_status()
        return {
            "configured": bool(status.get("gemini_api_key_configured")),
            "llm_provider": status.get("llm_provider"),
            "strategy_planner_provider": status.get("strategy_planner_provider"),
            "gemini_model": status.get("gemini_model"),
            "gemini_embedding_model": status.get("gemini_embedding_model"),
        }

    def local_llm_status(self) -> dict[str, Any]:
        return llm_config.llm_status(PROJECT_ROOT, self.upload_dir)

    def save_local_llm_provider(self, data: dict[str, Any]) -> dict[str, Any]:
        result = llm_config.save_provider_config(PROJECT_ROOT, data)
        result["comparison_models"] = llm_config.load_comparison_models(self.upload_dir)
        return result

    def save_local_llm_roles(self, data: dict[str, Any]) -> dict[str, Any]:
        return llm_config.save_role_config(PROJECT_ROOT, data)

    def local_llm_comparison_models(self) -> dict[str, Any]:
        return {"comparison_models": llm_config.load_comparison_models(self.upload_dir)}

    def save_local_llm_comparison_models(self, data: dict[str, Any]) -> dict[str, Any]:
        return llm_config.save_comparison_models(self.upload_dir, data)

    def test_local_llm_config(self, data: dict[str, Any]) -> dict[str, Any]:
        return llm_config.test_provider_config(PROJECT_ROOT, self.upload_dir, data)

    def list_llm_prompt_snapshots(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        filters = filters or {}
        document_filter = str(filters.get("document_id") or "").strip()
        highlight_filter = str(filters.get("highlight_id") or "").strip()
        stage_filter = str(filters.get("stage") or "").strip()
        snapshots = []
        for path in self._iter_prompt_snapshot_paths(document_filter=document_filter):
            snapshot = self._read_json(path, {})
            if not isinstance(snapshot, dict):
                continue
            if highlight_filter and str(snapshot.get("highlight_id") or "") != highlight_filter:
                continue
            if stage_filter and str(snapshot.get("stage") or "") != stage_filter:
                continue
            snapshots.append(self._prompt_snapshot_compact(snapshot))
        snapshots.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        filters_active = bool(document_filter or highlight_filter or stage_filter)
        any_snapshots = bool(self._iter_prompt_snapshot_paths())
        empty_message = (
            "No prompt snapshots match the current filters."
            if filters_active and any_snapshots
            else "No prompt snapshots found. Run an explanation in /pdf-chat first."
        )
        return {
            "prompt_snapshots": snapshots,
            "message": "" if snapshots else empty_message,
        }

    def get_llm_prompt_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        path = self._prompt_snapshot_path_by_id(snapshot_id)
        snapshot = self._read_json(path, {})
        if not isinstance(snapshot, dict) or not snapshot.get("snapshot_id"):
            raise KeyError(f"Unknown prompt snapshot: {snapshot_id}")
        return {"snapshot": snapshot}

    def run_llm_comparison(self, data: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = self._safe_file_id(data.get("snapshot_id") or "")
        snapshot = self.get_llm_prompt_snapshot(snapshot_id)["snapshot"]
        models = data.get("models") if isinstance(data.get("models"), list) else []
        comparison_id = self._safe_comparison_id(data.get("comparison_id") or f"comparison_{uuid.uuid4().hex[:12]}")
        results = [
            self._run_single_comparison_model(snapshot, model)
            for model in models
            if isinstance(model, dict)
        ]
        return {
            "comparison_id": comparison_id,
            "snapshot_id": snapshot_id,
            "stage": snapshot.get("stage") or "",
            "results": results,
        }

    def save_llm_comparison(self, data: dict[str, Any]) -> dict[str, Any]:
        comparison_id = self._safe_comparison_id(data.get("comparison_id") or f"comparison_{uuid.uuid4().hex[:12]}")
        payload = self._comparison_payload_with_strategy_details({**data, "comparison_id": comparison_id})
        payload = self._sanitize_comparison_payload(payload)
        payload.setdefault("created_at", self._iso_timestamp(time.time()))
        path = self._llm_comparison_dir() / f"{comparison_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"saved": True, "comparison": payload}

    def list_llm_comparisons(self) -> dict[str, Any]:
        comparisons = []
        for path in sorted(self._llm_comparison_dir().glob("*.json")):
            payload = self._read_json(path, {})
            if not isinstance(payload, dict):
                continue
            comparisons.append({
                "comparison_id": payload.get("comparison_id") or path.stem,
                "snapshot_id": payload.get("snapshot_id") or "",
                "stage": payload.get("stage") or "",
                "created_at": payload.get("created_at") or "",
                "model_count": len(payload.get("models") or []),
                "result_count": len(payload.get("results") or []),
            })
        comparisons.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"comparisons": comparisons}

    def get_llm_comparison(self, comparison_id: str) -> dict[str, Any]:
        safe_id = self._safe_comparison_id(comparison_id)
        path = self._llm_comparison_dir() / f"{safe_id}.json"
        payload = self._read_json(path, {})
        if not isinstance(payload, dict) or not payload.get("comparison_id"):
            raise KeyError(f"Unknown LLM comparison: {comparison_id}")
        return {"comparison": payload}

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        value = api_key.strip()
        if not value:
            return ""
        if len(value) <= 8:
            return "configured"
        return f"{value[:4]}...{value[-4:]}"

    @staticmethod
    def _normalize_face_detector_config_value(value: Any) -> str:
        normalized = str(value or "auto").strip().lower()
        aliases = {
            "": "auto",
            "default": "auto",
            "opencv": "opencv_haar",
            "haar": "opencv_haar",
            "opencv-haar": "opencv_haar",
            "center": "center_crop",
            "none": "center_crop",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in {"auto", "openface", "yolo", "opencv_haar", "center_crop"} else "auto"

    @classmethod
    def _safe_face_crop_settings(
        cls,
        overrides: dict[str, Any] | None = None,
        file_values: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        overrides = overrides if isinstance(overrides, dict) else {}
        values = parse_env_file(PROJECT_ROOT / LOCAL_ENV_FILE) if file_values is None else file_values

        def raw_value(env_key: str, alias: str, default: Any) -> Any:
            return (
                overrides.get(env_key)
                if env_key in overrides
                else overrides.get(alias)
                if alias in overrides
                else os.environ.get(env_key)
                or values.get(env_key)
                or default
            )

        return {
            "crop_mode": cls._safe_crop_mode(raw_value("FACE_CROP_MODE", "crop_mode", cls.FACE_CROP_DEFAULTS["crop_mode"])),
            "crop_scale": cls._clamped_float(raw_value("FACE_CROP_SCALE", "crop_scale", cls.FACE_CROP_DEFAULTS["crop_scale"]), 1.0, 2.6, cls.FACE_CROP_DEFAULTS["crop_scale"]),
            "crop_y_bias": cls._clamped_float(raw_value("FACE_CROP_Y_BIAS", "crop_y_bias", cls.FACE_CROP_DEFAULTS["crop_y_bias"]), -0.3, 0.5, cls.FACE_CROP_DEFAULTS["crop_y_bias"]),
            "crop_top_extra": cls._clamped_float(raw_value("FACE_CROP_TOP_EXTRA", "crop_top_extra", cls.FACE_CROP_DEFAULTS["crop_top_extra"]), 0.0, 0.6, cls.FACE_CROP_DEFAULTS["crop_top_extra"]),
            "crop_bottom_extra": cls._clamped_float(raw_value("FACE_CROP_BOTTOM_EXTRA", "crop_bottom_extra", cls.FACE_CROP_DEFAULTS["crop_bottom_extra"]), 0.0, 0.6, cls.FACE_CROP_DEFAULTS["crop_bottom_extra"]),
            "crop_make_square": cls._truthy_setting(raw_value("FACE_CROP_MAKE_SQUARE", "crop_make_square", cls.FACE_CROP_DEFAULTS["crop_make_square"]), cls.FACE_CROP_DEFAULTS["crop_make_square"]),
        }

    @classmethod
    def _face_crop_parameters_from_payload(cls, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = data if isinstance(data, dict) else {}
        overrides = payload.get("crop_settings") if isinstance(payload.get("crop_settings"), dict) else payload
        settings = cls._safe_face_crop_settings(overrides=overrides)
        return {
            "mode": settings["crop_mode"],
            "scale": settings["crop_scale"],
            "y_bias": settings["crop_y_bias"],
            "top_extra": settings["crop_top_extra"],
            "bottom_extra": settings["crop_bottom_extra"],
            "make_square": settings["crop_make_square"],
        }

    @classmethod
    def _safe_crop_mode(cls, value: Any) -> str:
        mode = str(value or "").strip()
        return mode if mode in cls.OPENFACE_CROP_MODE_DEFAULTS else cls.FACE_CROP_DEFAULTS["crop_mode"]

    @staticmethod
    def _clamped_float(value: Any, minimum: float, maximum: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        number = min(max(number, minimum), maximum)
        return round(number, 4)

    @staticmethod
    def _truthy_setting(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def emotion_model_status(self) -> dict[str, Any]:
        adapter = self._get_emotion_adapter()
        status = adapter.status()
        if not status.get("model_loaded") and not status.get("loading_error"):
            status = adapter.load()
        status["face_detector"] = self._face_detector_status_detail()
        status["emotion_pipeline_status"] = self._emotion_pipeline_status(status)
        return status

    def camera_debug_status(self) -> dict[str, Any]:
        model_status = self.emotion_model_status()
        emotion_pipeline_status = self._emotion_pipeline_status(model_status)
        return {
            "model_status": model_status,
            "emotion_pipeline_status": emotion_pipeline_status,
            "face_detector_status": self._face_detector_status_detail(),
            "face_crop_settings": self._safe_face_crop_settings(),
            "current_mode": emotion_pipeline_status.get("model_output_type") or "academic_state",
            "raw_emotion_available": bool(emotion_pipeline_status.get("raw_detection_available")),
            "allowed_support_cues": list(self.SUPPORT_CUE_STRATEGY_FAMILIES.keys()),
            "allowed_strategy_families_by_cue": {
                cue: list(families)
                for cue, families in self.SUPPORT_CUE_STRATEGY_FAMILIES.items()
                if cue != "neutral"
            },
            "mode_explanation": self._camera_debug_mode_explanation(),
        }

    def camera_debug_analyze_frame(self, data: dict[str, Any]) -> dict[str, Any]:
        image_data = str(data.get("image") or data.get("frame") or data.get("image_data") or data.get("frame_data") or "").strip()
        if not image_data:
            raise ValueError("frame image is required.")
        frame_image, warnings = self._decode_frame_image_for_model(image_data)
        frame_id = f"frame_{uuid.uuid4().hex[:12]}"
        frame_width, frame_height = self._image_size(frame_image)
        mirrored = self._truthy_setting(data.get("mirrored"), False)
        analyzed_frame_preview = self._image_preview_data_url(frame_image)
        crop_parameters = self._face_crop_parameters_from_payload(data)
        face_crop, face_detection = self._detect_face_crop(
            frame_image,
            crop_parameters=crop_parameters,
            mirrored=mirrored,
        )
        frame_warnings = list(warnings)
        if face_detection.get("warning"):
            frame_warnings.append(str(face_detection["warning"]))

        adapter = self._get_emotion_adapter()
        status = adapter.status()
        if not status.get("model_loaded"):
            status = adapter.load() if hasattr(adapter, "load") else status
        crop_preview = self._image_preview_data_url(face_crop)
        model_input_image = self._resize_image(face_crop, (224, 224))
        input_preview = self._image_preview_data_url(model_input_image)
        frame_warnings.extend(self._camera_debug_assertions(frame_width, frame_height, face_detection, model_input_image))
        frame_warnings = [item for item in dict.fromkeys(frame_warnings) if item]
        face_detection["warnings"] = [item for item in dict.fromkeys([*(face_detection.get("warnings") or []), *frame_warnings]) if item]
        prediction: dict[str, Any] = {}
        prediction_warnings: list[str] = []
        if status.get("model_loaded"):
            prediction = adapter.predict(face_crop)
            if prediction.get("model_output_type") != "raw_emotion":
                prediction["raw_emotion_available"] = False
                prediction["raw_emotion"] = None
            prediction_warnings.append(str(prediction.get("error") or ""))
        emotion_pipeline = self._emotion_pipeline_prediction(
            model_input_image,
            fallback_prediction=prediction,
            fallback_status=status,
        )
        response_base = {
            "frame_id": frame_id,
            "analyzed_frame_size": [frame_width, frame_height],
            "analyzed_frame_preview_data_url": analyzed_frame_preview,
            "face_detection": face_detection,
            "crop_preview_data_url": crop_preview,
            "model_input_preview_data_url": input_preview,
            "model_input_size": [224, 224],
            "emotion_pipeline": emotion_pipeline,
        }
        if self._truthy_setting(data.get("include_debug_previews"), False):
            response_base["annotated_frame_preview_data_url"] = self._annotated_frame_preview_data_url(
                frame_image,
                face_detection,
            )
        if not status.get("model_loaded"):
            return {
                **response_base,
                "ok": True,
                "prediction": {},
                "model_status": status,
                "warnings": frame_warnings,
            }

        ok = bool(prediction.get("model_loaded")) and not bool(prediction.get("error"))
        return {
            **response_base,
            "ok": ok,
            "prediction": self._safe_log_payload(prediction),
            "model_status": status,
            "warnings": [item for item in dict.fromkeys([*frame_warnings, *prediction_warnings]) if item],
        }

    def camera_debug_reaction_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        samples = data.get("samples") if isinstance(data.get("samples"), list) else []
        window_start = str(data.get("window_start") or self._first_sample_timestamp(samples) or self._iso_timestamp(time.time()))
        window_end = str(data.get("window_end") or self._last_sample_timestamp(samples) or self._iso_timestamp(time.time()))
        summary = self.summarize_reaction_window(
            samples=samples,
            source_turn_id=str(data.get("source_turn_id") or "camera_debug"),
            highlight_id=str(data.get("highlight_id") or "camera_debug"),
            window_start=window_start,
            window_end=window_end,
        )
        support_cue = str(summary.get("support_cue") or "")
        return {
            "reaction_window_summary": summary,
            "support_cue": support_cue,
            "support_cue_label": summary.get("support_cue_label") or "",
            "allowed_strategy_families": self._allowed_strategy_families_for_support_cue(support_cue),
        }

    def load_sample(self) -> dict[str, Any]:
        sample_path = create_sample_data(Path.cwd())
        document = self.session.load_document(str(sample_path))
        self._register_current_document(Path(sample_path), "txt")
        page = document.page(1)
        return self._document_response(page.text)

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix not in (".txt", ".pdf"):
            raise ValueError("Only TXT and PDF uploads are supported.")
        safe_name = self._safe_upload_name(filename)
        if suffix == ".pdf":
            document_id = uuid.uuid4().hex
            document_dir = self.documents_dir / document_id
            document_dir.mkdir(parents=True, exist_ok=True)
            path = document_dir / "original.pdf"
        else:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            document_id = None
            path = self.upload_dir / f"{int(time.time())}_{uuid.uuid4().hex[:10]}_{safe_name}"
        path.write_bytes(content)
        try:
            document = self.session.load_document(str(path))
        except Exception:
            if suffix != ".pdf":
                raise
            document = self._load_pdf_browser_fallback(path)
        self._register_current_document(path, "pdf" if suffix == ".pdf" else "txt", document_id=document_id)
        if self.current_document_id and self.current_document_id in self.documents:
            self.documents[self.current_document_id]["file_name"] = safe_name
        if suffix == ".pdf" and self.current_document_id:
            self.start_parse_job(self.current_document_id)
        page = document.page(1)
        return self._document_response(page.text)

    def get_page(self, page_number: int) -> dict[str, Any]:
        text = self.session.set_page(page_number)
        return {
            "document_id": self.current_document_id,
            "document_type": self.current_document_type,
            "page_number": page_number,
            "text": text,
        }

    def build_context(self, data: dict[str, Any]) -> dict[str, Any]:
        selected_text = str(data.get("selected_text") or "").strip()
        page_number = int(data.get("page_number") or self.session.current_page_number or 1)
        user_question = str(data.get("user_question") or "")
        document_id = str(data.get("document_id") or self.current_document_id or "").strip() or None
        document_type = self._document_type_for(document_id)
        if self.session.document is None:
            context = self.session.set_manual_context(selected_text)
        else:
            self.session.current_page_number = page_number
            page = self.session.document.page(page_number)
            start, end = self._find_selected_range(page.text, selected_text)
            if start >= 0:
                around = surrounding_text(page.text, start, end)
            else:
                around = page.text[:1600].strip()
            analysis = analyze_passage(selected_text or around)
            chunks = []
            retrieval_debug = {}
            if self.session.retriever:
                debug = self.session.retriever.retrieve_with_debug(
                    query=user_question,
                    selected_text=selected_text,
                    page_number=page_number,
                    top_k=int(self.config.get("paper", {}).get("top_k_chunks", 3)),
                )
                chunks = [chunk.text for chunk in debug["chunks"]]
                retrieval_debug = [
                    (key, value)
                    for key, value in debug.items()
                    if key != "chunks"
                ]
                retrieval_debug = dict(retrieval_debug)
            section_hint = self.session.document.section_for_page(page_number)
            context = PaperContext(
                document_title=self.session.document.title,
                page_number=page_number,
                selected_text=selected_text,
                surrounding_text=around,
                retrieved_chunks=chunks,
                passage_type=analysis.passage_type,
                page_title=section_hint or self.session.document.title,
                section_hint=section_hint,
                difficulty_hint=analysis.difficulty_hint,
                passage_analysis=analysis.as_dict(),
                retrieval_debug=retrieval_debug,
                document_id=document_id,
                document_type=document_type,
            )
        return {
            "document_id": context.document_id,
            "document_type": context.document_type,
            "document_title": context.document_title,
            "page_number": context.page_number,
            "selected_text": context.selected_text,
            "surrounding_text": context.surrounding_text,
            "retrieved_chunks": context.retrieved_chunks,
            "passage_type": context.passage_type,
            "page_title": context.page_title,
            "section_hint": context.section_hint,
            "difficulty_hint": context.difficulty_hint,
            "passage_analysis": context.passage_analysis,
            "retrieval_debug": context.retrieval_debug,
        }

    def add_highlight(self, data: dict[str, Any]) -> dict[str, Any]:
        document_id = str(data.get("document_id") or self.current_document_id or "").strip()
        if not document_id:
            raise ValueError("document_id is required before saving a highlight.")
        selected_text = str(data.get("selected_text") or "").strip()
        requested_type = str(data.get("highlight_type") or "text").strip().lower()
        cropped_image = str(data.get("cropped_image") or data.get("image") or "").strip()
        if not selected_text and requested_type not in {"area", "screenshot_fallback"} and not cropped_image:
            raise ValueError("selected_text or cropped_image is required before saving a highlight.")
        page_number = int(data.get("page_number") or self.session.current_page_number or 1)
        color = str(data.get("color") or "yellow").strip().lower() or "yellow"
        rects = self._sanitize_rects(data.get("rects", []))
        scaled_rects = self._sanitize_scaled_rects(data.get("scaled_rects", []))
        position = self._sanitize_position(data.get("position"))
        if not scaled_rects and position:
            scaled_rects = position.get("rects") or [position["boundingRect"]]
        page_text = self._page_text(page_number)
        text_confidence = self._text_confidence(selected_text, page_text)
        highlight_type = self._resolved_highlight_type(requested_type, selected_text, text_confidence, cropped_image)
        column_side = self._column_side(data.get("column_side"), scaled_rects, rects)
        highlight_id = uuid.uuid4().hex
        cropped_image_path = self._save_cropped_image(document_id, highlight_id, cropped_image) if cropped_image else None
        context = self.build_context(
            {
                "document_id": document_id,
                "selected_text": selected_text,
                "page_number": page_number,
                "user_question": data.get("user_question", ""),
            }
        )
        highlight = {
            "highlight_id": highlight_id,
            "document_id": document_id,
            "document_type": self._document_type_for(document_id),
            "page_number": page_number,
            "highlight_type": highlight_type,
            "selected_text": selected_text,
            "selected_text_preview": selected_text[:240],
            "text_confidence": text_confidence,
            "rects": rects,
            "scaled_rects": scaled_rects,
            "position": position,
            "color": color,
            "column_side": column_side,
            "cropped_image_path": str(cropped_image_path) if cropped_image_path else None,
            "passage_type": context.get("passage_type"),
            "difficulty_hint": context.get("difficulty_hint"),
            "explanation_thread": {
                "thread_id": f"thread-{highlight_id}",
                "highlight_id": highlight_id,
                "rail_side": self._opposite_rail(column_side),
                "status": "pending",
                "messages": [],
            },
            "created_at": time.time(),
        }
        self.highlights_by_document.setdefault(document_id, []).append(highlight)
        self.last_highlight_id = highlight_id
        return {"highlight_id": highlight_id, "highlight": highlight, "context": context}

    def get_highlights(self, document_id: str) -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        if not document_id:
            raise ValueError("document_id is required.")
        return {
            "document_id": document_id,
            "highlights": list(self.highlights_by_document.get(document_id, [])),
        }

    def start_parse_job(self, document_id: str) -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        record = self.documents.get(document_id)
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        if record.get("type") != "pdf":
            raise ValueError("Only PDF documents can be parsed.")
        parsed = parse_pdf_to_blocks(document_id, Path(record["path"]), self.documents_dir)
        record["parse_status"] = parsed
        return {"document_id": document_id, "parsed": parsed}

    def parse_debug_document(self) -> dict[str, Any]:
        document_id = "debug-pdf"
        source = self._debug_pdf_path()
        document_dir = self.documents_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        original = document_dir / "original.pdf"
        if source.resolve() != original.resolve():
            original.write_bytes(source.read_bytes())
        self.documents[document_id] = {
            "document_id": document_id,
            "type": "pdf",
            "path": original.resolve(),
            "title": original.stem,
        }
        self.highlights_by_document.setdefault(document_id, [])
        return self.start_parse_job(document_id)

    def parse_status(self, document_id: str) -> dict[str, Any]:
        document_id = str(document_id or "").strip()
        record = self.documents.get(document_id)
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        return {
            "document_id": document_id,
            "parse_status": record.get("parse_status") or {"status": "not_started"},
        }

    def match_parsed_blocks(self, data: dict[str, Any]) -> dict[str, Any]:
        document_id = str(data.get("document_id") or self.current_document_id or "").strip()
        if not document_id:
            raise ValueError("document_id is required.")
        record = self.documents.get(document_id)
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        parse_status = record.get("parse_status")
        if not parse_status:
            parse_status = self.start_parse_job(document_id)["parsed"]
        blocks_path = Path(parse_status["blocks_index_path"])
        blocks = load_blocks(blocks_path)
        page_number = int(data.get("page_number") or 1)
        rect_payload = self._rect_payload_for_matching(data, page_number)
        selected_text = str(data.get("selected_text") or "").strip()
        matched = match_blocks_for_rects(
            blocks,
            page_number,
            rect_payload["normalized_rects"],
            selected_text=selected_text,
        )
        return {
            "document_id": document_id,
            "page_number": page_number,
            "selected_text_secondary": selected_text,
            "viewport_rects": rect_payload["viewport_rects"],
            "normalized_rects": rect_payload["normalized_rects"],
            "parser_rects_1000": rect_payload["parser_rects_1000"],
            "rect_source": rect_payload["rect_source"],
            "parse_status": parse_status,
            **matched,
        }

    def retrieve_context(self, data: dict[str, Any]) -> dict[str, Any]:
        document_id = str(data.get("document_id") or self.current_document_id or "").strip()
        highlight_payload = data.get("highlight_payload") if isinstance(data.get("highlight_payload"), dict) else data
        if not document_id:
            raise ValueError("document_id is required.")
        record = self.documents.get(document_id)
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        parse_status = record.get("parse_status")
        if not parse_status:
            parse_status = self.start_parse_job(document_id)["parsed"]
        blocks_path = Path(parse_status["blocks_index_path"])
        blocks = load_blocks(blocks_path)
        document_dir = Path(record["path"]).resolve().parent
        return retrieve_paper_context(
            document_id=document_id,
            document_dir=document_dir,
            blocks=blocks,
            highlight_payload=highlight_payload,
        )

    def explain_debug_selection(self, data: dict[str, Any]) -> dict[str, Any]:
        payload = dict(data)
        document_id = str(payload.get("document_id") or "debug-pdf")
        payload["document_id"] = document_id
        try:
            payload["retrieval_context"] = self.retrieve_context(
                {"document_id": document_id, "highlight_payload": payload},
            )
        except Exception as exc:
            payload["retrieval_context"] = {
                "paper_profile": {},
                "matched_block": payload.get("matched_block") or {},
                "nearby_context": payload.get("nearby_useful_context") or [],
                "same_section_context": [],
                "related_blocks": [],
                "retrieval_strategy": "payload_fallback",
                "error": str(exc),
            }
        return explain_selection(payload)


    def document_file_path(self, document_id: str) -> Path:
        record = self._record_for_document(str(document_id))
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        if record.get("type") != "pdf":
            raise ValueError("Only uploaded PDF documents can be served through this endpoint.")
        path = Path(record["path"]).resolve()
        root = self.upload_dir.resolve()
        if path != root and root not in path.parents:
            raise ValueError("Refusing to serve a file outside runtime_uploads.")
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Document file is missing: {document_id}")
        return path

    def list_library_documents(self, include_hidden: bool = False) -> dict[str, Any]:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        documents = []
        for document_dir in sorted(self.documents_dir.iterdir()):
            if not document_dir.is_dir() or not (document_dir / "original.pdf").exists():
                continue
            meta = self._infer_document_meta(document_dir.name)
            if include_hidden or self._is_library_visible(meta):
                documents.append(meta)
        documents.sort(key=lambda item: float(item.get("last_opened_at") or item.get("updated_at") or 0), reverse=True)
        return {"documents": documents}

    def upload_library_document(self, filename: str, content: bytes) -> dict[str, Any]:
        if Path(filename or "").suffix.lower() != ".pdf":
            raise ValueError("Only PDF uploads are supported.")
        if not bytes(content).lstrip().startswith(b"%PDF"):
            raise ValueError("Uploaded file does not look like a PDF.")
        safe_name = self._safe_upload_name(filename)
        document_id = uuid.uuid4().hex
        document_dir = self._document_dir(document_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_document_runtime_dirs(document_id)
        original = document_dir / "original.pdf"
        original.write_bytes(content)
        now = time.time()
        parse_status = self._prepare_progress_payload(
            document_id=document_id,
            stage="extracting_text",
            status="preparing",
            progress_percent=18,
            started_at=now,
            updated_at=now,
            warnings=[],
        )
        self._write_prepare_status(document_id, parse_status)
        self.documents[document_id] = {
            "document_id": document_id,
            "type": "pdf",
            "path": original.resolve(),
            "file_name": safe_name,
            "title": safe_name,
            "page_count": 0,
            "parse_status": parse_status,
        }
        self.highlights_by_document.setdefault(document_id, [])
        meta = {
            "document_id": document_id,
            "file_name": safe_name,
            "title": safe_name,
            "page_count": 0,
            "created_at": now,
            "updated_at": now,
            "last_opened_at": None,
            "last_page": 1,
            "prepare_status": "preparing",
            "parsed_blocks_count": 0,
            "embedding_status": "pending",
            "retrieval_method": "unknown",
            "highlight_count": 0,
            "thread_count": 0,
            "uploaded_from": "pdf_chat",
            "library_visible": True,
        }
        self._write_meta(document_id, meta)
        self._log_interaction(document_id, "upload_pdf", success=True, retrieval_method="unknown")
        thread = threading.Thread(target=self._prepare_library_document_background, args=(document_id,), daemon=True)
        thread.start()
        return {
            "document_id": document_id,
            "meta": meta,
            "prepare_status": parse_status,
            "retrieval_method": "unknown",
            "warnings": [],
        }

    def library_document_detail(self, document_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        record = self._record_for_document(document_id)
        if not record:
            raise KeyError(f"Unknown document_id: {document_id}")
        meta = self._infer_document_meta(document_id)
        prepare_status = self._prepare_status_from_files(document_id) or record.get("parse_status") or {}
        document_dir = self._document_dir(document_id)
        files = {
            "original_pdf": (document_dir / "original.pdf").exists(),
            "meta": (document_dir / "meta.json").exists(),
            "document_md": (document_dir / "parsed" / "document.md").exists(),
            "blocks_index": (document_dir / "parsed" / "blocks_index.json").exists(),
            "paper_profile": (document_dir / "rag" / "paper_profile.json").exists(),
            "section_map": (document_dir / "rag" / "section_map.json").exists(),
            "keyword_index": (document_dir / "rag" / "keyword_index.json").exists(),
            "embeddings": (document_dir / "rag" / "embeddings.json").exists(),
            "highlights": (document_dir / "highlights" / "highlights.json").exists(),
            "threads": (document_dir / "threads").exists(),
        }
        warnings = [name for name, available in files.items() if name in {"document_md", "blocks_index", "paper_profile", "keyword_index"} and not available]
        return {"document_id": document_id, "meta": meta, "prepare_status": prepare_status, "files": files, "warnings": warnings}

    def _prepare_library_document_background(self, document_id: str) -> None:
        document_id = self._safe_document_id(document_id)
        initial = self._prepare_status_from_files(document_id)
        started_at = float(initial.get("started_at") or time.time())
        try:
            self._write_prepare_status(
                document_id,
                self._prepare_progress_payload(
                    document_id=document_id,
                    stage="extracting_text",
                    status="preparing",
                    progress_percent=24,
                    started_at=started_at,
                    updated_at=time.time(),
                    base=initial,
                ),
            )
            parsed = self.start_parse_job(document_id)["parsed"]
            completed = self._prepare_progress_payload(
                document_id=document_id,
                stage="ready",
                status="completed",
                progress_percent=100,
                started_at=started_at,
                updated_at=time.time(),
                base=parsed,
                warnings=self._prepare_warnings(parsed),
            )
            self._write_prepare_status(document_id, completed)
            meta = self._infer_document_meta(document_id)
            archived = bool(meta.get("archived_at")) or meta.get("library_visible") is False
            meta.update(
                {
                    "prepare_status": "completed",
                    "page_count": self._positive_int(parsed.get("page_count")) or meta.get("page_count") or 0,
                    "parsed_blocks_count": self._positive_int(parsed.get("block_count")) or meta.get("parsed_blocks_count") or 0,
                    "embedding_status": parsed.get("embedding_index_status") or meta.get("embedding_status") or "unavailable",
                    "retrieval_method": self._retrieval_method_for_status(parsed, document_id),
                    "updated_at": time.time(),
                    "uploaded_from": "pdf_chat",
                    "library_visible": not archived,
                }
            )
            self._write_meta(document_id, meta)
            if document_id in self.documents:
                self.documents[document_id]["parse_status"] = completed
            self._log_interaction(document_id, "prepare_pdf", success=True, retrieval_method=meta.get("retrieval_method"))
        except Exception as exc:
            failed = self._prepare_progress_payload(
                document_id=document_id,
                stage="extracting_text",
                status="failed",
                progress_percent=max(1, self._positive_int(initial.get("progress_percent")) or 18),
                started_at=started_at,
                updated_at=time.time(),
                base=initial,
                warnings=[str(exc)],
                error=str(exc),
            )
            self._write_prepare_status(document_id, failed)
            self._update_meta(
                document_id,
                {
                    "prepare_status": "failed",
                    "embedding_status": "unavailable",
                    "retrieval_method": "unknown",
                    "updated_at": time.time(),
                    "uploaded_from": "pdf_chat",
                },
            )
            self._log_interaction(document_id, "prepare_pdf", success=False, error=str(exc))

    def open_library_document(self, document_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        updates = {"last_opened_at": time.time(), "updated_at": time.time()}
        last_page = self._positive_int(data.get("last_page"))
        if last_page:
            updates["last_page"] = last_page
        meta = self._update_meta(document_id, updates)
        self._log_interaction(document_id, "open_document", success=True)
        return {"document_id": document_id, "meta": meta}

    def archive_library_document(self, document_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        self._ensure_document_runtime_dirs(document_id)
        now = time.time()
        meta = self._update_meta(
            document_id,
            {
                "library_visible": False,
                "archived_at": now,
                "updated_at": now,
            },
        )
        self._log_interaction(document_id, "archive_document", success=True)
        return {"document_id": document_id, "archived": True, "meta": meta}

    def start_reading_session(self, document_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        session_id = uuid.uuid4().hex
        now = time.time()
        session_dir = self._session_dir(session_id)
        learning_dir = self._learning_state_dir(session_id)
        learning_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "session_id": session_id,
            "document_id": document_id,
            "source": "simulated_camera",
            "model_output_type": "academic_state_model",
            "mode": "simulated_mode_b",
            "started_at": now,
            "scenario": "default_engagement_confusion_recovery",
            "thresholds": {
                "confusion": {"confidence": 0.65, "duration_sec": 8},
                "frustration": {"confidence": 0.60, "duration_sec": 6},
                "boredom": {"confidence": 0.60, "duration_sec": 12},
                "engagement": {"confidence": 0.75, "duration_sec": 0},
            },
        }
        (learning_dir / "simulator_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        state = self._simulated_learning_state(session_id, config, now)
        self._write_learning_state(session_id, state)
        self._append_session_event(
            session_id,
            {
                "event_type": "document_opened",
                "document_id": document_id,
                "session_id": session_id,
                "success": True,
            },
        )
        self._log_interaction(document_id, "reading_session_started", session_id=session_id, success=True)
        return {"session_id": session_id, "document_id": document_id, "learning_state": state}

    def current_learning_state(self, session_id: str) -> dict[str, Any]:
        session_id = self._safe_file_id(session_id)
        config = self._read_json(self._learning_state_dir(session_id) / "simulator_config.json", {})
        if not isinstance(config, dict) or not config.get("document_id"):
            raise KeyError(f"Unknown reading session: {session_id}")
        current = self._read_json(self._learning_state_dir(session_id) / "current_state.json", {})
        if isinstance(current, dict) and current.get("source") == "webcam_model":
            age = time.time() - self._epoch_from_iso_timestamp(str(current.get("timestamp") or ""))
            if 0 <= age <= 5:
                return {"session_id": session_id, "learning_state": current}
        state = self._simulated_learning_state(session_id, config, time.time())
        self._write_learning_state(session_id, state)
        return {"session_id": session_id, "learning_state": state}

    def record_reading_session_event(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        session_id = self._safe_file_id(session_id)
        config = self._read_json(self._learning_state_dir(session_id) / "simulator_config.json", {})
        if not isinstance(config, dict) or not config.get("document_id"):
            raise KeyError(f"Unknown reading session: {session_id}")
        event_type = str(data.get("event_type") or data.get("type") or "").strip()
        if not event_type:
            raise ValueError("event_type is required.")
        event = self._safe_log_payload(
            {
                **data,
                "event_type": event_type,
                "session_id": session_id,
                "document_id": data.get("document_id") or config.get("document_id"),
                "timestamp": time.time(),
            }
        )
        self._append_session_event(session_id, event)
        return {"session_id": session_id, "event": event}

    def strategy_candidates(self, document_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        request = self._strategy_request_payload(document_id, data)
        trigger_context = request.get("trigger_context") if isinstance(request.get("trigger_context"), dict) else {}
        if trigger_context.get("triggered_by") == "reaction_window" and (
            not request.get("source_turn_id")
            or not request.get("baseline_explanation")
            or not isinstance(request.get("reaction_window_summary"), dict)
            or not request.get("reaction_window_summary")
        ):
            raise ValueError("baseline_explanation, source_turn_id, and reaction_window_summary are required for reaction-window strategy candidates.")
        prompt_snapshot = self._save_strategy_planner_prompt_snapshot(document_id, request)
        llm_payload = self._call_strategy_planner_llm(request)
        if self._valid_strategy_payload(llm_payload):
            payload = dict(llm_payload)
            payload["planner_mode"] = "llm"
            payload["warnings"] = list(payload.get("warnings") or [])
        else:
            payload = self._heuristic_strategy_candidates(request)
        payload = self._normalize_strategy_payload(payload, request)
        payload["planner_prompt_version"] = "reaction_strategy_planner_v2" if request.get("reaction_window_summary") else "strategy_planner_v1"
        payload["support_cue"] = request.get("support_cue") or (request.get("reaction_window_summary") or {}).get("support_cue") or ""
        payload["support_cue_label"] = (request.get("reaction_window_summary") or {}).get("support_cue_label") or ""
        payload["reaction_window_summary"] = request.get("reaction_window_summary") or {}
        payload["planner_input_summary"] = request.get("planner_input_summary") if isinstance(request.get("planner_input_summary"), dict) else {}
        payload["document_id"] = document_id
        payload["highlight_id"] = request.get("highlight_id")
        payload["session_id"] = request.get("session_id")
        payload["prompt_snapshot_id"] = prompt_snapshot["snapshot_id"]
        event_type = "strategy_candidates_generated" if trigger_context.get("triggered_by") == "reaction_window" else "strategy_candidates_shown"
        self._log_strategy_event(document_id, event_type, request, payload, success=True)
        session_id = str(request.get("session_id") or "")
        if session_id:
            self._append_session_event(
                session_id,
                self._strategy_event_payload(event_type, document_id, request, payload, success=True),
            )
        return payload

    def get_library_highlights(self, document_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        self._record_for_document(document_id)
        loaded_highlights = self._load_highlights(document_id)
        highlights = [self._normalize_library_highlight(document_id, highlight) for highlight in loaded_highlights]
        if highlights != loaded_highlights:
            path = self._highlights_path(document_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"document_id": document_id, "highlights": highlights}, indent=2), encoding="utf-8")
            self.highlights_by_document[document_id] = highlights
        return {"document_id": document_id, "highlights": highlights}

    def save_library_highlights(self, document_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        raw_highlights = data.get("highlights", data if isinstance(data, list) else [])
        if not isinstance(raw_highlights, list):
            raise ValueError("highlights must be a list.")
        now = time.time()
        highlights = []
        for item in raw_highlights:
            if not isinstance(item, dict):
                continue
            highlight = dict(item)
            highlight_id = self._safe_file_id(highlight.get("highlight_id") or highlight.get("id") or uuid.uuid4().hex)
            highlight["id"] = highlight_id
            highlight["highlight_id"] = highlight_id
            highlight["document_id"] = document_id
            highlight["type"] = str(highlight.get("type") or highlight.get("highlight_type") or "text")
            highlight["highlight_type"] = highlight["type"]
            if "selected_text" in highlight:
                highlight["selected_text"] = normalize_pdf_text(highlight.get("selected_text"))
                highlight["text_preview"] = highlight["selected_text"][:240]
            highlight["updated_at"] = highlight.get("updated_at") or now
            highlight["created_at"] = highlight.get("created_at") or now
            highlight = self._with_crop_metadata(document_id, highlight)
            highlights.append(highlight)
        path = self._highlights_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"document_id": document_id, "highlights": highlights}, indent=2), encoding="utf-8")
        self.highlights_by_document[document_id] = highlights
        meta = self._update_meta(document_id, {"highlight_count": len(highlights), "updated_at": now})
        return {"document_id": document_id, "highlights": highlights, "meta": meta}

    def save_library_crop(self, document_id: str, highlight_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        image_data = str(data.get("crop_image_data_url") or data.get("image") or data.get("cropped_image") or "")
        crop_path = self._save_document_crop(document_id, highlight_id, image_data)
        if not crop_path:
            raise ValueError("A valid PNG or JPEG data URL is required.")
        relative = str(crop_path.relative_to(self._document_dir(document_id)))
        now = time.time()
        highlights = self._load_highlights(document_id)
        changed = False
        for highlight in highlights:
            if self._safe_file_id(highlight.get("highlight_id") or highlight.get("id") or "") != highlight_id:
                continue
            highlight["crop_path"] = relative
            highlight["crop_image_path"] = relative
            highlight["crop_url"] = self._crop_url(document_id, highlight_id)
            highlight["updated_at"] = now
            changed = True
        if changed:
            path = self._highlights_path(document_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"document_id": document_id, "highlights": highlights}, indent=2), encoding="utf-8")
            self.highlights_by_document[document_id] = highlights
            self._update_meta(document_id, {"updated_at": now})
        self._log_interaction(document_id, "save_crop", highlight_id=highlight_id, success=True)
        return {
            "document_id": document_id,
            "highlight_id": highlight_id,
            "crop_path": relative,
            "crop_image_path": relative,
            "crop_url": self._crop_url(document_id, highlight_id),
        }

    def get_library_thread(self, document_id: str, highlight_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        if not self._thread_path(document_id, highlight_id).exists() and not self._highlight_exists(document_id, highlight_id):
            raise KeyError(f"Unknown highlight_id: {highlight_id}")
        return self._load_thread(document_id, highlight_id)

    def save_library_thread(self, document_id: str, highlight_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        existing = self._load_thread(document_id, highlight_id)
        thread = self._thread_from_data(document_id, highlight_id, data)
        if self._thread_has_messages(existing) and not self._thread_has_messages(thread):
            meta = self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": time.time()})
            return {**existing, "meta": meta, "empty_overwrite_ignored": True}
        self._write_thread(document_id, highlight_id, thread)
        meta = self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": time.time()})
        return {**thread, "meta": meta}

    def delete_library_highlight(self, document_id: str, highlight_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        highlights = self._load_highlights(document_id)
        remaining = [
            highlight for highlight in highlights
            if self._safe_file_id(highlight.get("highlight_id") or highlight.get("id") or "") != highlight_id
        ]
        if len(remaining) == len(highlights):
            raise ValueError(f"Unknown highlight_id: {highlight_id}")
        path = self._highlights_path(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"document_id": document_id, "highlights": remaining}, indent=2), encoding="utf-8")
        self.highlights_by_document[document_id] = remaining
        thread_path = self._thread_path(document_id, highlight_id)
        thread_deleted = False
        if thread_path.exists():
            thread_path.unlink()
            thread_deleted = True
        now = time.time()
        meta = self._update_meta(
            document_id,
            {
                "highlight_count": len(remaining),
                "thread_count": self._thread_count(document_id),
                "updated_at": now,
            },
        )
        self._log_interaction(document_id, "highlight_deleted", highlight_id=highlight_id, success=True)
        return {
            "document_id": document_id,
            "highlight_id": highlight_id,
            "highlights": remaining,
            "thread_deleted": thread_deleted,
            "meta": meta,
        }

    def clear_library_thread(self, document_id: str, highlight_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        thread = self._load_thread(document_id, highlight_id)
        now = time.time()
        cleared = {
            **thread,
            "updated_at": now,
            "messages": [],
            "learning_state_snapshot": {},
            "strategy_candidates": [],
            "selected_strategy_id": "",
            "selected_strategy": {},
            "trigger_context": {},
            "reaction_window_summary": {},
            "turn_metadata": {},
            "support_cue": "",
            "support_cue_label": "",
            "strategy_reason": "",
            "planner_mode": "",
            "planner_prompt_version": "",
            "planner_input_summary": {},
        }
        self._write_thread(document_id, highlight_id, cleared)
        meta = self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": now})
        self._log_interaction(document_id, "thread_cleared", highlight_id=highlight_id, success=True)
        return {"document_id": document_id, "highlight_id": highlight_id, "thread": cleared, "meta": meta}

    def delete_library_thread_turn(self, document_id: str, highlight_id: str, turn_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        turn_id = self._safe_file_id(turn_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        if not self._thread_path(document_id, highlight_id).exists():
            raise ValueError(f"Unknown thread for highlight_id: {highlight_id}")
        thread = self._load_thread(document_id, highlight_id)
        messages = thread.get("messages") if isinstance(thread.get("messages"), list) else []
        remaining = [
            message for message in messages
            if self._safe_file_id(message.get("turn_id") or message.get("conversation_turn_id") or "") != turn_id
        ]
        if len(remaining) == len(messages):
            raise ValueError(f"Unknown turn_id: {turn_id}")
        thread["messages"] = remaining
        if isinstance(thread.get("turn_metadata"), dict):
            thread["turn_metadata"].pop(turn_id, None)
        if not remaining:
            thread["strategy_candidates"] = []
            thread["trigger_context"] = {}
            thread["reaction_window_summary"] = {}
            thread["turn_metadata"] = {}
        thread["updated_at"] = time.time()
        self._write_thread(document_id, highlight_id, thread)
        meta = self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": time.time()})
        self._log_interaction(document_id, "thread_turn_deleted", highlight_id=highlight_id, success=True, extra={"turn_id": turn_id})
        return {"document_id": document_id, "highlight_id": highlight_id, "turn_id": turn_id, "thread": thread, "meta": meta}

    def explain_library_selection(self, document_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        payload = dict(data)
        payload["document_id"] = document_id
        payload.setdefault("response_style", "chat_conversational")
        highlight_id = self._safe_file_id(payload.get("highlight_id") or uuid.uuid4().hex)
        payload["highlight_id"] = highlight_id
        if "selected_text" in payload:
            payload["selected_text"] = normalize_pdf_text(payload.get("selected_text"))
        if payload.get("crop_image_data_url"):
            crop_path = self._save_document_crop(document_id, highlight_id, str(payload.get("crop_image_data_url")))
            if crop_path:
                payload["crop_image_path"] = str(crop_path.relative_to(self._document_dir(document_id)))
        payload = self._attach_crop_data_url(document_id, payload)
        payload = self._with_strategy_payload(payload)
        turn_id = self._new_turn_id(payload.get("turn_id"))
        payload["turn_id"] = turn_id
        prompt_snapshot, prompt_snapshot_error = self._try_save_prompt_snapshot_for_payload(
            document_id=document_id,
            highlight_id=highlight_id,
            turn_id=turn_id,
            thread_id=f"thread-{highlight_id}",
            payload=payload,
        )
        result = self.explain_debug_selection(payload)
        context_used = self._context_used_summary(result)
        answer = str(result.get("answer") or "")
        error = str(result.get("error") or "").strip()
        thread = self._load_thread(document_id, highlight_id)
        if error or not answer.strip():
            failure_error = error or "Explanation provider returned an empty answer."
            self._log_interaction(
                document_id,
                "explain_selection",
                highlight_id=highlight_id,
                selection_type=payload.get("highlight_type") or payload.get("type"),
                session_id=payload.get("session_id"),
                provider=result.get("provider"),
                model=result.get("model"),
                retrieval_method=result.get("retrieval_method"),
                success=False,
                error=failure_error,
                extra=self._strategy_log_extra(payload, {**result, "error": failure_error}),
            )
            if payload.get("session_id"):
                self._append_session_event(
                    str(payload.get("session_id")),
                    self._strategy_event_payload(
                        "answer_generated",
                        document_id,
                        payload,
                        {"planner_mode": payload.get("planner_mode")},
                        success=False,
                        result={**result, "error": failure_error},
                    ),
                )
            return {
                **result,
                "ok": False,
                "answer": "",
                "error": failure_error,
                "assistant_message": None,
                "context_used": context_used,
                "thread": thread,
                "warnings": result.get("warnings") if isinstance(result.get("warnings"), list) else [],
            }
        if not thread.get("selection_snapshot"):
            thread["selection_snapshot"] = self._safe_selection_snapshot(payload)
        self._apply_strategy_thread_metadata(thread, payload)
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "created_at": time.time(),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "context_used": context_used,
            "turn_id": turn_id,
            "prompt_snapshot_id": prompt_snapshot.get("snapshot_id", ""),
            **self._assistant_strategy_metadata(payload),
        }
        if prompt_snapshot_error:
            assistant_message["prompt_snapshot_error"] = prompt_snapshot_error
        thread["messages"].append(assistant_message)
        thread["updated_at"] = time.time()
        self._write_thread(document_id, highlight_id, thread)
        self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": time.time()})
        self._log_interaction(
            document_id,
            "explain_selection",
            highlight_id=highlight_id,
            selection_type=payload.get("highlight_type") or payload.get("type"),
            session_id=payload.get("session_id"),
            provider=result.get("provider"),
            model=result.get("model"),
            retrieval_method=result.get("retrieval_method"),
            success=not bool(result.get("error")),
            error=result.get("error"),
            extra=self._strategy_log_extra(payload, result),
        )
        if payload.get("session_id"):
            self._append_session_event(
                str(payload.get("session_id")),
                self._strategy_event_payload("answer_generated", document_id, payload, {"planner_mode": payload.get("planner_mode")}, success=not bool(result.get("error")), result=result),
            )
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        if prompt_snapshot_error:
            warnings = [f"prompt snapshot could not be saved: {prompt_snapshot_error}", *warnings]
        return {
            **result,
            "ok": True,
            "answer": answer,
            "assistant_message": assistant_message,
            "context_used": context_used,
            "thread": thread,
            "prompt_snapshot_id": prompt_snapshot.get("snapshot_id", ""),
            "warnings": warnings,
        }

    def follow_up_library_thread(self, document_id: str, highlight_id: str, data: dict[str, Any]) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        question = str(data.get("question") or data.get("user_question") or "").strip()
        if not question:
            raise ValueError("question is required.")
        thread = self._load_thread(document_id, highlight_id)
        payload = dict(thread.get("selection_snapshot") or {})
        payload.update(data.get("selection_snapshot") if isinstance(data.get("selection_snapshot"), dict) else {})
        payload["document_id"] = document_id
        payload["highlight_id"] = highlight_id
        payload.setdefault("response_style", "chat_conversational")
        payload["follow_up_question"] = question
        payload["thread_history"] = thread.get("messages", [])[-8:]
        turn_id = self._new_turn_id(data.get("turn_id"))
        payload["turn_id"] = turn_id
        for key in ("session_id", "learning_state", "strategy_candidates", "selected_strategy_id", "selected_strategy", "trigger_context"):
            if key == "trigger_context" and thread.get(key) not in (None, "", [], {}):
                payload[key] = thread.get(key)
            elif key in data:
                payload[key] = data.get(key)
            elif thread.get(key) not in (None, "", [], {}):
                payload[key] = thread.get(key)
        payload = self._attach_crop_data_url(document_id, payload)
        payload = self._with_strategy_payload(payload)
        user_message = {"role": "user", "content": question, "created_at": time.time(), "turn_id": turn_id}
        thread["messages"].append(user_message)
        prompt_snapshot, prompt_snapshot_error = self._try_save_prompt_snapshot_for_payload(
            document_id=document_id,
            highlight_id=highlight_id,
            turn_id=turn_id,
            thread_id=str(thread.get("thread_id") or f"thread-{highlight_id}"),
            payload=payload,
        )
        result = self.explain_debug_selection(payload)
        context_used = self._context_used_summary(result)
        assistant_message = {
            "role": "assistant",
            "content": str(result.get("answer") or ""),
            "created_at": time.time(),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "context_used": context_used,
            "turn_id": turn_id,
            "prompt_snapshot_id": prompt_snapshot.get("snapshot_id", ""),
            **self._assistant_strategy_metadata(payload),
        }
        if prompt_snapshot_error:
            assistant_message["prompt_snapshot_error"] = prompt_snapshot_error
        thread["messages"].append(assistant_message)
        thread["selection_snapshot"] = self._safe_selection_snapshot(payload)
        self._apply_strategy_thread_metadata(thread, payload)
        thread["updated_at"] = time.time()
        self._write_thread(document_id, highlight_id, thread)
        self._update_meta(document_id, {"thread_count": self._thread_count(document_id), "updated_at": time.time()})
        self._log_interaction(
            document_id,
            "follow_up",
            highlight_id=highlight_id,
            selection_type=payload.get("highlight_type") or payload.get("type"),
            session_id=payload.get("session_id"),
            provider=result.get("provider"),
            model=result.get("model"),
            retrieval_method=result.get("retrieval_method"),
            success=not bool(result.get("error")),
            error=result.get("error"),
            extra=self._strategy_log_extra(payload, result),
        )
        if payload.get("session_id"):
            self._append_session_event(
                str(payload.get("session_id")),
                self._strategy_event_payload("follow_up_sent", document_id, payload, {"planner_mode": payload.get("planner_mode")}, success=not bool(result.get("error")), result=result),
            )
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        if prompt_snapshot_error:
            warnings = [f"prompt snapshot could not be saved: {prompt_snapshot_error}", *warnings]
        return {
            **result,
            "context_used": context_used,
            "thread": thread,
            "prompt_snapshot_id": prompt_snapshot.get("snapshot_id", ""),
            "warnings": warnings,
        }

    def manual_emotion(self, data: dict[str, Any]) -> dict[str, Any]:
        value = str(data.get("emotion") or data.get("state") or "").strip().lower()
        if not value:
            raise ValueError("Manual emotion or state is required.")
        snapshot = self.session.set_override(value)
        return self._state_response(snapshot, source="auto_reset" if value == "auto" else "manual")

    def frame_emotion(self, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("session_id"):
            return self.session_frame_emotion(str(data.get("session_id")), data)
        frame_data = str(data.get("image") or data.get("frame") or "")
        decoded_len = self._decode_frame_length(frame_data)
        emotion = self._heuristic_frame_emotion(decoded_len)
        raw = self._prediction_for(emotion, confidence=0.62, source="webcam_frame_dummy")
        smoothed = self.session.buffer.add(raw)
        candidate = map_prediction_to_learning_state(raw, smoothed, manual_override=False)
        snapshot = self.session.tracker.update(candidate)
        self.session.learning_state = snapshot
        payload = self._state_response(snapshot, source="webcam_frame_dummy")
        payload["frame_bytes"] = decoded_len
        payload["face_bbox"] = None
        self.last_frame_status = f"Processed {decoded_len} bytes using local dummy frame recognizer."
        return payload

    def session_frame_emotion(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        session_id = self._safe_file_id(session_id)
        config = self._read_json(self._learning_state_dir(session_id) / "simulator_config.json", {})
        if not isinstance(config, dict) or not config.get("document_id"):
            raise KeyError(f"Unknown reading session: {session_id}")
        document_id = str(data.get("document_id") or config.get("document_id") or "")
        image_data = str(data.get("image") or data.get("frame") or data.get("image_data") or data.get("frame_data") or "")
        decoded_len = self._decode_frame_length(image_data)
        frame_image, warnings = self._decode_frame_image_for_model(image_data)
        crop_parameters = self._face_crop_parameters_from_payload(data)
        mirrored = self._truthy_setting(data.get("mirrored"), False)
        face_crop, face_detection = self._detect_face_crop(
            frame_image,
            crop_parameters=crop_parameters,
            mirrored=mirrored,
        )
        compact_face_detection = self._compact_face_detection(face_detection)
        if face_detection.get("warning"):
            warnings.append(str(face_detection["warning"]))
        adapter = self._get_emotion_adapter()
        status = adapter.status()
        if not status.get("model_loaded"):
            status = adapter.load() if hasattr(adapter, "load") else status
        prediction: dict[str, Any] = {}
        if status.get("model_loaded"):
            prediction = adapter.predict(face_crop)
        model_input_image = self._resize_image(face_crop, (224, 224))
        emotion_pipeline = self._emotion_pipeline_prediction(
            model_input_image,
            fallback_prediction=prediction,
            fallback_status=status,
        )
        prediction_for_state = self._prediction_from_emotion_pipeline(emotion_pipeline, fallback_prediction=prediction)
        model_status = {
            **status,
            "emotion_pipeline_status": self._emotion_pipeline_status(status),
            "model_output_type": emotion_pipeline.get("model_output_type") or status.get("model_output_type"),
            "raw_emotion_available": bool(emotion_pipeline.get("raw_detection_available")),
        }
        if not prediction_for_state:
            failure_reason = (
                str(emotion_pipeline.get("error") or "")
                or str(prediction.get("error") or "")
                or str(status.get("loading_error") or "Emotion model is not loaded.")
            )
            state = self._current_or_simulated_learning_state(session_id, config)
            state = {
                **state,
                "warnings": [
                    *(state.get("warnings") or []),
                    *warnings,
                    "Live model unavailable. Using simulated learning signal.",
                    failure_reason,
                ],
                "face_detection": compact_face_detection,
            }
            self._write_learning_state(session_id, state)
            self._append_session_event(
                session_id,
                {
                    "event_type": "webcam_model_frame",
                    "document_id": document_id,
                    "source": "webcam_model",
                    "success": False,
                    "model_loaded": bool(status.get("model_loaded")),
                    "model_output_type": emotion_pipeline.get("model_output_type") or status.get("model_output_type"),
                    "raw_detection_available": bool(emotion_pipeline.get("raw_detection_available")),
                    "loading_error": status.get("loading_error"),
                    "frame_bytes": decoded_len,
                    "face_detection": compact_face_detection,
                    "detector": compact_face_detection.get("actual_detector"),
                    "crop_strategy": compact_face_detection.get("crop_strategy"),
                    "warnings": state.get("warnings") or [],
                },
            )
            return {
                "ok": False,
                "session_id": session_id,
                "learning_state": state,
                "model_status": model_status,
                "emotion_pipeline": emotion_pipeline,
                "face_detection": compact_face_detection,
                "warnings": state.get("warnings") or [],
            }

        state = self._learning_state_from_model_prediction(
            session_id=session_id,
            document_id=document_id,
            prediction=prediction_for_state,
            warnings=warnings,
            face_detection=compact_face_detection,
            now=time.time(),
        )
        self._write_learning_state(session_id, state)
        raw_detection = emotion_pipeline.get("raw_detection") if isinstance(emotion_pipeline.get("raw_detection"), dict) else {}
        self._append_session_event(
            session_id,
            {
                "event_type": "webcam_model_frame",
                "document_id": document_id,
                "source": "webcam_model",
                "model_output_type": state.get("model_output_type"),
                "raw_detection_available": state.get("raw_facial_emotion_available"),
                "raw_label": raw_detection.get("label"),
                "academic_state": state.get("academic_state"),
                "confidence": state.get("confidence"),
                "distribution": state.get("distribution"),
                "trend": state.get("trend"),
                "duration_sec": state.get("duration_sec"),
                "warnings": warnings,
                "face_detection": compact_face_detection,
                "detector": compact_face_detection.get("actual_detector") or compact_face_detection.get("detector"),
                "crop_strategy": compact_face_detection.get("crop_strategy"),
                "face_crop_fallback": bool(compact_face_detection.get("fallback_used")),
                "frame_bytes": decoded_len,
                "success": True,
                "model_loaded": True,
            },
        )
        self.last_frame_status = f"Processed {decoded_len} bytes using local learning-state model."
        return {
            "ok": True,
            "session_id": session_id,
            "learning_state": state,
            "model_status": model_status,
            "emotion_pipeline": emotion_pipeline,
            "face_detection": compact_face_detection,
            "warnings": warnings,
        }

    def emotion_state(self) -> dict[str, Any]:
        return self._state_response(self.session.learning_state, source="current")

    def chat(self, data: dict[str, Any]) -> dict[str, Any]:
        question = str(data.get("user_question") or data.get("question") or "").strip()
        if not question:
            raise ValueError("user_question is required.")
        context = self._context_from_chat_data(data, question)
        followup_action = data.get("followup_action")
        model_alias = str(data.get("model_alias") or data.get("model") or "dummy")
        response = self.session.ask(
            question,
            paper_context=context,
            followup_action=str(followup_action) if followup_action else None,
            model_alias=model_alias,
        )
        from emotion_aware_assistant.core.types import ChatRequest
        from emotion_aware_assistant.llm.prompt_builder import PromptBuilder

        prompt_request = ChatRequest(
            user_question=question,
            paper_context=context,
            learning_state=self.session.learning_state,
            conversation_history=self.session.conversation_history[-8:],
            followup_action=str(followup_action) if followup_action else None,
            model_name=model_alias,
        )
        self.last_prompt_preview = PromptBuilder().build_text_prompt(prompt_request)[:2400]
        self.last_request_summary = {
            "document_id": context.document_id,
            "document_type": context.document_type,
            "highlight_id": context.highlight_id,
            "page_number": context.page_number,
            "selected_text_length": len(context.selected_text),
            "retrieved_chunks": len(context.retrieved_chunks),
            "passage_type": context.passage_type,
            "difficulty_hint": context.difficulty_hint,
            "model_alias": model_alias,
            "followup_action": followup_action,
            "learning_state": self.session.learning_state.state,
            "trend": self.session.learning_state.trend,
        }
        policy = get_response_policy(
            self.session.learning_state.state,
            self.session.learning_state.trend,
            context.passage_type,
        )
        return {
            "answer": response.text,
            "followup_buttons": policy.followup_buttons,
            "followups": policy.followup_buttons,
            "model_name": response.model_name,
            "model": response.model_name,
            "client_type": self.session.llm.name,
            "latency": response.latency_sec,
            "strategy": self.session.learning_state.strategy,
            "learning_state": self._state_response(self.session.learning_state, source="chat"),
            "trend": self.session.learning_state.trend,
            "passage_type": context.passage_type,
            "document_id": context.document_id,
            "document_type": context.document_type,
            "highlight_id": context.highlight_id,
            "response_policy": policy.as_dict(),
            "prompt_preview": self.last_prompt_preview,
            "context_debug": self._context_debug(context),
            "request_summary": self.last_request_summary,
            "log_path": str(self.session.logger.path),
            "error": response.error,
        }

    def speech_transcribe(self) -> dict[str, Any]:
        return {"available": False, "text": "", "message": "Speech is optional and no speech backend is configured."}

    def _document_response(self, page_text: str) -> dict[str, Any]:
        document = self.session.document
        if document is None:
            raise RuntimeError("No document loaded.")
        record = self.documents.get(self.current_document_id or "", {})
        page_count = self._prepared_page_count(record, document)
        return {
            "document_id": self.current_document_id,
            "type": self.current_document_type or document.metadata.get("format", "txt"),
            "pdf_url": f"/api/document/file/{self.current_document_id}"
            if self.current_document_type == "pdf" and self.current_document_id
            else None,
            "title": document.title,
            "file_name": record.get("file_name") or Path(document.source_path).name,
            "source_path": str(document.source_path),
            "page_count": page_count,
            "parse_status": record.get("parse_status"),
            "current_page": self.session.current_page_number,
            "current_page_text": page_text,
            "metadata": document.metadata,
            "section_hints": document.section_hints,
            "pages": [
                {
                    "page_number": page.page_number,
                    "text_length": len(page.text),
                    "heading": page.heading,
                }
                for page in document.pages
            ],
        }

    def _document_dir(self, document_id: str) -> Path:
        document_id = self._safe_document_id(document_id)
        path = (self.documents_dir / document_id).resolve()
        root = self.documents_dir.resolve()
        if path != root and root not in path.parents:
            raise ValueError("Refusing to access a document outside runtime_uploads/documents.")
        return path

    def _record_for_document(self, document_id: str) -> dict[str, Any] | None:
        document_id = self._safe_document_id(document_id)
        record = self.documents.get(document_id)
        if record:
            self._ensure_document_runtime_dirs(document_id)
            return record
        document_dir = self._document_dir(document_id)
        original = document_dir / "original.pdf"
        if not original.exists():
            return None
        self._ensure_document_runtime_dirs(document_id)
        meta = self._read_json(document_dir / "meta.json", {})
        parse_status = self._prepare_status_from_files(document_id)
        record = {
            "document_id": document_id,
            "type": "pdf",
            "path": original.resolve(),
            "file_name": str(meta.get("file_name") or original.name),
            "title": str(meta.get("title") or meta.get("file_name") or original.stem),
            "page_count": self._positive_int(meta.get("page_count")),
            "parse_status": parse_status,
        }
        self.documents[document_id] = record
        self.highlights_by_document.setdefault(document_id, self._load_highlights(document_id))
        return record

    def _infer_document_meta(self, document_id: str) -> dict[str, Any]:
        document_id = self._safe_document_id(document_id)
        document_dir = self._document_dir(document_id)
        meta = self._read_json(document_dir / "meta.json", {})
        if not isinstance(meta, dict):
            meta = {}
        record = self.documents.get(document_id) or {}
        prepare_status = self._prepare_status_from_files(document_id) or record.get("parse_status") or {}
        paper_profile = self._read_json(document_dir / "rag" / "paper_profile.json", {})
        blocks = self._load_document_blocks(document_id)
        now = time.time()
        file_name = str(meta.get("file_name") or record.get("file_name") or "original.pdf")
        title = str(meta.get("title") or (paper_profile.get("title") if isinstance(paper_profile, dict) else "") or file_name)
        parsed_blocks_count = (
            self._positive_int(meta.get("parsed_blocks_count"))
            or self._positive_int(prepare_status.get("block_count"))
            or len(blocks)
        )
        page_count = (
            self._positive_int(meta.get("page_count"))
            or self._positive_int(prepare_status.get("page_count"))
            or self._page_count_from_loaded_blocks(blocks)
            or self._positive_int(record.get("page_count"))
            or 0
        )
        embedding_status = str(
            meta.get("embedding_status")
            or prepare_status.get("embedding_index_status")
            or self._embedding_status_from_file(document_id)
            or "unavailable"
        )
        if "library_visible" in meta:
            library_visible = bool(meta.get("library_visible"))
        else:
            library_visible = str(meta.get("uploaded_from") or "") == "pdf_chat"
        inferred = {
            "document_id": document_id,
            "file_name": file_name,
            "title": title,
            "page_count": page_count,
            "created_at": meta.get("created_at") or now,
            "updated_at": meta.get("updated_at") or now,
            "last_opened_at": meta.get("last_opened_at") or None,
            "last_page": self._positive_int(meta.get("last_page")) or 1,
            "prepare_status": meta.get("prepare_status") or prepare_status.get("status") or ("completed" if prepare_status else "unknown"),
            "parsed_blocks_count": parsed_blocks_count,
            "embedding_status": embedding_status,
            "retrieval_method": meta.get("retrieval_method") or self._retrieval_method_for_status(prepare_status, document_id),
            "highlight_count": self._highlight_count(document_id),
            "thread_count": self._thread_count(document_id),
            "uploaded_from": meta.get("uploaded_from"),
            "library_visible": library_visible,
            "archived_at": meta.get("archived_at") or None,
        }
        return inferred

    def _write_meta(self, document_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        document_dir = self._document_dir(document_id)
        document_dir.mkdir(parents=True, exist_ok=True)
        clean_meta = dict(meta)
        clean_meta["document_id"] = self._safe_document_id(document_id)
        (document_dir / "meta.json").write_text(json.dumps(clean_meta, indent=2), encoding="utf-8")
        record = self.documents.get(clean_meta["document_id"])
        if record:
            record["file_name"] = clean_meta.get("file_name") or record.get("file_name")
            record["title"] = clean_meta.get("title") or record.get("title")
            record["page_count"] = clean_meta.get("page_count") or record.get("page_count")
        return clean_meta

    def _ensure_document_runtime_dirs(self, document_id: str) -> None:
        document_dir = self._document_dir(document_id)
        for relative in self.REQUIRED_DOCUMENT_SUBDIRS:
            (document_dir / relative).mkdir(parents=True, exist_ok=True)

    def _update_meta(self, document_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        meta = self._infer_document_meta(document_id)
        meta.update(updates)
        return self._write_meta(document_id, meta)

    def _prepare_status_from_files(self, document_id: str) -> dict[str, Any]:
        document_dir = self._document_dir(document_id)
        status = self._read_json(document_dir / "rag" / "prepare_status.json", {})
        if not isinstance(status, dict):
            status = {}
        parsed_dir = document_dir / "parsed"
        blocks_path = parsed_dir / "blocks_index.json"
        if blocks_path.exists():
            blocks_payload = self._read_json(blocks_path, {})
            if isinstance(blocks_payload, dict):
                status.setdefault("document_id", document_id)
                status.setdefault("status", blocks_payload.get("status") or "completed")
                status.setdefault("parser", blocks_payload.get("parser"))
                status.setdefault("message", blocks_payload.get("message"))
                status.setdefault("page_count", blocks_payload.get("page_count"))
                status.setdefault("block_count", len(blocks_payload.get("blocks") or []))
        if parsed_dir.exists():
            status.setdefault("parsed_dir", str(parsed_dir))
            status.setdefault("document_md_path", str(parsed_dir / "document.md"))
            status.setdefault("blocks_index_path", str(blocks_path))
        rag_dir = document_dir / "rag"
        if rag_dir.exists():
            status.setdefault("paper_profile_path", str(rag_dir / "paper_profile.json"))
            status.setdefault("section_map_path", str(rag_dir / "section_map.json"))
            status.setdefault("keyword_index_path", str(rag_dir / "keyword_index.json"))
            status.setdefault("embeddings_path", str(rag_dir / "embeddings.json"))
            status.setdefault("rag_prepare_status_path", str(rag_dir / "prepare_status.json"))
            status.setdefault("embedding_index_status", self._embedding_status_from_file(document_id) or "unavailable")
        return self._ensure_prepare_progress(document_id, status)

    def _write_prepare_status(self, document_id: str, status: dict[str, Any]) -> None:
        rag_dir = self._document_dir(document_id) / "rag"
        rag_dir.mkdir(parents=True, exist_ok=True)
        status = dict(status)
        status.setdefault("rag_prepare_status_path", str(rag_dir / "prepare_status.json"))
        (rag_dir / "prepare_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    def _ensure_prepare_progress(self, document_id: str, status: dict[str, Any]) -> dict[str, Any]:
        status = dict(status or {})
        if not status:
            return self._prepare_progress_payload(
                document_id=document_id,
                stage="uploading_pdf",
                status="not_started",
                progress_percent=0,
                started_at=time.time(),
                updated_at=time.time(),
                base=status,
            )
        if "stage" in status and "progress_percent" in status and "steps" in status:
            return self._prepare_progress_payload(
                document_id=document_id,
                stage=str(status.get("stage") or "extracting_text"),
                status=str(status.get("status") or "preparing"),
                progress_percent=self._positive_int(status.get("progress_percent")),
                started_at=float(status.get("started_at") or status.get("prepared_at") or time.time()),
                updated_at=float(status.get("updated_at") or status.get("prepared_at") or time.time()),
                base=status,
                warnings=status.get("warnings") if isinstance(status.get("warnings"), list) else [],
                error=str(status.get("error") or "") or None,
            )
        completed = str(status.get("status") or "").lower() == "completed" or bool(status.get("block_count"))
        return self._prepare_progress_payload(
            document_id=document_id,
            stage="ready" if completed else "extracting_text",
            status="completed" if completed else str(status.get("status") or "preparing"),
            progress_percent=100 if completed else 24,
            started_at=float(status.get("started_at") or status.get("prepared_at") or time.time()),
            updated_at=float(status.get("updated_at") or status.get("prepared_at") or time.time()),
            base=status,
            warnings=status.get("warnings") if isinstance(status.get("warnings"), list) else [],
            error=str(status.get("error") or "") or None,
        )

    def _prepare_progress_payload(
        self,
        document_id: str,
        stage: str,
        status: str,
        progress_percent: int,
        started_at: float,
        updated_at: float,
        base: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        base_payload = dict(base or {})
        stage_info = self._prepare_stage_info(stage)
        progress = max(0, min(int(progress_percent or 0), 100))
        if status == "completed":
            progress = 100
            stage_info = self._prepare_stage_info("ready")
        elapsed = max(0.0, float(updated_at or time.time()) - float(started_at or updated_at or time.time()))
        estimated = 0.0 if status == "completed" else None
        if status not in {"completed", "failed"} and progress > 0:
            estimated = max(0.0, elapsed * (100 - progress) / progress)
        base_payload.update(
            {
                "document_id": document_id,
                "status": status,
                "stage": stage_info["stage"],
                "stage_label": stage_info["label"],
                "progress_percent": progress,
                "started_at": started_at,
                "updated_at": updated_at,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(estimated, 1) if estimated is not None else None,
                "steps": self._prepare_steps(stage_info["stage"], progress, status),
                "warnings": warnings or base_payload.get("warnings") or [],
            }
        )
        if error:
            base_payload["error"] = error
        return base_payload

    def _prepare_stage_info(self, stage: str) -> dict[str, Any]:
        for step in self.PREPARE_STEPS:
            if stage in {step["stage"], step["id"]}:
                return step
        return self.PREPARE_STEPS[1]

    def _prepare_steps(self, active_stage: str, progress_percent: int, status: str) -> list[dict[str, str]]:
        steps = []
        for step in self.PREPARE_STEPS:
            if status == "completed":
                step_status = "completed"
            elif status == "failed" and step["stage"] == active_stage:
                step_status = "failed"
            elif step["end"] <= progress_percent:
                step_status = "completed"
            elif step["stage"] == active_stage:
                step_status = "active"
            else:
                step_status = "pending"
            steps.append({"id": step["id"], "label": step["label"], "status": step_status})
        return steps

    @staticmethod
    def _is_library_visible(meta: dict[str, Any]) -> bool:
        if "library_visible" in meta:
            return bool(meta.get("library_visible"))
        return str(meta.get("uploaded_from") or "") == "pdf_chat"

    def _prepare_warnings(self, parse_status: dict[str, Any]) -> list[str]:
        warnings = []
        embedding_status = str(parse_status.get("embedding_index_status") or "")
        if embedding_status and embedding_status not in {"completed", "ready"}:
            warnings.append(f"Embedding index status: {embedding_status}. Keyword retrieval is available.")
        if parse_status.get("error"):
            warnings.append(str(parse_status["error"]))
        return warnings

    def _retrieval_method_for_status(self, status: dict[str, Any], document_id: str) -> str:
        embeddings_status = str(status.get("embedding_index_status") or self._embedding_status_from_file(document_id) or "").lower()
        if embeddings_status == "completed":
            return "embedding"
        if (self._document_dir(document_id) / "rag" / "keyword_index.json").exists():
            return "keyword"
        return str(status.get("retrieval_method") or "unknown")

    def _embedding_status_from_file(self, document_id: str) -> str:
        payload = self._read_json(self._document_dir(document_id) / "rag" / "embeddings.json", {})
        if isinstance(payload, dict):
            return str(payload.get("status") or "")
        return ""

    def _load_document_blocks(self, document_id: str) -> list[dict[str, Any]]:
        path = self._document_dir(document_id) / "parsed" / "blocks_index.json"
        if not path.exists():
            return []
        try:
            return load_blocks(path)
        except Exception:
            return []

    def _page_count_from_loaded_blocks(self, blocks: list[dict[str, Any]]) -> int:
        counts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            page_number = self._positive_int(block.get("page_number"))
            if page_number:
                counts.append(page_number)
            elif block.get("page_idx") is not None:
                counts.append(self._positive_int(block.get("page_idx")) + 1)
        return max(counts, default=0)

    def _session_dir(self, session_id: str) -> Path:
        session_id = self._safe_file_id(session_id)
        path = (self.upload_dir / "sessions" / session_id).resolve()
        root = (self.upload_dir / "sessions").resolve()
        if path != root and root not in path.parents:
            raise ValueError("Refusing to access a session outside runtime_uploads/sessions.")
        return path

    def _learning_state_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "learning_state"

    def _get_emotion_adapter(self) -> Any:
        if self._emotion_adapter is None:
            self._emotion_adapter = TeammateEmotionAdapter()
        return self._emotion_adapter

    def _get_emotion_pipeline(self) -> Any:
        if self._emotion_pipeline is None:
            self._emotion_pipeline = CombinedEmotionPipeline()
        return self._emotion_pipeline

    def _emotion_pipeline_status(self, fallback_status: dict[str, Any] | None = None) -> dict[str, Any]:
        pipeline = self._get_emotion_pipeline()
        try:
            return pipeline.status(fallback_status=fallback_status)
        except TypeError:
            return pipeline.status()

    def _emotion_pipeline_prediction(
        self,
        image: Any,
        fallback_prediction: dict[str, Any] | None = None,
        fallback_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pipeline = self._get_emotion_pipeline()
        try:
            return pipeline.predict(
                image,
                fallback_prediction=fallback_prediction,
                fallback_status=fallback_status,
            )
        except TypeError:
            return pipeline.predict(image)

    def _prediction_from_emotion_pipeline(
        self,
        emotion_pipeline: dict[str, Any],
        fallback_prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(emotion_pipeline, dict):
            return fallback_prediction or {}
        model_output_type = str(emotion_pipeline.get("model_output_type") or "")
        if model_output_type == "raw_emotion":
            raw_detection = emotion_pipeline.get("raw_detection") if isinstance(emotion_pipeline.get("raw_detection"), dict) else {}
            mapped = emotion_pipeline.get("mapped_academic_state") if isinstance(emotion_pipeline.get("mapped_academic_state"), dict) else {}
            smoothed = emotion_pipeline.get("smoothed_state") if isinstance(emotion_pipeline.get("smoothed_state"), dict) else {}
            scores = mapped.get("scores") if isinstance(mapped.get("scores"), dict) else {}
            academic_state = str(smoothed.get("state") or mapped.get("state") or "engagement")
            confidence = scores.get(academic_state, raw_detection.get("confidence", 0.0)) if isinstance(scores, dict) else raw_detection.get("confidence", 0.0)
            return {
                "model_loaded": True,
                "model_output_type": "raw_emotion",
                "raw_emotion_available": True,
                "raw_emotion": raw_detection.get("label"),
                "academic_state": academic_state,
                "confidence": confidence,
                "state_distribution": scores,
            }
        if model_output_type == "academic_state":
            academic = emotion_pipeline.get("academic_state") if isinstance(emotion_pipeline.get("academic_state"), dict) else {}
            mapped = emotion_pipeline.get("mapped_academic_state") if isinstance(emotion_pipeline.get("mapped_academic_state"), dict) else {}
            smoothed = emotion_pipeline.get("smoothed_state") if isinstance(emotion_pipeline.get("smoothed_state"), dict) else {}
            distribution = academic.get("distribution") if isinstance(academic.get("distribution"), dict) else mapped.get("scores")
            distribution = distribution if isinstance(distribution, dict) else {}
            academic_state = str(smoothed.get("state") or academic.get("state") or mapped.get("state") or "engagement")
            confidence = distribution.get(academic_state, academic.get("confidence", 0.0)) if isinstance(distribution, dict) else academic.get("confidence", 0.0)
            return {
                "model_loaded": True,
                "model_output_type": "academic_state",
                "raw_emotion_available": False,
                "raw_emotion": None,
                "academic_state": academic_state,
                "confidence": confidence,
                "state_distribution": distribution,
            }
        if fallback_prediction and not fallback_prediction.get("error"):
            return fallback_prediction
        return {}

    @staticmethod
    def _compact_face_detection(face_detection: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(face_detection, dict):
            return {}
        keys = (
            "detector",
            "requested_detector",
            "configured_detector",
            "actual_detector",
            "face_found",
            "confidence",
            "landmark_count",
            "crop_strategy",
            "crop_mode",
            "crop_parameters",
            "frame_size",
            "mirrored",
            "fallback_used",
            "warning",
            "warnings",
        )
        compact = {key: face_detection.get(key) for key in keys if key in face_detection}
        compact.setdefault("detector", compact.get("actual_detector") or face_detection.get("detector"))
        compact.setdefault("actual_detector", compact.get("detector") or face_detection.get("actual_detector"))
        return compact

    def _get_face_detector(self) -> Any:
        if self._face_detector is None:
            self._face_detector = create_face_detector()
        return self._face_detector

    def _face_detector_status_detail(self) -> dict[str, Any]:
        return detector_status(self._get_face_detector())

    def _current_or_simulated_learning_state(self, session_id: str, config: dict[str, Any]) -> dict[str, Any]:
        current = self._read_json(self._learning_state_dir(session_id) / "current_state.json", {})
        if isinstance(current, dict) and current.get("session_id"):
            return current
        return self._simulated_learning_state(session_id, config, time.time())

    def _decode_frame_image_for_model(self, image_data: str) -> tuple[Any, list[str]]:
        warnings: list[str] = []
        raw = b""
        if image_data:
            payload = image_data.split(",", 1)[1] if "," in image_data else image_data
            try:
                raw = base64.b64decode(payload, validate=False)
            except (binascii.Error, ValueError):
                warnings.append("Frame payload could not be decoded; using a neutral placeholder frame.")
        try:
            from PIL import Image  # type: ignore

            if raw:
                return Image.open(BytesIO(raw)).convert("RGB"), warnings
            return Image.new("RGB", (224, 224), (128, 128, 128)), warnings
        except Exception as exc:
            warnings.append(f"Pillow frame decoding unavailable; using byte placeholder: {exc}")
            try:
                import numpy as np  # type: ignore

                return np.full((224, 224, 3), 128, dtype="uint8"), warnings
            except Exception:
                return raw or b"", warnings

    @staticmethod
    def _image_preview_data_url(image: Any, size: tuple[int, int] | None = None) -> str:
        try:
            from PIL import Image  # type: ignore

            if isinstance(image, Image.Image):
                preview = image.copy().convert("RGB")
            else:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                if arr.ndim == 2:
                    preview = Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
                else:
                    preview = Image.fromarray(arr[:, :, :3].astype("uint8")).convert("RGB")
            if size:
                preview = preview.resize(size)
            output = BytesIO()
            preview.save(output, format="JPEG", quality=82)
            return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        except Exception:
            return ""

    def _annotated_frame_preview_data_url(self, image: Any, face_detection: dict[str, Any]) -> str:
        try:
            from PIL import Image, ImageDraw  # type: ignore

            if isinstance(image, Image.Image):
                preview = image.copy().convert("RGB")
            else:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                if arr.ndim == 2:
                    preview = Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB")
                else:
                    preview = Image.fromarray(arr[:, :, :3].astype("uint8")).convert("RGB")

            draw = ImageDraw.Draw(preview)
            crop_bbox = self._normalized_bbox(face_detection.get("crop_bbox_used"))
            landmark_bbox = self._normalized_bbox(face_detection.get("landmark_bbox"))
            if landmark_bbox:
                x, y, w, h = landmark_bbox
                draw.rectangle([x, y, x + w, y + h], outline=(56, 189, 248), width=3)
            if crop_bbox:
                x, y, w, h = crop_bbox
                draw.rectangle([x, y, x + w, y + h], outline=(245, 158, 11), width=3)

            landmarks = face_detection.get("landmarks")
            if not isinstance(landmarks, list):
                landmarks = (face_detection.get("openface") or {}).get("landmarks")
            if isinstance(landmarks, list):
                for point in landmarks[:68]:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    try:
                        px = float(point[0])
                        py = float(point[1])
                    except (TypeError, ValueError):
                        continue
                    radius = 2
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=(251, 146, 60),
                        outline=(255, 255, 255),
                    )

            output = BytesIO()
            preview.save(output, format="JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        except Exception:
            return ""

    @staticmethod
    def _resize_image(image: Any, size: tuple[int, int]) -> Any:
        try:
            from PIL import Image  # type: ignore

            if isinstance(image, Image.Image):
                return image.copy().convert("RGB").resize(size)
            import numpy as np  # type: ignore

            arr = np.asarray(image)
            if arr.ndim == 2:
                return Image.fromarray(arr.astype("uint8"), mode="L").convert("RGB").resize(size)
            return Image.fromarray(arr[:, :, :3].astype("uint8")).convert("RGB").resize(size)
        except Exception:
            return image

    @staticmethod
    def _camera_debug_mode_explanation() -> dict[str, Any]:
        return {
            "current_mode": "Academic-state checkpoint. Raw 8-class facial emotion output is unavailable.",
            "mapping_step": "Bypassed because the current checkpoint predicts boredom, confusion, engagement, and frustration directly.",
            "strategy_input": "Academic-state probabilities are used directly as a noisy support cue for strategy planning.",
            "current_mode_chain": "webcam frame -> face crop -> academic-state model -> support cue -> strategy families",
            "future_mode_chain": "webcam frame -> face crop -> raw-emotion model -> raw-to-academic mapping -> support cue -> strategy families",
        }

    @staticmethod
    def _first_sample_timestamp(samples: list[Any]) -> str:
        for sample in samples:
            if isinstance(sample, dict) and sample.get("timestamp"):
                return str(sample["timestamp"])
        return ""

    @staticmethod
    def _last_sample_timestamp(samples: list[Any]) -> str:
        for sample in reversed(samples):
            if isinstance(sample, dict) and sample.get("timestamp"):
                return str(sample["timestamp"])
        return ""

    @staticmethod
    def _center_crop_image(image: Any) -> Any:
        try:
            width, height = image.size
            side = min(width, height)
            left = max(0, (width - side) // 2)
            top = max(0, (height - side) // 2)
            return image.crop((left, top, left + side, top + side))
        except Exception:
            try:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                if arr.ndim < 2:
                    return image
                height, width = arr.shape[:2]
                side = min(width, height)
                left = max(0, (width - side) // 2)
                top = max(0, (height - side) // 2)
                return arr[top : top + side, left : left + side]
            except Exception:
                return image

    @staticmethod
    def _image_size(image: Any) -> tuple[int, int]:
        try:
            width, height = image.size
            return int(width), int(height)
        except Exception:
            try:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                height, width = arr.shape[:2]
                return int(width), int(height)
            except Exception:
                return 0, 0

    @staticmethod
    def _center_crop_bbox(image: Any) -> list[int] | None:
        width, height = WebState._image_size(image)
        if width <= 0 or height <= 0:
            return None
        side = min(width, height)
        left = max(0, (width - side) // 2)
        top = max(0, (height - side) // 2)
        return [int(left), int(top), int(side), int(side)]

    @staticmethod
    def _crop_to_bbox(image: Any, bbox: list[int]) -> Any:
        x, y, w, h = [int(value) for value in bbox]
        try:
            return image.crop((x, y, x + w, y + h))
        except Exception:
            try:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                return arr[y : y + h, x : x + w]
            except Exception:
                return image

    @staticmethod
    def _safe_openface_payload(metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        payload: dict[str, Any] = {}
        for key in (
            "configured",
            "binary_path",
            "binary_exists",
            "available",
            "success",
            "confidence",
            "landmark_count",
            "landmarks",
            "bbox",
            "pose",
            "aus",
            "aus_summary",
            "head_pose_available",
            "aus_available",
            "warning",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        return payload

    @staticmethod
    def _normalized_bbox(value: Any) -> list[int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            return [int(round(float(item))) for item in value[:4]]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _landmark_bbox_from_points(landmarks: Any) -> list[int] | None:
        if not isinstance(landmarks, list):
            return None
        xs: list[float] = []
        ys: list[float] = []
        for point in landmarks:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
        if not xs or not ys:
            return None
        min_x = math.floor(min(xs))
        min_y = math.floor(min(ys))
        return [
            int(min_x),
            int(min_y),
            max(1, int(math.ceil(max(xs) - min_x))),
            max(1, int(math.ceil(max(ys) - min_y))),
        ]

    def _effective_crop_parameters(self, crop_parameters: dict[str, Any], actual_detector: str) -> dict[str, Any]:
        mode = self._safe_crop_mode(crop_parameters.get("mode") or crop_parameters.get("crop_mode"))
        if actual_detector != "openface":
            return {
                "mode": mode,
                "scale": crop_parameters.get("scale", self.FACE_CROP_DEFAULTS["crop_scale"]),
                "y_bias": crop_parameters.get("y_bias", self.FACE_CROP_DEFAULTS["crop_y_bias"]),
                "top_extra": crop_parameters.get("top_extra", 0.0),
                "bottom_extra": crop_parameters.get("bottom_extra", self.FACE_CROP_DEFAULTS["crop_bottom_extra"]),
                "make_square": crop_parameters.get("make_square", self.FACE_CROP_DEFAULTS["crop_make_square"]),
            }
        defaults = dict(self.OPENFACE_CROP_MODE_DEFAULTS[mode])
        return {
            "mode": mode,
            "scale": crop_parameters.get("scale", defaults["scale"]),
            "y_bias": crop_parameters.get("y_bias", defaults["y_bias"]),
            "top_extra": crop_parameters.get("top_extra", defaults["top_extra"]),
            "bottom_extra": crop_parameters.get("bottom_extra", defaults["bottom_extra"]),
            "make_square": crop_parameters.get("make_square", defaults["make_square"]),
        }

    def _camera_debug_assertions(
        self,
        frame_width: int,
        frame_height: int,
        face_detection: dict[str, Any],
        model_input_image: Any,
    ) -> list[str]:
        warnings: list[str] = []
        landmarks = face_detection.get("landmarks")
        if not isinstance(landmarks, list):
            landmarks = (face_detection.get("openface") or {}).get("landmarks")
        if isinstance(landmarks, list):
            for point in landmarks:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    x = float(point[0])
                    y = float(point[1])
                except (TypeError, ValueError):
                    warnings.append("Landmark coordinates include non-numeric values.")
                    break
                if x < 0 or y < 0 or x > frame_width or y > frame_height:
                    warnings.append("Landmark coordinates outside frame bounds.")
                    break
        for key in ("bbox", "landmark_bbox", "crop_bbox_used"):
            bbox = self._normalized_bbox(face_detection.get(key))
            if not bbox:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                warnings.append(f"{key} width/height is not positive.")
                continue
            if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
                warnings.append(f"{key} is outside frame bounds.")
        model_width, model_height = self._image_size(model_input_image)
        if [model_width, model_height] != [224, 224]:
            warnings.append("Model input preview is not 224x224.")
        return list(dict.fromkeys(warnings))

    def _detect_face_crop(
        self,
        image: Any,
        crop_parameters: dict[str, Any] | None = None,
        mirrored: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        crop_parameters = crop_parameters or {
            "mode": self.FACE_CROP_DEFAULTS["crop_mode"],
            "scale": self.FACE_CROP_DEFAULTS["crop_scale"],
            "y_bias": self.FACE_CROP_DEFAULTS["crop_y_bias"],
            "top_extra": self.FACE_CROP_DEFAULTS["crop_top_extra"],
            "bottom_extra": self.FACE_CROP_DEFAULTS["crop_bottom_extra"],
            "make_square": self.FACE_CROP_DEFAULTS["crop_make_square"],
        }
        detector = self._get_face_detector()
        status = detector_status(detector)
        requested_detector = str(status.get("requested_detector") or status.get("mode") or "center_crop")
        width, height = self._image_size(image)
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                frame_bgr = arr[:, :, :3][:, :, ::-1]
            else:
                frame_bgr = arr
            boxes = detector.detect(frame_bgr) if getattr(detector, "is_available", False) else []
        except Exception as exc:
            boxes = []
            status = {
                "mode": "center_crop",
                "requested_detector": requested_detector,
                "actual_detector": "center_crop",
                "loaded": False,
                "fallback": "center_crop",
                "fallback_used": True,
                "warning": f"Face detector failed; using center crop fallback: {exc}",
            }
        if boxes:
            best = max(boxes, key=lambda box: float(getattr(box, "confidence", 0.0)))
            bbox = [int(best.x), int(best.y), int(best.w), int(best.h)]
            actual_detector = str(getattr(best, "source", "") or status.get("actual_detector") or status.get("mode") or "face_detector")
            openface_metadata = getattr(best, "openface", None) if actual_detector == "openface" else None
            openface_metadata = openface_metadata if isinstance(openface_metadata, dict) else {}
            landmarks = openface_metadata.get("landmarks") if actual_detector == "openface" else []
            landmarks = landmarks if isinstance(landmarks, list) else []
            landmark_bbox = self._normalized_bbox(openface_metadata.get("bbox")) if actual_detector == "openface" else None
            if actual_detector == "openface" and not landmark_bbox:
                landmark_bbox = self._landmark_bbox_from_points(landmarks)
            if actual_detector == "openface" and landmark_bbox:
                bbox = list(landmark_bbox)
            effective_crop_parameters = self._effective_crop_parameters(crop_parameters, actual_detector)
            crop_plan = expand_face_bbox(
                bbox,
                image_width=width,
                image_height=height,
                scale=float(effective_crop_parameters.get("scale", self.FACE_CROP_DEFAULTS["crop_scale"])),
                y_bias=float(effective_crop_parameters.get("y_bias", self.FACE_CROP_DEFAULTS["crop_y_bias"])),
                top_extra=float(effective_crop_parameters.get("top_extra", self.FACE_CROP_DEFAULTS["crop_top_extra"])),
                bottom_extra=float(effective_crop_parameters.get("bottom_extra", self.FACE_CROP_DEFAULTS["crop_bottom_extra"])),
                make_square=bool(effective_crop_parameters.get("make_square", self.FACE_CROP_DEFAULTS["crop_make_square"])),
            )
            expanded_bbox = list(crop_plan["expanded_bbox"])
            crop_bbox_used = list(crop_plan["crop_bbox_used"])
            crop = self._crop_to_bbox(image, crop_bbox_used)
            fallback_used = bool(status.get("fallback_used")) or (
                requested_detector == "yolo" and actual_detector != "yolo"
            ) or (
                requested_detector == "auto" and actual_detector != "yolo" and bool(status.get("yolo_loaded") is False)
            ) or (
                requested_detector == "openface" and actual_detector != "openface"
            )
            warning = status.get("warning")
            if actual_detector == "openface":
                warning = openface_metadata.get("warning") or None
            if requested_detector == "openface" and actual_detector != "openface":
                fallback_name = "OpenCV Haar" if actual_detector == "opencv_haar" else actual_detector.replace("_", " ")
                if actual_detector == "center_crop":
                    fallback_name = "center crop"
                warning = f"OpenFace unavailable or failed; using {fallback_name} fallback."
            elif getattr(detector, "last_primary_empty", False) and actual_detector != "yolo":
                warning = "YOLO found no face; using OpenCV Haar fallback." if actual_detector == "opencv_haar" else "YOLO found no face; using center crop fallback."
            if actual_detector == "opencv_haar" and requested_detector in {"auto", "yolo"} and status.get("yolo_loaded") is False:
                warning = "YOLO is unavailable, so the system is using calibrated OpenCV Haar crop fallback."
            if fallback_used and not warning:
                warning = f"{requested_detector.replace('_', ' ').upper()} unavailable; using {actual_detector.replace('_', ' ')} fallback."
            if actual_detector == "opencv_haar" and requested_detector == "yolo" and warning:
                warning = f"{warning} OpenCV Haar fallback uses an expanded square crop but may be less reliable."
            shape = "square" if crop_plan["crop_parameters"]["make_square"] else "rect"
            if actual_detector == "openface":
                crop_strategy = "openface_landmark_bbox"
            elif actual_detector == "opencv_haar":
                crop_strategy = f"expanded_{shape}_opencv_haar_fallback"
            else:
                crop_strategy = f"expanded_{shape}_{actual_detector}"
            crop_margin = round((float(crop_plan["crop_parameters"]["scale"]) - 1.0) / 2.0, 4)
            face_payload = {
                "detector": actual_detector,
                "requested_detector": requested_detector,
                "configured_detector": status.get("configured_detector") or requested_detector,
                "actual_detector": actual_detector,
                "face_found": True,
                "confidence": round(float(best.confidence), 4),
                "bbox": bbox,
                "expanded_bbox": expanded_bbox,
                "crop_bbox_used": crop_bbox_used,
                "crop_margin": crop_margin,
                "crop_strategy": crop_strategy,
                "crop_mode": str(effective_crop_parameters.get("mode") or ""),
                "crop_parameters": crop_plan["crop_parameters"],
                "frame_size": [width, height],
                "mirrored": bool(mirrored),
                "fallback_used": fallback_used,
                "warning": warning,
                "warnings": [warning] if warning else [],
            }
            if actual_detector == "openface":
                face_payload.update(
                    {
                        "landmarks_available": bool(openface_metadata.get("landmarks")),
                        "landmarks": landmarks,
                        "landmark_count": int(openface_metadata.get("landmark_count") or 0),
                        "landmark_bbox": landmark_bbox or bbox,
                        "head_pose_available": bool(openface_metadata.get("head_pose_available")),
                        "aus_available": bool(openface_metadata.get("aus_available")),
                        "openface": self._safe_openface_payload(openface_metadata),
                    }
                )
            elif status.get("openface"):
                face_payload["openface"] = self._safe_openface_payload(status.get("openface"))
            return crop, face_payload
        center_bbox = self._center_crop_bbox(image)
        actual_detector = "center_crop"
        if requested_detector == "openface":
            warning = "OpenFace unavailable or failed; using center crop fallback."
        elif requested_detector == "yolo" and status.get("actual_detector") == "opencv_haar":
            warning = f"{status.get('warning') or 'YOLO unavailable; using OpenCV Haar fallback.'} OpenCV Haar found no face; using center crop fallback."
        elif requested_detector == "auto" and status.get("actual_detector") == "opencv_haar":
            warning = f"{status.get('warning') or 'YOLO auto-detection unavailable; using OpenCV Haar fallback.'} OpenCV Haar found no face; using center crop fallback."
        else:
            warning = status.get("warning") or "No face detected; using center crop fallback."
        return self._center_crop_image(image), {
            "detector": "center_crop",
            "requested_detector": requested_detector,
            "configured_detector": status.get("configured_detector") or requested_detector,
            "actual_detector": actual_detector,
            "face_found": False,
            "confidence": 0.0,
            "bbox": None,
            "expanded_bbox": center_bbox,
            "crop_bbox_used": center_bbox,
            "crop_margin": 0.0,
            "crop_strategy": "center_crop",
            "crop_mode": str(crop_parameters.get("mode") or ""),
            "crop_parameters": {
                "scale": round(float(crop_parameters.get("scale", self.FACE_CROP_DEFAULTS["crop_scale"])), 4),
                "y_bias": round(float(crop_parameters.get("y_bias", self.FACE_CROP_DEFAULTS["crop_y_bias"])), 4),
                "top_extra": round(float(crop_parameters.get("top_extra", self.FACE_CROP_DEFAULTS["crop_top_extra"])), 4),
                "bottom_extra": round(float(crop_parameters.get("bottom_extra", self.FACE_CROP_DEFAULTS["crop_bottom_extra"])), 4),
                "make_square": bool(crop_parameters.get("make_square", self.FACE_CROP_DEFAULTS["crop_make_square"])),
            },
            "frame_size": [width, height],
            "mirrored": bool(mirrored),
            "fallback_used": True,
            "warning": warning,
            "warnings": [warning] if warning else [],
            "openface": self._safe_openface_payload(status.get("openface")) if status.get("openface") else {},
        }

    @staticmethod
    def _crop_with_margin(image: Any, x: int, y: int, w: int, h: int, margin_ratio: float = 0.18) -> Any:
        try:
            width, height = image.size
            margin = int(max(w, h) * margin_ratio)
            left = max(0, x - margin)
            top = max(0, y - margin)
            right = min(width, x + w + margin)
            bottom = min(height, y + h + margin)
            return image.crop((left, top, right, bottom))
        except Exception:
            try:
                import numpy as np  # type: ignore

                arr = np.asarray(image)
                height, width = arr.shape[:2]
                margin = int(max(w, h) * margin_ratio)
                left = max(0, x - margin)
                top = max(0, y - margin)
                right = min(width, x + w + margin)
                bottom = min(height, y + h + margin)
                return arr[top:bottom, left:right]
            except Exception:
                return image

    def _learning_state_from_model_prediction(
        self,
        session_id: str,
        document_id: str,
        prediction: dict[str, Any],
        warnings: list[str],
        now: float,
        face_detection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        distribution = prediction.get("state_distribution") or prediction.get("distribution") or {}
        if not isinstance(distribution, dict):
            distribution = {}
        academic_state = str(prediction.get("academic_state") or max(distribution, key=distribution.get, default="engagement"))
        if academic_state not in self.ACADEMIC_STATES:
            academic_state = "engagement"
        confidence = self._bounded_probability(prediction.get("confidence"), distribution.get(academic_state, 0.0))
        distribution = self._normalized_state_distribution(distribution, academic_state, confidence)
        current = self._read_json(self._learning_state_dir(session_id) / "current_state.json", {})
        previous_timestamp = self._epoch_from_iso_timestamp(str(current.get("timestamp") or "")) if isinstance(current, dict) else 0.0
        previous_state = str(current.get("academic_state") or "") if isinstance(current, dict) else ""
        previous_confidence = float(current.get("confidence") or 0.0) if isinstance(current, dict) else 0.0
        previous_duration = float(current.get("duration_sec") or 0.0) if isinstance(current, dict) else 0.0
        elapsed_since_previous = max(0.0, now - previous_timestamp) if previous_timestamp else 0.0
        same_state = previous_state == academic_state and current.get("source") == "webcam_model" if isinstance(current, dict) else False
        duration = previous_duration + elapsed_since_previous if same_state else 0.0
        trend = self._model_state_trend(academic_state, previous_state, confidence, previous_confidence, same_state)
        stability = "stable" if same_state and duration >= 6 else "transitioning"
        if academic_state == "engagement" and previous_state not in {"", "engagement"}:
            stability = "recovering"
        trigger_recommended, trigger_reason = self._learning_trigger_metadata(academic_state, confidence, trend, duration)
        return {
            "session_id": session_id,
            "document_id": document_id,
            "timestamp": self._iso_timestamp(now),
            "source": "webcam_model",
            "model_output_type": str(prediction.get("model_output_type") or "academic_state"),
            "raw_facial_emotion_available": bool(prediction.get("raw_emotion_available")) if prediction.get("model_output_type") == "raw_emotion" else False,
            "raw_facial_emotion": prediction.get("raw_emotion") if prediction.get("model_output_type") == "raw_emotion" else None,
            "academic_state": academic_state,
            "confidence": round(confidence, 4),
            "distribution": distribution,
            "trend": trend,
            "duration_sec": round(duration, 1),
            "intensity": round(confidence, 4),
            "stability": stability,
            "trigger_recommended": trigger_recommended,
            "trigger_reason": trigger_reason,
            "warnings": list(dict.fromkeys(warnings)),
            "face_detection": face_detection or {},
        }

    @staticmethod
    def _bounded_probability(value: Any, fallback: Any = 0.0) -> float:
        try:
            number = float(value)
        except Exception:
            try:
                number = float(fallback)
            except Exception:
                number = 0.0
        return max(0.0, min(1.0, number))

    def _normalized_state_distribution(self, distribution: dict[str, Any], academic_state: str, confidence: float) -> dict[str, float]:
        values = {state: self._bounded_probability(distribution.get(state), 0.0) for state in self.ACADEMIC_STATES}
        if sum(values.values()) <= 0:
            return self._state_distribution(academic_state, confidence)
        total = sum(values.values())
        return {state: round(value / total, 4) for state, value in values.items()}

    @staticmethod
    def _model_state_trend(academic_state: str, previous_state: str, confidence: float, previous_confidence: float, same_state: bool) -> str:
        if same_state:
            delta = confidence - previous_confidence
            if delta > 0.03:
                return "rising"
            if delta < -0.03:
                return "falling"
            return "stable"
        if academic_state == "engagement" and previous_state and previous_state != "engagement":
            return "rising"
        return "rising" if confidence >= previous_confidence else "stable"

    @staticmethod
    def _epoch_from_iso_timestamp(value: str) -> float:
        if not value:
            return 0.0
        try:
            from datetime import datetime, timezone

            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except Exception:
            return 0.0

    def _simulated_learning_state(self, session_id: str, config: dict[str, Any], now: float) -> dict[str, Any]:
        document_id = str(config.get("document_id") or "")
        started_at = float(config.get("started_at") or now)
        elapsed = max(0.0, now - started_at)
        last_answer_at = self._last_session_event_time(session_id, {"answer_generated", "strategy_selected"})
        if last_answer_at and last_answer_at >= started_at:
            recovery_elapsed = max(0.0, now - last_answer_at)
            academic_state = "engagement"
            confidence = min(0.78, 0.70 + recovery_elapsed * 0.01)
            trend = "rising" if recovery_elapsed < 12 else "stable"
            duration = recovery_elapsed
            stability = "recovering" if recovery_elapsed < 12 else "stable"
        elif elapsed < 10:
            academic_state = "engagement"
            confidence = 0.65
            trend = "stable"
            duration = elapsed
            stability = "stable"
        elif elapsed < 25:
            academic_state = "confusion"
            confidence = 0.60 + ((elapsed - 10) / 15) * 0.22
            trend = "rising"
            duration = elapsed - 10
            stability = "transitioning"
        elif elapsed < 40:
            academic_state = "confusion"
            confidence = 0.80
            trend = "stable"
            duration = elapsed - 10
            stability = "stable"
        else:
            academic_state = "frustration"
            confidence = min(0.74, 0.62 + ((elapsed - 40) / 30) * 0.12)
            trend = "rising" if elapsed < 55 else "stable"
            duration = elapsed - 40
            stability = "transitioning" if elapsed < 55 else "stable"
        confidence = round(max(0.0, min(confidence, 1.0)), 2)
        distribution = self._state_distribution(academic_state, confidence)
        trigger_recommended, trigger_reason = self._learning_trigger_metadata(academic_state, confidence, trend, duration)
        return {
            "session_id": session_id,
            "document_id": document_id,
            "timestamp": self._iso_timestamp(now),
            "source": "simulated_camera",
            "model_output_type": "academic_state_model",
            "raw_facial_emotion_available": False,
            "raw_facial_emotion": None,
            "academic_state": academic_state,
            "confidence": confidence,
            "distribution": distribution,
            "trend": trend,
            "duration_sec": round(duration, 1),
            "intensity": confidence,
            "stability": stability,
            "trigger_recommended": trigger_recommended,
            "trigger_reason": trigger_reason,
        }

    def _write_learning_state(self, session_id: str, state: dict[str, Any]) -> None:
        learning_dir = self._learning_state_dir(session_id)
        learning_dir.mkdir(parents=True, exist_ok=True)
        clean_state = self._safe_log_payload(state)
        (learning_dir / "current_state.json").write_text(json.dumps(clean_state, indent=2), encoding="utf-8")
        with (learning_dir / "state_stream.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean_state, ensure_ascii=False) + "\n")

    @staticmethod
    def _state_distribution(academic_state: str, confidence: float) -> dict[str, float]:
        states = ("boredom", "confusion", "engagement", "frustration")
        remaining = max(0.0, 1.0 - confidence)
        base = round(remaining / 3, 2)
        distribution = {state: base for state in states}
        distribution[academic_state] = round(confidence, 2)
        drift = round(1.0 - sum(distribution.values()), 2)
        if abs(drift) >= 0.01:
            for state in states:
                if state != academic_state:
                    distribution[state] = round(distribution[state] + drift, 2)
                    break
        return distribution

    @staticmethod
    def _learning_trigger_metadata(academic_state: str, confidence: float, trend: str, duration: float) -> tuple[bool, str]:
        thresholds = {
            "confusion": (0.65, 8),
            "frustration": (0.60, 6),
            "boredom": (0.60, 12),
            "engagement": (0.75, 0),
        }
        min_confidence, min_duration = thresholds.get(academic_state, (0.70, 10))
        trend_ok = trend in {"rising", "stable"}
        recommended = bool(confidence >= min_confidence and duration >= min_duration and trend_ok)
        if recommended:
            return True, f"{academic_state} confidence remained above threshold for {int(min_duration)} seconds while the user had an active selection"
        return False, "learning-state signal has not crossed a strategy trigger threshold"

    def _append_session_event(self, session_id: str, event: dict[str, Any]) -> None:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        payload = self._safe_log_payload({"timestamp": time.time(), **event, "session_id": session_id})
        with (session_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _session_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._session_dir(session_id) / "events.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _last_session_event_time(self, session_id: str, event_types: set[str]) -> float | None:
        latest = None
        for event in self._session_events(session_id):
            if event.get("event_type") not in event_types:
                continue
            timestamp = event.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = self._timestamp_from_iso(timestamp)
            try:
                value = float(timestamp)
            except (TypeError, ValueError):
                continue
            latest = value if latest is None else max(latest, value)
        return latest

    def summarize_reaction_window(
        self,
        samples: list[dict[str, Any]],
        source_turn_id: str,
        highlight_id: str,
        window_start: str,
        window_end: str,
    ) -> dict[str, Any]:
        valid_samples = [sample for sample in samples if isinstance(sample, dict)]
        states = list(self.ACADEMIC_STATES)
        distributions = []
        confidences = []
        state_weights = {state: 0.0 for state in states}
        trends = []
        face_modes: dict[str, int] = {}
        fallback_used = False
        for sample in valid_samples:
            state = str(sample.get("academic_state") or sample.get("state") or "").lower()
            confidence = self._bounded_probability(sample.get("confidence"), 0.0)
            if state in state_weights:
                state_weights[state] += max(confidence, 0.01)
            confidences.append(confidence)
            distribution = sample.get("distribution") if isinstance(sample.get("distribution"), dict) else sample.get("state_distribution")
            if isinstance(distribution, dict):
                normalized = self._normalized_state_distribution(distribution, state if state in states else "engagement", confidence)
                distributions.append(normalized)
                for item_state, value in normalized.items():
                    state_weights[item_state] = state_weights.get(item_state, 0.0) + float(value)
            if sample.get("trend"):
                trends.append(str(sample.get("trend")))
            face_detection = sample.get("face_detection") if isinstance(sample.get("face_detection"), dict) else {}
            mode = str(face_detection.get("detector") or face_detection.get("mode") or "")
            if mode:
                face_modes[mode] = face_modes.get(mode, 0) + 1
            fallback_used = fallback_used or bool(face_detection.get("fallback_used"))
        if distributions:
            avg_distribution = {
                state: round(sum(item.get(state, 0.0) for item in distributions) / len(distributions), 4)
                for state in states
            }
        else:
            avg_distribution = self._state_distribution("engagement", 0.25)
        sorted_states = sorted(states, key=lambda state: (state_weights.get(state, 0.0), avg_distribution.get(state, 0.0)), reverse=True)
        dominant_state = sorted_states[0] if sorted_states else "engagement"
        secondary_state = sorted_states[1] if len(sorted_states) > 1 else ""
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        max_confidence = round(max(confidences), 4) if confidences else 0.0
        duration = max(0.0, self._epoch_from_iso_timestamp(window_end) - self._epoch_from_iso_timestamp(window_start))
        trend = self._reaction_trend(trends, dominant_state)
        support_cue, support_cue_label = self._support_cue_for_reaction(avg_distribution, dominant_state, secondary_state, avg_confidence)
        face_mode = max(face_modes, key=face_modes.get) if face_modes else "center_crop"
        return {
            "source_turn_id": self._safe_file_id(source_turn_id) if source_turn_id else "",
            "highlight_id": self._safe_file_id(highlight_id) if highlight_id else "",
            "window_start": window_start,
            "window_end": window_end,
            "duration_sec": round(duration, 1),
            "dominant_state": dominant_state,
            "secondary_state": secondary_state,
            "avg_confidence": avg_confidence,
            "max_confidence": max_confidence,
            "avg_distribution": avg_distribution,
            "trend": trend,
            "stability": "stable" if max_confidence >= 0.6 and avg_confidence >= 0.55 else "low_confidence",
            "support_cue": support_cue,
            "support_cue_label": support_cue_label,
            "trigger_reason": f"The baseline explanation was being read while the learning signal showed a {support_cue_label.lower()}.",
            "face_detection_summary": {
                "mode": face_mode,
                "fallback_used": fallback_used,
                "sample_count": len(valid_samples),
            },
        }

    @staticmethod
    def _reaction_trend(trends: list[str], dominant_state: str) -> str:
        lowered = [trend.lower() for trend in trends]
        if any("rising" in trend for trend in lowered):
            return "rising"
        if any("falling" in trend for trend in lowered):
            return "falling"
        if lowered:
            return "stable"
        return "stable" if dominant_state else "uncertain"

    def _support_cue_for_reaction(
        self,
        distribution: dict[str, float],
        dominant_state: str,
        secondary_state: str,
        avg_confidence: float,
    ) -> tuple[str, str]:
        states = {state: self._bounded_probability(distribution.get(state), 0.0) for state in self.ACADEMIC_STATES}
        ordered = sorted(states.items(), key=lambda item: item[1], reverse=True)
        top_state, top_value = ordered[0] if ordered else ("", 0.0)
        second_value = ordered[1][1] if len(ordered) > 1 else 0.0
        spread = max(states.values() or [0.0]) - min(states.values() or [0.0])
        if states.get("confusion", 0.0) >= 0.35 and states.get("boredom", 0.0) >= 0.25:
            return "clarify_and_reengage", "Clarify and re-engage cue"
        if states.get("confusion", 0.0) >= 0.35 and states.get("frustration", 0.0) >= 0.25:
            return "gentle_clarification", "Gentle clarification cue"
        if avg_confidence < 0.55 or top_value < 0.45 or top_value - second_value < 0.05 or spread < 0.18:
            return "neutral_or_uncertain", "Possible ways to continue"
        if states.get("engagement", 0.0) >= 0.65 or top_state == "engagement":
            return "deepening", "Deepening cue"
        if states.get("boredom", 0.0) >= 0.50 or top_state == "boredom":
            return "re_engagement", "Re-engagement cue"
        if states.get("frustration", 0.0) >= 0.45 or top_state == "frustration":
            return "reduce_load", "Reduce cognitive load cue"
        if states.get("confusion", 0.0) >= 0.45 or top_state == "confusion":
            return "sustained_clarification", "Sustained clarification cue"
        return "neutral_or_uncertain", "Possible ways to continue"

    def _allowed_strategy_families_for_support_cue(self, support_cue: str) -> list[str]:
        cue = str(support_cue or "").strip().lower()
        if cue in self.SUPPORT_CUE_STRATEGY_FAMILIES:
            return list(self.SUPPORT_CUE_STRATEGY_FAMILIES[cue])
        if cue in self.STRATEGY_FAMILY:
            return list(self.STRATEGY_FAMILY[cue])
        return list(self.SUPPORT_CUE_STRATEGY_FAMILIES["neutral_or_uncertain"])

    def _strategy_request_payload(self, document_id: str, data: dict[str, Any]) -> dict[str, Any]:
        learning_state = data.get("learning_state") if isinstance(data.get("learning_state"), dict) else {}
        paper_context = data.get("paper_context") if isinstance(data.get("paper_context"), dict) else {}
        trigger_context = data.get("trigger_context") if isinstance(data.get("trigger_context"), dict) else {}
        recent_conversation = data.get("recent_conversation") if isinstance(data.get("recent_conversation"), list) else []
        reaction_window_summary = data.get("reaction_window_summary") if isinstance(data.get("reaction_window_summary"), dict) else {}
        planner_input_summary = data.get("planner_input_summary") if isinstance(data.get("planner_input_summary"), dict) else {}
        support_cue = str(data.get("support_cue") or reaction_window_summary.get("support_cue") or "")
        request = {
            "session_id": str(data.get("session_id") or ""),
            "document_id": document_id,
            "highlight_id": self._safe_file_id(data.get("highlight_id") or uuid.uuid4().hex),
            "source_turn_id": self._safe_file_id(data.get("source_turn_id") or reaction_window_summary.get("source_turn_id") or "") if (data.get("source_turn_id") or reaction_window_summary.get("source_turn_id")) else "",
            "selection_type": str(data.get("selection_type") or data.get("highlight_type") or "text"),
            "page_number": self._positive_int(data.get("page_number")),
            "selected_text": normalize_pdf_text(data.get("selected_text")),
            "caption": normalize_pdf_text(data.get("caption")),
            "baseline_explanation": normalize_pdf_text(data.get("baseline_explanation")),
            "reaction_window_summary": self._safe_log_payload(reaction_window_summary),
            "support_cue": support_cue,
            "crop_available": bool(data.get("crop_available") or data.get("crop_image_available") or data.get("crop_url") or data.get("crop_image_path")),
            "user_question": normalize_pdf_text(data.get("user_question") or data.get("question")),
            "learning_state": learning_state,
            "paper_context": paper_context,
            "planner_input_summary": self._safe_log_payload(planner_input_summary),
            "recent_conversation": self._sanitize_strategy_recent_conversation(recent_conversation[-6:]),
            "trigger_context": trigger_context,
            "previous_strategy_id": str(data.get("previous_strategy_id") or data.get("selected_strategy_id") or ""),
            "previous_strategy_family": str(data.get("previous_strategy_family") or ""),
        }
        request["allowed_strategy_families"] = self._allowed_strategy_families_for_support_cue(support_cue) if support_cue else []
        request["planner_input_summary"] = self._strategy_planner_input_summary(request, planner_input_summary)
        return request

    def _strategy_planner_input_summary(self, request: dict[str, Any], provided: dict[str, Any] | None = None) -> dict[str, Any]:
        paper_context = request.get("paper_context") if isinstance(request.get("paper_context"), dict) else {}
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        recent_conversation = request.get("recent_conversation") if isinstance(request.get("recent_conversation"), list) else []
        selected_strategy = request.get("selected_strategy") if isinstance(request.get("selected_strategy"), dict) else {}
        summary = dict(provided or {})
        summary.update({
            "support_cue": request.get("support_cue") or reaction_summary.get("support_cue") or "",
            "allowed_strategy_families": list(request.get("allowed_strategy_families") or []),
            "selected_text_length": len(normalize_pdf_text(request.get("selected_text"))),
            "caption_length": len(normalize_pdf_text(request.get("caption"))),
            "baseline_explanation_length": len(normalize_pdf_text(request.get("baseline_explanation"))),
            "recent_conversation_count": len(recent_conversation),
            "previous_strategy_id": str(selected_strategy.get("strategy_id") or summary.get("previous_strategy_id") or ""),
            "passage_type": str(paper_context.get("passage_type") or "unknown"),
            "difficulty_hint": str(paper_context.get("difficulty_hint") or "unknown"),
            "reaction_window_duration": float(reaction_summary.get("duration_sec") or summary.get("reaction_window_duration_sec") or 0),
        })
        return self._safe_log_payload(summary)

    def _strategy_planning_context(self, request: dict[str, Any]) -> dict[str, Any]:
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        planner_summary = request.get("planner_input_summary") if isinstance(request.get("planner_input_summary"), dict) else {}
        learning_state = request.get("learning_state") if isinstance(request.get("learning_state"), dict) else {}
        recent_conversation = self._sanitize_strategy_recent_conversation(request.get("recent_conversation") if isinstance(request.get("recent_conversation"), list) else [])
        previous_strategy = self._latest_strategy_metadata(recent_conversation)
        support_cue = str(request.get("support_cue") or reaction_summary.get("support_cue") or "")
        avg_distribution = (
            reaction_summary.get("avg_distribution")
            if isinstance(reaction_summary.get("avg_distribution"), dict)
            else learning_state.get("distribution")
            if isinstance(learning_state.get("distribution"), dict)
            else {}
        )
        return {
            "selected_evidence": {
                "selection_type": str(request.get("selection_type") or "text"),
                "page_number": self._positive_int(request.get("page_number")),
                "selected_text": self._bounded_strategy_text(request.get("selected_text"), 2400),
                "caption": self._bounded_strategy_text(request.get("caption"), 1000),
                "crop_available": bool(request.get("crop_available")),
                "selection_note": self._bounded_strategy_text(request.get("user_question"), 600),
            },
            "paper_context": self._strategy_paper_context(request),
            "previous_explanation": {
                "source_turn_id": str(request.get("source_turn_id") or ""),
                "baseline_explanation": self._bounded_baseline_for_strategy(request.get("baseline_explanation")),
            },
            "reaction_context": {
                "reaction_window_summary": self._safe_log_payload(reaction_summary),
                "support_cue": support_cue,
                "support_cue_label": str(reaction_summary.get("support_cue_label") or planner_summary.get("support_cue_label") or support_cue or ""),
                "dominant_state": str(reaction_summary.get("dominant_state") or learning_state.get("academic_state") or ""),
                "secondary_state": str(reaction_summary.get("secondary_state") or ""),
                "avg_confidence": self._float_between(reaction_summary.get("avg_confidence") or learning_state.get("confidence"), 0.0, 1.0),
                "avg_distribution": self._safe_log_payload(avg_distribution),
                "trend": str(reaction_summary.get("trend") or learning_state.get("trend") or ""),
                "duration_sec": float(reaction_summary.get("duration_sec") or planner_summary.get("reaction_window_duration") or 0),
            },
            "strategy_constraints": {
                "allowed_strategy_families": list(request.get("allowed_strategy_families") or []),
                "previous_strategy_id": str(request.get("previous_strategy_id") or planner_summary.get("previous_strategy_id") or previous_strategy.get("strategy_id") or ""),
                "previous_strategy_family": str(request.get("previous_strategy_family") or planner_summary.get("previous_strategy_family") or previous_strategy.get("strategy_family") or ""),
            },
            "recent_conversation": recent_conversation,
        }

    def _strategy_paper_context(self, request: dict[str, Any]) -> dict[str, Any]:
        paper_context = request.get("paper_context") if isinstance(request.get("paper_context"), dict) else {}
        matched_block = self._context_item_text(
            paper_context.get("matched_block")
            or paper_context.get("matched_block_text")
            or request.get("matched_block")
            or ""
        )
        seen = {self._strategy_dedupe_key(matched_block)} if matched_block else set()
        nearby_context = self._bounded_context_texts(
            paper_context.get("nearby_context")
            or paper_context.get("nearby_useful_context")
            or paper_context.get("nearby_blocks")
            or [],
            limit=4,
            seen=seen,
        )
        rag_chunks = self._bounded_rag_chunks(
            paper_context.get("retrieved_rag_chunks")
            or paper_context.get("retrieved_chunks")
            or paper_context.get("global_rag_context")
            or paper_context.get("retrieved_blocks")
            or [],
            limit=5,
            seen=seen,
        )
        profile = paper_context.get("paper_profile") if isinstance(paper_context.get("paper_profile"), dict) else {}
        profile_summary = (
            paper_context.get("paper_profile_summary")
            or profile.get("summary")
            or profile.get("abstract")
            or profile.get("title")
            or ""
        )
        return {
            "matched_block": self._bounded_strategy_text(matched_block, 1600),
            "nearby_context": nearby_context,
            "retrieved_rag_chunks": rag_chunks,
            "paper_profile_summary": self._bounded_strategy_text(profile_summary, 1200),
            "passage_type": str(paper_context.get("passage_type") or "unknown"),
            "difficulty_hint": str(paper_context.get("difficulty_hint") or "unknown"),
        }

    def _bounded_context_texts(self, items: Any, *, limit: int, seen: set[str]) -> list[str]:
        output: list[str] = []
        source = items if isinstance(items, list) else [items] if items else []
        for item in source:
            text = self._context_item_text(item)
            key = self._strategy_dedupe_key(text)
            if not text or key in seen:
                continue
            seen.add(key)
            output.append(self._bounded_strategy_text(text, 1200))
            if len(output) >= limit:
                break
        return output

    def _bounded_rag_chunks(self, items: Any, *, limit: int, seen: set[str]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        source = items if isinstance(items, list) else [items] if items else []
        for item in source:
            text = self._context_item_text(item)
            key = self._strategy_dedupe_key(text)
            if not text or key in seen:
                continue
            seen.add(key)
            page_number = 0
            if isinstance(item, dict):
                page_number = item.get("page_number") or item.get("page") or 0
            chunk: dict[str, Any] = {
                "page_number": self._positive_int(page_number),
                "content": self._bounded_strategy_text(text, 1200),
            }
            if isinstance(item, dict) and item.get("score") is not None:
                chunk["score"] = self._float_between(item.get("score"), 0.0, 1.0)
            output.append(chunk)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _strategy_dedupe_key(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()[:500]

    def _context_item_text(self, item: Any) -> str:
        if isinstance(item, dict):
            for key in ("content", "markdown_content", "text", "block_text", "caption", "selected_text"):
                value = normalize_pdf_text(item.get(key))
                if value:
                    return value
            return normalize_pdf_text(json.dumps(self._safe_log_payload(item), ensure_ascii=False))
        return normalize_pdf_text(item)

    def _bounded_baseline_for_strategy(self, text: Any) -> str:
        baseline = normalize_pdf_text(text)
        if len(baseline) <= 1800:
            return baseline
        return f"{baseline[:1600].rstrip()}\n\n[Baseline explanation truncated for planner context; original length: {len(baseline)} characters.]"

    @staticmethod
    def _bounded_strategy_text(text: Any, limit: int) -> str:
        value = WebState._sanitize_prompt_text(normalize_pdf_text(text))
        if len(value) <= limit:
            return value
        return f"{value[: max(0, limit - 80)].rstrip()} [truncated; original length: {len(value)} characters]"

    def _sanitize_strategy_recent_conversation(self, recent_conversation: Any) -> list[dict[str, Any]]:
        if not isinstance(recent_conversation, list):
            return []
        keep_keys = {
            "role",
            "content",
            "turn_id",
            "conversation_turn_id",
            "turn_type",
            "strategy_id",
            "selected_strategy_id",
            "strategy_family",
            "pedagogical_move",
            "context_focus",
            "why_recommended",
        }
        sanitized: list[dict[str, Any]] = []
        for item in recent_conversation[-6:]:
            if not isinstance(item, dict):
                continue
            clean: dict[str, Any] = {}
            for key in keep_keys:
                if key in item and item.get(key) not in (None, ""):
                    clean[key] = self._bounded_strategy_text(item.get(key), 1800) if key == "content" else self._safe_log_payload(item.get(key))
            selected_strategy = item.get("selected_strategy") if isinstance(item.get("selected_strategy"), dict) else {}
            for source_key, target_key in (
                ("strategy_id", "strategy_id"),
                ("strategy_family", "strategy_family"),
                ("pedagogical_move", "pedagogical_move"),
                ("context_focus", "context_focus"),
                ("why_recommended", "why_recommended"),
            ):
                if target_key not in clean and selected_strategy.get(source_key):
                    clean[target_key] = self._safe_log_payload(selected_strategy.get(source_key))
            if clean.get("role") or clean.get("content"):
                clean["role"] = str(clean.get("role") or "assistant")
                clean["content"] = str(clean.get("content") or "")
                sanitized.append(clean)
        return sanitized

    @staticmethod
    def _latest_strategy_metadata(recent_conversation: list[dict[str, Any]]) -> dict[str, Any]:
        for item in reversed(recent_conversation):
            if item.get("strategy_id") or item.get("strategy_family"):
                return item
        return {}

    def _call_strategy_planner_llm(self, request: dict[str, Any]) -> dict[str, Any] | None:
        role = llm_config.role_config_from_env("strategy_planner_model")
        provider = str(role.get("provider") or "").strip().lower()
        model = str(role.get("model") or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        messages = self._strategy_planner_messages(request)
        if provider in {"openrouter", "openai_compatible"}:
            api_key = llm_config.provider_api_key_from_env(provider)
            base_url = (
                llm_config.DEFAULT_OPENROUTER_BASE_URL
                if provider == "openrouter"
                else llm_config.provider_base_url_from_env("openai_compatible")
            )
            if not api_key or not base_url or not model:
                return None
            return self._call_chat_completions_strategy_planner(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
            )
        if provider not in {"gemini", "google"}:
            return None
        api_key = llm_config.provider_api_key_from_env("gemini")
        if not api_key:
            return None
        prompt = self._messages_to_prompt_text(messages)
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.35,
                "response_mime_type": "application/json",
            },
        }
        http_request = urllib.request.Request(
            GEMINI_ENDPOINT_TEMPLATE.format(model=model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            text = self._gemini_strategy_text(response_payload)
            parsed = self._parse_strict_json_output(text)
            if isinstance(parsed, dict):
                parsed.setdefault("warnings", [])
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return None

    def _call_chat_completions_strategy_planner(
        self,
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        body = {
            "model": model,
            "messages": self._clean_snapshot_messages(messages),
            "temperature": 0.35,
        }
        http_request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            text = self._chat_completions_strategy_text(response_payload)
            parsed = self._parse_strict_json_output(text)
            if isinstance(parsed, dict):
                parsed.setdefault("warnings", [])
                parsed.setdefault("provider", provider)
                parsed.setdefault("model", model)
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _gemini_strategy_text(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") if isinstance(payload, dict) else []
        if not candidates:
            return ""
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else {}
        parts = content.get("parts") if isinstance(content, dict) else []
        return "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict)).strip()

    @staticmethod
    def _chat_completions_strategy_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") if isinstance(payload, dict) else []
        if not choices:
            return ""
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict)).strip()
        return ""

    def _strategy_planner_prompt(self, request: dict[str, Any]) -> str:
        return self._messages_to_prompt_text(self._strategy_planner_messages(request))

    def _strategy_planner_messages(self, request: dict[str, Any], context: dict[str, Any] | None = None) -> list[dict[str, str]]:
        strategy_context = context if isinstance(context, dict) else self._strategy_planning_context(request)
        system_prompt = """
You are a pedagogical strategy planner for an academic paper reading assistant.

You are given a structured context package containing selected evidence, paper context, previous explanation, recent learning signal, strategy constraints, and recent conversation. Use all of these to propose pedagogical strategies.

The context is rich on purpose. Do not ignore the paper context. Do not choose strategies solely from the support cue.
The support cue suggests the learner's possible support need.
The paper context determines the strategy focus.
The previous explanation tells you what the user has already seen.

The strategy should combine:
1. support need from reaction_context
2. pedagogical move from allowed_strategy_families
3. context focus from selected_evidence and paper_context
4. continuation logic from previous_explanation and recent_conversation

The learning-state signal is a noisy support cue, not a diagnosis.
Do not say the user is confused, bored, frustrated, or engaged.
Use neutral phrasing such as "clarification cue", "re-engagement cue", or "deepening cue".
Do not mention webcam, face, facial expression, or camera detection.
Do not include Chinese translations.

Only use the signal to decide what support style may be helpful. If confidence is low, make strategies less state-specific and more user-choice-oriented. Do not generate the final answer.

The rule layer provides allowed_strategy_families. Choose strategy_family only from that list. Do not invent a new strategy family.

You are not generating a topic title. You are generating a pedagogical support strategy.
A good strategy title should describe how the assistant will help the user reason about the passage, not merely name the passage topic.

Examples:
Bad: "The Tension Between AI and Authenticity"
Good: "Critique the core assumption about authenticity"
Bad: "DiaryMate Writing Flow"
Good: "Map the writing flow step by step"

Separate:
- pedagogical_move: the learning support action
- context_focus: the paper-specific content focus

Return strict JSON only with:
{
  "state_interpretation": {
    "support_need": "clarification | reassurance | re-engagement | deepening | neutral",
    "confidence_handling": "...",
    "context_reasoning": "...",
    "safety_note": "Affective signal used only as a support cue, not as diagnosis."
  },
  "candidates": [
    {
      "strategy_id": "...",
      "strategy_family": "...",
      "pedagogical_move": "...",
      "context_focus": "...",
      "title": "...",
      "short_description": "...",
      "why_recommended": "...",
      "prompt_instruction": "...",
      "expected_answer_shape": ["..."],
      "recommended": true,
      "recommended_score": 0.0
    }
  ],
  "planner_prompt_version": "reaction_strategy_planner_v2",
  "warnings": []
}
""".strip()
        return [
            {"role": "system", "content": self._sanitize_prompt_text(system_prompt)},
            {
                "role": "user",
                "content": self._sanitize_prompt_text(
                    "strategy_planning_context:\n" + json.dumps(strategy_context, ensure_ascii=False, indent=2)
                ),
            },
        ]

    def _valid_strategy_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not 3 <= len(candidates) <= 4:
            return False
        required = {"strategy_id", "title", "short_description", "why_recommended", "prompt_instruction", "expected_answer_shape"}
        for candidate in candidates:
            if not isinstance(candidate, dict) or not required.issubset(candidate):
                return False
        return True

    def _normalize_strategy_payload(self, payload: dict[str, Any], request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(payload or {})
        candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        normalized = self._normalize_strategy_candidates(candidates, request=request)
        payload["candidates"] = normalized
        return payload

    def _normalize_strategy_candidates(self, candidates: list[Any], request: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        allowed_families = list((request or {}).get("allowed_strategy_families") or [])
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            item = self._normalize_strategy_candidate(candidate, index, request=request, allowed_families=allowed_families)
            score = item.get("recommended_score")
            if score is None:
                score = max(0.5, 0.72 - index * 0.04)
            item["recommended_score"] = round(self._float_between(score, 0.0, 1.0), 2)
            normalized.append(item)
        if not normalized:
            return []
        best_index = max(range(len(normalized)), key=lambda index: (normalized[index]["recommended_score"], -index))
        for index, candidate in enumerate(normalized):
            candidate["recommended"] = index == best_index
        normalized.sort(key=lambda candidate: (not candidate.get("recommended"), -float(candidate.get("recommended_score") or 0)))
        return normalized

    def _normalize_strategy_candidate(
        self,
        candidate: dict[str, Any],
        index: int = 0,
        request: dict[str, Any] | None = None,
        allowed_families: list[str] | None = None,
    ) -> dict[str, Any]:
        item = dict(candidate)
        all_families = self._all_strategy_families()
        allowed = list(allowed_families or [])
        raw_family = str(item.get("strategy_family") or item.get("strategy_id") or "").strip().lower()
        family = raw_family if raw_family in all_families else ""
        if allowed and family not in allowed:
            family = allowed[min(index, len(allowed) - 1)]
            item["strategy_id"] = family
        elif family:
            item["strategy_id"] = str(item.get("strategy_id") or family)
        else:
            family = "custom_strategy"
            item["strategy_id"] = str(item.get("strategy_id") or family)
        item["strategy_family"] = family
        item.setdefault("pedagogical_move", self._pedagogical_move_for_family(family))
        item.setdefault("context_focus", self._context_focus_for_strategy(request or {}, item))
        item.setdefault("title", self._strategy_title_for_family(family, item.get("context_focus")))
        item.setdefault("short_description", self._strategy_description_for_family(family))
        item.setdefault("why_recommended", self._strategy_reason_for_family(family, request or {}))
        item.setdefault("prompt_instruction", self._strategy_instruction_for_family(family))
        item.setdefault("expected_answer_shape", self._strategy_shape_for_family(family))
        if not isinstance(item.get("expected_answer_shape"), list):
            item["expected_answer_shape"] = [str(item.get("expected_answer_shape") or "Explanation")]
        return item

    def _all_strategy_families(self) -> set[str]:
        families: set[str] = set()
        for values in self.STRATEGY_FAMILY.values():
            families.update(values)
        for values in self.SUPPORT_CUE_STRATEGY_FAMILIES.values():
            families.update(values)
        return families

    @staticmethod
    def _pedagogical_move_for_family(strategy_family: str) -> str:
        moves = {
            "input_process_output_map": "Map the method as input, process, and output",
            "step_by_step_breakdown": "Walk through the passage step by step",
            "concrete_example": "Anchor the idea in a concrete example",
            "define_key_terms": "Define the key terms first",
            "formula_intuition": "Build intuition for the formula",
            "mechanism_walkthrough": "Trace the mechanism from cause to outcome",
            "simplest_version_first": "Start with the simplest accurate version",
            "analogy_or_reframe": "Reframe the idea with a careful analogy",
            "one_small_next_step": "Focus on one small next step",
            "reduce_information_density": "Reduce the information density",
            "key_takeaway_first": "Start with the key takeaway",
            "one_sentence_takeaway": "Start with a one-sentence takeaway",
            "why_it_matters": "Connect the passage to why it matters",
            "make_it_relevant": "Make the passage relevant to the paper goal",
            "compare_with_familiar_method": "Compare with a familiar method",
            "quick_quiz": "Use a quick active-reading check",
            "deep_technical_explanation": "Deepen the technical explanation",
            "critique_assumptions": "Critique the core assumption",
            "connect_to_related_work": "Connect the passage to related work",
            "limitations_and_implications": "Explore limitations and implications",
            "compare_methods": "Compare the methods and trade-offs",
            "concise_explanation": "Give a concise grounded explanation",
            "structured_breakdown": "Break the passage into claim, evidence, and purpose",
            "example_based_explanation": "Explain through an illustrative example",
            "connect_to_paper_argument": "Connect the passage to the paper argument",
        }
        return moves.get(strategy_family, "Use a custom support strategy")

    @staticmethod
    def _strategy_description_for_family(strategy_family: str) -> str:
        descriptions = {
            "critique_assumptions": "Examine the assumption the passage relies on and how the paper complicates it.",
            "deep_technical_explanation": "Add technical depth while staying grounded in the selected passage.",
            "input_process_output_map": "Break the method into what goes in, what happens, and what comes out.",
            "step_by_step_breakdown": "Turn the passage into a small sequence of ordered reasoning steps.",
            "concrete_example": "Use one grounded example before returning to the paper evidence.",
            "define_key_terms": "Clarify specialized terms before explaining the full claim.",
            "simplest_version_first": "Start with the smallest accurate explanation, then add detail.",
            "why_it_matters": "Connect the selected passage to the paper's larger purpose.",
        }
        return descriptions.get(strategy_family, "Adapt the explanation style to the selected passage and recent support cue.")

    @staticmethod
    def _strategy_instruction_for_family(strategy_family: str) -> str:
        instructions = {
            "critique_assumptions": "Identify the core assumption, explain why it matters, and show how the paper challenges or depends on it.",
            "deep_technical_explanation": "Explain the technical mechanism, assumptions, and implications using only the provided paper evidence.",
            "input_process_output_map": "Explain the passage as Input, Process, Output, and one small example.",
            "step_by_step_breakdown": "Explain the passage as numbered steps tied to paper evidence.",
            "concrete_example": "Explain with one simple example, then connect it back to the selected evidence.",
            "define_key_terms": "Define key terms briefly, then explain how they fit together in the passage.",
            "simplest_version_first": "Give the simplest accurate version first, then add only necessary detail.",
            "why_it_matters": "State the takeaway, then explain why it matters for the paper's argument.",
        }
        return instructions.get(strategy_family, "Explain the passage with this pedagogical move while staying grounded in the paper context.")

    @staticmethod
    def _strategy_shape_for_family(strategy_family: str) -> list[str]:
        shapes = {
            "critique_assumptions": ["Core assumption", "Why it matters", "How the paper challenges it", "Implications"],
            "deep_technical_explanation": ["Technical reading", "Assumptions", "Implications"],
            "input_process_output_map": ["Core intuition", "Input", "Process", "Output", "Mini example"],
            "step_by_step_breakdown": ["Main point", "Step 1", "Step 2", "Step 3", "Why it matters"],
            "concrete_example": ["Plain-language idea", "Mini example", "Back to the paper"],
            "define_key_terms": ["Key terms", "How they connect", "Paper-specific meaning"],
            "simplest_version_first": ["Simplest version", "Necessary detail", "One small next step"],
            "why_it_matters": ["Takeaway", "Connection to paper", "Why it matters"],
        }
        return shapes.get(strategy_family, ["Pedagogical move", "Passage focus", "Grounded explanation"])

    def _context_focus_for_strategy(self, request: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
        explicit = normalize_pdf_text((candidate or {}).get("context_focus"))
        if explicit:
            return explicit
        selected = normalize_pdf_text(request.get("selected_text"))
        caption = normalize_pdf_text(request.get("caption"))
        baseline = normalize_pdf_text(request.get("baseline_explanation"))
        paper_context = request.get("paper_context") if isinstance(request.get("paper_context"), dict) else {}
        passage_type = str(paper_context.get("passage_type") or request.get("selection_type") or "passage").replace("_", " ")
        source = selected or caption or baseline
        if source:
            return source[:96].rstrip(" .,;:")
        return f"the selected {passage_type}"

    def _strategy_title_for_family(self, strategy_family: str, context_focus: Any) -> str:
        move = self._pedagogical_move_for_family(strategy_family)
        focus = normalize_pdf_text(context_focus)
        if not focus:
            return move
        short_focus = focus[:56].rstrip(" .,;:")
        if strategy_family == "critique_assumptions":
            return f"{move} about {short_focus.lower()}"
        if strategy_family in {"step_by_step_breakdown", "input_process_output_map", "mechanism_walkthrough"}:
            return f"{move} for {short_focus.lower()}"
        return f"{move}: {short_focus}"

    def _strategy_reason_for_family(self, strategy_family: str, request: dict[str, Any]) -> str:
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        trigger_reason = normalize_pdf_text(reaction_summary.get("trigger_reason"))
        if trigger_reason:
            return f"{trigger_reason} This strategy uses {self._pedagogical_move_for_family(strategy_family).lower()} for the selected passage."
        support_label = normalize_pdf_text(reaction_summary.get("support_cue_label") or request.get("support_cue") or "support cue")
        return f"The current {support_label.lower()} suggests this pedagogical move may be useful for the selected passage."

    def _heuristic_strategy_candidates(self, request: dict[str, Any]) -> dict[str, Any]:
        learning_state = request.get("learning_state") if isinstance(request.get("learning_state"), dict) else {}
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        support_cue = str(request.get("support_cue") or reaction_summary.get("support_cue") or "").lower()
        state = str(reaction_summary.get("dominant_state") or learning_state.get("academic_state") or learning_state.get("state") or "neutral").lower()
        confidence = self._float_between(learning_state.get("confidence"), 0.0, 1.0)
        if reaction_summary.get("avg_confidence") is not None:
            confidence = self._float_between(reaction_summary.get("avg_confidence"), 0.0, 1.0)
        if confidence < 0.55 or state not in self.ACADEMIC_STATES:
            state = "neutral"
        paper_context = request.get("paper_context") if isinstance(request.get("paper_context"), dict) else {}
        passage_type = str(paper_context.get("passage_type") or "unknown").lower()
        difficulty_hint = str(paper_context.get("difficulty_hint") or "unknown").lower()
        selection_type = str(request.get("selection_type") or "text").lower()
        strategy_ids = self._strategy_ids_for_context(state, passage_type, difficulty_hint, selection_type, support_cue)
        candidates = [
            self._strategy_candidate(strategy_id, state, passage_type, difficulty_hint, request, index)
            for index, strategy_id in enumerate(strategy_ids[:3])
        ]
        support_need = {
            "confusion": "clarification",
            "frustration": "reassurance",
            "boredom": "re-engagement",
            "engagement": "deepening",
        }.get(state, "neutral")
        return {
            "planner_mode": "heuristic",
            "state_interpretation": {
                "support_need": support_need,
                "confidence_handling": "Low confidence signals are treated as neutral suggestions." if state == "neutral" else "Signal is used only to choose a support style.",
                "context_reasoning": self._context_reasoning(support_cue or state, passage_type, difficulty_hint, selection_type),
                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
            },
            "candidates": candidates,
            "planner_prompt_version": "reaction_strategy_planner_v2" if reaction_summary else "strategy_planner_v1",
            "support_cue": support_cue,
            "reaction_window_summary": reaction_summary,
            "warnings": ["LLM strategy planner unavailable or invalid; used heuristic fallback."],
        }

    def _strategy_ids_for_context(self, state: str, passage_type: str, difficulty_hint: str, selection_type: str, support_cue: str = "") -> list[str]:
        support_map = {
            "sustained_clarification": ["step_by_step_breakdown", "define_key_terms", "concrete_example", "input_process_output_map", "mechanism_walkthrough", "formula_intuition"],
            "reduce_load": ["simplest_version_first", "one_small_next_step", "analogy_or_reframe", "reduce_information_density", "key_takeaway_first"],
            "re_engagement": ["why_it_matters", "one_sentence_takeaway", "make_it_relevant", "compare_with_familiar_method", "quick_quiz"],
            "deepening": ["deep_technical_explanation", "critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"],
            "clarify_and_reengage": ["concise_explanation", "concrete_example", "why_it_matters", "step_by_step_breakdown", "compare_with_familiar_method"],
            "gentle_clarification": ["simplest_version_first", "one_small_next_step", "define_key_terms", "analogy_or_reframe", "concrete_example"],
            "neutral_or_uncertain": ["concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"],
            "neutral": ["concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"],
        }
        if support_cue in support_map:
            return support_map[support_cue]
        if state == "confusion" and ("formula" in passage_type or "formula" in difficulty_hint):
            return ["formula_intuition", "define_key_terms", "concrete_example"]
        if state == "confusion" and any(token in passage_type + difficulty_hint for token in ("method", "process", "mechanism", "multi_step")):
            return ["input_process_output_map", "step_by_step_breakdown", "concrete_example"]
        if state == "confusion" and selection_type == "area":
            return ["mechanism_walkthrough", "concrete_example", "define_key_terms"]
        if state == "frustration" and any(token in difficulty_hint for token in ("dense_theory", "unclear", "technical")):
            return ["simplest_version_first", "analogy_or_reframe", "one_small_next_step"]
        if state == "boredom" and any(token in passage_type for token in ("discussion", "result")):
            return ["why_it_matters", "compare_with_familiar_method", "quick_quiz"]
        if state == "engagement" and any(token in passage_type for token in ("method", "result", "figure")):
            return ["deep_technical_explanation", "critique_assumptions", "limitations_and_implications"]
        if state in self.STRATEGY_FAMILY:
            return list(self.STRATEGY_FAMILY[state][:3])
        return list(self.STRATEGY_FAMILY["neutral"][:3])

    def _strategy_candidate(
        self,
        strategy_id: str,
        state: str,
        passage_type: str,
        difficulty_hint: str,
        request: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        passage_label = self._friendly_passage_label(passage_type, request.get("selection_type"))
        difficulty_label = self._friendly_difficulty_label(difficulty_hint)
        selected_length = len(str(request.get("selected_text") or ""))
        templates = {
            "input_process_output_map": (
                f"Map this {passage_label} as input -> process -> output",
                "Break the selected method into what goes in, what happens, and what comes out.",
                "Explain the passage as Input, Process, Output, and one small example.",
                ["Core intuition", "Input", "Process", "Output", "Mini example"],
            ),
            "step_by_step_breakdown": (
                f"Walk through the {passage_label} step by step",
                "Turn the dense passage into a short sequence of moves.",
                "Explain the passage as numbered steps, keeping each step tied to paper evidence.",
                ["Main point", "Step 1", "Step 2", "Step 3", "Why it matters"],
            ),
            "concrete_example": (
                f"Use a concrete example for this {passage_label}",
                "Anchor the explanation in a small example before returning to the paper.",
                "Explain the passage using one simple example, then connect it back to the selected evidence.",
                ["Plain-language idea", "Mini example", "Back to the paper"],
            ),
            "define_key_terms": (
                f"Define the key terms in this {passage_label}",
                "Clarify specialized terms before explaining the full claim.",
                "List key terms, define them briefly, then explain how they fit together.",
                ["Key terms", "How they connect", "Paper-specific meaning"],
            ),
            "formula_intuition": (
                "Build intuition for the formula",
                "Explain what each symbol or operation is doing before the formal reading.",
                "Explain the formula intuitively, define symbols only when supported, and give one small numeric or conceptual example.",
                ["What the formula is for", "Term meanings", "Intuition", "Small example"],
            ),
            "mechanism_walkthrough": (
                f"Trace the mechanism in this {passage_label}",
                "Follow the causal or procedural chain shown by the selected passage or figure.",
                "Explain the mechanism from starting condition to outcome, naming uncertainties when context is missing.",
                ["Starting point", "Mechanism", "Outcome", "Uncertainties"],
            ),
            "simplest_version_first": (
                "Start with the simplest version",
                "Reduce the passage to the smallest accurate statement before adding detail.",
                "Give the simplest accurate version first, then add only the necessary detail.",
                ["Simplest version", "Necessary detail", "One small next step"],
            ),
            "analogy_or_reframe": (
                f"Reframe the {passage_label} with an analogy",
                "Use a careful analogy to lower the density without changing the paper's claim.",
                "Use one analogy or reframe, then state exactly where the analogy stops.",
                ["Reframe", "Paper mapping", "Limits of the analogy"],
            ),
            "one_small_next_step": (
                "Focus on one small next step",
                "Identify the next useful reading move instead of expanding everything at once.",
                "Explain only the next important idea and one action the reader can take.",
                ["Next idea", "Why it matters", "Small reading move"],
            ),
            "reduce_information_density": (
                "Reduce the information density",
                "Separate the essential claim from supporting detail.",
                "Give the essential version first, then add one layer of detail at a time.",
                ["Essential claim", "Needed detail", "What to read next"],
            ),
            "one_sentence_takeaway": (
                f"Start with a one-sentence takeaway",
                "Condense the previous explanation into the central point before expanding.",
                "Give one sentence, then one short paragraph connecting it back to the paper.",
                ["One-sentence takeaway", "Paper connection", "Optional detail"],
            ),
            "why_it_matters": (
                f"Explain why this {passage_label} matters",
                "Connect the selected passage to the paper's larger argument.",
                "State the takeaway, then explain why it matters for the method, result, or argument.",
                ["Takeaway", "Connection to paper", "Why it matters"],
            ),
            "compare_with_familiar_method": (
                "Compare with a familiar method",
                "Make the passage easier to place by contrasting it with a common baseline.",
                "Compare the passage with a familiar method or baseline only where the provided context supports it.",
                ["This paper's move", "Familiar contrast", "Important difference"],
            ),
            "quick_quiz": (
                f"Use a quick check on this {passage_label}",
                "Turn the passage into a short active-reading check.",
                "Explain briefly, then ask one grounded check question with the answer.",
                ["Brief explanation", "Check question", "Answer"],
            ),
            "deep_technical_explanation": (
                f"Give a deeper technical read of this {passage_label}",
                "Use the current passage and context for a more detailed technical explanation.",
                "Explain technical assumptions, mechanism, and implications without adding unsupported facts.",
                ["Technical reading", "Assumptions", "Implications"],
            ),
            "critique_assumptions": (
                "Examine the assumptions",
                "Look at what the selected passage seems to rely on.",
                "Identify assumptions that are visible in the provided context and separate them from speculation.",
                ["Visible assumptions", "Why they matter", "What remains unknown"],
            ),
            "limitations_and_implications": (
                "Look at limitations and implications",
                "Move from explanation into what the passage enables or leaves open.",
                "Explain the passage, then discuss supported limitations and implications.",
                ["Explanation", "Limitations", "Implications"],
            ),
            "concise_explanation": (
                f"Get a concise explanation of this {passage_label}",
                "Keep the answer short and evidence-focused.",
                "Explain the passage in a concise, grounded way.",
                ["Main idea", "Paper context", "Why it matters"],
            ),
            "structured_breakdown": (
                f"Use a structured breakdown for this {passage_label}",
                "Organize the explanation without assuming a specific support need.",
                "Break the passage into claim, evidence, and purpose.",
                ["Claim", "Evidence", "Purpose"],
            ),
            "example_based_explanation": (
                f"Try an example-based explanation for this {passage_label}",
                "Use an example while keeping the factual claims grounded.",
                "Explain with one example and clearly mark it as illustrative.",
                ["Paper idea", "Illustrative example", "Return to evidence"],
            ),
            "connect_to_paper_argument": (
                "Connect this to the paper argument",
                "Show how the selected passage fits the paper's broader point.",
                "Explain the passage and connect it to the paper argument using available context.",
                ["Passage meaning", "Connection", "Open context gaps"],
            ),
        }
        title, description, instruction, shape = templates.get(strategy_id, templates["structured_breakdown"])
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        trigger_reason = str(reaction_summary.get("trigger_reason") or "")
        support_label = str(reaction_summary.get("support_cue_label") or "support cue")
        if trigger_reason:
            why = f"{trigger_reason} This strategy may help by adapting the previous explanation with {description[0].lower() + description[1:]}"
        else:
            why = (
                f"This may help with a {difficulty_label} {passage_label}. "
                f"The current {support_label.lower()} suggests this support style could be useful for the selected passage."
            )
        if selected_length < 40 and request.get("selection_type") == "area":
            why = f"This strategy uses the area selection and caption context to keep the explanation anchored."
        context_focus = self._context_focus_for_strategy(request, {"strategy_family": strategy_id})
        return {
            "strategy_id": strategy_id,
            "strategy_family": strategy_id,
            "pedagogical_move": self._pedagogical_move_for_family(strategy_id),
            "context_focus": context_focus,
            "title": title,
            "short_description": description,
            "why_recommended": why,
            "prompt_instruction": instruction,
            "expected_answer_shape": shape,
            "recommended": index == 0,
            "recommended_score": round(max(0.58, 0.88 - index * 0.08), 2),
        }

    @staticmethod
    def _friendly_passage_label(passage_type: str, selection_type: Any) -> str:
        if str(selection_type or "").lower() == "area":
            return "figure or visual area"
        normalized = str(passage_type or "").replace("_", " ").replace("-", " ").strip().lower()
        if not normalized or normalized == "unknown":
            return "passage"
        if "method" in normalized or "process" in normalized:
            return "method passage"
        if "formula" in normalized:
            return "formula"
        if "result" in normalized:
            return "result passage"
        if "definition" in normalized:
            return "definition"
        if "discussion" in normalized:
            return "discussion passage"
        return normalized

    @staticmethod
    def _friendly_difficulty_label(difficulty_hint: str) -> str:
        normalized = str(difficulty_hint or "").replace("_", " ").replace("-", " ").strip().lower()
        return normalized if normalized and normalized != "unknown" else "dense"

    @staticmethod
    def _context_reasoning(state: str, passage_type: str, difficulty_hint: str, selection_type: str) -> str:
        return (
            f"Strategies are based on a {selection_type or 'text'} selection, "
            f"passage type {passage_type or 'unknown'}, difficulty hint {difficulty_hint or 'unknown'}, "
            f"and support cue {state}."
        )

    def _with_strategy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        reaction_summary = payload.get("reaction_window_summary") if isinstance(payload.get("reaction_window_summary"), dict) else {}
        support_cue = str(payload.get("support_cue") or reaction_summary.get("support_cue") or "")
        payload["allowed_strategy_families"] = self._allowed_strategy_families_for_support_cue(support_cue) if support_cue else []
        selected_strategy = payload.get("selected_strategy") if isinstance(payload.get("selected_strategy"), dict) else {}
        if selected_strategy:
            payload["selected_strategy"] = self._safe_log_payload(
                self._normalize_strategy_candidate(selected_strategy, 0, request=payload, allowed_families=payload["allowed_strategy_families"])
            )
            payload["selected_strategy_id"] = str(payload.get("selected_strategy_id") or selected_strategy.get("strategy_id") or "")
            payload["selected_strategy_id"] = str(payload["selected_strategy"].get("strategy_id") or payload["selected_strategy_id"])
        if isinstance(payload.get("learning_state"), dict):
            payload["learning_state"] = self._safe_log_payload(payload["learning_state"])
        if isinstance(payload.get("reaction_window_summary"), dict):
            payload["reaction_window_summary"] = self._safe_log_payload(payload["reaction_window_summary"])
            payload["support_cue"] = str(payload.get("support_cue") or payload["reaction_window_summary"].get("support_cue") or "")
            payload["support_cue_label"] = str(payload.get("support_cue_label") or payload["reaction_window_summary"].get("support_cue_label") or "")
            payload["source_turn_id"] = str(payload.get("source_turn_id") or payload["reaction_window_summary"].get("source_turn_id") or "")
            payload["face_detection_summary"] = payload["reaction_window_summary"].get("face_detection_summary") if isinstance(payload["reaction_window_summary"].get("face_detection_summary"), dict) else {}
        if isinstance(payload.get("strategy_candidates"), list):
            payload["strategy_candidates"] = self._normalize_strategy_candidates([
                self._safe_log_payload(candidate) for candidate in payload["strategy_candidates"] if isinstance(candidate, dict)
            ], request=payload)[:4]
        if isinstance(payload.get("trigger_context"), dict):
            payload["trigger_context"] = self._safe_log_payload(payload["trigger_context"])
        if isinstance(payload.get("planner_input_summary"), dict):
            payload["planner_input_summary"] = self._safe_log_payload(payload["planner_input_summary"])
        if payload.get("selected_strategy"):
            payload["strategy_reason"] = str(payload.get("strategy_reason") or payload["selected_strategy"].get("why_recommended") or "")
            payload["planner_prompt_version"] = str(payload.get("planner_prompt_version") or "reaction_strategy_planner_v2")
        return payload

    @staticmethod
    def _apply_strategy_thread_metadata(thread: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("session_id"):
            thread["session_id"] = str(payload.get("session_id"))
        if isinstance(payload.get("learning_state"), dict) and payload["learning_state"]:
            thread["learning_state_snapshot"] = payload["learning_state"]
        if isinstance(payload.get("strategy_candidates"), list):
            thread["strategy_candidates"] = payload["strategy_candidates"]
        if payload.get("selected_strategy_id"):
            thread["selected_strategy_id"] = str(payload.get("selected_strategy_id"))
        if isinstance(payload.get("selected_strategy"), dict):
            thread["selected_strategy"] = payload["selected_strategy"]
        if isinstance(payload.get("trigger_context"), dict):
            thread["trigger_context"] = payload["trigger_context"]
        if payload.get("planner_mode"):
            thread["planner_mode"] = str(payload.get("planner_mode"))
        if isinstance(payload.get("reaction_window_summary"), dict) and payload["reaction_window_summary"]:
            thread["reaction_window_summary"] = payload["reaction_window_summary"]
        if payload.get("support_cue"):
            thread["support_cue"] = str(payload.get("support_cue"))
        if payload.get("support_cue_label"):
            thread["support_cue_label"] = str(payload.get("support_cue_label"))
        if payload.get("planner_prompt_version"):
            thread["planner_prompt_version"] = str(payload.get("planner_prompt_version"))
        if payload.get("strategy_reason"):
            thread["strategy_reason"] = str(payload.get("strategy_reason"))
        if isinstance(payload.get("planner_input_summary"), dict):
            thread["planner_input_summary"] = payload["planner_input_summary"]

    @staticmethod
    def _assistant_strategy_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        selected_strategy = payload.get("selected_strategy") if isinstance(payload.get("selected_strategy"), dict) else {}
        learning_state = payload.get("learning_state") if isinstance(payload.get("learning_state"), dict) else {}
        trigger_context = payload.get("trigger_context") if isinstance(payload.get("trigger_context"), dict) else {}
        reaction_summary = payload.get("reaction_window_summary") if isinstance(payload.get("reaction_window_summary"), dict) else {}
        turn_type = "follow_up" if payload.get("follow_up_question") else "strategy_reexplanation" if selected_strategy else "baseline_explanation"
        if not selected_strategy and not learning_state and not trigger_context and not reaction_summary:
            return {"turn_type": turn_type}
        return {
            "turn_type": turn_type,
            "source_turn_id": str(payload.get("source_turn_id") or reaction_summary.get("source_turn_id") or ""),
            "reaction_window_summary": reaction_summary,
            "support_cue": str(payload.get("support_cue") or reaction_summary.get("support_cue") or ""),
            "support_cue_label": str(payload.get("support_cue_label") or reaction_summary.get("support_cue_label") or ""),
            "strategy_id": str(payload.get("selected_strategy_id") or selected_strategy.get("strategy_id") or ""),
            "strategy_family": str(selected_strategy.get("strategy_family") or selected_strategy.get("strategy_id") or ""),
            "pedagogical_move": str(selected_strategy.get("pedagogical_move") or ""),
            "context_focus": str(selected_strategy.get("context_focus") or ""),
            "strategy_title": str(selected_strategy.get("title") or ""),
            "strategy_short_description": str(selected_strategy.get("short_description") or ""),
            "why_recommended": str(selected_strategy.get("why_recommended") or ""),
            "strategy_reason": str(payload.get("strategy_reason") or selected_strategy.get("why_recommended") or ""),
            "learning_state_snapshot": learning_state,
            "trigger_context": trigger_context,
            "planner_mode": str(payload.get("planner_mode") or ""),
            "planner_prompt_version": str(payload.get("planner_prompt_version") or ("reaction_strategy_planner_v2" if selected_strategy else "")),
            "face_detection_summary": payload.get("face_detection_summary") if isinstance(payload.get("face_detection_summary"), dict) else reaction_summary.get("face_detection_summary") if isinstance(reaction_summary.get("face_detection_summary"), dict) else {},
            "planner_input_summary": payload.get("planner_input_summary") if isinstance(payload.get("planner_input_summary"), dict) else {},
        }

    def _new_turn_id(self, value: Any = None) -> str:
        raw = str(value or "").strip()
        if raw:
            return self._safe_file_id(raw)
        return f"turn_{uuid.uuid4().hex}"

    def _strategy_log_extra(self, payload: dict[str, Any], result: dict[str, Any] | None = None) -> dict[str, Any]:
        learning_state = payload.get("learning_state") if isinstance(payload.get("learning_state"), dict) else {}
        selected_strategy = payload.get("selected_strategy") if isinstance(payload.get("selected_strategy"), dict) else {}
        trigger_context = payload.get("trigger_context") if isinstance(payload.get("trigger_context"), dict) else {}
        return {
            "session_id": payload.get("session_id"),
            "strategy_candidates": payload.get("strategy_candidates") if isinstance(payload.get("strategy_candidates"), list) else [],
            "selected_strategy_id": payload.get("selected_strategy_id"),
            "selected_strategy_title": selected_strategy.get("title") or "",
            "academic_state": learning_state.get("academic_state") or learning_state.get("state") or "",
            "confidence": learning_state.get("confidence"),
            "distribution": learning_state.get("distribution") if isinstance(learning_state.get("distribution"), dict) else {},
            "trend": learning_state.get("trend"),
            "duration_sec": learning_state.get("duration_sec"),
            "intensity": learning_state.get("intensity"),
            "trigger_reason": trigger_context.get("trigger_reason") or "",
            "passage_type": payload.get("passage_type") or (payload.get("paper_context") or {}).get("passage_type") if isinstance(payload.get("paper_context"), dict) else "",
            "difficulty_hint": payload.get("difficulty_hint") or (payload.get("paper_context") or {}).get("difficulty_hint") if isinstance(payload.get("paper_context"), dict) else "",
            "selected_text_length": len(normalize_pdf_text(payload.get("selected_text"))),
            "provider": (result or {}).get("provider"),
            "model": (result or {}).get("model"),
        }

    def _log_strategy_event(
        self,
        document_id: str,
        event_type: str,
        request: dict[str, Any],
        planner_payload: dict[str, Any],
        success: bool,
        error: str | None = None,
    ) -> None:
        self._log_interaction(
            document_id,
            event_type,
            highlight_id=request.get("highlight_id"),
            selection_type=request.get("selection_type"),
            session_id=request.get("session_id"),
            provider="strategy_planner",
            model=str(planner_payload.get("planner_mode") or "heuristic"),
            success=success,
            error=error,
            extra=self._strategy_event_payload(event_type, document_id, request, planner_payload, success=success, error=error),
        )

    def _strategy_event_payload(
        self,
        event_type: str,
        document_id: str,
        request: dict[str, Any],
        planner_payload: dict[str, Any],
        success: bool,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        learning_state = request.get("learning_state") if isinstance(request.get("learning_state"), dict) else {}
        trigger_context = request.get("trigger_context") if isinstance(request.get("trigger_context"), dict) else {}
        paper_context = request.get("paper_context") if isinstance(request.get("paper_context"), dict) else {}
        selected_strategy = request.get("selected_strategy") if isinstance(request.get("selected_strategy"), dict) else {}
        reaction_summary = request.get("reaction_window_summary") if isinstance(request.get("reaction_window_summary"), dict) else {}
        selected_text = normalize_pdf_text(request.get("selected_text"))
        return self._safe_log_payload(
            {
                "event_type": event_type,
                "session_id": request.get("session_id"),
                "document_id": document_id,
                "highlight_id": request.get("highlight_id"),
                "source_turn_id": request.get("source_turn_id") or reaction_summary.get("source_turn_id") or "",
                "reaction_window_summary": reaction_summary,
                "support_cue": request.get("support_cue") or reaction_summary.get("support_cue") or "",
                "support_cue_label": reaction_summary.get("support_cue_label") or "",
                "strategy_candidates": planner_payload.get("candidates") or request.get("strategy_candidates") or [],
                "selected_strategy_id": request.get("selected_strategy_id") or selected_strategy.get("strategy_id") or "",
                "selected_strategy_title": selected_strategy.get("title") or "",
                "selected_strategy_family": selected_strategy.get("strategy_family") or selected_strategy.get("strategy_id") or "",
                "selected_pedagogical_move": selected_strategy.get("pedagogical_move") or "",
                "selected_context_focus": selected_strategy.get("context_focus") or "",
                "planner_mode": planner_payload.get("planner_mode") or "",
                "planner_prompt_version": planner_payload.get("planner_prompt_version") or request.get("planner_prompt_version") or "",
                "academic_state": learning_state.get("academic_state") or learning_state.get("state") or "",
                "confidence": learning_state.get("confidence"),
                "distribution": learning_state.get("distribution") if isinstance(learning_state.get("distribution"), dict) else {},
                "trend": learning_state.get("trend"),
                "duration_sec": learning_state.get("duration_sec"),
                "intensity": learning_state.get("intensity"),
                "trigger_reason": trigger_context.get("trigger_reason") or "",
                "passage_type": paper_context.get("passage_type") or request.get("passage_type") or "",
                "difficulty_hint": paper_context.get("difficulty_hint") or request.get("difficulty_hint") or "",
                "selected_text_length": len(selected_text),
                "provider": (result or {}).get("provider"),
                "model": (result or {}).get("model"),
                "success": success,
                "failure": not success,
                "error": error or (result or {}).get("error") or "",
            }
        )

    @staticmethod
    def _float_between(value: Any, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = minimum
        return max(minimum, min(number, maximum))

    @staticmethod
    def _iso_timestamp(value: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))

    @staticmethod
    def _timestamp_from_iso(value: str) -> float | None:
        try:
            return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError):
            return None

    def _highlights_path(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "highlights" / "highlights.json"

    def _load_highlights(self, document_id: str) -> list[dict[str, Any]]:
        path = self._highlights_path(document_id)
        payload = self._read_json(path, {"highlights": []})
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("highlights"), list):
            return payload["highlights"]
        return []

    def _highlight_count(self, document_id: str) -> int:
        return len(self._load_highlights(document_id))

    def _thread_dir(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "threads"

    def _thread_path(self, document_id: str, highlight_id: str) -> Path:
        return self._thread_dir(document_id) / f"{self._safe_file_id(highlight_id)}.json"

    def _load_thread(self, document_id: str, highlight_id: str) -> dict[str, Any]:
        path = self._thread_path(document_id, highlight_id)
        payload = self._read_json(path, {})
        if not isinstance(payload, dict) or not payload:
            now = time.time()
            return {
                "document_id": document_id,
                "highlight_id": highlight_id,
                "created_at": now,
                "updated_at": now,
                "selection_snapshot": {},
                "session_id": "",
                "learning_state_snapshot": {},
                "strategy_candidates": [],
                "selected_strategy_id": "",
                "selected_strategy": {},
                "trigger_context": {},
                "reaction_window_summary": {},
                "turn_metadata": {},
                "support_cue": "",
                "support_cue_label": "",
                "planner_mode": "",
                "planner_prompt_version": "",
                "planner_input_summary": {},
                "messages": [],
            }
        payload.setdefault("document_id", document_id)
        payload.setdefault("highlight_id", highlight_id)
        payload.setdefault("selection_snapshot", {})
        payload.setdefault("session_id", "")
        payload.setdefault("learning_state_snapshot", {})
        payload.setdefault("strategy_candidates", [])
        payload.setdefault("selected_strategy_id", "")
        payload.setdefault("selected_strategy", {})
        payload.setdefault("trigger_context", {})
        payload.setdefault("reaction_window_summary", {})
        payload.setdefault("turn_metadata", {})
        payload["turn_metadata"] = self._safe_turn_metadata(payload.get("turn_metadata"))
        payload.setdefault("support_cue", "")
        payload.setdefault("support_cue_label", "")
        payload.setdefault("planner_mode", "")
        payload.setdefault("planner_prompt_version", "")
        payload.setdefault("planner_input_summary", {})
        payload.setdefault("messages", [])
        payload.setdefault("created_at", time.time())
        payload.setdefault("updated_at", time.time())
        return payload

    def _thread_from_data(self, document_id: str, highlight_id: str, data: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        normalized_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip() or "user"
            normalized_messages.append(
                {
                    **message,
                    "role": role,
                    "content": str(message.get("content") or ""),
                    "created_at": message.get("created_at") or now,
                }
            )
        return {
            "document_id": document_id,
            "highlight_id": highlight_id,
            "created_at": data.get("created_at") or now,
            "updated_at": now,
            "selection_snapshot": self._safe_selection_snapshot(data.get("selection_snapshot") if isinstance(data.get("selection_snapshot"), dict) else {}),
            "session_id": str(data.get("session_id") or ""),
            "learning_state_snapshot": data.get("learning_state_snapshot") if isinstance(data.get("learning_state_snapshot"), dict) else data.get("learning_state") if isinstance(data.get("learning_state"), dict) else {},
            "strategy_candidates": data.get("strategy_candidates") if isinstance(data.get("strategy_candidates"), list) else [],
            "selected_strategy_id": str(data.get("selected_strategy_id") or ""),
            "selected_strategy": data.get("selected_strategy") if isinstance(data.get("selected_strategy"), dict) else {},
            "trigger_context": data.get("trigger_context") if isinstance(data.get("trigger_context"), dict) else {},
            "reaction_window_summary": data.get("reaction_window_summary") if isinstance(data.get("reaction_window_summary"), dict) else {},
            "turn_metadata": self._safe_turn_metadata(data.get("turn_metadata")),
            "support_cue": str(data.get("support_cue") or ""),
            "support_cue_label": str(data.get("support_cue_label") or ""),
            "planner_mode": str(data.get("planner_mode") or ""),
            "planner_prompt_version": str(data.get("planner_prompt_version") or ""),
            "planner_input_summary": data.get("planner_input_summary") if isinstance(data.get("planner_input_summary"), dict) else {},
            "messages": normalized_messages,
        }

    def _write_thread(self, document_id: str, highlight_id: str, thread: dict[str, Any]) -> None:
        path = self._thread_path(document_id, highlight_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(thread, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _thread_has_messages(thread: dict[str, Any] | None) -> bool:
        if not isinstance(thread, dict):
            return False
        messages = thread.get("messages") if isinstance(thread.get("messages"), list) else []
        return any(normalize_pdf_text(message.get("content") if isinstance(message, dict) else "") for message in messages)

    def _thread_count(self, document_id: str) -> int:
        thread_dir = self._thread_dir(document_id)
        if not thread_dir.exists():
            return 0
        count = 0
        for path in thread_dir.glob("*.json"):
            if not path.is_file():
                continue
            payload = self._read_json(path, {})
            if isinstance(payload, dict) and isinstance(payload.get("messages"), list) and payload["messages"]:
                count += 1
        return count

    def _save_document_crop(self, document_id: str, highlight_id: str, image_data: str) -> Path | None:
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", image_data, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        ext = "jpg" if match.group(1).lower() in {"jpg", "jpeg"} else "png"
        try:
            raw = base64.b64decode(match.group(2), validate=False)
        except (binascii.Error, ValueError):
            return None
        crop_dir = self._document_dir(document_id) / "highlights" / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        path = (crop_dir / f"{self._safe_file_id(highlight_id)}.{ext}").resolve()
        root = self._document_dir(document_id).resolve()
        if root not in path.parents:
            raise ValueError("Refusing to save cropped image outside the document folder.")
        path.write_bytes(raw)
        return path

    def library_crop_path(self, document_id: str, highlight_id: str) -> Path:
        document_id = self._safe_document_id(document_id)
        highlight_id = self._safe_file_id(highlight_id)
        if not self._record_for_document(document_id):
            raise KeyError(f"Unknown document_id: {document_id}")
        path = self._existing_crop_path(document_id, highlight_id)
        if not path:
            raise KeyError(f"Unknown crop for highlight_id: {highlight_id}")
        return path

    def _existing_crop_path(self, document_id: str, highlight_id: str) -> Path | None:
        crop_dir = self._document_dir(document_id) / "highlights" / "crops"
        safe_id = self._safe_file_id(highlight_id)
        for ext in ("png", "jpg", "jpeg"):
            path = (crop_dir / f"{safe_id}.{ext}").resolve()
            root = self._document_dir(document_id).resolve()
            if root in path.parents and path.exists() and path.is_file():
                return path
        return None

    def _with_crop_metadata(self, document_id: str, highlight: dict[str, Any]) -> dict[str, Any]:
        highlight = dict(highlight)
        highlight_id = self._safe_file_id(highlight.get("highlight_id") or highlight.get("id") or "")
        if not highlight_id:
            return highlight
        crop_path = str(highlight.get("crop_image_path") or highlight.get("crop_path") or "").strip()
        if not crop_path:
            existing = self._existing_crop_path(document_id, highlight_id)
            if existing:
                crop_path = str(existing.relative_to(self._document_dir(document_id)))
        if crop_path:
            highlight["crop_path"] = crop_path
            highlight["crop_image_path"] = crop_path
            highlight["crop_url"] = highlight.get("crop_url") or self._crop_url(document_id, highlight_id)
        return highlight

    def _normalize_library_highlight(self, document_id: str, highlight: dict[str, Any]) -> dict[str, Any]:
        highlight = self._with_crop_metadata(document_id, highlight)
        highlight_id = self._safe_file_id(highlight.get("highlight_id") or highlight.get("id") or "")
        if not highlight_id:
            return highlight
        highlight["id"] = highlight_id
        highlight["highlight_id"] = highlight_id
        highlight["document_id"] = document_id
        highlight["type"] = str(highlight.get("type") or highlight.get("highlight_type") or "text")
        highlight["highlight_type"] = highlight["type"]
        return highlight

    def _highlight_exists(self, document_id: str, highlight_id: str) -> bool:
        safe_highlight_id = self._safe_file_id(highlight_id)
        return any(
            self._safe_file_id(item.get("highlight_id") or item.get("id") or "") == safe_highlight_id
            for item in self._load_highlights(document_id)
            if isinstance(item, dict)
        )

    @staticmethod
    def _crop_url(document_id: str, highlight_id: str) -> str:
        return f"/api/documents/{document_id}/highlights/{highlight_id}/crop"

    def _attach_crop_data_url(self, document_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("crop_image_data_url"):
            return payload
        crop_path = str(payload.get("crop_image_path") or "").strip()
        if not crop_path:
            return payload
        absolute = (self._document_dir(document_id) / crop_path).resolve()
        root = self._document_dir(document_id).resolve()
        if root not in absolute.parents or not absolute.exists():
            return payload
        mime = "image/jpeg" if absolute.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        payload = dict(payload)
        payload["crop_image_data_url"] = f"data:{mime};base64,{base64.b64encode(absolute.read_bytes()).decode('ascii')}"
        return payload

    def _safe_selection_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(payload or {})
        data_url = str(snapshot.pop("crop_image_data_url", "") or "")
        if data_url:
            snapshot["crop_image_available"] = True
            snapshot["crop_image_data_url_length"] = len(data_url)
        return snapshot

    def _save_prompt_snapshot_for_payload(
        self,
        *,
        document_id: str,
        highlight_id: str,
        turn_id: str,
        thread_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        stage = "emotion_strategy" if isinstance(payload.get("selected_strategy"), dict) and payload.get("selected_strategy") else "rag_baseline"
        messages = self._prompt_snapshot_messages(payload)
        identity = self._active_answer_model_identity()
        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        prompt_text = self._messages_to_prompt_text(messages)
        snapshot = {
            "snapshot_id": snapshot_id,
            "document_id": document_id,
            "highlight_id": highlight_id,
            "turn_id": turn_id,
            "thread_id": thread_id,
            "created_at": self._iso_timestamp(time.time()),
            "stage": stage,
            "source": "pdf-chat",
            "provider": identity["provider"],
            "model": identity["model"],
            "messages": messages,
            "prompt_text": prompt_text,
            "context_summary": self._prompt_context_summary(payload, stage),
            "redaction": {
                "api_keys_removed": True,
                "raw_frames_removed": True,
                "large_images_removed": bool(payload.get("crop_image_data_url") or payload.get("crop_image_path")),
            },
        }
        path = self._prompt_snapshot_dir(document_id) / f"{snapshot_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    def _try_save_prompt_snapshot_for_payload(
        self,
        *,
        document_id: str,
        highlight_id: str,
        turn_id: str,
        thread_id: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        try:
            return self._save_prompt_snapshot_for_payload(
                document_id=document_id,
                highlight_id=highlight_id,
                turn_id=turn_id,
                thread_id=thread_id,
                payload=payload,
            ), ""
        except Exception as exc:
            return {}, f"{type(exc).__name__}: {str(exc)[:240]}"

    def _save_strategy_planner_prompt_snapshot(self, document_id: str, request: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        strategy_context = self._strategy_planning_context(request)
        messages = self._strategy_planner_messages(request, strategy_context)
        identity = self._active_strategy_model_identity()
        snapshot = {
            "snapshot_id": snapshot_id,
            "document_id": document_id,
            "highlight_id": str(request.get("highlight_id") or ""),
            "source_turn_id": str(request.get("source_turn_id") or ""),
            "created_at": self._iso_timestamp(time.time()),
            "stage": "strategy_planner",
            "source": "pdf-chat",
            "provider": identity["provider"],
            "model": identity["model"],
            "messages": messages,
            "prompt_text": self._messages_to_prompt_text(messages),
            "strategy_planning_context": strategy_context,
            "context_summary": self._strategy_planner_context_summary(request, strategy_context),
            "allowed_strategy_families": strategy_context.get("strategy_constraints", {}).get("allowed_strategy_families", []),
            "support_cue": strategy_context.get("reaction_context", {}).get("support_cue", ""),
            "redaction": {
                "api_keys_removed": True,
                "raw_frames_removed": True,
                "large_images_removed": True,
            },
        }
        path = self._prompt_snapshot_dir(document_id) / f"{snapshot_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return snapshot

    def _prompt_snapshot_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        messages = []
        for item in build_prompt_messages(payload):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = self._sanitize_prompt_text(str(item.get("content") or ""))
            messages.append({"role": role, "content": content})
        return messages or [{"role": "user", "content": ""}]

    def _prompt_context_summary(self, payload: dict[str, Any], stage: str) -> dict[str, Any]:
        retrieval_context = payload.get("retrieval_context") if isinstance(payload.get("retrieval_context"), dict) else {}
        selected_strategy = payload.get("selected_strategy") if isinstance(payload.get("selected_strategy"), dict) else {}
        reaction_summary = payload.get("reaction_window_summary") if isinstance(payload.get("reaction_window_summary"), dict) else {}
        selected_text = normalize_pdf_text(payload.get("selected_text"))
        rag_chunks = (
            retrieval_context.get("global_rag_context")
            or retrieval_context.get("related_blocks")
            or payload.get("global_rag_context")
            or []
        )
        nearby_context = retrieval_context.get("nearby_context") or payload.get("nearby_useful_context") or []
        matched_block = retrieval_context.get("matched_block") or payload.get("matched_block") or {}
        paper_profile = retrieval_context.get("paper_profile") or payload.get("paper_profile") or {}
        baseline = normalize_pdf_text(payload.get("baseline_explanation"))
        return {
            "selection_type": str(payload.get("highlight_type") or payload.get("type") or "text"),
            "page_number": self._positive_int(payload.get("page_number")),
            "selected_text_preview": selected_text[:240],
            "selected_text_length": len(selected_text),
            "has_area_crop": bool(payload.get("crop_image_data_url") or payload.get("crop_image_path") or payload.get("crop_available") or payload.get("crop_image_available")),
            "has_caption": bool(normalize_pdf_text(payload.get("caption")) or payload.get("selected_caption")),
            "matched_block_present": bool(matched_block),
            "nearby_context_present": bool(nearby_context),
            "rag_chunk_count": len(rag_chunks) if isinstance(rag_chunks, list) else 0,
            "paper_profile_present": bool(paper_profile),
            "has_baseline_explanation": bool(baseline),
            "baseline_explanation_length": len(baseline),
            "has_reaction_window_summary": bool(reaction_summary),
            "support_cue": reaction_summary.get("support_cue") or payload.get("support_cue") if stage == "emotion_strategy" else None,
            "selected_strategy_id": payload.get("selected_strategy_id") or selected_strategy.get("strategy_id") if stage == "emotion_strategy" else None,
            "strategy_family": selected_strategy.get("strategy_family") or selected_strategy.get("strategy_id") if stage == "emotion_strategy" else None,
            "pedagogical_move": selected_strategy.get("pedagogical_move") if stage == "emotion_strategy" else None,
            "context_focus": selected_strategy.get("context_focus") if stage == "emotion_strategy" else None,
            "reaction_window_duration": reaction_summary.get("duration_sec") if stage == "emotion_strategy" else None,
            "reaction_window_avg_confidence": reaction_summary.get("avg_confidence") if stage == "emotion_strategy" else None,
        }

    def _strategy_planner_context_summary(self, request: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        strategy_context = context if isinstance(context, dict) else self._strategy_planning_context(request)
        selected = strategy_context.get("selected_evidence") if isinstance(strategy_context.get("selected_evidence"), dict) else {}
        paper_context = strategy_context.get("paper_context") if isinstance(strategy_context.get("paper_context"), dict) else {}
        previous = strategy_context.get("previous_explanation") if isinstance(strategy_context.get("previous_explanation"), dict) else {}
        reaction = strategy_context.get("reaction_context") if isinstance(strategy_context.get("reaction_context"), dict) else {}
        constraints = strategy_context.get("strategy_constraints") if isinstance(strategy_context.get("strategy_constraints"), dict) else {}
        selected_text = normalize_pdf_text(selected.get("selected_text"))
        baseline = normalize_pdf_text(previous.get("baseline_explanation"))
        nearby_context = paper_context.get("nearby_context") if isinstance(paper_context.get("nearby_context"), list) else []
        rag_chunks = paper_context.get("retrieved_rag_chunks") if isinstance(paper_context.get("retrieved_rag_chunks"), list) else []
        reaction_summary = reaction.get("reaction_window_summary") if isinstance(reaction.get("reaction_window_summary"), dict) else {}
        recent_conversation = strategy_context.get("recent_conversation") if isinstance(strategy_context.get("recent_conversation"), list) else []
        return {
            "selection_type": str(selected.get("selection_type") or "text"),
            "page_number": self._positive_int(selected.get("page_number")),
            "selected_text_preview": selected_text[:240],
            "selected_text_length": len(selected_text),
            "baseline_explanation_length": len(baseline),
            "has_reaction_window_summary": bool(reaction_summary),
            "support_cue": reaction.get("support_cue") or "",
            "dominant_state": reaction.get("dominant_state") or "",
            "secondary_state": reaction.get("secondary_state") or "",
            "reaction_window_duration": float(reaction.get("duration_sec") or 0),
            "reaction_window_avg_confidence": self._float_between(reaction.get("avg_confidence"), 0.0, 1.0),
            "allowed_strategy_families": list(constraints.get("allowed_strategy_families") or []),
            "passage_type": str(paper_context.get("passage_type") or "unknown"),
            "difficulty_hint": str(paper_context.get("difficulty_hint") or "unknown"),
            "rag_chunk_count": len(rag_chunks),
            "nearby_context_count": len(nearby_context),
            "recent_conversation_count": len(recent_conversation),
        }

    def _active_answer_model_identity(self) -> dict[str, str]:
        if not os.environ.get("LLM_PROVIDER", "").strip():
            return {"provider": "mock", "model": "mock"}
        role = llm_config.role_config_from_env("answer_model")
        provider = str(role.get("provider") or "mock").strip().lower() or "mock"
        model = str(role.get("model") or "").strip() or ("mock" if provider == "mock" else "")
        return {"provider": provider, "model": model}

    def _active_strategy_model_identity(self) -> dict[str, str]:
        if not os.environ.get("STRATEGY_PLANNER_PROVIDER", "").strip() and not os.environ.get("LLM_PROVIDER", "").strip():
            return {"provider": "mock", "model": "mock"}
        role = llm_config.role_config_from_env("strategy_planner_model")
        provider = str(role.get("provider") or "mock").strip().lower() or "mock"
        model = str(role.get("model") or "").strip() or ("mock" if provider == "mock" else "")
        return {"provider": provider, "model": model}

    def _prompt_snapshot_dir(self, document_id: str) -> Path:
        return self._document_dir(document_id) / "prompt_snapshots"

    def _iter_prompt_snapshot_paths(self, document_filter: str = "") -> list[Path]:
        root = self.documents_dir.resolve()
        if document_filter:
            document_id = self._safe_document_id(document_filter)
            snapshot_dir = self._prompt_snapshot_dir(document_id)
            return sorted(snapshot_dir.glob("*.json")) if snapshot_dir.exists() else []
        if not root.exists():
            return []
        paths: list[Path] = []
        for document_dir in sorted(root.iterdir()):
            snapshot_dir = document_dir / "prompt_snapshots"
            if snapshot_dir.is_dir():
                paths.extend(sorted(snapshot_dir.glob("*.json")))
        return paths

    def _prompt_snapshot_path_by_id(self, snapshot_id: str) -> Path:
        safe_id = self._safe_file_id(snapshot_id)
        for path in self._iter_prompt_snapshot_paths():
            if path.stem == safe_id:
                return path
        raise KeyError(f"Unknown prompt snapshot: {snapshot_id}")

    @staticmethod
    def _prompt_snapshot_compact(snapshot: dict[str, Any]) -> dict[str, Any]:
        summary = snapshot.get("context_summary") if isinstance(snapshot.get("context_summary"), dict) else {}
        return {
            "snapshot_id": snapshot.get("snapshot_id") or "",
            "document_id": snapshot.get("document_id") or "",
            "highlight_id": snapshot.get("highlight_id") or "",
            "turn_id": snapshot.get("turn_id") or "",
            "stage": snapshot.get("stage") or "",
            "created_at": snapshot.get("created_at") or "",
            "provider": snapshot.get("provider") or "",
            "model": snapshot.get("model") or "",
            "summary": {
                "selection_type": summary.get("selection_type") or "",
                "page_number": summary.get("page_number") or 0,
                "selected_text_preview": summary.get("selected_text_preview") or "",
                "selected_text_length": summary.get("selected_text_length") or 0,
                "rag_chunk_count": summary.get("rag_chunk_count") or 0,
                "nearby_context_count": summary.get("nearby_context_count") or 0,
                "support_cue": summary.get("support_cue"),
                "dominant_state": summary.get("dominant_state") or "",
                "secondary_state": summary.get("secondary_state") or "",
                "allowed_strategy_families": summary.get("allowed_strategy_families") if isinstance(summary.get("allowed_strategy_families"), list) else [],
                "baseline_explanation_length": summary.get("baseline_explanation_length") or 0,
                "reaction_window_duration": summary.get("reaction_window_duration") or 0,
                "strategy_family": summary.get("strategy_family"),
                "pedagogical_move": summary.get("pedagogical_move"),
            },
        }

    def _run_single_comparison_model(self, snapshot: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
        label = str(model_config.get("label") or model_config.get("model") or "Model").strip()
        provider = self._normalize_llm_provider(model_config.get("provider"))
        model = str(model_config.get("model") or "").strip()
        temperature = self._float_between(model_config.get("temperature", 0.2), 0.0, 2.0)
        max_tokens_value = model_config.get("max_tokens")
        if max_tokens_value in (None, ""):
            max_tokens_value = self._default_compare_max_tokens(snapshot)
        max_tokens = int(self._float_between(max_tokens_value, 1, 8192))
        base = {"label": label, "provider": provider, "model": model}
        if provider not in {"gemini", "openrouter", "openai_compatible"} or not model:
            return {**base, "ok": False, "latency_ms": 0, "output": "", "error": "Provider is not configured.", "auto_checks": self._llm_compare_auto_checks("", snapshot)}
        key = llm_config.provider_api_key_from_env(provider)
        base_url = ""
        if provider == "openrouter":
            base_url = llm_config.DEFAULT_OPENROUTER_BASE_URL
        elif provider == "openai_compatible":
            base_url = llm_config.provider_base_url_from_env(provider)
        if not key or (provider == "openai_compatible" and not base_url):
            return {**base, "ok": False, "latency_ms": 0, "output": "", "error": "Provider is not configured.", "auto_checks": self._llm_compare_auto_checks("", snapshot)}

        started = time.time()
        try:
            if provider == "gemini":
                output, finish_reason = self._run_gemini_snapshot(snapshot, api_key=key, model=model, temperature=temperature, max_tokens=max_tokens)
            else:
                output, finish_reason = self._run_chat_completions_snapshot(
                    snapshot,
                    provider=provider,
                    api_key=key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            latency_ms = int((time.time() - started) * 1000)
            auto_checks = self._llm_compare_auto_checks(output, snapshot)
            if finish_reason:
                auto_checks["finish_reason"] = finish_reason
            return {
                **base,
                "ok": bool(output.strip()),
                "latency_ms": latency_ms,
                "output": output,
                "error": None if output.strip() else "Provider returned an empty output.",
                "finish_reason": finish_reason,
                "auto_checks": auto_checks,
            }
        except Exception as exc:
            latency_ms = int((time.time() - started) * 1000)
            return {
                **base,
                "ok": False,
                "latency_ms": latency_ms,
                "output": "",
                "error": f"{type(exc).__name__}: {exc}",
                "auto_checks": self._llm_compare_auto_checks("", snapshot),
            }

    @staticmethod
    def _default_compare_max_tokens(snapshot: dict[str, Any]) -> int:
        return 3000 if str(snapshot.get("stage") or "") == "strategy_planner" else 800

    def _run_gemini_snapshot(self, snapshot: dict[str, Any], *, api_key: str, model: str, temperature: float, max_tokens: int) -> tuple[str, str]:
        prompt = self._messages_to_prompt_text(snapshot.get("messages") if isinstance(snapshot.get("messages"), list) else [])
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        request = urllib.request.Request(
            GEMINI_ENDPOINT_TEMPLATE.format(model=model),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._gemini_strategy_text(payload), self._gemini_finish_reason(payload)

    def _run_chat_completions_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        messages = snapshot.get("messages") if isinstance(snapshot.get("messages"), list) else []
        body = {
            "model": model,
            "messages": self._clean_snapshot_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        if provider == "openrouter":
            site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
            site_name = os.environ.get("OPENROUTER_SITE_NAME", "").strip()
            if site_url:
                headers["HTTP-Referer"] = site_url
            if site_name:
                headers["X-OpenRouter-Title"] = site_name
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._chat_completions_strategy_text(payload), self._chat_completions_finish_reason(payload)

    @staticmethod
    def _gemini_finish_reason(payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates") if isinstance(payload, dict) else []
        if not candidates or not isinstance(candidates[0], dict):
            return ""
        return str(candidates[0].get("finishReason") or candidates[0].get("finish_reason") or "")

    @staticmethod
    def _chat_completions_finish_reason(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") if isinstance(payload, dict) else []
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        return str(choices[0].get("finish_reason") or message.get("finish_reason") or "")

    def _llm_compare_auto_checks(self, output: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        text = str(output or "")
        lower = text.lower()
        unsafe_phrases = [
            "you are confused",
            "you look frustrated",
            "camera detected",
            "your face shows",
            "i can see you are",
        ]
        parsed_json = self._parse_strict_json_output(text)
        json_valid = parsed_json is not None
        summary = snapshot.get("context_summary") if isinstance(snapshot.get("context_summary"), dict) else {}
        strategy_terms = [
            str(summary.get("strategy_family") or ""),
            str(summary.get("pedagogical_move") or ""),
        ]
        checks = {
            "non_empty": bool(text.strip()),
            "length_chars": len(text),
            "output_length_chars": len(text),
            "json_valid": json_valid,
            "json_parse_error": "" if json_valid or not text.strip() else "Could not parse a complete JSON object.",
            "output_truncated_likely": self._json_output_looks_truncated(text) if text.strip() else False,
            "unsafe_affect_phrases_found": [phrase for phrase in unsafe_phrases if phrase in lower],
            "strategy_terms_found": [
                term for term in strategy_terms
                if term and term.lower().replace("_", " ") in lower
            ],
        }
        if str(snapshot.get("stage") or "") == "strategy_planner":
            checks.update(self._strategy_planner_output_checks(parsed_json, summary))
        return checks

    @staticmethod
    def _parse_strict_json_output(text: str) -> Any | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidates = [raw]
        fenced = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", raw, re.DOTALL)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        first_object = WebState._first_json_object_text(raw)
        if first_object:
            candidates.append(first_object)
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    @staticmethod
    def _first_json_object_text(text: str) -> str:
        start = -1
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(str(text or "")):
            if start < 0:
                if char == "{":
                    start = index
                    depth = 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return str(text)[start:index + 1]
        return ""

    @staticmethod
    def _json_output_looks_truncated(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        if raw.endswith((",", ":", "{", "[")):
            return True
        depth = 0
        in_string = False
        escaped = False
        for char in raw:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
        return in_string or depth > 0

    def _strategy_planner_output_checks(self, parsed_json: Any, summary: dict[str, Any]) -> dict[str, Any]:
        candidates = []
        if isinstance(parsed_json, dict) and isinstance(parsed_json.get("candidates"), list):
            candidates = [candidate for candidate in parsed_json["candidates"] if isinstance(candidate, dict)]
        required = {
            "strategy_id",
            "strategy_family",
            "pedagogical_move",
            "context_focus",
            "title",
            "short_description",
            "why_recommended",
            "prompt_instruction",
            "expected_answer_shape",
            "recommended",
            "recommended_score",
        }
        missing_by_candidate = []
        for index, candidate in enumerate(candidates):
            missing = []
            for field in required:
                value = candidate.get(field)
                if field not in candidate or value is None or value == "":
                    missing.append(field)
            missing = sorted(missing)
            if missing:
                missing_by_candidate.append({"index": index, "missing": missing})
        recommended_count = sum(1 for candidate in candidates if candidate.get("recommended") is True)
        allowed_families = set(str(item) for item in (summary.get("allowed_strategy_families") or []) if str(item))
        disallowed_families = sorted({
            str(candidate.get("strategy_family") or "")
            for candidate in candidates
            if allowed_families and str(candidate.get("strategy_family") or "") not in allowed_families
        })
        topic_title_warning = any(self._strategy_candidate_title_warning(candidate) for candidate in candidates)
        return {
            "has_candidates": bool(candidates),
            "candidate_count": len(candidates),
            "exactly_one_recommended": recommended_count == 1,
            "required_fields_present": bool(candidates) and not missing_by_candidate,
            "missing_required_fields": missing_by_candidate,
            "allowed_strategy_family": bool(candidates) and bool(allowed_families) and not disallowed_families,
            "allowed_strategy_families": sorted(allowed_families),
            "disallowed_strategy_families": disallowed_families,
            "topic_title_warning": topic_title_warning,
        }

    @staticmethod
    def _strategy_candidate_title_warning(candidate: dict[str, Any]) -> bool:
        title = normalize_pdf_text(candidate.get("title")).lower()
        move = normalize_pdf_text(candidate.get("pedagogical_move")).lower()
        context_focus = normalize_pdf_text(candidate.get("context_focus")).lower()
        if not title:
            return False
        if not move and (title or context_focus):
            return True
        if move and title == move:
            return True
        if context_focus and title == context_focus and not move:
            return True
        return False

    @classmethod
    def _clean_snapshot_messages(cls, messages: list[Any]) -> list[dict[str, str]]:
        cleaned = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            cleaned.append({"role": role, "content": cls._sanitize_prompt_text(str(item.get("content") or ""))})
        return cleaned or [{"role": "user", "content": ""}]

    @classmethod
    def _messages_to_prompt_text(cls, messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        cleaned = cls._clean_snapshot_messages(messages)
        return "\n\n".join(f"{item['role']}:\n{item['content']}" for item in cleaned).strip()

    @staticmethod
    def _sanitize_prompt_text(text: str) -> str:
        text = re.sub(r"data:image/[^\\s\"']+", "[image data omitted]", str(text or ""))
        text = re.sub(r"base64,[A-Za-z0-9+/=]+", "base64,[omitted]", text)
        for marker in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "X-goog-api-key", "AI" + "za"):
            text = text.replace(marker, "[omitted]")
        return text

    def _llm_comparison_dir(self) -> Path:
        return self.upload_dir / "llm_comparisons"

    @staticmethod
    def _safe_comparison_id(value: Any) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
        return clean or f"comparison_{uuid.uuid4().hex[:12]}"

    def _comparison_payload_with_strategy_details(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean = dict(payload or {})
        if str(clean.get("stage") or "") != "strategy_planner":
            return clean
        prompt_summary = clean.get("prompt_summary") if isinstance(clean.get("prompt_summary"), dict) else {}
        allowed_families = [
            str(item)
            for item in (prompt_summary.get("allowed_strategy_families") or clean.get("allowed_strategy_families") or [])
            if str(item)
        ]
        if allowed_families:
            clean["allowed_strategy_families"] = allowed_families
        results = []
        for result in (clean.get("results") if isinstance(clean.get("results"), list) else []):
            if not isinstance(result, dict):
                continue
            item = dict(result)
            if "parsed_json" not in item and isinstance(item.get("output"), str):
                parsed = self._parse_strict_json_output(item["output"])
                if parsed is not None:
                    item["parsed_json"] = parsed
            results.append(item)
        clean["results"] = results
        return clean

    @staticmethod
    def _normalize_llm_provider(value: Any) -> str:
        provider = str(value or "").strip().lower().replace("-", "_")
        aliases = {"openai": "openai_compatible", "openai-compatible": "openai_compatible"}
        return aliases.get(provider, provider)

    def _sanitize_comparison_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            clean = {}
            for key, value in payload.items():
                lower = str(key).lower()
                if any(secret in lower for secret in ("api_key", "apikey", "authorization", "token", "secret", "password")):
                    continue
                clean[str(key)] = self._sanitize_comparison_payload(value)
            return clean
        if isinstance(payload, list):
            return [self._sanitize_comparison_payload(item) for item in payload]
        if isinstance(payload, str):
            return self._sanitize_prompt_text(payload)
        return payload

    def _safe_turn_metadata(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        metadata: dict[str, Any] = {}
        for turn_id, value in payload.items():
            safe_turn_id = self._safe_file_id(turn_id)
            if not safe_turn_id or not isinstance(value, dict):
                continue
            safe_value = self._safe_log_payload(value)
            if isinstance(safe_value, dict):
                metadata[safe_turn_id] = safe_value
        return metadata

    @staticmethod
    def _context_used_summary(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": result.get("provider"),
            "model": result.get("model"),
            "mode": result.get("mode") or result.get("recommended_llm_mode"),
            "used_image": bool(result.get("used_image")),
            "paper_profile_used": bool(result.get("paper_profile_used")),
            "paper_profile_summary": result.get("paper_profile_summary") or "",
            "retrieved_block_count": result.get("retrieved_block_count") or 0,
            "retrieval_method": result.get("retrieval_method") or "",
            "retrieved_blocks": result.get("retrieved_blocks") or [],
            "global_rag_context": result.get("global_rag_context") or [],
            "nearby_context": result.get("nearby_context") or [],
            "prompt_preview": result.get("prompt_preview") or "",
        }

    def _log_interaction(
        self,
        document_id: str,
        event_type: str,
        highlight_id: str | None = None,
        selection_type: str | None = None,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        retrieval_method: str | None = None,
        success: bool = True,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        log_dir = self._document_dir(document_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.time(),
            "event_type": event_type,
            "document_id": document_id,
            "session_id": session_id,
            "highlight_id": highlight_id,
            "selection_type": selection_type,
            "provider": provider,
            "model": model,
            "retrieval_method": retrieval_method,
            "success": bool(success),
            "error": str(error or "")[:500],
        }
        if extra:
            payload.update(extra)
        payload = self._safe_log_payload(payload)
        with (log_dir / "interactions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _safe_log_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            clean = {}
            for key, item in value.items():
                key_text = str(key)
                lower_key = key_text.lower()
                if any(secret in lower_key for secret in ("api_key", "apikey", "authorization", "token", "secret")):
                    continue
                if lower_key in {"crop_image_data_url", "image", "cropped_image", "frame", "image_data", "frame_data", "data_url", "raw_frame", "frame_base64"}:
                    clean[key_text] = "[omitted]"
                    continue
                clean[key_text] = self._safe_log_payload(item)
            return clean
        if isinstance(value, list):
            return [self._safe_log_payload(item) for item in value[:20]]
        if isinstance(value, str):
            if "data:image/" in value or "base64," in value:
                return "[omitted]"
            for marker in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "X-goog-api-key", "AI" + "za"):
                value = value.replace(marker, "[omitted]")
            return value[:2000]
        return value

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        try:
            if not path.exists():
                return fallback
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return fallback

    @staticmethod
    def _safe_document_id(document_id: Any) -> str:
        value = re.sub(r"[^A-Za-z0-9._-]+", "", str(document_id or "")).strip("._")
        if not value:
            raise ValueError("document_id is required.")
        return value

    @staticmethod
    def _safe_file_id(value: Any) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
        return clean or uuid.uuid4().hex

    def _prepared_page_count(self, record: dict[str, Any], document: Document) -> int:
        for value in (
            (record.get("parse_status") or {}).get("page_count") if isinstance(record.get("parse_status"), dict) else None,
            record.get("page_count"),
            document.metadata.get("page_count") if isinstance(document.metadata, dict) else None,
        ):
            count = self._positive_int(value)
            if count:
                return count
        if self.current_document_type == "pdf" and document.page_count == 1:
            count = self._page_count_from_blocks_index(record)
            if count:
                return count
        return document.page_count

    def _page_count_from_blocks_index(self, record: dict[str, Any]) -> int:
        parse_status = record.get("parse_status")
        if not isinstance(parse_status, dict):
            return 0
        blocks_path = parse_status.get("blocks_index_path")
        if not blocks_path:
            return 0
        try:
            blocks = load_blocks(Path(blocks_path))
        except Exception:
            return 0
        counts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("page_number") is not None:
                counts.append(self._positive_int(block.get("page_number")))
            elif block.get("page_idx") is not None:
                page_idx = self._positive_int(block.get("page_idx"))
                if page_idx is not None:
                    counts.append(page_idx + 1)
        return max((count for count in counts if count), default=0)

    def _context_from_chat_data(self, data: dict[str, Any], question: str) -> PaperContext:
        context_data = self.build_context(
            {
                "selected_text": data.get("selected_text", ""),
                "page_number": data.get("page_number") or self.session.current_page_number,
                "user_question": question,
                "document_id": data.get("document_id") or self.current_document_id,
            }
        )
        return PaperContext(
            document_title=context_data["document_title"],
            page_number=context_data["page_number"],
            selected_text=context_data["selected_text"],
            surrounding_text=context_data["surrounding_text"],
            retrieved_chunks=context_data["retrieved_chunks"],
            passage_type=context_data["passage_type"],
            page_title=context_data.get("page_title"),
            section_hint=context_data.get("section_hint"),
            difficulty_hint=context_data.get("difficulty_hint", "moderate"),
            passage_analysis=context_data.get("passage_analysis", {}),
            retrieval_debug=context_data.get("retrieval_debug", {}),
            document_id=context_data.get("document_id"),
            document_type=context_data.get("document_type"),
            highlight_id=str(data.get("highlight_id") or "").strip() or None,
        )

    def _state_response(self, snapshot, source: str) -> dict[str, Any]:
        buffer_debug = self.session.buffer.debug_snapshot()
        trend_debug = self.session.tracker.debug_snapshot()
        return {
            "raw_emotion": snapshot.raw_emotion,
            "smoothed_emotion": snapshot.smoothed_emotion,
            "state": snapshot.state,
            "learning_state": snapshot.state,
            "trend": snapshot.trend,
            "confidence": snapshot.confidence,
            "duration_sec": snapshot.duration_sec,
            "strategy": snapshot.strategy,
            "valence": snapshot.valence,
            "arousal": snapshot.arousal,
            "probabilities": buffer_debug["probabilities"],
            "buffer": buffer_debug,
            "history": trend_debug["history"],
            "dominant_state": trend_debug["dominant_state"],
            "manual_override": snapshot.manual_override,
            "source": source,
            "source_mode": "manual" if snapshot.manual_override else "auto",
        }

    def _register_current_document(self, path: Path, document_type: str, document_id: str | None = None) -> str:
        document_id = document_id or uuid.uuid4().hex
        document_path = Path(path).resolve()
        self.current_document_id = document_id
        self.current_document_type = document_type
        self.documents[document_id] = {
            "document_id": document_id,
            "type": document_type,
            "path": document_path,
            "file_name": document_path.name,
            "title": self.session.document.title if self.session.document else document_path.stem,
        }
        self.highlights_by_document.setdefault(document_id, [])
        self.last_highlight_id = None
        return document_id

    @staticmethod
    def _debug_pdf_path() -> Path:
        configured = os.environ.get("PDF_DEBUG_PATH")
        target = Path(configured).expanduser().resolve() if configured else Path("runtime_uploads/debug/test.pdf").resolve()
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(
                "Debug PDF not found. Set PDF_DEBUG_PATH or copy a PDF to runtime_uploads/debug/test.pdf."
            )
        return target

    def _load_pdf_browser_fallback(self, path: Path) -> Document:
        fallback_text = (
            "PDF text extraction unavailable on the backend. The browser PDF workspace "
            "can still render this file with PDF.js and create screenshot-backed highlights."
        )
        document = Document(
            title=path.stem,
            source_path=path.resolve(),
            pages=[Page(page_number=1, text=fallback_text, heading="PDF browser viewer")],
            metadata={
                "format": "pdf",
                "source_name": path.name,
                "text_extraction_status": "missing_pymupdf",
            },
            section_hints=["PDF browser viewer"],
        )
        chunk_document(document)
        self.session.document = document
        self.session.retriever = None
        self.session.current_page_number = 1
        self.session.selected_range = None
        self.session.manual_context = None
        return document

    def _document_type_for(self, document_id: str | None) -> str | None:
        if document_id and document_id in self.documents:
            return str(self.documents[document_id].get("type") or "")
        return self.current_document_type

    @staticmethod
    def _safe_upload_name(filename: str) -> str:
        name = Path(filename or "uploaded.txt").name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return name or "uploaded.txt"

    @staticmethod
    def _find_selected_range(page_text: str, selected_text: str) -> tuple[int, int]:
        if not selected_text:
            return -1, -1
        start = page_text.find(selected_text)
        if start >= 0:
            return start, start + len(selected_text)
        words = selected_text.split()
        if not words:
            return -1, -1
        pattern = r"\s+".join(re.escape(word) for word in words)
        match = re.search(pattern, page_text)
        if match:
            return match.start(), match.end()
        return -1, -1

    @staticmethod
    def _sanitize_rects(value: Any) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        sanitized: list[dict[str, float]] = []
        for item in value[:100]:
            if not isinstance(item, dict):
                continue
            rect: dict[str, float] = {}
            for key in ("left", "top", "width", "height"):
                try:
                    rect[key] = round(float(item.get(key, 0.0)), 2)
                except (TypeError, ValueError):
                    rect[key] = 0.0
            if rect["width"] > 0 and rect["height"] > 0:
                sanitized.append(rect)
        return sanitized

    @staticmethod
    def _sanitize_scaled_rects(value: Any) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        sanitized: list[dict[str, float]] = []
        for item in value[:100]:
            if not isinstance(item, dict):
                continue
            rect: dict[str, float] = {}
            for key in ("x1", "y1", "x2", "y2", "width", "height", "pageNumber"):
                try:
                    rect[key] = round(float(item.get(key, 0.0)), 6)
                except (TypeError, ValueError):
                    rect[key] = 0.0
            if rect["x2"] >= rect["x1"] and rect["y2"] >= rect["y1"]:
                sanitized.append(rect)
        return sanitized

    def _sanitize_position(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        bounding = value.get("boundingRect")
        rects = value.get("rects", [])
        sanitized_rects = self._sanitize_scaled_rects(rects if isinstance(rects, list) else [])
        sanitized_bounding = self._sanitize_scaled_rects([bounding])[0] if isinstance(bounding, dict) else None
        if not sanitized_bounding:
            return None
        return {"boundingRect": sanitized_bounding, "rects": sanitized_rects}

    def _rect_payload_for_matching(self, data: dict[str, Any], page_number: int) -> dict[str, Any]:
        normalized_rects = self._sanitize_normalized_rects(data.get("normalized_rects", []), page_number)
        parser_rects_1000 = self._sanitize_parser_rects_1000(data.get("parser_rects_1000", []), page_number)
        if not normalized_rects and parser_rects_1000:
            normalized_rects = [self._parser_1000_to_normalized(rect) for rect in parser_rects_1000]

        viewport_rects: list[dict[str, float]] = []
        rect_source = ""
        for source_name, raw_rects in self._matching_rect_sources(data):
            viewport_rects = self._sanitize_viewer_rects(raw_rects, page_number)
            if viewport_rects:
                rect_source = source_name
                break

        if not normalized_rects and viewport_rects:
            normalized_rects = [
                rect for rect in (self._viewer_rect_to_normalized(rect) for rect in viewport_rects)
                if rect
            ]
        if not parser_rects_1000 and normalized_rects:
            parser_rects_1000 = [self._normalized_to_parser_1000(rect) for rect in normalized_rects]
        if normalized_rects and not rect_source:
            rect_source = "normalized_rects"
        if parser_rects_1000 and not rect_source:
            rect_source = "parser_rects_1000"

        return {
            "viewport_rects": viewport_rects,
            "normalized_rects": normalized_rects,
            "parser_rects_1000": parser_rects_1000,
            "rect_source": rect_source or "none",
        }

    @staticmethod
    def _matching_rect_sources(data: dict[str, Any]) -> list[tuple[str, Any]]:
        sources: list[tuple[str, Any]] = []
        sources.append(("viewport_rects", data.get("viewport_rects", [])))
        sources.append(("scaled_rects_legacy", data.get("scaled_rects", [])))
        sources.append(("rects_legacy", data.get("rects", [])))
        position = data.get("position")
        if isinstance(position, dict):
            rects = position.get("rects", [])
            bounding = position.get("boundingRect")
            sources.append(("position.rects", rects if isinstance(rects, list) else []))
            sources.append(("position.boundingRect", [bounding] if isinstance(bounding, dict) else []))
        return sources

    @staticmethod
    def _sanitize_viewer_rects(value: Any, page_number: int) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        sanitized: list[dict[str, float]] = []
        for item in value[:100]:
            if not isinstance(item, dict):
                continue
            rect = WebState._viewer_rect_from_any(item, page_number)
            if rect:
                sanitized.append(rect)
        return sanitized

    @staticmethod
    def _viewer_rect_from_any(item: dict[str, Any], page_number: int) -> dict[str, float] | None:
        raw_page = WebState._float_or_zero(item.get("pageNumber"))
        if raw_page and int(raw_page) != int(page_number):
            return None
        if {"x1", "y1", "x2", "y2"}.issubset(item):
            x1 = WebState._float_or_zero(item.get("x1"))
            y1 = WebState._float_or_zero(item.get("y1"))
            x2 = WebState._float_or_zero(item.get("x2"))
            y2 = WebState._float_or_zero(item.get("y2"))
            width = WebState._float_or_zero(item.get("width"))
            height = WebState._float_or_zero(item.get("height"))
        elif {"left", "top", "width", "height"}.issubset(item):
            x1 = WebState._float_or_zero(item.get("left"))
            y1 = WebState._float_or_zero(item.get("top"))
            rect_width = WebState._float_or_zero(item.get("width"))
            rect_height = WebState._float_or_zero(item.get("height"))
            x2 = x1 + rect_width
            y2 = y1 + rect_height
            width = WebState._float_or_zero(item.get("pageWidth")) or WebState._float_or_zero(item.get("page_width")) or rect_width
            height = WebState._float_or_zero(item.get("pageHeight")) or WebState._float_or_zero(item.get("page_height")) or rect_height
        else:
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 <= x1 or y2 <= y1:
            return None
        rect_page = int(WebState._float_or_zero(item.get("pageNumber")) or page_number)
        return {
            "x1": round(x1, 6),
            "y1": round(y1, 6),
            "x2": round(x2, 6),
            "y2": round(y2, 6),
            "width": round(width, 6),
            "height": round(height, 6),
            "pageNumber": float(max(1, rect_page)),
        }

    @staticmethod
    def _sanitize_normalized_rects(value: Any, page_number: int) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        sanitized: list[dict[str, float]] = []
        for item in value[:100]:
            if not isinstance(item, dict) or not {"x1", "y1", "x2", "y2"}.issubset(item):
                continue
            raw_page = WebState._float_or_zero(item.get("pageNumber"))
            if raw_page and int(raw_page) != int(page_number):
                continue
            rect = WebState._clamped_rect(
                WebState._float_or_zero(item.get("x1")),
                WebState._float_or_zero(item.get("y1")),
                WebState._float_or_zero(item.get("x2")),
                WebState._float_or_zero(item.get("y2")),
                page_number=int(WebState._float_or_zero(item.get("pageNumber")) or page_number),
            )
            if rect:
                sanitized.append(rect)
        return sanitized

    @staticmethod
    def _sanitize_parser_rects_1000(value: Any, page_number: int) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        sanitized: list[dict[str, float]] = []
        for item in value[:100]:
            if not isinstance(item, dict) or not {"x1", "y1", "x2", "y2"}.issubset(item):
                continue
            raw_page = WebState._float_or_zero(item.get("pageNumber"))
            if raw_page and int(raw_page) != int(page_number):
                continue
            rect = WebState._bounded_rect(
                WebState._float_or_zero(item.get("x1")),
                WebState._float_or_zero(item.get("y1")),
                WebState._float_or_zero(item.get("x2")),
                WebState._float_or_zero(item.get("y2")),
                lower=0.0,
                upper=1000.0,
                page_number=int(WebState._float_or_zero(item.get("pageNumber")) or page_number),
            )
            if rect:
                sanitized.append(rect)
        return sanitized

    @staticmethod
    def _viewer_rect_to_normalized(rect: dict[str, float]) -> dict[str, float] | None:
        x1 = float(rect["x1"])
        y1 = float(rect["y1"])
        x2 = float(rect["x2"])
        y2 = float(rect["y2"])
        page_width = float(rect.get("width") or 0.0)
        page_height = float(rect.get("height") or 0.0)
        if max(x2, y2) <= 1.0:
            return WebState._clamped_rect(x1, y1, x2, y2, int(rect.get("pageNumber") or 1))
        if page_width > 1.0 and page_height > 1.0 and x2 <= page_width * 1.1 and y2 <= page_height * 1.1:
            return WebState._clamped_rect(
                x1 / page_width,
                y1 / page_height,
                x2 / page_width,
                y2 / page_height,
                int(rect.get("pageNumber") or 1),
            )
        if max(x2, y2) <= 1000.0:
            return WebState._clamped_rect(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0, int(rect.get("pageNumber") or 1))
        return None

    @staticmethod
    def _parser_1000_to_normalized(rect: dict[str, float]) -> dict[str, float]:
        return WebState._clamped_rect(
            float(rect["x1"]) / 1000.0,
            float(rect["y1"]) / 1000.0,
            float(rect["x2"]) / 1000.0,
            float(rect["y2"]) / 1000.0,
            int(rect.get("pageNumber") or 1),
        ) or {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "pageNumber": float(rect.get("pageNumber") or 1)}

    @staticmethod
    def _normalized_to_parser_1000(rect: dict[str, float]) -> dict[str, float]:
        return {
            "x1": round(float(rect["x1"]) * 1000.0, 3),
            "y1": round(float(rect["y1"]) * 1000.0, 3),
            "x2": round(float(rect["x2"]) * 1000.0, 3),
            "y2": round(float(rect["y2"]) * 1000.0, 3),
            "pageNumber": float(rect.get("pageNumber") or 1),
        }

    @staticmethod
    def _clamped_rect(x1: float, y1: float, x2: float, y2: float, page_number: int) -> dict[str, float] | None:
        return WebState._bounded_rect(x1, y1, x2, y2, lower=0.0, upper=1.0, page_number=page_number)

    @staticmethod
    def _bounded_rect(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        lower: float,
        upper: float,
        page_number: int,
    ) -> dict[str, float] | None:
        left = max(lower, min(float(x1), float(x2)))
        top = max(lower, min(float(y1), float(y2)))
        right = min(upper, max(float(x1), float(x2)))
        bottom = min(upper, max(float(y1), float(y2)))
        if right <= left or bottom <= top:
            return None
        return {
            "x1": round(left, 6),
            "y1": round(top, 6),
            "x2": round(right, 6),
            "y2": round(bottom, 6),
            "pageNumber": float(max(1, int(page_number))),
        }

    @staticmethod
    def _float_or_zero(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _positive_int(value: Any) -> int:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return 0
        return number if number >= 0 else 0

    def _page_text(self, page_number: int) -> str:
        if self.session.document is None:
            return ""
        try:
            return self.session.document.page(page_number).text
        except Exception:
            return ""

    def _text_confidence(self, selected_text: str, page_text: str) -> float:
        text = selected_text.strip()
        if not text:
            return 0.0
        if "\ufffd" in text:
            return 0.15
        printable = sum(1 for char in text if char.isprintable() and not char.isspace())
        alnum = sum(1 for char in text if char.isalnum())
        compact = re.sub(r"\s+", " ", text).strip().lower()
        if not compact or printable / max(len(text), 1) < 0.75 or alnum / max(printable, 1) < 0.45:
            return 0.2
        normalized_page = re.sub(r"\s+", " ", page_text or "").strip().lower()
        if normalized_page and compact in normalized_page:
            return 0.98
        words = re.findall(r"[a-z0-9]+", compact)
        if len(words) <= 2:
            return 0.45
        if normalized_page:
            page_words = set(re.findall(r"[a-z0-9]+", normalized_page))
            overlap = sum(1 for word in words if word in page_words) / max(len(words), 1)
            if overlap < 0.35:
                return 0.25
            ordered_pattern = r".*".join(re.escape(word) for word in words[:8])
            return 0.75 if re.search(ordered_pattern, normalized_page) else round(0.35 + overlap * 0.35, 3)
        return 0.6

    @staticmethod
    def _resolved_highlight_type(requested_type: str, selected_text: str, text_confidence: float, cropped_image: str) -> str:
        if requested_type == "area":
            return "area"
        if requested_type == "screenshot_fallback":
            return "screenshot_fallback"
        if selected_text.strip() and text_confidence >= 0.5:
            return "text"
        return "screenshot_fallback" if cropped_image else "text"

    @staticmethod
    def _column_side(value: Any, scaled_rects: list[dict[str, float]], rects: list[dict[str, float]]) -> str:
        explicit = str(value or "").strip().lower()
        if explicit in {"left", "right", "full_width"}:
            return explicit
        if scaled_rects:
            rect = scaled_rects[0]
            width = max(float(rect.get("width") or 1.0), 1e-6)
            x1 = float(rect.get("x1") or 0.0) / width
            x2 = float(rect.get("x2") or 0.0) / width
            if x2 - x1 > 0.55 or (x1 < 0.40 and x2 > 0.60):
                return "full_width"
            return "left" if (x1 + x2) / 2 < 0.5 else "right"
        if rects:
            rect = rects[0]
            left = float(rect.get("left") or 0.0)
            width = float(rect.get("width") or 0.0)
            return "left" if left + width / 2 < 320 else "right"
        return "full_width"

    @staticmethod
    def _opposite_rail(column_side: str) -> str:
        if column_side == "left":
            return "right"
        if column_side == "right":
            return "left"
        return "right"

    def _save_cropped_image(self, document_id: str, highlight_id: str, image_data: str) -> Path | None:
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", image_data, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        ext = "jpg" if match.group(1).lower() in {"jpg", "jpeg"} else "png"
        try:
            raw = base64.b64decode(match.group(2), validate=False)
        except (binascii.Error, ValueError):
            return None
        crop_dir = self.upload_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        path = (crop_dir / f"{document_id}_{highlight_id}.{ext}").resolve()
        root = self.upload_dir.resolve()
        if root not in path.parents:
            raise ValueError("Refusing to save cropped image outside runtime_uploads.")
        path.write_bytes(raw)
        return path

    @staticmethod
    def _context_debug(context: PaperContext) -> dict[str, Any]:
        return {
            "document_id": context.document_id,
            "document_type": context.document_type,
            "page_number": context.page_number,
            "highlight_id": context.highlight_id,
            "selected_text": context.selected_text,
            "surrounding_text": context.surrounding_text,
            "retrieved_chunks": context.retrieved_chunks,
            "passage_type": context.passage_type,
            "difficulty_hint": context.difficulty_hint,
            "section_hint": context.section_hint,
            "passage_analysis": context.passage_analysis,
            "retrieval_debug": context.retrieval_debug,
        }

    @staticmethod
    def _prediction_for(emotion: str, confidence: float, source: str) -> EmotionPrediction:
        floor = (1.0 - confidence) / (len(ALLOWED_EMOTIONS) - 1)
        probabilities = {label: floor for label in ALLOWED_EMOTIONS}
        probabilities[emotion] = confidence
        return EmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=time.time(),
            source=source,
        )

    @staticmethod
    def _decode_frame_length(frame_data: str) -> int:
        if not frame_data:
            return 0
        if "," in frame_data:
            frame_data = frame_data.split(",", 1)[1]
        try:
            return len(base64.b64decode(frame_data, validate=False))
        except (binascii.Error, ValueError):
            return 0

    @staticmethod
    def _heuristic_frame_emotion(decoded_len: int) -> str:
        if decoded_len <= 0:
            return "neutral"
        labels = ["neutral", "happy", "surprise", "fear"]
        return labels[decoded_len % len(labels)]

    @staticmethod
    def _face_detector_status() -> str:
        try:
            import cv2  # type: ignore

            return "opencv available"
        except Exception:
            return "opencv missing; frame endpoint uses center crop fallback"

    @staticmethod
    def _dependency_status() -> dict[str, str]:
        import importlib.util

        modules = {
            "PyMuPDF": "fitz",
            "OpenCV": "cv2",
            "PyTorch": "torch",
            "timm": "timm",
            "PyQt5": "PyQt5",
            "scikit-learn": "sklearn",
            "requests": "requests",
            "faster-whisper": "faster_whisper",
        }
        return {name: ("available" if importlib.util.find_spec(module) else "missing") for name, module in modules.items()}
