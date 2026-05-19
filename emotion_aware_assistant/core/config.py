from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {
        "default_mode": "web",
        "log_dir": "logs",
        "demo_mode_if_missing_dependencies": True,
    },
    "emotion": {
        "allowed_labels": [
            "neutral",
            "happy",
            "angry",
            "sad",
            "fear",
            "surprise",
            "disgust",
            "contempt",
        ],
        "default_recognizer": "auto",
        "buffer_size": 10,
        "confidence_threshold": 0.35,
        "trend_window_sec": 6,
        "trend_update_interval_sec": 0.5,
        "hysteresis_updates": 3,
        "high_confidence_switch_threshold": 0.80,
        "teammate_model_dir": "models/emotion_model",
    },
    "face_detection": {
        "preferred": "yolo",
        "yolo_weights": "models/face_detector/yolov8n-face.pt",
        "fallback": "haar",
    },
    "llm": {
        "default_client": "openrouter",
        "fallback_client": "dummy",
        "default_model": "gpt4o_mini",
        "timeout_sec": 60,
        "models": [
            {"alias": "gpt4o_mini", "name": "openai/gpt-4o-mini"},
            {"alias": "claude_haiku", "name": "anthropic/claude-3.5-haiku"},
            {"alias": "gemini_flash", "name": "google/gemini-flash-1.5"},
            {"alias": "deepseek", "name": "deepseek/deepseek-chat"},
        ],
    },
    "paper": {"chunk_size": 1000, "chunk_overlap": 150, "top_k_chunks": 3},
    "ui": {"show_webcam_preview": True, "start_camera_on_launch": False},
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ENV_FILE = ".env.local"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = re.sub(r"^export\s+", "", key.strip())
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_env_file(path: str | Path = ".env", override: bool = False) -> dict[str, Any]:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    env_path = Path(path)
    values = parse_env_file(env_path)
    loaded_keys: list[str] = []
    skipped_existing_keys: list[str] = []
    for key, value in values.items():
        if not override and key in os.environ:
            skipped_existing_keys.append(key)
            continue
        os.environ[key] = value
        loaded_keys.append(key)
    return {
        "path": str(env_path),
        "present": env_path.exists(),
        "loaded_keys": loaded_keys,
        "skipped_existing_keys": skipped_existing_keys,
    }


def load_project_local_env(project_root: str | Path | None = None, override: bool = False) -> dict[str, Any]:
    """Load the project-root .env.local file without printing secret values."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return load_env_file(root / LOCAL_ENV_FILE, override=override)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load configuration, falling back to defaults when PyYAML is unavailable."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path)
    if not cfg_path.exists():
        return config

    try:
        import yaml  # type: ignore
    except Exception:
        return config

    loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return config
    return _deep_merge(config, loaded)


def get_nested(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
