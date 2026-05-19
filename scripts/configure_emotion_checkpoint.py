from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from emotion_aware_assistant.core.config import LOCAL_ENV_FILE, PROJECT_ROOT
from emotion_aware_assistant.emotion.raw_emotion_pipeline import inspect_checkpoint_file
from scripts.configure_api_key import _ensure_gitignore_entry, _replace_or_append


def configure_emotion_checkpoint(
    project_root: str | Path,
    checkpoint: str | Path,
    mode: str = "auto",
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    mode = _normalize_mode(mode)
    info = inspect_checkpoint_file(checkpoint_path)
    detected_mode = str(info.get("detected_model_mode") or info.get("model_output_type") or "unknown")

    env_path = root / LOCAL_ENV_FILE
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updates = {
        "EMOTION_CHECKPOINT_PATH": str(checkpoint_path),
        "EMOTION_MODEL_MODE": mode,
    }
    if detected_mode == "raw_emotion":
        updates["RAW_EMOTION_CHECKPOINT_PATH"] = str(checkpoint_path)
    for key, value in updates.items():
        lines, _ = _replace_or_append(lines, key, value)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    _ensure_gitignore_entry(root)
    os.environ.update(updates)
    result = {
        "saved": True,
        "env_local_present": env_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "mode": mode,
        "detected_model_mode": detected_mode,
        "classes": list(info.get("classes") or []),
        "architecture": info.get("architecture") or info.get("arch") or "",
        "raw_detection_available": detected_mode == "raw_emotion",
    }
    if not quiet:
        print(f"checkpoint_path: {result['checkpoint_path']}")
        print(f"mode: {result['mode']}")
        print(f"detected_model_mode: {result['detected_model_mode']}")
        print(f"classes: {result['classes']}")
        print(f"raw_detection_available: {result['raw_detection_available']}")
        print(f"wrote: {env_path}")
    return result


def _normalize_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "raw": "raw_emotion",
        "raw_emotion_model": "raw_emotion",
        "academic": "academic_state",
        "academic_state_model": "academic_state",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "raw_emotion", "academic_state"}:
        raise ValueError("mode must be auto, raw_emotion, or academic_state.")
    return mode


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure the local emotion checkpoint path.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt/.pth/.ckpt checkpoint.")
    parser.add_argument("--mode", default="auto", choices=("auto", "raw_emotion", "academic_state"))
    args = parser.parse_args()
    configure_emotion_checkpoint(PROJECT_ROOT, args.checkpoint, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
