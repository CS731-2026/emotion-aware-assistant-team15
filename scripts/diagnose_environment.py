from __future__ import annotations

import importlib.util
import importlib
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.core.config import parse_env_file  # noqa: E402


PACKAGES = {
    "PyMuPDF": "fitz",
    "OpenCV": "cv2",
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "Pillow": "PIL",
    "PyQt5": "PyQt5",
    "scikit-learn": "sklearn",
    "requests": "requests",
    "httpx": "httpx",
    "faster-whisper": "faster_whisper",
}


def available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def availability_label(module: str) -> str:
    if importlib.util.find_spec(module) is None:
        return "missing"
    try:
        importlib.import_module(module)
    except Exception as exc:
        return f"broken: {exc}"
    return "available"


def webcam_status() -> str:
    if availability_label("cv2") != "available":
        return "not checked: OpenCV missing"
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(0)
        ok = bool(cap.isOpened())
        cap.release()
        return "available" if ok else "not available"
    except Exception as exc:
        return f"check failed: {exc}"


def environment_config_status(root: str | Path = ROOT) -> dict[str, object]:
    root_path = Path(root)
    env_path = root_path / ".env.local"
    local_values = parse_env_file(env_path)

    def configured_value(key: str) -> str:
        return (os.environ.get(key) or local_values.get(key) or "").strip()

    return {
        "env_local_present": env_path.exists(),
        "gemini_api_key_configured": bool(configured_value("GEMINI_API_KEY")),
        "llm_provider": configured_value("LLM_PROVIDER") or "not configured",
        "strategy_planner_provider": configured_value("STRATEGY_PLANNER_PROVIDER") or "not configured",
    }


def main() -> int:
    print("Environment diagnostic")
    print(f"Python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform: {platform.platform()}")
    print()
    print("Package availability:")
    for name, module in PACKAGES.items():
        print(f"- {name}: {availability_label(module)}")
    print()
    print(f"Webcam: {webcam_status()}")
    env_status = environment_config_status(ROOT)
    print(f".env.local present: {str(env_status['env_local_present']).lower()}")
    print(f"GEMINI_API_KEY configured: {str(env_status['gemini_api_key_configured']).lower()}")
    print(f"LLM_PROVIDER: {env_status['llm_provider']}")
    print(f"STRATEGY_PLANNER_PROVIDER: {env_status['strategy_planner_provider']}")
    model_dir = Path("models/emotion_model")
    metadata_exists = (model_dir / "metadata.json").exists()
    checkpoint_exists = any((model_dir / name).exists() for name in ("best_model.pt", "best_model.pth", "best_model.ckpt"))
    print(f"Emotion metadata: {'present' if metadata_exists else 'absent'}")
    print(f"Emotion checkpoint: {'present' if checkpoint_exists else 'absent'}")
    try:
        from emotion_aware_assistant.emotion.teammate_emotion_adapter import TeammateEmotionAdapter

        adapter = TeammateEmotionAdapter()
        status = adapter.load()
        print(f"Emotion model load: {'loaded' if status.get('model_loaded') else 'not loaded'}")
        if status.get("loading_error"):
            print(f"Emotion model loading error: {status.get('loading_error')}")
        print(f"Emotion model output type: {status.get('model_output_type')}")
        print(f"Raw facial emotion available: {status.get('raw_emotion_available')}")
    except Exception as exc:
        print(f"Emotion model load: check failed: {exc}")
    print()
    print("Recommended installs:")
    print("- Minimal local demo: pip install -e .")
    print("- Web/PDF/vision extras: pip install -r requirements.txt")
    print("- If PDF upload fails: pip install pymupdf")
    print("- If webcam/model inference is needed: pip install opencv-python torch torchvision timm")
    print("- If TF-IDF retrieval is needed: pip install scikit-learn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
