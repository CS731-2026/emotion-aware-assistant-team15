from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.core.config import LOCAL_ENV_FILE, parse_env_file
from emotion_aware_assistant.emotion.face_detector import parse_openface_csv, run_openface_feature_extraction
from scripts.diagnose_openface import check_candidate, diagnose_openface


def find_feature_extraction(binary_arg: str | Path | None = None, project_root: str | Path = ROOT) -> Path | None:
    if binary_arg:
        return Path(binary_arg).expanduser()
    values = parse_env_file(Path(project_root) / LOCAL_ENV_FILE)
    configured = values.get("OPENFACE_FEATURE_EXTRACTION_BIN", "").strip()
    if configured:
        return Path(configured).expanduser()
    discovered = diagnose_openface()
    return Path(discovered["binary_path"]) if discovered.get("binary_path") else None


def run_binary_check(binary_path: str | Path | None) -> dict[str, Any]:
    if not binary_path:
        return {"ok": False, "warning": "OpenFace FeatureExtraction binary was not found."}
    status = check_candidate(binary_path)
    return {
        "ok": bool(status["exists"] and status["executable"] and status["can_run"]),
        "binary_path": status["path"],
        "warning": status.get("error") or "",
        "help_output_first_lines": status.get("help_output_first_lines") or [],
    }


def run_image_check(
    binary_path: str | Path,
    image_path: str | Path,
    *,
    keep_output: bool = False,
    timeout: float = 8.0,
) -> dict[str, Any]:
    binary = Path(binary_path).expanduser()
    image = Path(image_path).expanduser()
    if not image.exists() or not image.is_file():
        return {"ok": False, "warning": f"Image not found: {image}"}
    temp_dir = tempfile.mkdtemp(prefix="openface-test-")
    output_dir = Path(temp_dir) / "out"
    try:
        result = run_openface_feature_extraction(binary, image, output_dir, timeout=timeout)
        if result.returncode != 0:
            return {"ok": False, "warning": "FeatureExtraction failed.", "returncode": result.returncode}
        csv_files = sorted(output_dir.glob("*.csv"))
        if not csv_files:
            return {"ok": False, "warning": "FeatureExtraction did not produce a CSV output."}
        parsed = parse_openface_csv(csv_files[0])
        return {
            "ok": bool(parsed.get("success")),
            "success": bool(parsed.get("success")),
            "confidence": parsed.get("confidence"),
            "landmark_count": parsed.get("landmark_count"),
            "bbox": parsed.get("bbox"),
            "pose": parsed.get("pose") or {},
            "au_count": len(parsed.get("aus") or {}),
            "warning": parsed.get("warning") or "",
            "output_dir": str(output_dir) if keep_output else "",
        }
    finally:
        if not keep_output:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a safe FeatureExtraction smoke test.")
    parser.add_argument("--bin", default="", help="Path to FeatureExtraction.")
    parser.add_argument("--image", default="", help="Optional image path for one-image OpenFace analysis.")
    parser.add_argument("--keep-output", action="store_true", help="Keep the temporary OpenFace output directory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    binary = find_feature_extraction(args.bin or None)
    check = run_binary_check(binary)
    if not check["ok"]:
        if args.json:
            print(json.dumps(check, indent=2))
        else:
            print(f"OpenFace binary check failed: {check['warning']}")
        return 1
    if not args.image:
        check["image_test"] = "not run; provide --image /path/to/face_image.jpg to parse landmarks"
        if args.json:
            print(json.dumps(check, indent=2))
        else:
            print("OpenFace binary can run.")
            print("No image provided; binary run test only.")
        return 0

    result = run_image_check(check["binary_path"], args.image, keep_output=args.keep_output)
    payload = {"binary_check": check, "image_check": result}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"success: {result.get('success')}")
        print(f"confidence: {result.get('confidence')}")
        print(f"landmark_count: {result.get('landmark_count')}")
        print(f"bbox: {result.get('bbox')}")
        print(f"pose fields: {', '.join((result.get('pose') or {}).keys()) or 'none'}")
        print(f"AU fields count: {result.get('au_count')}")
        if result.get("warning"):
            print(f"warning: {result['warning']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
