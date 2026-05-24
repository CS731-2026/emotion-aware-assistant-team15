from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_api_key import _ensure_gitignore_entry, _replace_or_append
from scripts.diagnose_openface import check_candidate


def configure_openface(project_root: str | Path, binary_path: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    binary = Path(binary_path).expanduser()
    if not binary.is_absolute():
        binary = (root / binary).resolve()
    status = check_candidate(binary)
    if not status["exists"]:
        raise FileNotFoundError(f"OpenFace FeatureExtraction binary not found: {binary}")
    if not status["executable"]:
        raise ValueError(f"OpenFace FeatureExtraction binary is not executable: {binary}")
    if not status["can_run"]:
        raise ValueError(f"OpenFace FeatureExtraction binary could not be run safely: {binary}")

    env_path = root / ".env.local"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for key, value in {
        "FACE_DETECTOR": "openface",
        "OPENFACE_FEATURE_EXTRACTION_BIN": str(binary),
    }.items():
        lines, _ = _replace_or_append(lines, key, value)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    _ensure_gitignore_entry(root)
    ensure_openface_gitignore_entries(root)
    return {
        "saved": True,
        "env_path": str(env_path),
        "binary_path": str(binary),
        "can_run": bool(status["can_run"]),
    }


def ensure_openface_gitignore_entries(project_root: str | Path = ROOT) -> None:
    root = Path(project_root)
    gitignore = root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    stripped = {line.strip() for line in existing}
    entries = [
        "external/",
        "external/OpenFace/",
        "runtime_uploads/openface_build_logs/",
        "FeatureExtraction",
        "FeatureExtraction.exe",
        "*.exe",
        "*.csv",
        "*.hog",
        "*.params",
        "*.dat",
        "*.model",
    ]
    changed = False
    for entry in entries:
        if entry not in stripped:
            existing.append(entry)
            changed = True
    if changed:
        gitignore.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure this project to use TadasBaltrusaitis/OpenFace FeatureExtraction.")
    parser.add_argument("--bin", required=True, help="Path to the FeatureExtraction binary.")
    parser.add_argument("--project-root", default=str(ROOT), help="Project root containing .env.local.")
    args = parser.parse_args(argv)

    try:
        result = configure_openface(args.project_root, args.bin)
    except Exception as exc:
        print(f"OpenFace configuration failed: {exc}", file=sys.stderr)
        return 1
    print("Saved OpenFace configuration to .env.local")
    print(f"FeatureExtraction: {result['binary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
