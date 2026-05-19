from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_api_key import _ensure_gitignore_entry

DEFAULT_DESTINATION = ROOT / "models" / "face_detector" / "yolov8n-face.pt"
SEARCH_PATTERNS = (
    "*yolo*face*.pt",
    "*yolov8*face*.pt",
    "yolov8n-face.pt",
    "yolo-face.pt",
    "best.pt",
    "*.onnx",
)


def default_search_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home,
        ROOT,
        home / "Downloads",
        home / "下载",
        ROOT / "models" / "face_detector",
    ]
    return _dedupe_paths(roots)


def find_weight_candidates(search_roots: Iterable[str | Path] | None = None) -> list[dict[str, object]]:
    roots = _dedupe_paths(Path(root).expanduser() for root in (search_roots or default_search_roots()))
    candidates: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if _looks_like_weight(root):
                candidates[str(root.resolve())] = root.resolve()
            continue
        for pattern in SEARCH_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file() and _looks_like_weight(path):
                    candidates[str(path.resolve())] = path.resolve()
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
        for path in sorted(candidates.values(), key=lambda item: str(item).lower())
    ]


def install_weight(source: str | Path, destination: str | Path = DEFAULT_DESTINATION) -> Path:
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {source_path}")
    if not _looks_like_weight(source_path):
        raise ValueError("Expected a .pt, .pth, .ckpt, .onnx, or .engine weight file.")
    destination_path = Path(destination).expanduser()
    if not destination_path.is_absolute():
        destination_path = ROOT / destination_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    _ensure_gitignore_entry(ROOT)
    _ensure_face_detector_gitignore_entries()
    return destination_path


def _ensure_face_detector_gitignore_entries() -> None:
    gitignore = ROOT / ".gitignore"
    entries = [
        "models/face_detector/*.pt",
        "models/face_detector/*.pth",
        "models/face_detector/*.ckpt",
        "models/face_detector/*.onnx",
        "models/face_detector/*.engine",
    ]
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    stripped = {line.strip() for line in existing}
    changed = False
    for entry in entries:
        if entry not in stripped:
            existing.append(entry)
            changed = True
    if changed:
        gitignore.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")


def _looks_like_weight(path: Path) -> bool:
    return path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = os.path.normcase(str(path.expanduser()))
        if key in seen:
            continue
        deduped.append(path.expanduser())
        seen.add(key)
    return deduped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find local YOLO face-detector weight files.")
    parser.add_argument("--root", action="append", default=[], help="Search root. Can be passed multiple times.")
    parser.add_argument("--install", default="", help="Copy this weight file to models/face_detector/yolov8n-face.pt.")
    args = parser.parse_args(argv)

    if args.install:
        destination = install_weight(args.install)
        print(f"Installed face detector weights to {destination}")
        print("Model weights remain ignored by Git.")
        return 0

    roots = [Path(root).expanduser() for root in args.root] if args.root else default_search_roots()
    candidates = find_weight_candidates(roots)
    if not candidates:
        print("No YOLO face-detector weight candidates found.")
        return 0
    print("YOLO face-detector weight candidates:")
    for index, item in enumerate(candidates, start=1):
        print(f"{index}. {item['path']} ({item['size_mb']} MB)")
    print("No files were copied. Use --install <path> to install one locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
