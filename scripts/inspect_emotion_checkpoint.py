from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


def find_checkpoint_candidate(source: Path) -> tuple[Path, str]:
    if source.is_file():
        return source, "file"
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint source does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Checkpoint source is not a file or directory: {source}")
    candidates = [path for suffix in CHECKPOINT_SUFFIXES for path in source.rglob(f"*{suffix}") if path.is_file()]
    if candidates:
        candidates.sort(key=lambda path: (("best" not in path.name.lower()), -path.stat().st_size, -path.stat().st_mtime))
        return candidates[0], "file"
    if (source / "data.pkl").exists() and (source / "version").exists() and (source / "data").is_dir():
        return source, "extracted_torch_archive"
    raise FileNotFoundError(f"No .pt/.pth/.ckpt checkpoint found under {source}")


def zip_extracted_checkpoint(source: Path, dest: Path) -> Path:
    root_name = source.name or "checkpoint"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, strict_timestamps=False) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.write(path, f"{root_name}/{path.relative_to(source).as_posix()}")
    return dest


def tensor_shape(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    try:
        return "x".join(str(item) for item in shape)
    except Exception:
        return str(shape)


def inspect_checkpoint_file(checkpoint_path: str | Path) -> dict[str, Any]:
    from emotion_aware_assistant.emotion.raw_emotion_pipeline import inspect_checkpoint_file as inspect_file

    return inspect_file(Path(checkpoint_path).expanduser())


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a raw-emotion or academic-state checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or directory to inspect.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    source = Path(args.checkpoint).expanduser()
    try:
        candidate, candidate_type = find_checkpoint_candidate(source)
        load_path = candidate
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if candidate_type == "extracted_torch_archive":
            temp_dir = tempfile.TemporaryDirectory()
            load_path = zip_extracted_checkpoint(candidate, Path(temp_dir.name) / f"{candidate.name}.pt")
        info = inspect_checkpoint_file(load_path)
        info.update({"resolved_checkpoint_path": str(candidate), "load_path": str(load_path), "candidate_type": candidate_type})
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"checkpoint path: {info['resolved_checkpoint_path']}")
            print(f"load path: {info['load_path']}")
            print(f"arch: {info.get('arch') or info.get('architecture') or ''}")
            print(f"num_classes: {info.get('num_classes')}")
            print(f"classes: {info.get('classes')}")
            print(f"detected model mode: {info.get('detected_model_mode')}")
            print(f"model_state_dict presence: {bool(info.get('model_state_dict_present'))}")
            print(f"checkpoint keys: {info.get('checkpoint_keys')}")
            print(f"sample keys: {info.get('sample_keys')}")
        if temp_dir:
            temp_dir.cleanup()
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
