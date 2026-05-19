from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")
DEFAULT_METADATA = {
    "model_output_type": "academic_state",
    "architecture": "convnext_tiny.fb_in22k_ft_in1k",
    "framework": "timm",
    "num_classes": 4,
    "classes": ["boredom", "confusion", "engagement", "frustration"],
    "class_to_idx": {
        "boredom": 0,
        "confusion": 1,
        "engagement": 2,
        "frustration": 3,
    },
    "input_size": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "checkpoint_key": "model_state_dict",
    "source_path": "/home/rli/下载/best",
    "best_epoch": 19,
    "best_val_acc": 80.67,
    "notes": "4-class academic-state model. Raw 8-class facial emotion is unavailable for this checkpoint.",
}


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


def zip_extracted_checkpoint(source: Path, dest: Path) -> None:
    root_name = source.name or "checkpoint"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED, strict_timestamps=False) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.write(path, f"{root_name}/{path.relative_to(source).as_posix()}")


def install_checkpoint(source: Path, dest: Path) -> tuple[Path, str]:
    candidate, candidate_type = find_checkpoint_candidate(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if candidate_type == "extracted_torch_archive":
        zip_extracted_checkpoint(candidate, dest)
    else:
        shutil.copy2(candidate, dest)
    metadata = dict(DEFAULT_METADATA)
    metadata["source_path"] = str(source)
    (dest.parent / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return candidate, candidate_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the teammate academic-state checkpoint for local inference.")
    parser.add_argument("--source", required=True, help="Source checkpoint file or directory.")
    parser.add_argument("--dest", default="models/emotion_model/best_model.pt", help="Destination checkpoint path.")
    args = parser.parse_args()
    source = Path(args.source).expanduser()
    dest = Path(args.dest)
    try:
        candidate, candidate_type = install_checkpoint(source, dest)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"source_candidate: {candidate}")
    print(f"source_type: {candidate_type}")
    print(f"installed_checkpoint: {dest}")
    print(f"metadata: {dest.parent / 'metadata.json'}")
    print("note: model weight files are ignored by .gitignore and should not be staged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
