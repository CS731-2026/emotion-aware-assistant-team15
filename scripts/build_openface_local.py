from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.configure_openface import configure_openface, ensure_openface_gitignore_entries
from scripts.diagnose_openface import diagnose_build_environment
from scripts.test_openface_feature_extraction import run_binary_check


OPENFACE_REPO = "https://github.com/TadasBaltrusaitis/OpenFace.git"
DEFAULT_SOURCE_DIR = ROOT / "external" / "OpenFace"
LOG_DIR = ROOT / "runtime_uploads" / "openface_build_logs"


def build_openface_local(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    *,
    jobs: int | None = None,
    skip_model_download: bool = False,
    configure_project: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser()
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    ensure_openface_gitignore_entries(ROOT)
    environment = diagnose_build_environment()
    if not environment["build_feasible_without_sudo"]:
        return {
            "ok": False,
            "stage": "dependency_check",
            "warning": "OpenFace build dependencies appear incomplete.",
            "build_environment": environment,
        }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"openface_build_{time.strftime('%Y%m%d_%H%M%S')}.log"
    build_jobs = jobs or safe_job_count()

    with log_path.open("w", encoding="utf-8") as log:
        if not source.exists():
            result = run_logged(["git", "clone", "--depth", "1", OPENFACE_REPO, str(source)], log, cwd=ROOT)
            if result.returncode != 0:
                return _failure("clone", log_path, result)
        elif not (source / ".git").exists():
            return {
                "ok": False,
                "stage": "source_check",
                "warning": f"{source} exists but is not an OpenFace git checkout.",
                "log_path": str(log_path),
            }

        result = run_logged(["git", "submodule", "update", "--init", "--recursive"], log, cwd=source)
        if result.returncode != 0:
            return _failure("submodule", log_path, result)

        if not skip_model_download:
            model_script = first_existing(source, ["download_models.sh", "download_models.ps1"])
            if model_script and model_script.suffix != ".ps1":
                result = run_logged(["bash", str(model_script)], log, cwd=source)
                if result.returncode != 0:
                    return _failure("model_download", log_path, result)

        build_dir = source / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        result = run_logged(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."], log, cwd=build_dir)
        if result.returncode != 0:
            return _failure("cmake_configure", log_path, result)

        result = run_logged(["cmake", "--build", ".", "--config", "Release", "-j", str(build_jobs)], log, cwd=build_dir)
        if result.returncode != 0:
            return _failure("compile", log_path, result)

    binary = find_built_feature_extraction(source)
    if not binary:
        return {"ok": False, "stage": "binary_verification", "warning": "FeatureExtraction binary was not found after build.", "log_path": str(log_path)}
    check = run_binary_check(binary)
    if not check["ok"]:
        return {"ok": False, "stage": "binary_verification", "warning": check.get("warning") or "FeatureExtraction could not run.", "log_path": str(log_path)}
    configured = configure_openface(ROOT, binary) if configure_project else None
    return {
        "ok": True,
        "stage": "complete",
        "binary_path": str(binary),
        "log_path": str(log_path),
        "configured": bool(configured),
    }


def run_logged(command: list[str], log, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    log.write(f"\n$ {' '.join(command)}\n")
    log.flush()
    completed = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    log.write(f"\nexit_code={completed.returncode}\n")
    log.flush()
    return completed


def safe_job_count() -> int:
    return max(1, min(os.cpu_count() or 2, 4))


def first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def find_built_feature_extraction(source: Path) -> Path | None:
    candidates = [
        source / "build" / "bin" / "FeatureExtraction",
        source / "build" / "bin" / "FeatureExtraction.exe",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in source.glob("build/**/FeatureExtraction*"):
        if candidate.is_file() and candidate.name in {"FeatureExtraction", "FeatureExtraction.exe"}:
            return candidate
    return None


def _failure(stage: str, log_path: Path, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "returncode": result.returncode,
        "warning": f"OpenFace build failed during {stage}.",
        "log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clone and build TadasBaltrusaitis/OpenFace locally without sudo.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="OpenFace source directory, default external/OpenFace.")
    parser.add_argument("--jobs", type=int, default=0, help="Build jobs. Defaults to a safe capped CPU count.")
    parser.add_argument("--skip-model-download", action="store_true", help="Skip repository model download script.")
    parser.add_argument("--configure-project", action="store_true", help="Write .env.local after a successful build.")
    args = parser.parse_args(argv)

    result = build_openface_local(
        args.source_dir,
        jobs=args.jobs or None,
        skip_model_download=args.skip_model_download,
        configure_project=args.configure_project,
    )
    print(f"ok: {result['ok']}")
    print(f"stage: {result.get('stage')}")
    if result.get("binary_path"):
        print(f"FeatureExtraction: {result['binary_path']}")
    if result.get("log_path"):
        print(f"log: {result['log_path']}")
    if result.get("warning"):
        print(f"warning: {result['warning']}")
    if result.get("build_environment", {}).get("suggested_commands"):
        print("Suggested install commands, not executed:")
        for command in result["build_environment"]["suggested_commands"]:
            print(f"  {command}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
