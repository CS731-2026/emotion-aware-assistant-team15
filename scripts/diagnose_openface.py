from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.emotion.face_detector import OPENFACE_CANDIDATE_PATHS


SUGGESTED_APT_COMMANDS = [
    "sudo apt-get update",
    "sudo apt-get install -y build-essential cmake git wget unzip pkg-config",
    "sudo apt-get install -y libopenblas-dev libopencv-dev libdlib-dev libboost-all-dev libtbb-dev",
]


def candidate_feature_extraction_paths(
    extra_paths: Iterable[str | Path] | None = None,
    *,
    include_path: bool = True,
    project_root: str | Path = ROOT,
) -> list[Path]:
    paths: list[Path] = []
    provided_paths = list(extra_paths) if extra_paths is not None else None
    for item in provided_paths or []:
        paths.append(Path(item).expanduser())
    if provided_paths is not None and not include_path:
        return _dedupe_paths(paths)
    if include_path:
        path_candidate = shutil.which("FeatureExtraction")
        if path_candidate:
            paths.append(Path(path_candidate))
    root = Path(project_root)
    for candidate in OPENFACE_CANDIDATE_PATHS:
        path = Path(candidate).expanduser()
        paths.append(path if path.is_absolute() else root / path)
    return _dedupe_paths(paths)


def diagnose_openface(
    candidate_paths: Iterable[str | Path] | None = None,
    *,
    include_path: bool = True,
    timeout: float = 5.0,
    project_root: str | Path = ROOT,
) -> dict[str, Any]:
    candidates = candidate_feature_extraction_paths(
        candidate_paths,
        include_path=include_path,
        project_root=project_root,
    )
    checked = [check_candidate(path, timeout=timeout) for path in candidates]
    runnable = next((item for item in checked if item["can_run"]), None)
    existing = next((item for item in checked if item["exists"]), None)
    selected = runnable or existing
    warning = None
    if not selected:
        warning = "OpenFace FeatureExtraction binary was not found."
    elif not selected["executable"]:
        warning = "OpenFace FeatureExtraction binary exists but is not executable."
    elif not selected["can_run"]:
        warning = "OpenFace FeatureExtraction binary exists but could not be run."
    return {
        "found": bool(selected and selected["exists"]),
        "binary_path": selected["path"] if selected and selected["exists"] else None,
        "executable": bool(selected and selected["executable"]),
        "can_run": bool(selected and selected["can_run"]),
        "help_output_first_lines": selected["help_output_first_lines"] if selected else [],
        "warning": warning,
        "candidates": checked,
    }


def check_candidate(path: str | Path, *, timeout: float = 5.0) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    exists = candidate.exists() and candidate.is_file()
    executable = bool(exists and os.access(candidate, os.X_OK))
    result = {
        "path": str(candidate),
        "exists": exists,
        "executable": executable,
        "can_run": False,
        "help_output_first_lines": [],
        "error": "",
    }
    if not exists:
        result["error"] = "not found"
        return result
    if not executable:
        result["error"] = "not executable"
        return result
    for args in (["-help"], ["--help"], []):
        probe = _run_command([str(candidate), *args], timeout=timeout)
        output = _first_lines("\n".join([probe.get("stdout", ""), probe.get("stderr", "")]))
        if output:
            result["help_output_first_lines"] = output
        if probe["returncode"] == 0 or output:
            result["can_run"] = True
            return result
        result["error"] = probe.get("error") or f"exit {probe['returncode']}"
    return result


def diagnose_build_environment() -> dict[str, Any]:
    tools = {
        name: tool_status(name, version_args)
        for name, version_args in {
            "cmake": ["--version"],
            "git": ["--version"],
            "make": ["--version"],
            "gcc": ["--version"],
            "g++": ["--version"],
            "pkg-config": ["--version"],
            "wget": ["--version"],
            "curl": ["--version"],
            "unzip": ["-v"],
        }.items()
    }
    cxx17 = check_cpp17_support() if tools["g++"]["available"] else {"available": False, "warning": "g++ not available"}
    libraries = {
        "opencv": pkg_config_status(["opencv4", "opencv"]),
        "openblas": pkg_config_status(["openblas"]) or header_or_library_status("openblas", ["/usr/include/openblas_config.h", "/usr/lib/x86_64-linux-gnu/libopenblas.so"]),
        "dlib": pkg_config_status(["dlib-1", "dlib"]) or header_or_library_status("dlib", ["/usr/include/dlib"]),
        "boost": header_or_library_status("boost", ["/usr/include/boost"]),
        "tbb": pkg_config_status(["tbb"]) or header_or_library_status("tbb", ["/usr/include/tbb"]),
    }
    required_tools = ["cmake", "git", "make", "gcc", "g++", "pkg-config", "unzip"]
    missing_tools = [name for name in required_tools if not tools[name]["available"]]
    missing_libraries = [name for name, status in libraries.items() if not status["available"]]
    feasible = not missing_tools and not missing_libraries and bool(cxx17["available"])
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "tools": tools,
        "cxx17": cxx17,
        "libraries": libraries,
        "missing_tools": missing_tools,
        "missing_libraries": missing_libraries,
        "build_feasible_without_sudo": feasible,
        "suggested_commands": [] if feasible else SUGGESTED_APT_COMMANDS,
    }


def tool_status(name: str, version_args: list[str]) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None, "version": "", "warning": f"{name} was not found."}
    result = _run_command([path, *version_args], timeout=5)
    first_line = _first_lines("\n".join([result.get("stdout", ""), result.get("stderr", "")]), limit=1)
    return {
        "available": True,
        "path": path,
        "version": first_line[0] if first_line else "",
        "warning": result.get("error", ""),
    }


def check_cpp17_support() -> dict[str, Any]:
    source = "#include <optional>\nint main(){std::optional<int> value=1; return *value == 1 ? 0 : 1;}\n"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "cpp17-check"
        result = _run_command(["g++", "-std=c++17", "-x", "c++", "-", "-o", str(output)], timeout=10, input_text=source)
    return {
        "available": result["returncode"] == 0,
        "warning": "" if result["returncode"] == 0 else _first_lines(result.get("stderr", ""), limit=3),
    }


def pkg_config_status(names: list[str]) -> dict[str, Any]:
    if not shutil.which("pkg-config"):
        return {"available": False, "method": "pkg-config", "package": "", "warning": "pkg-config was not found."}
    for name in names:
        exists = _run_command(["pkg-config", "--exists", name], timeout=5)
        if exists["returncode"] != 0:
            continue
        version = _run_command(["pkg-config", "--modversion", name], timeout=5)
        lines = _first_lines(version.get("stdout", ""), limit=1)
        return {"available": True, "method": "pkg-config", "package": name, "version": lines[0] if lines else ""}
    return {"available": False, "method": "pkg-config", "package": "", "warning": f"None found: {', '.join(names)}"}


def header_or_library_status(name: str, paths: list[str]) -> dict[str, Any]:
    found = [path for path in paths if Path(path).exists()]
    return {
        "available": bool(found),
        "method": "filesystem",
        "path": found[0] if found else "",
        "warning": "" if found else f"{name} development files were not detected.",
    }


def _run_command(command: list[str], *, timeout: float, input_text: str | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[:4000],
            "stderr": completed.stderr[:4000],
            "error": "",
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "stdout": "", "stderr": "", "error": "timeout"}
    except OSError as exc:
        return {"returncode": 127, "stdout": "", "stderr": "", "error": str(exc)}


def _first_lines(text: str, *, limit: int = 6) -> list[str]:
    return [line[:240] for line in text.splitlines() if line.strip()][:limit]


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
    parser = argparse.ArgumentParser(description="Diagnose TadasBaltrusaitis/OpenFace FeatureExtraction availability and build readiness.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--candidate", action="append", default=[], help="Extra FeatureExtraction candidate path.")
    parser.add_argument("--no-path", action="store_true", help="Do not search PATH.")
    args = parser.parse_args(argv)

    payload = {
        "openface": diagnose_openface(args.candidate, include_path=not args.no_path),
        "build_environment": diagnose_build_environment(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    openface = payload["openface"]
    print("OpenFace FeatureExtraction diagnostic")
    print(f"found: {openface['found']}")
    print(f"binary_path: {openface['binary_path'] or 'not found'}")
    print(f"executable: {openface['executable']}")
    print(f"can_run: {openface['can_run']}")
    if openface.get("warning"):
        print(f"warning: {openface['warning']}")
    if openface["help_output_first_lines"]:
        print("help output:")
        for line in openface["help_output_first_lines"]:
            print(f"  {line}")

    environment = payload["build_environment"]
    print("\nBuild readiness")
    print(f"platform: {environment['platform']['system']} {environment['platform']['release']} {environment['platform']['machine']}")
    print(f"build_feasible_without_sudo: {environment['build_feasible_without_sudo']}")
    if environment["missing_tools"]:
        print(f"missing tools: {', '.join(environment['missing_tools'])}")
    if environment["missing_libraries"]:
        print(f"missing libraries: {', '.join(environment['missing_libraries'])}")
    if environment["suggested_commands"]:
        print("Suggested install commands, not executed:")
        for command in environment["suggested_commands"]:
            print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
