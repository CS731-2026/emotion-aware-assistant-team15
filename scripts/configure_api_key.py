from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "LLM_PROVIDER": "gemini",
    "GEMINI_MODEL": "gemini-flash-latest",
    "GEMINI_EMBEDDING_MODEL": "gemini-embedding-001",
    "STRATEGY_PLANNER_PROVIDER": "gemini",
}


def _key_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=", re.ASCII)


def _replace_or_append(lines: list[str], key: str, value: str) -> tuple[list[str], bool]:
    replacement = f"{key}={value}"
    updated = False
    existed = False
    next_lines: list[str] = []
    for line in lines:
        if _key_pattern(key).match(line):
            existed = True
            if not updated:
                next_lines.append(replacement)
                updated = True
            continue
        next_lines.append(line)
    if not existed:
        next_lines.append(replacement)
    return next_lines, existed


def _ensure_gitignore_entry(project_root: Path) -> None:
    gitignore = project_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if ".env.local" not in {line.strip() for line in existing}:
        existing.append(".env.local")
        gitignore.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")


def configure_gemini_key(
    project_root: str | Path,
    api_key: str,
    *,
    llm_provider: str = DEFAULTS["LLM_PROVIDER"],
    gemini_model: str = DEFAULTS["GEMINI_MODEL"],
    gemini_embedding_model: str = DEFAULTS["GEMINI_EMBEDDING_MODEL"],
    strategy_planner_provider: str = DEFAULTS["STRATEGY_PLANNER_PROVIDER"],
    quiet: bool = False,
) -> dict[str, object]:
    root = Path(project_root)
    env_path = root / ".env.local"
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("Gemini API key cannot be empty.")

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    settings = {
        "LLM_PROVIDER": llm_provider.strip() or DEFAULTS["LLM_PROVIDER"],
        "GEMINI_MODEL": gemini_model.strip() or DEFAULTS["GEMINI_MODEL"],
        "GEMINI_EMBEDDING_MODEL": gemini_embedding_model.strip() or DEFAULTS["GEMINI_EMBEDDING_MODEL"],
        "STRATEGY_PLANNER_PROVIDER": strategy_planner_provider.strip() or DEFAULTS["STRATEGY_PLANNER_PROVIDER"],
        "GEMINI_API_KEY": api_key,
    }
    existing_key = any(_key_pattern("GEMINI_API_KEY").match(line) for line in lines)
    for key, value in settings.items():
        lines, _ = _replace_or_append(lines, key, value)

    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    _ensure_gitignore_entry(root)

    if not quiet:
        if existing_key:
            print("Updated GEMINI_API_KEY in .env.local")
        else:
            print("Saved Gemini configuration to .env.local")
    return {
        "env_path": str(env_path),
        "updated_existing_key": existing_key,
        "gitignore_path": str(root / ".gitignore"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure local Gemini API access for the web app.")
    parser.add_argument("--key", default="", help="Gemini API key. Prefer interactive input to avoid shell history.")
    parser.add_argument("--model", default=DEFAULTS["GEMINI_MODEL"])
    parser.add_argument("--embedding-model", default=DEFAULTS["GEMINI_EMBEDDING_MODEL"])
    parser.add_argument("--llm-provider", default=DEFAULTS["LLM_PROVIDER"])
    parser.add_argument("--strategy-planner-provider", default=DEFAULTS["STRATEGY_PLANNER_PROVIDER"])
    args = parser.parse_args(argv)

    api_key = args.key.strip()
    if api_key:
        print("Warning: command-line keys may be stored in shell history. Interactive hidden input is safer.", file=sys.stderr)
    else:
        api_key = getpass.getpass("Gemini API key: ").strip()
    configure_gemini_key(
        ROOT,
        api_key,
        llm_provider=args.llm_provider,
        gemini_model=args.model,
        gemini_embedding_model=args.embedding_model,
        strategy_planner_provider=args.strategy_planner_provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
