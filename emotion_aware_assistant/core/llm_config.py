from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from emotion_aware_assistant.core.config import LOCAL_ENV_FILE, parse_env_file


DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = ""
SUPPORTED_PROVIDERS = {"gemini", "openrouter", "openai_compatible"}
ROLE_ENV_KEYS = {
    "answer_model": ("LLM_PROVIDER", "LLM_MODEL"),
    "strategy_planner_model": ("STRATEGY_PLANNER_PROVIDER", "STRATEGY_PLANNER_MODEL"),
    "embedding_model": ("EMBEDDING_PROVIDER", "EMBEDDING_MODEL"),
}
SUPPORTED_ENV_KEYS = {
    "LLM_PROVIDER",
    "LLM_MODEL",
    "STRATEGY_PLANNER_PROVIDER",
    "STRATEGY_PLANNER_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_SITE_URL",
    "OPENROUTER_SITE_NAME",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
}
SECRET_FIELDS = {"api_key", "key", "secret", "token", "password"}
DEFAULT_COMPARISON_MODELS = [
    {
        "id": "gemini_flash",
        "label": "Gemini Flash",
        "provider": "gemini",
        "model": DEFAULT_GEMINI_MODEL,
        "enabled": True,
        "role": "comparison",
    },
    {
        "id": "deepseek_chat_v3_free",
        "label": "DeepSeek Chat V3 Free",
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "enabled": False,
        "role": "comparison",
    },
    {
        "id": "deepseek_r1_free",
        "label": "DeepSeek R1 Free",
        "provider": "openrouter",
        "model": "deepseek/deepseek-r1:free",
        "enabled": False,
        "role": "comparison",
    },
    {
        "id": "gpt_via_openrouter",
        "label": "GPT via OpenRouter",
        "provider": "openrouter",
        "model": "openai/gpt-5.2",
        "enabled": False,
        "role": "comparison",
    },
    {
        "id": "custom_openrouter_model",
        "label": "Custom OpenRouter Model",
        "provider": "openrouter",
        "model": "",
        "enabled": False,
        "role": "comparison",
    },
]


def read_llm_values(project_root: str | Path | None = None) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    values = parse_env_file(root / LOCAL_ENV_FILE)
    for key in SUPPORTED_ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def read_process_llm_values() -> dict[str, str]:
    return {key: os.environ[key] for key in SUPPORTED_ENV_KEYS if key in os.environ}


def mask_key(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if len(raw) <= 8:
        return "configured"
    return f"{raw[:4]}...{raw[-4:]}"


def role_config(role: str, values: dict[str, str] | None = None) -> dict[str, Any]:
    values = read_llm_values() if values is None else values
    if role == "answer_model":
        provider = _provider(values.get("LLM_PROVIDER") or "gemini")
        model = _text(values.get("LLM_MODEL")) or provider_default_model(provider, values)
    elif role == "strategy_planner_model":
        answer = role_config("answer_model", values)
        provider = _provider(values.get("STRATEGY_PLANNER_PROVIDER") or answer["provider"])
        model = _text(values.get("STRATEGY_PLANNER_MODEL")) or _text(answer.get("model")) or provider_default_model(provider, values)
    elif role == "embedding_model":
        provider = _provider(values.get("EMBEDDING_PROVIDER") or "gemini")
        model = (
            _text(values.get("EMBEDDING_MODEL"))
            or _text(values.get("GEMINI_EMBEDDING_MODEL"))
            or DEFAULT_GEMINI_EMBEDDING_MODEL
        )
    else:
        raise ValueError(f"Unsupported LLM role: {role}")
    return {
        "provider": provider,
        "model": model,
        "configured": role_provider_configured(provider, model, values),
    }


def role_config_from_env(role: str) -> dict[str, Any]:
    return role_config(role, read_process_llm_values())


def provider_default_model(provider: str, values: dict[str, str] | None = None) -> str:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if provider == "openrouter":
        return _text(values.get("OPENROUTER_MODEL"))
    if provider == "openai_compatible":
        return _text(values.get("OPENAI_MODEL"))
    return _text(values.get("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL


def provider_api_key(provider: str, values: dict[str, str] | None = None) -> str:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if provider == "openrouter":
        return _text(values.get("OPENROUTER_API_KEY"))
    if provider == "openai_compatible":
        return _text(values.get("OPENAI_API_KEY"))
    return _text(values.get("GEMINI_API_KEY"))


def provider_api_key_from_env(provider: str) -> str:
    return provider_api_key(provider, read_process_llm_values())


def provider_base_url(provider: str, values: dict[str, str] | None = None) -> str:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_BASE_URL
    if provider == "openai_compatible":
        return _text(values.get("OPENAI_BASE_URL")) or DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    return ""


def provider_base_url_from_env(provider: str) -> str:
    return provider_base_url(provider, read_process_llm_values())


def provider_configured(provider: str, values: dict[str, str] | None = None, *, require_model: bool = False) -> bool:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if not provider_api_key(provider, values):
        return False
    if require_model and not provider_default_model(provider, values):
        return provider == "gemini"
    if provider == "openai_compatible" and not provider_base_url(provider, values):
        return False
    return True


def role_provider_configured(provider: str, model: str, values: dict[str, str] | None = None) -> bool:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if not provider_api_key(provider, values) or not _text(model):
        return False
    if provider == "openai_compatible" and not provider_base_url(provider, values):
        return False
    return True


def llm_status(project_root: str | Path, profiles_dir: str | Path) -> dict[str, Any]:
    values = read_llm_values(project_root)
    roles = {name: role_config(name, values) for name in ROLE_ENV_KEYS}
    warnings = role_warnings(roles)
    return {
        "providers": {
            "gemini": {
                "configured": provider_configured("gemini", values),
                "masked_key": mask_key(provider_api_key("gemini", values)),
                "models": {
                    "default": _text(values.get("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL,
                    "embedding": _text(values.get("GEMINI_EMBEDDING_MODEL")) or DEFAULT_GEMINI_EMBEDDING_MODEL,
                },
            },
            "openrouter": {
                "configured": provider_configured("openrouter", values),
                "masked_key": mask_key(provider_api_key("openrouter", values)),
                "base_url": DEFAULT_OPENROUTER_BASE_URL,
                "model": _text(values.get("OPENROUTER_MODEL")),
                "site_url": _text(values.get("OPENROUTER_SITE_URL")) or None,
                "site_name": _text(values.get("OPENROUTER_SITE_NAME")) or None,
            },
            "openai_compatible": {
                "configured": provider_configured("openai_compatible", values),
                "masked_key": mask_key(provider_api_key("openai_compatible", values)),
                "base_url": provider_base_url("openai_compatible", values) or None,
                "model": _text(values.get("OPENAI_MODEL")),
            },
        },
        "roles": roles,
        "comparison_models": load_comparison_models(profiles_dir),
        "warnings": warnings,
    }


def save_provider_config(project_root: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    provider = _provider(data.get("provider"))
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported LLM provider.")
    updates: dict[str, str] = {}
    api_key = _text(data.get("api_key"))
    default_model = _text(data.get("default_model") or data.get("model"))
    embedding_model = _text(data.get("embedding_model"))
    base_url = _text(data.get("base_url"))
    if provider == "gemini":
        if api_key:
            updates["GEMINI_API_KEY"] = api_key
        if default_model:
            updates["GEMINI_MODEL"] = default_model
        if embedding_model:
            updates["GEMINI_EMBEDDING_MODEL"] = embedding_model
    elif provider == "openrouter":
        if api_key:
            updates["OPENROUTER_API_KEY"] = api_key
        if default_model:
            updates["OPENROUTER_MODEL"] = default_model
        site_url = _text(data.get("site_url") or data.get("OPENROUTER_SITE_URL"))
        site_name = _text(data.get("site_name") or data.get("OPENROUTER_SITE_NAME"))
        if site_url:
            updates["OPENROUTER_SITE_URL"] = site_url
        if site_name:
            updates["OPENROUTER_SITE_NAME"] = site_name
    elif provider == "openai_compatible":
        if api_key:
            updates["OPENAI_API_KEY"] = api_key
        if base_url:
            updates["OPENAI_BASE_URL"] = base_url
        if default_model:
            updates["OPENAI_MODEL"] = default_model
    if not updates:
        raise ValueError("No provider settings were provided.")
    _write_env_updates(project_root, updates)
    os.environ.update(updates)
    status = llm_status(project_root, _profiles_dir_from_root(project_root))
    return {"saved": True, "restart_required": False, **status}


def save_role_config(project_root: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, str] = {}
    for role, keys in ROLE_ENV_KEYS.items():
        payload = data.get(role)
        if not isinstance(payload, dict):
            continue
        provider = _provider(payload.get("provider"))
        model = _text(payload.get("model"))
        if provider:
            updates[keys[0]] = provider
        if model:
            updates[keys[1]] = model
    if not updates:
        raise ValueError("No role model settings were provided.")
    _write_env_updates(project_root, updates)
    os.environ.update(updates)
    values = read_llm_values(project_root)
    roles = {name: role_config(name, values) for name in ROLE_ENV_KEYS}
    return {
        "saved": True,
        "restart_required": False,
        "roles": roles,
        "warnings": role_warnings(roles),
    }


def load_comparison_models(profiles_dir: str | Path) -> list[dict[str, Any]]:
    path = _profiles_path(profiles_dir)
    if not path.exists():
        return [dict(profile) for profile in DEFAULT_COMPARISON_MODELS]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return _sanitize_profiles(payload.get("comparison_models") if isinstance(payload, dict) else [])


def save_comparison_models(profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    raw_profiles = data.get("comparison_models")
    if not isinstance(raw_profiles, list):
        raise ValueError("comparison_models must be a list.")
    profiles = _sanitize_profiles(raw_profiles)
    path = _profiles_path(profiles_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"comparison_models": profiles}, indent=2), encoding="utf-8")
    return {"saved": True, "comparison_models": profiles}


def test_provider_config(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    provider = _provider(data.get("provider"))
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported LLM provider.")
    values = read_llm_values(project_root)
    model = _text(data.get("model")) or _text(role_config(_text(data.get("role")) or "answer_model", values).get("model"))
    test_type = _text(data.get("test_type")) or "configured_only"
    configured = bool(provider_api_key(provider, values)) and bool(model)
    error = None
    if provider == "openai_compatible" and not provider_base_url(provider, values):
        configured = False
        error = "OpenAI-compatible base URL is not configured."
    if provider == "openrouter" and not model:
        error = "OpenRouter model is not configured."
    if provider == "gemini" and not provider_api_key(provider, values):
        error = "Gemini API key is not configured."
    if test_type != "configured_only":
        return {
            "ok": configured,
            "provider": provider,
            "model": model,
            "configured": configured,
            "tested": "configured_only",
            "error": error or "Lightweight provider calls are not enabled by default.",
            "status": llm_status(project_root, profiles_dir),
        }
    return {
        "ok": configured,
        "provider": provider,
        "model": model,
        "configured": configured,
        "tested": "configured_only",
        "error": error,
    }


def role_warnings(roles: dict[str, dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    embedding_provider = _provider((roles.get("embedding_model") or {}).get("provider"))
    if embedding_provider and embedding_provider != "gemini":
        warnings.append(
            f"Embedding provider {embedding_provider} is not supported by the current RAG embedding path; Gemini embeddings or keyword retrieval will be used."
        )
    return warnings


def _sanitize_profiles(raw_profiles: Any) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    if not isinstance(raw_profiles, list):
        return profiles
    for index, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            continue
        provider = _provider(item.get("provider"))
        model = _text(item.get("model"))
        if provider not in SUPPORTED_PROVIDERS or not model:
            continue
        profile = {
            "id": _safe_profile_id(item.get("id") or f"profile_{index + 1}"),
            "label": _text(item.get("label")) or model,
            "provider": provider,
            "model": model,
            "enabled": bool(item.get("enabled")),
            "role": _safe_profile_role(item.get("role")),
        }
        notes = _text(item.get("notes"))
        if notes:
            profile["notes"] = notes[:500]
        profiles.append(profile)
    return profiles


def _write_env_updates(project_root: str | Path, updates: dict[str, str]) -> None:
    root = Path(project_root)
    env_path = root / LOCAL_ENV_FILE
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for key, value in updates.items():
        lines = _replace_or_append(lines, key, value)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    _ensure_env_gitignored(root)


def _replace_or_append(lines: list[str], key: str, value: str) -> list[str]:
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=", re.ASCII)
    replacement = f"{key}={value}"
    next_lines: list[str] = []
    replaced = False
    for line in lines:
        if pattern.match(line):
            if not replaced:
                next_lines.append(replacement)
                replaced = True
            continue
        next_lines.append(line)
    if not replaced:
        next_lines.append(replacement)
    return next_lines


def _ensure_env_gitignored(root: Path) -> None:
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    if LOCAL_ENV_FILE not in {line.strip() for line in lines}:
        lines.append(LOCAL_ENV_FILE)
        gitignore.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _profiles_dir_from_root(project_root: str | Path) -> Path:
    return Path(project_root) / "runtime_uploads"


def _profiles_path(profiles_dir: str | Path) -> Path:
    root = Path(profiles_dir)
    return root / "config" / "llm_profiles.json"


def _safe_profile_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned[:80] or "profile"


def _safe_profile_role(value: Any) -> str:
    role = str(value or "comparison").strip().lower()
    return role if role in {"answer", "strategy", "embedding", "comparison"} else "comparison"


def _provider(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {"openai": "openai_compatible", "openai-compatible": "openai_compatible"}
    return aliases.get(normalized, normalized)


def _text(value: Any) -> str:
    return str(value or "").strip()
