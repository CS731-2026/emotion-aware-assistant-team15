from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from emotion_aware_assistant.core.config import LOCAL_ENV_FILE, PROJECT_ROOT, parse_env_file


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = ""
LOCAL_LLM_SETTINGS_FILE = "local_llm_settings.json"
STALE_GEMINI_MODEL_IDS = {"gemini-flash-latest"}
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
SUPPORTED_PROVIDERS = {"gemini", "openrouter", "openai_compatible"}
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "openai_compatible": "OpenAI-compatible",
}
MODEL_PRESETS = [
    {
        "id": "openrouter_free",
        "provider": "openrouter",
        "display_name": "OpenRouter Free",
        "model_id": "openrouter/free",
        "family": "OpenRouter Free",
    },
    {
        "id": "claude_opus_47_fast",
        "provider": "openrouter",
        "display_name": "Claude Opus 4.7 Fast",
        "model_id": "anthropic/claude-opus-4.7-fast",
        "family": "Claude",
    },
    {
        "id": "gemini_25_flash",
        "provider": "gemini",
        "display_name": "Gemini 2.5 Flash",
        "model_id": DEFAULT_GEMINI_MODEL,
        "family": "Gemini",
    },
]
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
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_EMBEDDING_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
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
        "id": "gemini_25_flash",
        "label": "Gemini 2.5 Flash",
        "provider": "gemini",
        "model": DEFAULT_GEMINI_MODEL,
        "enabled": False,
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


def local_llm_settings_path(
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
) -> Path:
    if profiles_dir is not None:
        return Path(profiles_dir) / LOCAL_LLM_SETTINGS_FILE
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "runtime_uploads" / LOCAL_LLM_SETTINGS_FILE


def read_local_llm_settings(
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = local_llm_settings_path(project_root=project_root, profiles_dir=profiles_dir)
    if not path.exists():
        return _canonical_settings({})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _canonical_settings({})
    return _canonical_settings(payload if isinstance(payload, dict) else {})


def save_llm_settings(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    provider_items = data.get("providers")
    if isinstance(data.get("provider"), str):
        provider_items = [data]
    if isinstance(provider_items, list):
        for item in provider_items:
            if not isinstance(item, dict):
                continue
            provider = _provider(item.get("provider"))
            if provider == "openrouter":
                settings = _apply_openrouter_settings(settings, item)
                profile_id = _profile_id_for_model(settings, item, create=True)
                if profile_id:
                    settings["roles"]["default_answer_model_profile_id"] = profile_id
            elif provider == "gemini":
                settings = _apply_gemini_settings(settings, item)
                settings = _apply_embedding_settings(settings, {"provider": "gemini", **item})
                profile_id = _profile_id_for_model(settings, item, create=True)
                if profile_id:
                    settings["roles"]["default_answer_model_profile_id"] = profile_id

    if isinstance(data.get("openrouter"), dict):
        settings = _apply_openrouter_settings(settings, data["openrouter"])
    if isinstance(data.get("gemini"), dict):
        settings = _apply_gemini_settings(settings, data["gemini"])
    if isinstance(data.get("embedding"), dict):
        settings = _apply_embedding_settings(settings, data["embedding"])

    model_profiles = data.get("model_profiles")
    if model_profiles is None:
        model_profiles = data.get("compare_model_slots") or data.get("comparison_models")
    if isinstance(model_profiles, list):
        settings["model_profiles"] = [_sanitize_model_profile(item) for item in model_profiles if isinstance(item, dict)]

    default_model = data.get("default_model")
    if isinstance(default_model, dict):
        profile_id = _profile_id_for_model(settings, default_model, create=True)
        if profile_id:
            settings["roles"]["default_answer_model_profile_id"] = profile_id

    roles = data.get("roles")
    if isinstance(roles, dict):
        if any(key in roles for key in ("default_answer_model_profile_id", "strategy_model_profile_id", "compare_model_profile_ids")):
            settings["roles"] = _updated_profile_roles(settings, roles)
        else:
            default_profile = _profile_id_for_role_payload(settings, roles.get("answer_model"), create=True)
            strategy_profile = _profile_id_for_role_payload(settings, roles.get("strategy_planner_model"), create=True)
            if default_profile:
                settings["roles"]["default_answer_model_profile_id"] = default_profile
            if strategy_profile:
                settings["roles"]["strategy_model_profile_id"] = strategy_profile
            embedding_role = roles.get("embedding_model")
            if isinstance(embedding_role, dict):
                settings = _apply_embedding_settings(settings, embedding_role)
    elif any(role in data for role in ROLE_ENV_KEYS):
        settings["roles"] = _sanitize_roles(data)

    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, **llm_settings_status(project_root, profiles_dir)}


def llm_settings_status(project_root: str | Path, profiles_dir: str | Path) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    status = llm_status(project_root, profiles_dir)
    values = read_llm_values(project_root, profiles_dir)
    openrouter = _safe_openrouter_status(settings, values)
    gemini = _safe_gemini_status(settings, values)
    embedding = _safe_embedding_status(settings, values)
    answer = role_config("answer_model", values)
    default_provider = _provider(answer.get("provider"))
    default_model = _text(answer.get("model"))
    comparison_models = load_comparison_models(profiles_dir)
    warnings = list(status.get("warnings") if isinstance(status.get("warnings"), list) else [])
    warnings.extend(_role_profile_warnings(settings))
    status.update(
        {
            "openrouter": openrouter,
            "gemini": gemini,
            "model_profiles": [_public_model_profile(profile) for profile in settings.get("model_profiles", []) if isinstance(profile, dict)],
            "model_presets": [dict(preset) for preset in MODEL_PRESETS],
            "roles": _public_profile_roles(settings),
            "embedding": embedding,
            "default_model": {
                "provider": default_provider,
                "model": default_model,
                "label": _profile_display_for_model(settings, default_model) or _provider_display_name(settings, default_provider, default_model),
                "configured": bool(answer.get("configured")),
            },
            "compare_model_slots": comparison_models,
            "comparison_models": comparison_models,
            "settings_file": str(local_llm_settings_path(project_root=project_root, profiles_dir=profiles_dir)),
            "settings_revision": int(settings.get("settings_revision") or 0),
            "warnings": warnings,
        }
    )
    providers = status.get("providers") if isinstance(status.get("providers"), dict) else {}
    openrouter_provider = providers.get("openrouter") if isinstance(providers.get("openrouter"), dict) else {}
    openrouter_provider.update(
        {
            "configured": bool(openrouter.get("key_configured")),
            "key_configured": bool(openrouter.get("key_configured")),
            "masked_key_display": openrouter.get("masked_key_display"),
            "base_url": openrouter.get("base_url"),
            "model": default_model,
            "site_url": openrouter.get("site_url") or None,
            "site_name": openrouter.get("site_name") or None,
            "status": openrouter.get("status"),
            "last_test": openrouter.get("last_test"),
        }
    )
    providers["openrouter"] = openrouter_provider
    gemini_provider = providers.get("gemini") if isinstance(providers.get("gemini"), dict) else {}
    gemini_provider.update(
        {
            "configured": bool(gemini.get("key_configured")),
            "key_configured": bool(gemini.get("key_configured")),
            "masked_key_display": gemini.get("masked_key_display"),
            "display_name": gemini.get("display_name"),
            "model": gemini.get("model"),
            "embedding_model": gemini.get("embedding_model"),
            "status": gemini.get("status"),
            "last_test": gemini.get("last_test"),
        }
    )
    providers["gemini"] = gemini_provider
    status["providers"] = providers
    return status


def test_llm_connection(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    values = read_llm_values(project_root, profiles_dir)
    provider = _provider(data.get("provider") or role_config("answer_model", values).get("provider"))
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported LLM provider.")
    model = _text(data.get("model")) if "model" in data else _text(role_config("answer_model", values).get("model"))
    if provider == "gemini":
        return _run_gemini_direct_test(project_root, profiles_dir, model_id=model, test_type="chat")
    api_key = provider_api_key(provider, values)
    base = {
        "ok": False,
        "provider": provider,
        "model": model,
        "latency_ms": 0,
        "status": "skipped",
        "error_type": "",
        "message": "",
    }
    if not api_key:
        return _store_test_result(
            project_root,
            profiles_dir,
            provider,
            {**base, "error_type": "missing_key", "message": "API key is not configured."},
        )
    if not model:
        return _store_test_result(
            project_root,
            profiles_dir,
            provider,
            {**base, "error_type": "missing_model", "message": "Model ID is not configured."},
        )

    started = time.time()
    try:
        request = _provider_test_request(provider, model, api_key, values)
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
        result = {
            **base,
            "ok": True,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "success",
            "error_type": "",
            "message": "Connection test succeeded.",
        }
    except urllib.error.HTTPError as exc:
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": _http_error_type(exc.code),
            "message": f"Provider returned HTTP {exc.code}.",
        }
    except TimeoutError:
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": "timeout",
            "message": "Provider request timed out.",
        }
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        error_type = "timeout" if isinstance(reason, TimeoutError) else "provider_error"
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": error_type,
            "message": "Provider request failed.",
        }
    except Exception:
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": "unknown_error",
            "message": "Provider request failed unexpectedly.",
        }
    return _store_test_result(project_root, profiles_dir, provider, result)


def _run_gemini_direct_test(
    project_root: str | Path,
    profiles_dir: str | Path,
    *,
    model_id: str,
    test_type: str,
) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    values = read_llm_values(project_root, profiles_dir)
    key, key_source = _provider_key_and_source(project_root, settings, values, "gemini")
    model = _gemini_model_name(model_id)
    if not model:
        model = _gemini_model_name((settings.get("gemini") or {}).get("embedding_model" if test_type == "embedding" else "model"))
    if not model:
        model = DEFAULT_GEMINI_EMBEDDING_MODEL if test_type == "embedding" else DEFAULT_GEMINI_MODEL
    endpoint_url = _gemini_endpoint_url(model, test_type)
    base = {
        "ok": False,
        "provider": "gemini",
        "model": model,
        "model_id": model,
        "test_type": test_type,
        "endpoint_url": endpoint_url,
        "payload_shape": "embedding.content.parts" if test_type == "embedding" else "contents.parts",
        "latency_ms": 0,
        "status": "skipped",
        "status_code": None,
        "google_error_status": "",
        "error_type": "",
        "message": "",
        "key_source": key_source,
        "masked_key_suffix": _masked_key_suffix(key),
    }
    if not key:
        result = {**base, "error_type": "missing_key", "message": "Gemini API key is not configured."}
        return _store_gemini_test_result(project_root, profiles_dir, result)
    if not model:
        result = {**base, "error_type": "missing_model", "message": "Gemini model ID is not configured."}
        return _store_gemini_test_result(project_root, profiles_dir, result)

    started = time.time()
    try:
        request = _gemini_embedding_test_request(model, key) if test_type == "embedding" else _gemini_chat_test_request(model, key)
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
        result = {
            **base,
            "ok": True,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "success",
            "status_code": 200,
            "message": "Connection test succeeded.",
        }
    except urllib.error.HTTPError as exc:
        error_details = _safe_provider_error_details(exc, key)
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "status_code": int(getattr(exc, "code", 0) or 0),
            "error_type": _http_error_type(int(getattr(exc, "code", 0) or 0)),
            "google_error_status": error_details.get("google_error_status", ""),
            "message": error_details.get("message", ""),
        }
    except TimeoutError:
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": "timeout",
            "message": "Gemini provider request timed out.",
        }
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": "timeout" if isinstance(reason, TimeoutError) else "provider_error",
            "message": "Gemini provider request failed.",
        }
    except Exception:
        result = {
            **base,
            "latency_ms": int((time.time() - started) * 1000),
            "status": "failed",
            "error_type": "unknown_error",
            "message": "Gemini provider request failed unexpectedly.",
        }
    return _store_gemini_test_result(project_root, profiles_dir, result)


def _provider_key_and_source(project_root: str | Path, settings: dict[str, Any], values: dict[str, str], provider: str) -> tuple[str, str]:
    provider = _provider(provider)
    if provider == "gemini":
        settings_key = _settings_provider_key(settings, "gemini")
        resolved = (
            {"key": settings_key, "key_source": "local_settings"}
            if settings_key
            else resolve_gemini_api_key(project_root, None, values=values)
        )
        return _text(resolved.get("key")), _text(resolved.get("key_source"))
    env_key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENROUTER_API_KEY"
    settings_key = _settings_provider_key(settings, provider)
    if settings_key:
        return settings_key, "local_settings"
    if _text(os.environ.get(env_key_name)):
        return _text(os.environ.get(env_key_name)), "environment"
    env_file_value = _text(parse_env_file(Path(project_root) / LOCAL_ENV_FILE).get(env_key_name))
    if env_file_value:
        return env_file_value, "environment"
    return _text(provider_api_key(provider, values)), ""


def _settings_provider_key(settings: dict[str, Any], provider: str) -> str:
    provider = _provider(provider)
    if provider == "gemini":
        return _text((settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}).get("api_key"))
    if provider == "openrouter":
        return _text((settings.get("openrouter") if isinstance(settings.get("openrouter"), dict) else {}).get("api_key"))
    return ""


def resolve_gemini_api_key(
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    *,
    include_env_file: bool = True,
    values: dict[str, str] | None = None,
) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    settings = read_local_llm_settings(project_root=root, profiles_dir=profiles_dir)
    settings_key = _settings_provider_key(settings, "gemini")
    if settings_key:
        return {
            "key": settings_key,
            "key_source": "local_settings",
            "masked_suffix": _masked_key_suffix(settings_key),
        }
    env_values: dict[str, str] = {}
    if values:
        env_values.update({key: _text(value) for key, value in values.items()})
    if include_env_file:
        env_values.update(parse_env_file(root / LOCAL_ENV_FILE))
        env_values.update({key: value for key, value in parse_env_file(root / ".env").items() if key in SUPPORTED_ENV_KEYS})
    for key in SUPPORTED_ENV_KEYS:
        if key in os.environ:
            env_values[key] = os.environ[key]
    env_key = _text(env_values.get("GEMINI_API_KEY")) or _text(env_values.get("GOOGLE_API_KEY"))
    if env_key:
        return {
            "key": env_key,
            "key_source": "environment",
            "masked_suffix": _masked_key_suffix(env_key),
        }
    return {"key": "", "key_source": "", "masked_suffix": ""}


def _masked_key_suffix(value: str | None) -> str:
    raw = _text(value)
    return raw[-4:] if raw else ""


def _safe_provider_error_details(exc: urllib.error.HTTPError, api_key: str = "") -> dict[str, str]:
    text = ""
    error_status = ""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        error = payload.get("error") if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            text = _text(error.get("message"))
            error_status = _text(error.get("status"))
    except Exception:
        text = ""
    if not text:
        text = f"Provider returned HTTP {getattr(exc, 'code', '')}."
    if api_key:
        text = text.replace(api_key, "[redacted]")
        error_status = error_status.replace(api_key, "[redacted]")
    return {"message": text[:500], "google_error_status": error_status[:120]}


def _safe_provider_error_message(exc: urllib.error.HTTPError, api_key: str = "") -> str:
    return _safe_provider_error_details(exc, api_key).get("message", "")


def _store_gemini_test_result(project_root: str | Path, profiles_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    safe = _safe_test_result(result)
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    if result.get("test_type") == "embedding":
        embedding = settings.get("embedding") if isinstance(settings.get("embedding"), dict) else {}
        embedding["last_test"] = {**safe, "tested_at": str(int(time.time()))}
        settings["embedding"] = embedding
    else:
        gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
        gemini["last_test"] = {**safe, "tested_at": str(int(time.time()))}
        settings["gemini"] = gemini
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return safe


def has_explicit_llm_config(values: dict[str, str] | None = None) -> bool:
    values = read_llm_values(PROJECT_ROOT, include_env_file=False) if values is None else values
    relevant = {
        "LLM_PROVIDER",
        "LLM_MODEL",
        "STRATEGY_PLANNER_PROVIDER",
        "STRATEGY_PLANNER_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    }
    return any(_text(values.get(key)) for key in relevant)


def read_llm_values(
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    *,
    include_env_file: bool = True,
) -> dict[str, str]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    settings = read_local_llm_settings(project_root=root, profiles_dir=profiles_dir)
    settings_values = _settings_env_values(settings)
    values: dict[str, str] = {}
    if include_env_file:
        values.update(parse_env_file(root / LOCAL_ENV_FILE))
        values.update({key: value for key, value in parse_env_file(root / ".env").items() if key in SUPPORTED_ENV_KEYS})
    for key in SUPPORTED_ENV_KEYS:
        if key in os.environ:
            values[key] = os.environ[key]
    for key, value in settings_values.items():
        if _text(value):
            values[key] = _text(value)
    if not _text(values.get("GEMINI_API_KEY")) and _text(values.get("GOOGLE_API_KEY")):
        values["GEMINI_API_KEY"] = _text(values.get("GOOGLE_API_KEY"))
    return values


def read_process_llm_values() -> dict[str, str]:
    return {key: os.environ[key] for key in SUPPORTED_ENV_KEYS if key in os.environ}


def settings_revision_info(
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = local_llm_settings_path(project_root=project_root, profiles_dir=profiles_dir)
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    stat = path.stat() if path.exists() else None
    return {
        "settings_revision": int(settings.get("settings_revision") or 0),
        "settings_file_mtime": stat.st_mtime if stat else None,
        "settings_file_path": str(path),
    }


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
    return role_config(role, read_llm_values(PROJECT_ROOT, include_env_file=False))


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
    return _text(values.get("GEMINI_API_KEY")) or _text(values.get("GOOGLE_API_KEY"))


def provider_api_key_from_env(provider: str) -> str:
    if _provider(provider) == "gemini":
        return _text(resolve_gemini_api_key(PROJECT_ROOT, None, include_env_file=False).get("key"))
    return provider_api_key(provider, read_llm_values(PROJECT_ROOT, include_env_file=False))


def resolve_provider_api_key(
    provider: str,
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    *,
    include_env_file: bool = True,
    values: dict[str, str] | None = None,
) -> dict[str, str]:
    provider = _provider(provider)
    if provider == "gemini":
        return resolve_gemini_api_key(project_root, profiles_dir, include_env_file=include_env_file, values=values)
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    settings = read_local_llm_settings(project_root=root, profiles_dir=profiles_dir)
    settings_key = _settings_provider_key(settings, provider)
    if settings_key:
        return {"key": settings_key, "key_source": "local_settings", "masked_suffix": _masked_key_suffix(settings_key)}
    env_values: dict[str, str] = {}
    if values:
        env_values.update({key: _text(value) for key, value in values.items()})
    if include_env_file:
        env_values.update(parse_env_file(root / LOCAL_ENV_FILE))
        env_values.update({key: value for key, value in parse_env_file(root / ".env").items() if key in SUPPORTED_ENV_KEYS})
    for key in SUPPORTED_ENV_KEYS:
        if key in os.environ:
            env_values[key] = os.environ[key]
    key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    env_key = _text(env_values.get(key_name))
    if env_key:
        return {"key": env_key, "key_source": "environment", "masked_suffix": _masked_key_suffix(env_key)}
    return {"key": "", "key_source": "missing", "masked_suffix": ""}


def role_config_with_source(
    role: str,
    project_root: str | Path | None = None,
    profiles_dir: str | Path | None = None,
    *,
    include_env_file: bool = True,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    settings = read_local_llm_settings(project_root=root, profiles_dir=profiles_dir)
    values = read_llm_values(root, profiles_dir, include_env_file=include_env_file)
    config = role_config(role, values)
    return {**config, "model_source": _role_model_source(settings, role, config, values)}


def provider_base_url(provider: str, values: dict[str, str] | None = None) -> str:
    values = read_llm_values() if values is None else values
    provider = _provider(provider)
    if provider == "openrouter":
        return _text(values.get("OPENROUTER_BASE_URL")) or DEFAULT_OPENROUTER_BASE_URL
    if provider == "openai_compatible":
        return _text(values.get("OPENAI_BASE_URL")) or DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    return ""


def provider_base_url_from_env(provider: str) -> str:
    return provider_base_url(provider, read_llm_values(PROJECT_ROOT, include_env_file=False))


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
    values = read_llm_values(project_root, profiles_dir)
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
                "base_url": provider_base_url("openrouter", values),
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
    settings = read_local_llm_settings(profiles_dir=profiles_dir)
    role_ids = (settings.get("roles") or {}).get("compare_model_profile_ids") if isinstance(settings.get("roles"), dict) else []
    if isinstance(role_ids, list) and role_ids:
        profiles = []
        for profile_id in role_ids[:6]:
            profile = _profile_by_id(settings, profile_id)
            if not profile or not bool(profile.get("enabled", True)) or _is_stale_gemini_profile(profile):
                continue
            profiles.append(_comparison_profile_from_model_profile(profile))
        return profiles
    path = _profiles_path(profiles_dir)
    if not path.exists():
        return []
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
    values = read_llm_values(project_root, profiles_dir)
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


def _settings_env_values(settings: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    settings = _canonical_settings(settings)
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    gemini_key = _text(gemini.get("api_key"))
    if gemini_key:
        values["GEMINI_API_KEY"] = gemini_key
        if _text(gemini.get("model")):
            values["GEMINI_MODEL"] = _text(gemini.get("model"))
        if _text(gemini.get("embedding_model")):
            values["GEMINI_EMBEDDING_MODEL"] = _text(gemini.get("embedding_model"))

    openrouter = settings.get("openrouter") if isinstance(settings.get("openrouter"), dict) else {}
    if _text(openrouter.get("api_key")):
        values["OPENROUTER_API_KEY"] = _text(openrouter.get("api_key"))
    values["OPENROUTER_BASE_URL"] = _text(openrouter.get("base_url")) or DEFAULT_OPENROUTER_BASE_URL
    if _text(openrouter.get("site_url")):
        values["OPENROUTER_SITE_URL"] = _text(openrouter.get("site_url"))
    if _text(openrouter.get("site_name")):
        values["OPENROUTER_SITE_NAME"] = _text(openrouter.get("site_name"))
    if _text(openrouter.get("api_key")):
        values.setdefault("LLM_PROVIDER", "openrouter")

    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    default_profile = _profile_by_id(settings, roles.get("default_answer_model_profile_id"))
    strategy_profile = _profile_by_id(settings, roles.get("strategy_model_profile_id")) or default_profile
    if default_profile:
        default_provider = _profile_provider(default_profile)
        values["LLM_PROVIDER"] = default_provider
        values["LLM_MODEL"] = _text(default_profile.get("model_id"))
        if default_provider == "openrouter":
            values["OPENROUTER_MODEL"] = _text(default_profile.get("model_id"))
        elif default_provider == "gemini":
            values["GEMINI_MODEL"] = _text(default_profile.get("model_id"))
    if strategy_profile:
        strategy_provider = _profile_provider(strategy_profile)
        values["STRATEGY_PLANNER_PROVIDER"] = strategy_provider
        values["STRATEGY_PLANNER_MODEL"] = _text(strategy_profile.get("model_id"))

    embedding = settings.get("embedding") if isinstance(settings.get("embedding"), dict) else {}
    embedding_provider = _provider(embedding.get("provider"))
    if embedding_provider:
        values["EMBEDDING_PROVIDER"] = embedding_provider
    if _text(embedding.get("model")):
        values["EMBEDDING_MODEL"] = _text(embedding.get("model"))
    if embedding_provider == "gemini":
        if _text(embedding.get("api_key")) and not _text(values.get("GEMINI_API_KEY")):
            values["GEMINI_API_KEY"] = _text(embedding.get("api_key"))
        if _text(embedding.get("model")):
            values["GEMINI_EMBEDDING_MODEL"] = _text(embedding.get("model"))
    return values


def _role_model_source(settings: dict[str, Any], role: str, config: dict[str, Any], values: dict[str, str]) -> str:
    settings = _canonical_settings(settings)
    provider = _provider(config.get("provider"))
    model = _text(config.get("model"))
    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    profile_id = ""
    if role == "answer_model":
        profile_id = _text(roles.get("default_answer_model_profile_id"))
    elif role == "strategy_planner_model":
        profile_id = _text(roles.get("strategy_model_profile_id")) or _text(roles.get("default_answer_model_profile_id"))
    if profile_id:
        profile = _profile_by_id(settings, profile_id)
        if profile and _text(profile.get("model_id")) == model and _profile_provider(profile) == provider:
            return "role_profile"
    if role == "embedding_model":
        embedding = settings.get("embedding") if isinstance(settings.get("embedding"), dict) else {}
        if _text(embedding.get("model")) == model:
            return "settings"
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    if provider == "gemini" and model in {_text(gemini.get("model")), _text(gemini.get("embedding_model"))}:
        return "settings"
    role_model_keys = {
        "answer_model": ("LLM_MODEL", "GEMINI_MODEL", "OPENROUTER_MODEL", "OPENAI_MODEL"),
        "strategy_planner_model": ("STRATEGY_PLANNER_MODEL", "LLM_MODEL", "GEMINI_MODEL", "OPENROUTER_MODEL", "OPENAI_MODEL"),
        "embedding_model": ("EMBEDDING_MODEL", "GEMINI_EMBEDDING_MODEL"),
    }
    if any(_text(values.get(key)) == model for key in role_model_keys.get(role, ())):
        return "environment"
    return "missing" if not model else "settings"


def _updated_provider_settings(provider: str, current: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    updated = {key: value for key, value in current.items() if key not in SECRET_FIELDS}
    existing_key = _text(current.get("api_key"))
    if bool(item.get("clear_key")):
        updated["api_key"] = ""
    elif "api_key" in item and _text(item.get("api_key")):
        updated["api_key"] = _text(item.get("api_key"))
    elif existing_key:
        updated["api_key"] = existing_key
    for source, target in (
        ("model", "model"),
        ("default_model", "model"),
        ("display_name", "display_name"),
        ("label", "display_name"),
        ("base_url", "base_url"),
        ("embedding_model", "embedding_model"),
        ("site_url", "site_url"),
        ("site_name", "site_name"),
    ):
        if source in item:
            updated[target] = _text(item.get(source))
    updated["provider"] = provider
    return updated


def _sanitize_roles(data: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ROLE_ENV_KEYS:
        payload = data.get(role)
        if not isinstance(payload, dict):
            continue
        provider = _provider(payload.get("provider"))
        model = _text(payload.get("model"))
        if provider or model:
            roles[role] = {"provider": provider, "model": model}
    return roles


def _write_local_llm_settings(project_root: str | Path, profiles_dir: str | Path, settings: dict[str, Any]) -> None:
    path = local_llm_settings_path(project_root=project_root, profiles_dir=profiles_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = _canonical_settings(settings)
    settings["settings_revision"] = int(settings.get("settings_revision") or 0) + 1
    path.write_text(json.dumps(_sanitized_settings_for_storage(settings), indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    _ensure_settings_gitignored(Path(project_root))


def _sanitized_settings_for_storage(settings: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_settings(settings)
    openrouter = canonical.get("openrouter") if isinstance(canonical.get("openrouter"), dict) else {}
    gemini = canonical.get("gemini") if isinstance(canonical.get("gemini"), dict) else {}
    embedding = canonical.get("embedding") if isinstance(canonical.get("embedding"), dict) else {}
    stored_openrouter = {
        "api_key": _text(openrouter.get("api_key")),
        "base_url": _text(openrouter.get("base_url")) or DEFAULT_OPENROUTER_BASE_URL,
        "site_url": _text(openrouter.get("site_url")),
        "site_name": _text(openrouter.get("site_name")),
    }
    if isinstance(openrouter.get("last_test"), dict):
        stored_openrouter["last_test"] = _safe_test_result(openrouter.get("last_test"))
    stored_gemini = {
        "api_key": _text(gemini.get("api_key")),
        "display_name": _text(gemini.get("display_name")) or "Gemini Direct",
        "model": _text(gemini.get("model")) or DEFAULT_GEMINI_MODEL,
        "embedding_model": _text(gemini.get("embedding_model")) or DEFAULT_GEMINI_EMBEDDING_MODEL,
    }
    if isinstance(gemini.get("last_test"), dict):
        stored_gemini["last_test"] = _safe_test_result(gemini.get("last_test"))
    stored_embedding = {
        "provider": _provider(embedding.get("provider")) or None,
        "api_key": _text(embedding.get("api_key")) or None,
        "model": _text(embedding.get("model")) or None,
    }
    if isinstance(embedding.get("last_test"), dict):
        stored_embedding["last_test"] = _safe_test_result(embedding.get("last_test"))
    return {
        "settings_revision": int(canonical.get("settings_revision") or 0),
        "openrouter": stored_openrouter,
        "gemini": stored_gemini,
        "model_profiles": [
            _sanitize_model_profile(profile)
            for profile in canonical.get("model_profiles", [])
            if isinstance(profile, dict) and _text(profile.get("model_id"))
        ],
        "roles": _updated_profile_roles(canonical, canonical.get("roles") if isinstance(canonical.get("roles"), dict) else {}),
        "embedding": stored_embedding,
        "warnings": [_text(item) for item in canonical.get("warnings", []) if _text(item)][:20],
    }


def _ensure_settings_gitignored(root: Path) -> None:
    gitignore = root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    entries = {line.strip() for line in lines}
    additions = ["runtime_uploads/", f"runtime_uploads/{LOCAL_LLM_SETTINGS_FILE}"]
    changed = False
    for entry in additions:
        if entry not in entries:
            lines.append(entry)
            changed = True
    if changed:
        gitignore.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _provider_display_name(settings: dict[str, Any], provider: str, model: str) -> str:
    if provider == "gemini":
        gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
        return _text(gemini.get("display_name")) or PROVIDER_LABELS.get(provider, provider) or model
    providers = settings.get("providers") if isinstance(settings.get("providers"), dict) else {}
    details = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
    return _text(details.get("display_name")) or PROVIDER_LABELS.get(provider, provider) or model


def _masked_key_display(value: str | None) -> str:
    raw = _text(value)
    if not raw:
        return "not configured"
    if len(raw) <= 4:
        return "configured"
    return f"configured (...{raw[-4:]})"


def _provider_settings_state(provider: str, values: dict[str, str], model: str) -> str:
    if not provider_api_key(provider, values):
        return "missing key"
    if not _text(model):
        return "missing model"
    if provider == "openai_compatible" and not provider_base_url(provider, values):
        return "missing base URL"
    return "configured"


def _provider_last_test(settings: dict[str, Any], provider: str) -> dict[str, Any]:
    providers = settings.get("providers") if isinstance(settings.get("providers"), dict) else {}
    details = providers.get(provider) if isinstance(providers.get(provider), dict) else {}
    last_test = details.get("last_test")
    if not isinstance(last_test, dict):
        return {}
    return {
        "ok": bool(last_test.get("ok")),
        "status": _text(last_test.get("status")),
        "error_type": _text(last_test.get("error_type")),
        "latency_ms": int(float(last_test.get("latency_ms") or 0)),
        "message": _text(last_test.get("message")),
        "tested_at": _text(last_test.get("tested_at")),
    }


def _provider_test_request(provider: str, model: str, api_key: str, values: dict[str, str]) -> urllib.request.Request:
    prompt = "Reply with exactly: OK"
    if provider == "gemini":
        return _gemini_chat_test_request(model, api_key)
    base_url = provider_base_url(provider, values)
    if not base_url:
        raise ValueError("Provider base URL is not configured.")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if provider == "openrouter":
        site_url = _text(values.get("OPENROUTER_SITE_URL"))
        site_name = _text(values.get("OPENROUTER_SITE_NAME"))
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-OpenRouter-Title"] = site_name
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,
    }
    return urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _gemini_model_name(model: Any) -> str:
    value = _text(model).strip().strip("/")
    while value.startswith("models/"):
        value = value.removeprefix("models/").strip("/")
    return value


def _gemini_endpoint_url(model: Any, test_type: str = "chat") -> str:
    model_name = _gemini_model_name(model)
    method = "embedContent" if test_type == "embedding" else "generateContent"
    return f"{GEMINI_API_BASE}/{model_name}:{method}"


def _gemini_chat_test_request(model: str, api_key: str) -> urllib.request.Request:
    model_name = _gemini_model_name(model)
    body = {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]}
    return urllib.request.Request(
        _gemini_endpoint_url(model_name, "chat"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )


def _gemini_embedding_test_request(model: str, api_key: str) -> urllib.request.Request:
    model_name = _gemini_model_name(model)
    body = {
        "model": f"models/{model_name}",
        "content": {"parts": [{"text": "Reply with exactly: OK"}]},
        "taskType": "RETRIEVAL_DOCUMENT",
    }
    return urllib.request.Request(
        _gemini_endpoint_url(model_name, "embedding"),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )


def _http_error_type(status_code: int) -> str:
    if status_code in {401, 403}:
        return "unauthorized"
    if status_code == 404:
        return "model_not_found"
    if status_code == 429:
        return "rate_limited"
    return "provider_error"


def _store_test_result(project_root: str | Path, profiles_dir: str | Path, provider: str, result: dict[str, Any]) -> dict[str, Any]:
    safe_result = {
        "ok": bool(result.get("ok")),
        "provider": _provider(result.get("provider")),
        "model": _text(result.get("model")),
        "model_id": _text(result.get("model_id") or result.get("model")),
        "test_type": _text(result.get("test_type")),
        "endpoint_url": _text(result.get("endpoint_url")),
        "payload_shape": _text(result.get("payload_shape")),
        "latency_ms": int(float(result.get("latency_ms") or 0)),
        "status": _text(result.get("status")) or "failed",
        "status_code": result.get("status_code"),
        "google_error_status": _text(result.get("google_error_status")),
        "error_type": _text(result.get("error_type")),
        "message": _text(result.get("message")),
        "key_source": _text(result.get("key_source")),
        "masked_key_suffix": _text(result.get("masked_key_suffix")),
    }
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    if provider == "openrouter":
        settings.setdefault("openrouter", {})["last_test"] = {**safe_result, "tested_at": str(int(time.time()))}
    elif provider == "gemini":
        settings.setdefault("gemini", {})["last_test"] = {**safe_result, "tested_at": str(int(time.time()))}
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return safe_result


def save_openrouter_settings(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    settings = _apply_openrouter_settings(read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir), data)
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, **llm_settings_status(project_root, profiles_dir)}


def test_openrouter_key(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    model_id = _text((data or {}).get("model_id")) or _first_enabled_profile_model(settings, provider="openrouter") or "openrouter/free"
    result = _run_provider_model_test(project_root, profiles_dir, provider="openrouter", model_id=model_id, display_name="OpenRouter key")
    return result


def save_gemini_settings(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    settings = _apply_gemini_settings(read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir), data)
    if "embedding_model" in data or "model" in data:
        settings = _apply_embedding_settings(
            settings,
            {
                "provider": "gemini",
                "model": data.get("embedding_model") or (settings.get("gemini") or {}).get("embedding_model"),
            },
        )
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, **llm_settings_status(project_root, profiles_dir)}


def test_gemini_key(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    model_id = _text((data or {}).get("model_id")) or _text(gemini.get("model")) or _first_enabled_profile_model(settings, provider="gemini") or DEFAULT_GEMINI_MODEL
    return test_gemini_chat(project_root, profiles_dir, {"model_id": model_id})


def test_gemini_chat(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    model_id = _text((data or {}).get("model_id")) or _text(gemini.get("model")) or DEFAULT_GEMINI_MODEL
    result = _run_gemini_direct_test(project_root, profiles_dir, model_id=model_id, test_type="chat")
    return {**result, "provider": "gemini", "display_name": _text(gemini.get("display_name")) or "Gemini Direct"}


def test_gemini_embedding(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    embedding = settings.get("embedding") if isinstance(settings.get("embedding"), dict) else {}
    model_id = (
        _text((data or {}).get("model_id"))
        or _text((data or {}).get("model"))
        or _text(embedding.get("model"))
        or _text(gemini.get("embedding_model"))
        or DEFAULT_GEMINI_EMBEDDING_MODEL
    )
    result = _run_gemini_direct_test(project_root, profiles_dir, model_id=model_id, test_type="embedding")
    return {**result, "provider": "gemini", "display_name": "Gemini embedding"}


def create_model_profile(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    profile = _sanitize_model_profile(data)
    if not profile.get("model_id"):
        raise ValueError("model_id is required.")
    existing_ids = {_text(item.get("id")) for item in settings.get("model_profiles", []) if isinstance(item, dict)}
    base_id = _safe_profile_id(profile.get("id") or profile.get("display_name") or profile.get("model_id"))
    profile["id"] = _unique_profile_id(base_id, existing_ids)
    settings["model_profiles"].append(profile)
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, "model_profile": _public_model_profile(profile), **llm_settings_status(project_root, profiles_dir)}


def update_model_profile(project_root: str | Path, profiles_dir: str | Path, profile_id: str, data: dict[str, Any]) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    profile = _profile_by_id(settings, profile_id, include_disabled=True)
    if not profile:
        raise KeyError(f"Unknown model profile: {profile_id}")
    updated = _sanitize_model_profile({**profile, **data, "id": profile.get("id")})
    profiles = []
    for item in settings.get("model_profiles", []):
        if isinstance(item, dict) and _text(item.get("id")) == _text(profile_id):
            profiles.append(updated)
        elif isinstance(item, dict):
            profiles.append(item)
    settings["model_profiles"] = profiles
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, "model_profile": _public_model_profile(updated), **llm_settings_status(project_root, profiles_dir)}


def delete_model_profile(project_root: str | Path, profiles_dir: str | Path, profile_id: str) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    profile_id = _text(profile_id)
    original_count = len(settings.get("model_profiles", []))
    settings["model_profiles"] = [
        profile
        for profile in settings.get("model_profiles", [])
        if isinstance(profile, dict) and _text(profile.get("id")) != profile_id
    ]
    if len(settings["model_profiles"]) == original_count:
        raise KeyError(f"Unknown model profile: {profile_id}")
    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    deleted_warning = f"Model profile {profile_id} was deleted; affected role assignments were cleared."
    if roles.get("default_answer_model_profile_id") == profile_id:
        roles["default_answer_model_profile_id"] = None
    if roles.get("strategy_model_profile_id") == profile_id:
        roles["strategy_model_profile_id"] = None
    compare_ids = roles.get("compare_model_profile_ids") if isinstance(roles.get("compare_model_profile_ids"), list) else []
    roles["compare_model_profile_ids"] = [item for item in compare_ids if _text(item) != profile_id]
    settings["roles"] = roles
    settings.setdefault("warnings", [])
    if isinstance(settings["warnings"], list):
        settings["warnings"].append(deleted_warning)
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"deleted": True, **llm_settings_status(project_root, profiles_dir)}


def test_model_profile(project_root: str | Path, profiles_dir: str | Path, profile_id: str) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    profile = _profile_by_id(settings, profile_id, include_disabled=True)
    if not profile:
        raise KeyError(f"Unknown model profile: {profile_id}")
    result = _run_provider_model_test(
        project_root,
        profiles_dir,
        provider=_profile_provider(profile),
        model_id=_text(profile.get("model_id")),
        display_name=_text(profile.get("display_name")) or _text(profile.get("model_id")),
    )
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    for item in settings.get("model_profiles", []):
        if isinstance(item, dict) and _text(item.get("id")) == _text(profile_id):
            item["last_test_status"] = result["status"]
            item["last_test_latency_ms"] = result["latency_ms"]
            item["last_test_error_type"] = result["error_type"] or None
            item["last_test_at"] = str(int(time.time()))
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return result


def test_all_enabled_model_profiles(project_root: str | Path, profiles_dir: str | Path) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    results = []
    for profile in settings.get("model_profiles", []):
        if isinstance(profile, dict) and bool(profile.get("enabled", True)):
            results.append(test_model_profile(project_root, profiles_dir, _text(profile.get("id"))))
    return {"results": results}


def save_profile_roles(project_root: str | Path, profiles_dir: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    settings["roles"] = _updated_profile_roles(settings, data)
    _write_local_llm_settings(project_root, profiles_dir, settings)
    return {"saved": True, **llm_settings_status(project_root, profiles_dir)}


def _canonical_settings(raw: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "settings_revision": _safe_int(raw.get("settings_revision") or raw.get("config_version")),
        "openrouter": {
            "api_key": "",
            "base_url": DEFAULT_OPENROUTER_BASE_URL,
            "site_url": "",
            "site_name": "",
        },
        "gemini": {
            "api_key": "",
            "display_name": "Gemini Direct",
            "model": DEFAULT_GEMINI_MODEL,
            "embedding_model": DEFAULT_GEMINI_EMBEDDING_MODEL,
        },
        "model_profiles": [],
        "roles": {
            "default_answer_model_profile_id": None,
            "strategy_model_profile_id": None,
            "compare_model_profile_ids": [],
        },
        "embedding": {
            "provider": None,
            "api_key": None,
            "model": None,
        },
        "warnings": [],
    }
    openrouter = raw.get("openrouter") if isinstance(raw.get("openrouter"), dict) else {}
    if openrouter:
        settings["openrouter"].update({
            "api_key": _text(openrouter.get("api_key")),
            "base_url": _text(openrouter.get("base_url")) or DEFAULT_OPENROUTER_BASE_URL,
            "site_url": _text(openrouter.get("site_url")),
            "site_name": _text(openrouter.get("site_name")),
        })
        if isinstance(openrouter.get("last_test"), dict):
            settings["openrouter"]["last_test"] = _safe_test_result(openrouter["last_test"])

    gemini = raw.get("gemini") if isinstance(raw.get("gemini"), dict) else {}
    if gemini:
        settings = _apply_gemini_settings(settings, gemini)

    providers = raw.get("providers") if isinstance(raw.get("providers"), dict) else {}
    if isinstance(providers.get("openrouter"), dict):
        settings = _apply_openrouter_settings(settings, providers["openrouter"])
    if isinstance(providers.get("gemini"), dict):
        settings = _apply_gemini_settings(settings, providers["gemini"])
        settings = _apply_embedding_settings(settings, {"provider": "gemini", **providers["gemini"]})

    raw_profiles = raw.get("model_profiles")
    if isinstance(raw_profiles, list):
        settings["model_profiles"] = [_sanitize_model_profile(item) for item in raw_profiles if isinstance(item, dict) and _text(item.get("model_id") or item.get("model"))]

    legacy_default = raw.get("default_model") if isinstance(raw.get("default_model"), dict) else {}
    if legacy_default and _provider(legacy_default.get("provider") or "openrouter") in {"openrouter", "gemini"}:
        profile_id = _profile_id_for_model(settings, legacy_default, create=True)
        if profile_id:
            settings["roles"]["default_answer_model_profile_id"] = profile_id

    roles = raw.get("roles") if isinstance(raw.get("roles"), dict) else {}
    if roles:
        if any(key in roles for key in ("default_answer_model_profile_id", "strategy_model_profile_id", "compare_model_profile_ids")):
            settings["roles"] = _updated_profile_roles(settings, roles)
        else:
            default_profile = _profile_id_for_role_payload(settings, roles.get("answer_model"), create=True)
            strategy_profile = _profile_id_for_role_payload(settings, roles.get("strategy_planner_model"), create=True)
            if default_profile:
                settings["roles"]["default_answer_model_profile_id"] = default_profile
            if strategy_profile:
                settings["roles"]["strategy_model_profile_id"] = strategy_profile
            if isinstance(roles.get("embedding_model"), dict):
                settings = _apply_embedding_settings(settings, roles["embedding_model"])

    embedding = raw.get("embedding") if isinstance(raw.get("embedding"), dict) else {}
    if embedding:
        settings = _apply_embedding_settings(settings, embedding)
        if isinstance(embedding.get("last_test"), dict):
            settings["embedding"]["last_test"] = _safe_test_result(embedding.get("last_test"))
        if _provider(embedding.get("provider")) == "gemini" and _text(embedding.get("api_key")) and not _text(settings["gemini"].get("api_key")):
            settings["gemini"]["api_key"] = _text(embedding.get("api_key"))
        if _provider(embedding.get("provider")) == "gemini" and _text(embedding.get("model")):
            settings["gemini"]["embedding_model"] = _text(embedding.get("model"))

    if isinstance(raw.get("warnings"), list):
        settings["warnings"] = [_text(item) for item in raw["warnings"] if _text(item)]
    return settings


def _apply_openrouter_settings(settings: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    settings = _canonical_settings(settings) if "openrouter" not in settings else settings
    current = settings["openrouter"]
    existing_key = _text(current.get("api_key"))
    if bool(data.get("clear_key")):
        current["api_key"] = ""
    elif "api_key" in data and _text(data.get("api_key")):
        current["api_key"] = _text(data.get("api_key"))
    elif existing_key:
        current["api_key"] = existing_key
    if "base_url" in data:
        current["base_url"] = _text(data.get("base_url")) or DEFAULT_OPENROUTER_BASE_URL
    current.setdefault("base_url", DEFAULT_OPENROUTER_BASE_URL)
    for key in ("site_url", "site_name"):
        if key in data:
            current[key] = _text(data.get(key))
    return settings


def _apply_gemini_settings(settings: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    settings = _canonical_settings(settings) if "gemini" not in settings else settings
    current = settings["gemini"]
    existing_key = _text(current.get("api_key"))
    if bool(data.get("clear_key")):
        current["api_key"] = ""
    elif "api_key" in data and _text(data.get("api_key")):
        current["api_key"] = _text(data.get("api_key"))
    elif existing_key:
        current["api_key"] = existing_key
    if "display_name" in data or "label" in data:
        current["display_name"] = _text(data.get("display_name") or data.get("label")) or "Gemini Direct"
    elif not _text(current.get("display_name")):
        current["display_name"] = "Gemini Direct"
    if "model" in data or "default_model" in data:
        current["model"] = _text(data.get("model") or data.get("default_model")) or DEFAULT_GEMINI_MODEL
    elif not _text(current.get("model")):
        current["model"] = DEFAULT_GEMINI_MODEL
    if "embedding_model" in data:
        current["embedding_model"] = _text(data.get("embedding_model")) or DEFAULT_GEMINI_EMBEDDING_MODEL
    elif not _text(current.get("embedding_model")):
        current["embedding_model"] = DEFAULT_GEMINI_EMBEDDING_MODEL
    if isinstance(data.get("last_test"), dict):
        current["last_test"] = _safe_test_result(data.get("last_test"))
    return settings


def _apply_embedding_settings(settings: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    settings = _canonical_settings(settings) if "embedding" not in settings else settings
    embedding = settings["embedding"]
    provider = _provider(data.get("provider")) or embedding.get("provider")
    if provider:
        embedding["provider"] = provider
    existing_key = _text(embedding.get("api_key"))
    if bool(data.get("clear_key")):
        embedding["api_key"] = None
    elif "api_key" in data and _text(data.get("api_key")):
        embedding["api_key"] = _text(data.get("api_key"))
    elif existing_key:
        embedding["api_key"] = existing_key
    model = _text(data.get("embedding_model") or data.get("model") or data.get("default_model"))
    if model:
        embedding["model"] = model
        if provider == "gemini" and "gemini" in settings:
            settings["gemini"]["embedding_model"] = model
    if isinstance(data.get("last_test"), dict):
        embedding["last_test"] = _safe_test_result(data.get("last_test"))
    return settings


def _sanitize_model_profile(item: dict[str, Any]) -> dict[str, Any]:
    provider = _provider(item.get("provider") or "openrouter")
    if provider not in {"openrouter", "gemini"}:
        provider = "openrouter"
    model_id = _text(item.get("model_id") or item.get("model"))
    display_name = _text(item.get("display_name") or item.get("label")) or model_id
    profile = {
        "id": _safe_profile_id(item.get("id") or display_name or model_id),
        "provider": provider,
        "display_name": display_name,
        "model_id": model_id,
        "family": _text(item.get("family") or item.get("tag")),
        "enabled": bool(item.get("enabled", True)),
        "notes": _text(item.get("notes"))[:500],
        "last_test_status": _text(item.get("last_test_status")) or None,
        "last_test_latency_ms": item.get("last_test_latency_ms") if item.get("last_test_latency_ms") in (None, "") else int(float(item.get("last_test_latency_ms") or 0)),
        "last_test_error_type": _text(item.get("last_test_error_type")) or None,
        "last_test_at": _text(item.get("last_test_at")) or None,
    }
    return profile


def _updated_profile_roles(settings: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    profile_ids = {
        _text(profile.get("id"))
        for profile in settings.get("model_profiles", [])
        if isinstance(profile, dict) and not _is_stale_gemini_profile(profile)
    }
    next_roles = {
        "default_answer_model_profile_id": roles.get("default_answer_model_profile_id"),
        "strategy_model_profile_id": roles.get("strategy_model_profile_id"),
        "compare_model_profile_ids": list(roles.get("compare_model_profile_ids") or []),
    }
    if "default_answer_model_profile_id" in data:
        value = _text(data.get("default_answer_model_profile_id"))
        next_roles["default_answer_model_profile_id"] = value if value in profile_ids else None
    if "strategy_model_profile_id" in data:
        value = _text(data.get("strategy_model_profile_id"))
        next_roles["strategy_model_profile_id"] = value if value in profile_ids else None
    if isinstance(data.get("compare_model_profile_ids"), list):
        next_roles["compare_model_profile_ids"] = [
            _text(item)
            for item in data.get("compare_model_profile_ids", [])[:6]
            if _text(item) in profile_ids
        ]
    return next_roles


def _profile_id_for_role_payload(settings: dict[str, Any], payload: Any, *, create: bool = False) -> str:
    return _profile_id_for_model(settings, payload, create=create) if isinstance(payload, dict) else ""


def _profile_id_for_model(settings: dict[str, Any], payload: dict[str, Any], *, create: bool = False) -> str:
    provider = _provider(payload.get("provider") or "openrouter")
    if provider not in {"openrouter", "gemini"}:
        return ""
    model_id = _text(payload.get("model_id") or payload.get("model"))
    if not model_id:
        return ""
    for profile in settings.get("model_profiles", []):
        if isinstance(profile, dict) and _profile_provider(profile) == provider and _text(profile.get("model_id")) == model_id:
            return _text(profile.get("id"))
    if not create:
        return ""
    profile = _sanitize_model_profile({
        "provider": provider,
        "display_name": payload.get("display_name") or payload.get("label") or model_id,
        "model_id": model_id,
        "enabled": True,
    })
    existing_ids = {_text(item.get("id")) for item in settings.get("model_profiles", []) if isinstance(item, dict)}
    profile["id"] = _unique_profile_id(profile["id"], existing_ids)
    settings.setdefault("model_profiles", []).append(profile)
    return _text(profile.get("id"))


def _profile_by_id(settings: dict[str, Any], profile_id: Any, *, include_disabled: bool = False) -> dict[str, Any] | None:
    profile_id = _text(profile_id)
    if not profile_id:
        return None
    for profile in settings.get("model_profiles", []):
        if (
            isinstance(profile, dict)
            and _text(profile.get("id")) == profile_id
            and (include_disabled or bool(profile.get("enabled", True)))
        ):
            return profile
    return None


def _profile_provider(profile: dict[str, Any]) -> str:
    provider = _provider(profile.get("provider") or "openrouter")
    return provider if provider in {"openrouter", "gemini"} else "openrouter"


def _public_model_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _text(profile.get("id")),
        "provider": _profile_provider(profile),
        "display_name": _text(profile.get("display_name")),
        "model_id": _text(profile.get("model_id")),
        "family": _text(profile.get("family")),
        "enabled": bool(profile.get("enabled", True)),
        "notes": _text(profile.get("notes")),
        "last_test_status": profile.get("last_test_status"),
        "last_test_latency_ms": profile.get("last_test_latency_ms"),
        "last_test_error_type": profile.get("last_test_error_type"),
        "last_test_at": profile.get("last_test_at"),
    }


def _public_profile_roles(settings: dict[str, Any]) -> dict[str, Any]:
    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    return {
        "default_answer_model_profile_id": roles.get("default_answer_model_profile_id") if _profile_available_for_role(settings, roles.get("default_answer_model_profile_id")) else None,
        "strategy_model_profile_id": roles.get("strategy_model_profile_id") if _profile_available_for_role(settings, roles.get("strategy_model_profile_id")) else None,
        "compare_model_profile_ids": [
            _text(profile_id)
            for profile_id in roles.get("compare_model_profile_ids", [])
            if _profile_available_for_role(settings, profile_id)
        ] if isinstance(roles.get("compare_model_profile_ids"), list) else [],
    }


def _profile_exists(settings: dict[str, Any], profile_id: Any) -> bool:
    profile_id = _text(profile_id)
    return any(isinstance(profile, dict) and _text(profile.get("id")) == profile_id for profile in settings.get("model_profiles", []))


def _profile_available_for_role(settings: dict[str, Any], profile_id: Any) -> bool:
    profile_id = _text(profile_id)
    for profile in settings.get("model_profiles", []):
        if isinstance(profile, dict) and _text(profile.get("id")) == profile_id:
            return not _is_stale_gemini_profile(profile)
    return False


def _is_stale_gemini_profile(profile: dict[str, Any]) -> bool:
    return _profile_provider(profile) == "gemini" and _text(profile.get("model_id")) in STALE_GEMINI_MODEL_IDS


def _role_profile_warnings(settings: dict[str, Any]) -> list[str]:
    warnings = list(settings.get("warnings") if isinstance(settings.get("warnings"), list) else [])
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    if _text(gemini.get("model")) in STALE_GEMINI_MODEL_IDS:
        warnings.append("Gemini chat model setting uses stale model gemini-flash-latest; update it to gemini-2.5-flash.")
    roles = settings.get("roles") if isinstance(settings.get("roles"), dict) else {}
    for key in ("default_answer_model_profile_id", "strategy_model_profile_id"):
        if roles.get(key) and not _profile_exists(settings, roles.get(key)):
            warnings.append(f"Role {key} points to a missing model profile. Select another saved model.")
        elif roles.get(key) and not _profile_available_for_role(settings, roles.get(key)):
            warnings.append(f"Role {key} points to stale Gemini model gemini-flash-latest. Select another saved model such as gemini-2.5-flash.")
    compare_ids = roles.get("compare_model_profile_ids") if isinstance(roles.get("compare_model_profile_ids"), list) else []
    missing = [profile_id for profile_id in compare_ids if not _profile_exists(settings, profile_id)]
    if missing:
        warnings.append("One or more compare model profile assignments point to missing saved models.")
    stale_assigned = [profile_id for profile_id in compare_ids if _profile_exists(settings, profile_id) and not _profile_available_for_role(settings, profile_id)]
    if stale_assigned:
        warnings.append("One or more compare model profile assignments use stale Gemini model gemini-flash-latest. Select another saved model such as gemini-2.5-flash.")
    for profile in settings.get("model_profiles", []):
        if isinstance(profile, dict) and _is_stale_gemini_profile(profile):
            warnings.append("Gemini model profile uses stale model gemini-flash-latest; update it to gemini-2.5-flash.")
    return warnings


def _safe_openrouter_status(settings: dict[str, Any], values: dict[str, str] | None = None) -> dict[str, Any]:
    values = values or {}
    openrouter = settings.get("openrouter") if isinstance(settings.get("openrouter"), dict) else {}
    key = _text(openrouter.get("api_key")) or _text(values.get("OPENROUTER_API_KEY"))
    status = "configured" if key else "missing key"
    last_test = _safe_test_result(openrouter.get("last_test") if isinstance(openrouter.get("last_test"), dict) else {})
    if last_test.get("status") == "success":
        status = "last key test success"
    elif last_test.get("status") == "failed":
        status = "last key test failed"
    return {
        "provider": "openrouter",
        "key_configured": bool(key),
        "masked_key_display": _masked_key_display(key),
        "base_url": _text(values.get("OPENROUTER_BASE_URL")) or _text(openrouter.get("base_url")) or DEFAULT_OPENROUTER_BASE_URL,
        "site_url": _text(values.get("OPENROUTER_SITE_URL")) or _text(openrouter.get("site_url")),
        "site_name": _text(values.get("OPENROUTER_SITE_NAME")) or _text(openrouter.get("site_name")),
        "status": status,
        "last_test": last_test or None,
    }


def _safe_gemini_status(settings: dict[str, Any], values: dict[str, str] | None = None) -> dict[str, Any]:
    values = values or {}
    gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
    key = _text(gemini.get("api_key")) or _text(values.get("GEMINI_API_KEY"))
    model = _text(gemini.get("model")) or _text(values.get("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL
    embedding_model = (
        _text(gemini.get("embedding_model"))
        or _text(values.get("GEMINI_EMBEDDING_MODEL"))
        or _text(values.get("EMBEDDING_MODEL"))
        or DEFAULT_GEMINI_EMBEDDING_MODEL
    )
    status = "configured" if key else "missing key"
    if not model:
        status = "missing model"
    last_test = _safe_test_result(gemini.get("last_test") if isinstance(gemini.get("last_test"), dict) else {})
    if last_test.get("status") == "success":
        status = "last key test success"
    elif last_test.get("status") == "failed":
        status = "last key test failed"
    return {
        "provider": "gemini",
        "key_configured": bool(key),
        "masked_key_display": _masked_key_display(key),
        "display_name": _text(gemini.get("display_name")) or "Gemini Direct",
        "model": model,
        "embedding_model": embedding_model,
        "status": status,
        "last_test": last_test or None,
    }


def _safe_embedding_status(settings: dict[str, Any], values: dict[str, str] | None = None) -> dict[str, Any]:
    values = values or {}
    embedding = settings.get("embedding") if isinstance(settings.get("embedding"), dict) else {}
    provider = _provider(embedding.get("provider")) or _provider(values.get("EMBEDDING_PROVIDER")) or None
    key = (provider_api_key(provider, values) if provider else "") or _text(embedding.get("api_key"))
    model = _text(embedding.get("model")) or _text(values.get("EMBEDDING_MODEL")) or _text(values.get("GEMINI_EMBEDDING_MODEL"))
    last_test = _safe_test_result(embedding.get("last_test") if isinstance(embedding.get("last_test"), dict) else {})
    return {
        "provider": provider,
        "key_configured": bool(key),
        "masked_key_display": _masked_key_display(key),
        "model": model or None,
        "last_test": last_test or None,
    }


def _safe_test_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or not result:
        return {}
    return {
        "ok": bool(result.get("ok")),
        "status": _text(result.get("status")),
        "error_type": _text(result.get("error_type")),
        "latency_ms": int(float(result.get("latency_ms") or result.get("last_test_latency_ms") or 0)),
        "message": _text(result.get("message")),
        "tested_at": _text(result.get("tested_at") or result.get("last_test_at")),
        "test_type": _text(result.get("test_type")),
        "model_id": _text(result.get("model_id") or result.get("model")),
        "endpoint_url": _text(result.get("endpoint_url")),
        "payload_shape": _text(result.get("payload_shape")),
        "status_code": result.get("status_code"),
        "google_error_status": _text(result.get("google_error_status")),
        "key_source": _text(result.get("key_source")),
        "masked_key_suffix": _text(result.get("masked_key_suffix")),
    }


def _profile_display_for_model(settings: dict[str, Any], model_id: str) -> str:
    for profile in settings.get("model_profiles", []):
        if isinstance(profile, dict) and _text(profile.get("model_id")) == _text(model_id):
            return _text(profile.get("display_name"))
    return ""


def _comparison_profile_from_model_profile(profile: dict[str, Any]) -> dict[str, Any]:
    provider = _profile_provider(profile)
    return {
        "id": _text(profile.get("id")),
        "label": _text(profile.get("display_name")) or _text(profile.get("model_id")),
        "display_name": _text(profile.get("display_name")) or _text(profile.get("model_id")),
        "provider": provider,
        "model": _text(profile.get("model_id")),
        "model_id": _text(profile.get("model_id")),
        "enabled": bool(profile.get("enabled", True)),
        "role": "comparison",
        "notes": _text(profile.get("notes")),
    }


def _first_enabled_profile_model(settings: dict[str, Any], *, provider: str = "") -> str:
    provider = _provider(provider)
    for profile in settings.get("model_profiles", []):
        if (
            isinstance(profile, dict)
            and bool(profile.get("enabled", True))
            and _text(profile.get("model_id"))
            and (not provider or _profile_provider(profile) == provider)
            and not _is_stale_gemini_profile(profile)
        ):
            return _text(profile.get("model_id"))
    return ""


def _run_provider_model_test(project_root: str | Path, profiles_dir: str | Path, *, provider: str, model_id: str, display_name: str) -> dict[str, Any]:
    provider = _provider(provider)
    result = test_llm_connection(project_root, profiles_dir, {"provider": provider, "model": model_id})
    safe = {
        "ok": bool(result.get("ok")),
        "provider": provider,
        "display_name": display_name,
        "model_id": model_id,
        "test_type": _text(result.get("test_type")),
        "endpoint_url": _text(result.get("endpoint_url")),
        "payload_shape": _text(result.get("payload_shape")),
        "latency_ms": int(float(result.get("latency_ms") or 0)),
        "status": _text(result.get("status")) or "failed",
        "status_code": result.get("status_code"),
        "google_error_status": _text(result.get("google_error_status")),
        "error_type": _text(result.get("error_type")),
        "message": _text(result.get("message")),
        "key_source": _text(result.get("key_source")),
        "masked_key_suffix": _text(result.get("masked_key_suffix")),
    }
    settings = read_local_llm_settings(project_root=project_root, profiles_dir=profiles_dir)
    if provider == "openrouter" and display_name == "OpenRouter key":
        openrouter = settings.get("openrouter") if isinstance(settings.get("openrouter"), dict) else {}
        openrouter["last_test"] = {**safe, "tested_at": str(int(time.time()))}
        settings["openrouter"] = openrouter
        _write_local_llm_settings(project_root, profiles_dir, settings)
    elif provider == "gemini" and display_name == "Gemini key":
        gemini = settings.get("gemini") if isinstance(settings.get("gemini"), dict) else {}
        gemini["last_test"] = {**safe, "tested_at": str(int(time.time()))}
        settings["gemini"] = gemini
        _write_local_llm_settings(project_root, profiles_dir, settings)
    return safe


def _unique_profile_id(base_id: str, existing_ids: set[str]) -> str:
    base = _safe_profile_id(base_id)
    if base not in existing_ids:
        return base
    index = 2
    while f"{base}_{index}" in existing_ids:
        index += 1
    return f"{base}_{index}"


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


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _text(value: Any) -> str:
    return str(value or "").strip()
