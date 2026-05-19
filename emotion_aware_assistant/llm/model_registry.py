from __future__ import annotations

from typing import Any


def configured_models(config: dict[str, Any]) -> dict[str, str]:
    models = config.get("llm", {}).get("models", [])
    registry: dict[str, str] = {"dummy": "dummy"}
    for item in models:
        if isinstance(item, dict) and item.get("alias") and item.get("name"):
            registry[str(item["alias"])] = str(item["name"])
    return registry


def resolve_model(config: dict[str, Any], alias_or_name: str | None = None) -> str:
    registry = configured_models(config)
    requested = alias_or_name or config.get("llm", {}).get("default_model", "dummy")
    return registry.get(str(requested), str(requested))


def default_model_alias(config: dict[str, Any]) -> str:
    return str(config.get("llm", {}).get("default_model", "dummy"))
