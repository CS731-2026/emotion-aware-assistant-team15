from __future__ import annotations

import os
import time
from typing import Any

import requests

from emotion_aware_assistant.core.config import load_env_file
from emotion_aware_assistant.core import llm_config
from emotion_aware_assistant.core.types import ChatRequest, ChatResponse

from .prompt_builder import PromptBuilder


class OpenRouterClient:
    def __init__(self, timeout_sec: int = 60):
        load_env_file()
        self.timeout_sec = timeout_sec
        self.prompt_builder = PromptBuilder()

    @property
    def is_available(self) -> bool:
        return bool(self._runtime_config()["api_key"])

    @property
    def name(self) -> str:
        return "openrouter"

    def chat(self, request: ChatRequest) -> ChatResponse:
        config = self._runtime_config()
        if not config["api_key"]:
            return ChatResponse(
                text="OpenRouter API key is not configured; using dummy mode is recommended.",
                model_name=request.model_name,
                latency_sec=0.0,
                error="missing_api_key",
            )
        messages = self.prompt_builder.build_messages(request)
        payload: dict[str, Any] = {
            "model": request.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 900,
        }
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": config["site_url"],
            "X-Title": config["site_name"],
        }
        start = time.perf_counter()
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return ChatResponse(
                text=text,
                model_name=data.get("model", request.model_name),
                latency_sec=time.perf_counter() - start,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                estimated_cost=None,
            )
        except Exception as exc:
            return ChatResponse(
                text=f"OpenRouter request failed: {exc}",
                model_name=request.model_name,
                latency_sec=time.perf_counter() - start,
                error=str(exc),
            )

    def _runtime_config(self) -> dict[str, str]:
        values = llm_config.read_llm_values(include_env_file=False)
        key_info = llm_config.resolve_provider_api_key("openrouter", include_env_file=False, values=values)
        return {
            "api_key": str(key_info.get("key") or ""),
            "site_url": str(values.get("OPENROUTER_SITE_URL") or os.getenv("OPENROUTER_SITE_URL") or "http://localhost"),
            "site_name": str(values.get("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_APP_NAME") or "EmotionAwareAcademicAssistant"),
        }
