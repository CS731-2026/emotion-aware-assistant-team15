from __future__ import annotations

import os
import time
from typing import Any

import requests

from emotion_aware_assistant.core.config import load_env_file
from emotion_aware_assistant.core.types import ChatRequest, ChatResponse

from .prompt_builder import PromptBuilder


class OpenRouterClient:
    def __init__(self, timeout_sec: int = 60):
        load_env_file()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "EmotionAwareAcademicAssistant")
        self.timeout_sec = timeout_sec
        self.prompt_builder = PromptBuilder()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def name(self) -> str:
        return "openrouter"

    def chat(self, request: ChatRequest) -> ChatResponse:
        if not self.is_available:
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
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
