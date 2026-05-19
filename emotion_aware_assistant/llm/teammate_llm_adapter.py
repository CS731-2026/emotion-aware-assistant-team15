from __future__ import annotations

from emotion_aware_assistant.core.types import ChatRequest, ChatResponse

from .dummy_llm import DummyLLMClient


class TeammateLLMAdapter:
    """Stable replacement point for teammate B's custom LLM integration."""

    def __init__(self):
        self._fallback = DummyLLMClient()

    @property
    def is_available(self) -> bool:
        return self._fallback.is_available

    @property
    def name(self) -> str:
        return "teammate_llm_adapter"

    def chat(self, request: ChatRequest) -> ChatResponse:
        # Teammate B can replace this method while preserving the LLMClient interface.
        response = self._fallback.chat(request)
        return ChatResponse(
            text=response.text,
            model_name=self.name,
            latency_sec=response.latency_sec,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=response.estimated_cost,
            error=response.error,
        )
