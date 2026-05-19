from __future__ import annotations

from typing import Protocol

from emotion_aware_assistant.core.types import ChatRequest, ChatResponse


class LLMClient(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse:
        ...

    @property
    def is_available(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...
