from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from emotion_aware_assistant.core.types import ChatRequest, ChatResponse


class InteractionLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"session_{stamp}.jsonl"

    def log_chat(self, request: ChatRequest, response: ChatResponse) -> None:
        event = {
            "timestamp": time.time(),
            "document_id": request.paper_context.document_id,
            "document_type": request.paper_context.document_type,
            "document_title": request.paper_context.document_title,
            "page_number": request.paper_context.page_number,
            "highlight_id": request.paper_context.highlight_id,
            "selected_text_preview": request.paper_context.selected_text[:240],
            "selected_passage_length": len(request.paper_context.selected_text),
            "passage_type": request.paper_context.passage_type,
            "difficulty_hint": request.paper_context.difficulty_hint,
            "user_question": request.user_question,
            "followup_action": request.followup_action,
            "raw_emotion": request.learning_state.raw_emotion,
            "smoothed_emotion": request.learning_state.smoothed_emotion,
            "learning_state": request.learning_state.state,
            "trend": request.learning_state.trend,
            "confidence": request.learning_state.confidence,
            "strategy": request.learning_state.strategy,
            "model_name": response.model_name,
            "latency": response.latency_sec,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "response_length": len(response.text),
            "manual_override": request.learning_state.manual_override,
            "errors": response.error,
        }
        self.log_event(event)

    def log_event(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._jsonable(event), ensure_ascii=False) + "\n")

    def _jsonable(self, value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value
