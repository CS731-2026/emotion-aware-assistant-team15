from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmotionPrediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    timestamp: float
    face_bbox: tuple[int, int, int, int] | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class LearningState:
    state: str
    confidence: float
    raw_emotion: str
    smoothed_emotion: str
    valence: float
    arousal: float
    strategy: str
    trend: str
    duration_sec: float
    explanation: str
    manual_override: bool = False


@dataclass(frozen=True)
class PaperContext:
    document_title: str
    page_number: int | None
    selected_text: str
    surrounding_text: str
    retrieved_chunks: list[str] = field(default_factory=list)
    passage_type: str = "general"
    page_title: str | None = None
    section_hint: str | None = None
    difficulty_hint: str = "moderate"
    passage_analysis: dict[str, Any] = field(default_factory=dict)
    retrieval_debug: dict[str, Any] = field(default_factory=dict)
    document_id: str | None = None
    document_type: str | None = None
    highlight_id: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    user_question: str
    paper_context: PaperContext
    learning_state: LearningState
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    followup_action: str | None = None
    model_name: str = "dummy"


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model_name: str
    latency_sec: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class SystemStatus:
    webcam_available: bool
    emotion_model_loaded: bool
    llm_available: bool
    speech_available: bool
    current_mode: str


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: str


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
