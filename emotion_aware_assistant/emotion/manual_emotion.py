from __future__ import annotations

from .dummy_emotion import DummyEmotionRecognizer
from .labels import ALLOWED_EMOTIONS, normalize_emotion


class ManualEmotionRecognizer(DummyEmotionRecognizer):
    """Recognizer controlled by the UI/CLI for demos and user override."""

    def __init__(self, emotion: str = "neutral", confidence: float = 0.95):
        normalized = normalize_emotion(emotion)
        if normalized not in ALLOWED_EMOTIONS:
            normalized = "neutral"
        super().__init__(fixed_emotion=normalized, confidence=confidence)
        self.load()

    def set_emotion(self, emotion: str) -> None:
        normalized = normalize_emotion(emotion)
        if normalized not in ALLOWED_EMOTIONS:
            raise ValueError(f"Unsupported manual emotion: {emotion}")
        self.fixed_emotion = normalized
