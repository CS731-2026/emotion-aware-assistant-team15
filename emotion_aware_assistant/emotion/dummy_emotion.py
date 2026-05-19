from __future__ import annotations

import itertools
import time

from emotion_aware_assistant.core.types import EmotionPrediction

from .labels import ALLOWED_EMOTIONS, normalize_emotion


class DummyEmotionRecognizer:
    """Offline recognizer used for smoke checks and demos without model weights."""

    def __init__(self, fixed_emotion: str | None = None, confidence: float = 0.82):
        self.fixed_emotion = normalize_emotion(fixed_emotion or "")
        self.confidence = confidence
        self._loaded = False
        self._cycle = itertools.cycle(ALLOWED_EMOTIONS)

    def load(self) -> None:
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def label_order(self) -> list[str]:
        return list(ALLOWED_EMOTIONS)

    def predict(self, face_bgr_or_rgb=None) -> EmotionPrediction:
        if not self._loaded:
            self.load()
        emotion = self.fixed_emotion if self.fixed_emotion in ALLOWED_EMOTIONS else next(self._cycle)
        floor = (1.0 - self.confidence) / (len(ALLOWED_EMOTIONS) - 1)
        probabilities = {label: floor for label in ALLOWED_EMOTIONS}
        probabilities[emotion] = self.confidence
        return EmotionPrediction(
            emotion=emotion,
            confidence=self.confidence,
            probabilities=probabilities,
            timestamp=time.time(),
            source="dummy",
        )
