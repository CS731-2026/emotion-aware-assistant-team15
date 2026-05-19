from __future__ import annotations

import time
from collections import deque

from emotion_aware_assistant.core.types import EmotionPrediction

from .labels import ALLOWED_EMOTIONS


class EmotionBuffer:
    """Fast smoothing buffer that averages probability vectors, not labels."""

    def __init__(self, maxlen: int = 10, confidence_threshold: float = 0.35):
        self.maxlen = maxlen
        self.confidence_threshold = confidence_threshold
        self._items: deque[EmotionPrediction] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._items.clear()

    def add(self, prediction: EmotionPrediction) -> EmotionPrediction:
        self._items.append(prediction)
        return self.smoothed()

    @property
    def size(self) -> int:
        return len(self._items)

    def confidence_history(self) -> list[float]:
        return [item.confidence for item in self._items]

    def probability_history(self) -> list[dict[str, float]]:
        return [dict(item.probabilities) for item in self._items]

    def distribution(self) -> dict[str, float]:
        return self.smoothed().probabilities

    def debug_snapshot(self) -> dict:
        smoothed = self.smoothed()
        return {
            "buffer_size": self.size,
            "maxlen": self.maxlen,
            "confidence_threshold": self.confidence_threshold,
            "confidence_history": self.confidence_history(),
            "probabilities": smoothed.probabilities,
            "smoothed_emotion": smoothed.emotion,
            "smoothed_confidence": smoothed.confidence,
        }

    def smoothed(self) -> EmotionPrediction:
        if not self._items:
            probabilities = {label: 1.0 / len(ALLOWED_EMOTIONS) for label in ALLOWED_EMOTIONS}
            return EmotionPrediction(
                emotion="uncertain",
                confidence=0.0,
                probabilities=probabilities,
                timestamp=time.time(),
                source="buffer-empty",
            )

        totals = {label: 0.0 for label in ALLOWED_EMOTIONS}
        weight_total = 0.0
        for item in self._items:
            weight = max(0.01, item.confidence)
            weight_total += weight
            for label in ALLOWED_EMOTIONS:
                totals[label] += float(item.probabilities.get(label, 0.0)) * weight
        averaged = {label: value / weight_total for label, value in totals.items()}
        total_prob = sum(averaged.values()) or 1.0
        averaged = {label: value / total_prob for label, value in averaged.items()}
        best_label = max(averaged, key=averaged.get)
        best_confidence = averaged[best_label]
        emotion = best_label if best_confidence >= self.confidence_threshold else "uncertain"
        return EmotionPrediction(
            emotion=emotion,
            confidence=best_confidence,
            probabilities=averaged,
            timestamp=self._items[-1].timestamp,
            face_bbox=self._items[-1].face_bbox,
            source="emotion_buffer",
        )
