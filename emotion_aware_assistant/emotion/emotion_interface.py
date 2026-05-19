from __future__ import annotations

from typing import Protocol

from emotion_aware_assistant.core.types import EmotionPrediction


class EmotionRecognizer(Protocol):
    def load(self) -> None:
        ...

    def predict(self, face_bgr_or_rgb) -> EmotionPrediction:
        ...

    @property
    def is_loaded(self) -> bool:
        ...

    @property
    def label_order(self) -> list[str]:
        ...
