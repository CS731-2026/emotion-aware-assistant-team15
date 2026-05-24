from __future__ import annotations

import time
from collections import deque

from emotion_aware_assistant.core.types import LearningState

from .labels import STATE_TO_STRATEGY, STATE_TO_VALENCE_AROUSAL


class AffectiveTrendTracker:
    """Slow temporal model over smoothed learning-state snapshots."""

    def __init__(
        self,
        window_sec: float = 6.0,
        hysteresis_updates: int = 3,
        high_confidence_switch_threshold: float = 0.80,
    ):
        self.window_sec = window_sec
        self.hysteresis_updates = hysteresis_updates
        self.high_confidence_switch_threshold = high_confidence_switch_threshold
        self._history: deque[tuple[float, LearningState]] = deque()
        self._current_state = "uncertain"

    def update(self, candidate: LearningState, now: float | None = None) -> LearningState:
        now = now or time.time()
        self._prune(now)
        previous_state = self._current_state
        next_state = self._choose_state(candidate)
        self._current_state = next_state

        provisional = self._with_state(candidate, next_state, trend="uncertain", duration_sec=0.0)
        self._history.append((now, provisional))
        duration = self._duration_for(next_state, now)
        trend = self._trend(next_state, previous_state, duration)
        return self._with_state(candidate, next_state, trend=trend, duration_sec=duration)

    def snapshot(self) -> LearningState:
        if not self._history:
            from .state_mapper import uncertain_learning_state

            return uncertain_learning_state()
        timestamp, last = self._history[-1]
        duration = self._duration_for(last.state, time.time())
        trend = self._trend(last.state, last.state, duration)
        return self._with_state(last, last.state, trend=trend, duration_sec=duration)

    def history(self, limit: int = 10) -> list[dict]:
        items = list(self._history)[-limit:]
        return [
            {
                "timestamp": timestamp,
                "state": item.state,
                "trend": item.trend,
                "confidence": item.confidence,
                "raw_emotion": item.raw_emotion,
                "smoothed_emotion": item.smoothed_emotion,
                "strategy": item.strategy,
                "manual_override": item.manual_override,
            }
            for timestamp, item in items
        ]

    def dominant_state(self) -> str:
        if not self._history:
            return "uncertain"
        weights: dict[str, float] = {}
        for _, item in self._history:
            weights[item.state] = weights.get(item.state, 0.0) + max(0.01, item.confidence)
        return max(weights, key=weights.get) if weights else "uncertain"

    def debug_snapshot(self, limit: int = 10) -> dict:
        current = self.snapshot()
        return {
            "current_state": current.state,
            "trend": current.trend,
            "duration_sec": current.duration_sec,
            "dominant_state": self.dominant_state(),
            "history": self.history(limit),
            "window_sec": self.window_sec,
            "hysteresis_updates": self.hysteresis_updates,
        }

    def _choose_state(self, candidate: LearningState) -> str:
        if candidate.manual_override:
            return candidate.state
        if candidate.state == "uncertain":
            return self._current_state if self._current_state != "uncertain" else "uncertain"
        if candidate.confidence >= self.high_confidence_switch_threshold:
            return candidate.state
        recent_same = sum(1 for _, item in list(self._history)[-self.hysteresis_updates :] if item.state == candidate.state)
        if recent_same + 1 >= self.hysteresis_updates:
            return candidate.state
        return self._current_state if self._current_state != "uncertain" else candidate.state

    def _duration_for(self, state: str, now: float) -> float:
        if not self._history:
            return 0.0
        first_timestamp = now
        for timestamp, item in reversed(self._history):
            if item.state != state:
                break
            first_timestamp = timestamp
        return max(0.0, now - first_timestamp)

    def _trend(self, state: str, previous_state: str, duration_sec: float) -> str:
        if state == "uncertain":
            return "uncertain"
        if state == "engagement":
            return "recovering_engagement" if previous_state not in ("engagement", "uncertain") else "stable_engagement"
        if state == "confusion":
            return "rising_confusion" if previous_state != "confusion" else "stable_confusion"
        if state == "frustration":
            return "persistent_frustration" if duration_sec >= max(2.0, self.window_sec / 2.0) else "rising_frustration"
        if state == "boredom":
            return "boredom_or_disengagement"
        return "uncertain"

    def _prune(self, now: float) -> None:
        while self._history and now - self._history[0][0] > self.window_sec:
            self._history.popleft()

    @staticmethod
    def _with_state(candidate: LearningState, state: str, trend: str, duration_sec: float) -> LearningState:
        valence, arousal = STATE_TO_VALENCE_AROUSAL.get(state, (candidate.valence, candidate.arousal))
        return LearningState(
            state=state,
            confidence=candidate.confidence,
            raw_emotion=candidate.raw_emotion,
            smoothed_emotion=candidate.smoothed_emotion,
            valence=valence,
            arousal=arousal,
            strategy=STATE_TO_STRATEGY.get(state, candidate.strategy),
            trend=trend,
            duration_sec=duration_sec,
            explanation=candidate.explanation,
            manual_override=candidate.manual_override,
        )
