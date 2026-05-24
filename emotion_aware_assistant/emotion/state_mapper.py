from __future__ import annotations

import time

from emotion_aware_assistant.core.types import EmotionPrediction, LearningState

from .labels import (
    EMOTION_TO_STATE,
    LEARNING_STATES,
    STATE_TO_STRATEGY,
    STATE_TO_VALENCE_AROUSAL,
    VALENCE_AROUSAL,
)


STATE_EXPLANATIONS: dict[str, str] = {
    "confusion": "A recent pattern resembles uncertainty during difficult reading, so the assistant slows down and clarifies.",
    "frustration": "A recent pattern resembles difficulty with high effort, so the assistant reduces density and supports a simpler route.",
    "boredom": "A recent pattern resembles low engagement, so the assistant uses concise takeaways and interaction.",
    "engagement": "A recent pattern resembles steady engagement, so the assistant can provide more technical depth.",
    "uncertain": "The signal is weak or unavailable, so the assistant uses a neutral academic support style.",
}


def map_prediction_to_learning_state(
    raw_prediction: EmotionPrediction,
    smoothed_prediction: EmotionPrediction | None = None,
    trend: str = "uncertain",
    duration_sec: float = 0.0,
    manual_override: bool = False,
) -> LearningState:
    smoothed = smoothed_prediction or raw_prediction
    emotion = smoothed.emotion
    if emotion not in EMOTION_TO_STATE:
        state = "uncertain"
        valence, arousal = STATE_TO_VALENCE_AROUSAL[state]
    else:
        state = EMOTION_TO_STATE[emotion]
        valence, arousal = VALENCE_AROUSAL[emotion]
    return LearningState(
        state=state,
        confidence=smoothed.confidence,
        raw_emotion=raw_prediction.emotion,
        smoothed_emotion=emotion,
        valence=valence,
        arousal=arousal,
        strategy=STATE_TO_STRATEGY[state],
        trend=trend,
        duration_sec=duration_sec,
        explanation=STATE_EXPLANATIONS[state],
        manual_override=manual_override,
    )


def state_to_learning_state(
    state: str,
    confidence: float = 0.9,
    trend: str = "uncertain",
    duration_sec: float = 0.0,
    manual_override: bool = True,
) -> LearningState:
    state = state.strip().lower()
    if state not in LEARNING_STATES:
        state = "uncertain"
    valence, arousal = STATE_TO_VALENCE_AROUSAL[state]
    return LearningState(
        state=state,
        confidence=confidence,
        raw_emotion="manual_state",
        smoothed_emotion="manual_state",
        valence=valence,
        arousal=arousal,
        strategy=STATE_TO_STRATEGY[state],
        trend=trend,
        duration_sec=duration_sec,
        explanation=STATE_EXPLANATIONS[state],
        manual_override=manual_override,
    )


def uncertain_learning_state() -> LearningState:
    return state_to_learning_state(
        "uncertain",
        confidence=0.0,
        trend="uncertain",
        duration_sec=0.0,
        manual_override=False,
    )
