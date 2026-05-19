from __future__ import annotations

ALLOWED_EMOTIONS: list[str] = [
    "neutral",
    "happy",
    "angry",
    "sad",
    "fear",
    "surprise",
    "disgust",
    "contempt",
]

LEARNING_STATES: list[str] = ["confusion", "frustration", "boredom", "engagement", "uncertain"]

EMOTION_TO_STATE: dict[str, str] = {
    "sad": "frustration",
    "angry": "frustration",
    "disgust": "frustration",
    "fear": "confusion",
    "surprise": "confusion",
    "contempt": "boredom",
    "happy": "engagement",
    "neutral": "engagement",
}

VALENCE_AROUSAL: dict[str, tuple[float, float]] = {
    "sad": (-0.7, 0.4),
    "angry": (-0.8, 0.8),
    "disgust": (-0.7, 0.7),
    "fear": (-0.8, 0.9),
    "surprise": (0.0, 0.8),
    "contempt": (-0.5, 0.3),
    "happy": (0.8, 0.6),
    "neutral": (0.0, 0.4),
}

STATE_TO_STRATEGY: dict[str, str] = {
    "confusion": "step_by_step_clarification",
    "frustration": "supportive_simplification",
    "boredom": "concise_reengagement",
    "engagement": "deeper_academic_expansion",
    "uncertain": "neutral_adaptive_support",
}

STATE_TO_VALENCE_AROUSAL: dict[str, tuple[float, float]] = {
    "frustration": (-0.75, 0.75),
    "confusion": (-0.3, 0.75),
    "boredom": (-0.35, 0.25),
    "engagement": (0.4, 0.5),
    "uncertain": (0.0, 0.4),
}


def normalize_emotion(label: str) -> str:
    label = label.strip().lower()
    aliases = {"anger": "angry", "sadness": "sad", "happiness": "happy"}
    return aliases.get(label, label)
