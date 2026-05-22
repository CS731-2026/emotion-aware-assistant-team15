from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


RawEmotionSource = Literal["raw_8class_checkpoint", "simulated", "unavailable"]
AcademicState = Literal["boredom", "confusion", "engagement", "frustration", "uncertain"]
SupportCue = Literal["re_engagement", "clarification", "deepening", "difficulty_support", "neutral", "uncertain"]
QuestionIntent = Literal[
    "explain",
    "simplify",
    "example",
    "compare",
    "why_how",
    "still_confused",
    "critique",
    "general_followup",
]
ResponseMode = Literal["baseline", "adaptive", "unknown"]

ACADEMIC_STATES = ("boredom", "confusion", "engagement", "frustration")
SUPPORT_CUES = ("re_engagement", "clarification", "deepening", "difficulty_support", "neutral", "uncertain")
SUPPORT_CUE_BY_ACADEMIC_STATE = {
    "boredom": "re_engagement",
    "confusion": "clarification",
    "engagement": "deepening",
    "frustration": "difficulty_support",
    "uncertain": "uncertain",
}
SUPPORT_CUE_ALIASES = {
    "re_engagement": "re_engagement",
    "clarification": "clarification",
    "sustained_clarification": "clarification",
    "gentle_clarification": "clarification",
    "clarify_and_reengage": "clarification",
    "deepening": "deepening",
    "engagement_extension": "deepening",
    "difficulty_support": "difficulty_support",
    "reassurance": "difficulty_support",
    "reduce_load": "difficulty_support",
    "neutral": "neutral",
    "neutral_or_uncertain": "uncertain",
    "mixed": "uncertain",
    "uncertain": "uncertain",
}
MAPPING_RULE_VERSION = "required_8class_probability_aggregation_v1"
NON_DIAGNOSTIC_DISCLAIMER = "Use this as a lightweight learning-support signal, not as a psychological diagnosis."


@dataclass(frozen=True)
class RawEmotionEvidence:
    source: RawEmotionSource
    checkpoint_path: str | None
    raw_emotion_label: str | None
    raw_emotion_confidence: float | None
    raw_emotion_probabilities: dict[str, float]
    raw_top_emotions: list[dict[str, float | str]]
    frame_timestamp: str | None
    crop_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class AcademicStateMapping:
    mapping_rule_version: str
    mapped_academic_state: AcademicState
    mapped_academic_scores: dict[str, float]
    mapping_explanation: str
    confidence_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ReactionWindowSummary:
    window_start: str | None
    window_end: str | None
    dominant_raw_emotions: list[dict[str, float | str]]
    dominant_mapped_states: list[dict[str, float | str]]
    trend: str
    evidence_count: int
    reliability_notes: list[str]
    avg_mapped_scores: dict[str, float]
    support_cue: str
    start_time: str | None = None
    end_time: str | None = None
    duration_sec: float | None = None
    end_reason: str | None = None
    read_progress_estimate: float | None = None
    sample_count: int | None = None
    early_distribution: dict[str, float] | None = None
    middle_distribution: dict[str, float] | None = None
    late_distribution: dict[str, float] | None = None
    early_dominant_state: str | None = None
    middle_dominant_state: str | None = None
    late_dominant_state: str | None = None
    state_transition: str | None = None
    transition_label: str | None = None
    dominant_state: str | None = None
    secondary_state: str | None = None
    stability: str | None = None
    confidence_handling: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class LearningProcessContext:
    document_id: str | None
    highlight_id: str | None
    turn_index: int
    followup_count_for_highlight: int
    same_highlight_repeated_questions: bool
    last_user_question: str | None
    current_user_question: str | None
    question_intent: QuestionIntent
    previous_strategy: str | None
    recent_strategies: list[str]
    last_response_was_baseline_or_adaptive: ResponseMode
    time_since_last_assistant_answer: float | None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class LearningSignalPackage:
    active_source: Literal["raw_8class_process_aware"]
    raw_emotion_evidence: RawEmotionEvidence | None
    academic_state_mapping: AcademicStateMapping | None
    reaction_window_summary: ReactionWindowSummary | None
    learning_process_context: LearningProcessContext
    academic_state_scores: dict[str, float]
    dominant_academic_state: AcademicState
    secondary_academic_state: AcademicState | None
    support_cue: SupportCue
    inferred_process_state: str
    strategy_state: str
    recommended_strategy: str
    strategy_reason: str
    reason_text: str
    academic_state_evidence_text: str
    confidence_handling: str
    source_turn_type: str
    support_cue_label: str
    prompt_guidance: list[str]
    diagnostic_only_direct_4class: dict[str, Any] | None
    non_diagnostic_disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def detect_question_intent(question: Any) -> QuestionIntent:
    text = str(question or "").strip().lower()
    if not text:
        return "general_followup"
    if any(phrase in text for phrase in ("still don't understand", "still do not understand", "still confused", "i don't get", "i do not get", "not understand", "lost")):
        return "still_confused"
    if any(token in text for token in ("simplify", "simpler", "plain english", "define", "term", "jargon")):
        return "simplify"
    if any(token in text for token in ("example", "instance", "case", "walk me through")):
        return "example"
    if any(token in text for token in ("compare", "contrast", "difference", "similar", "versus", " vs ")):
        return "compare"
    if any(token in text for token in ("why", "how", "mechanism", "intuition")):
        return "why_how"
    if any(token in text for token in ("assumption", "limitation", "valid", "critique", "weakness", "problem")):
        return "critique"
    if any(token in text for token in ("explain", "what does", "what is", "meaning")):
        return "explain"
    return "general_followup"


def build_learning_signal_package(
    *,
    raw_emotion_evidence: RawEmotionEvidence | None,
    academic_state_mapping: AcademicStateMapping | None,
    reaction_window_summary: ReactionWindowSummary | None,
    learning_process_context: LearningProcessContext,
    diagnostic_only_direct_4class: dict[str, Any] | None = None,
) -> LearningSignalPackage:
    academic_state_scores = _effective_scores(academic_state_mapping, reaction_window_summary)
    dominant_academic_state, secondary_academic_state = _academic_state_layer(
        academic_state_scores,
        academic_state_mapping,
    )
    support_cue = canonical_support_cue(
        reaction_window_summary.support_cue if reaction_window_summary else None,
        dominant_academic_state=dominant_academic_state,
    )
    inferred_process_state, strategy_state, recommended_strategy, strategy_reason = infer_process_state(
        raw_emotion_evidence=raw_emotion_evidence,
        academic_state_mapping=academic_state_mapping,
        reaction_window_summary=reaction_window_summary,
        learning_process_context=learning_process_context,
    )
    reason_metadata = build_strategy_reason_text(
        source_turn_type=(
            "strategy_reexplanation"
            if learning_process_context.last_response_was_baseline_or_adaptive == "adaptive"
            else "baseline_explanation"
            if learning_process_context.last_response_was_baseline_or_adaptive == "baseline"
            else "previous_explanation"
        ),
        academic_state_scores=academic_state_scores,
        dominant_academic_state=dominant_academic_state,
        secondary_academic_state=secondary_academic_state,
        support_cue=support_cue,
        trend=reaction_window_summary.trend if reaction_window_summary else "",
        avg_confidence=None,
        recommended_strategy=recommended_strategy,
        pedagogical_move=recommended_strategy,
    )
    diagnostic = dict(diagnostic_only_direct_4class or {}) if diagnostic_only_direct_4class is not None else None
    if diagnostic is not None:
        diagnostic["used_for_strategy"] = False
    return LearningSignalPackage(
        active_source="raw_8class_process_aware",
        raw_emotion_evidence=raw_emotion_evidence,
        academic_state_mapping=academic_state_mapping,
        reaction_window_summary=reaction_window_summary,
        learning_process_context=learning_process_context,
        academic_state_scores=academic_state_scores,
        dominant_academic_state=dominant_academic_state,
        secondary_academic_state=secondary_academic_state,
        support_cue=support_cue,  # type: ignore[arg-type]
        inferred_process_state=inferred_process_state,
        strategy_state=strategy_state,
        recommended_strategy=recommended_strategy,
        strategy_reason=reason_metadata["reason_text"] or strategy_reason,
        reason_text=reason_metadata["reason_text"] or strategy_reason,
        academic_state_evidence_text=reason_metadata["academic_state_evidence_text"],
        confidence_handling=reason_metadata["confidence_handling"],
        source_turn_type=reason_metadata["source_turn_type"],
        support_cue_label=reason_metadata["support_cue_label"],
        prompt_guidance=prompt_guidance_for_process_state(inferred_process_state),
        diagnostic_only_direct_4class=diagnostic,
    )


def infer_process_state(
    *,
    raw_emotion_evidence: RawEmotionEvidence | None,
    academic_state_mapping: AcademicStateMapping | None,
    reaction_window_summary: ReactionWindowSummary | None,
    learning_process_context: LearningProcessContext,
) -> tuple[str, str, str, str]:
    intent = learning_process_context.question_intent
    previous_strategy = str(learning_process_context.previous_strategy or "").lower()
    followup_count = int(learning_process_context.followup_count_for_highlight or 0)
    repeated = bool(learning_process_context.same_highlight_repeated_questions or followup_count > 1)
    scores = _effective_scores(academic_state_mapping, reaction_window_summary)
    confusion_like = scores.get("confusion", 0.0)
    frustration_like = scores.get("frustration", 0.0)
    boredom_like = scores.get("boredom", 0.0)
    engagement_like = scores.get("engagement", 0.0)
    evidence_count = reaction_window_summary.evidence_count if reaction_window_summary else 0
    dominant_academic_state, _secondary_academic_state = _academic_state_layer(scores, academic_state_mapping)
    support_cue = canonical_support_cue(
        reaction_window_summary.support_cue if reaction_window_summary else None,
        dominant_academic_state=dominant_academic_state,
    )
    raw_label = str(raw_emotion_evidence.raw_emotion_label if raw_emotion_evidence else "").lower()
    after_answer = learning_process_context.last_response_was_baseline_or_adaptive in {"baseline", "adaptive"}
    has_reaction_context = bool(
        reaction_window_summary
        and (
            evidence_count > 0
            or str(reaction_window_summary.trend or "").lower() != "single_frame_only"
        )
    )

    if (
        not has_reaction_context
        and not after_answer
        and followup_count <= 0
        and learning_process_context.turn_index <= 1
        and intent in {"explain", "general_followup"}
    ):
        return (
            "baseline_ready",
            "uncertain",
            "baseline_explanation",
            "first explanation should stay paper-grounded before adapting from process signals",
        )

    if intent == "still_confused" and ("simpl" in previous_strategy or "define" in previous_strategy):
        return (
            "continued_difficulty",
            "confusion",
            "worked_example",
            "recent learning signal suggested continued difficulty after the previous explanation",
        )

    if repeated and (confusion_like >= 0.45 or frustration_like >= 0.35 or support_cue in {"clarification", "difficulty_support"}):
        strategy = "worked_example" if intent in {"example", "still_confused"} else "step_by_step_decomposition"
        return (
            "continued_difficulty",
            "frustration" if frustration_like > confusion_like else "confusion",
            strategy,
            "recent learning signal suggested continued difficulty after the previous explanation",
        )

    if intent in {"compare", "why_how"} and engagement_like >= max(boredom_like, 0.25):
        return (
            "engaged_deepening",
            "engagement",
            "compare_or_contrast" if intent == "compare" else "deepen_or_extend",
            "the user is asking an active deeper question, so neutral expression is treated as engaged reading",
        )

    if has_reaction_context and (dominant_academic_state == "engagement" or support_cue == "deepening"):
        return (
            "engaged_deepening",
            "engagement",
            "deepen_or_extend",
            "recent learning signal supports a deeper or extended explanation",
        )

    if has_reaction_context and support_cue == "clarification":
        return (
            "clarification_needed",
            "confusion",
            "simplify_and_define_terms",
            "reaction-window learning signal suggests clarification would be useful",
        )

    if has_reaction_context and support_cue == "difficulty_support":
        return (
            "continued_difficulty",
            "frustration",
            "step_by_step_decomposition",
            "reaction-window learning signal suggests continued difficulty after the previous explanation",
        )

    if has_reaction_context and support_cue == "re_engagement":
        return (
            "possible_low_engagement",
            "boredom",
            "relevance_hook",
            "reaction-window learning signal suggests a light re-engagement move",
        )

    if intent == "example":
        return (
            "continued_difficulty",
            "confusion",
            "worked_example",
            "the follow-up asks for an example, so a concrete worked example is the safest next support",
        )

    if after_answer and raw_label in {"fear", "surprise"} and evidence_count >= 1:
        strategy = "deepen_or_extend" if intent in {"why_how", "compare"} else "simplify_and_define_terms"
        return (
            "possible_confusion_or_curiosity",
            "engagement" if strategy == "deepen_or_extend" else "confusion",
            strategy,
            "reaction-window signal may indicate confusion or curiosity; the question wording resolves the next support style",
        )

    if after_answer and (frustration_like >= 0.45 or raw_label in {"anger", "sad", "disgust"}) and evidence_count >= 3:
        return (
            "frustration_like_difficulty",
            "frustration",
            "step_by_step_decomposition",
            "persistent post-answer difficulty signal suggests a supportive step-by-step explanation",
        )

    if boredom_like >= 0.35 and evidence_count < 3:
        return (
            "possible_low_engagement",
            "boredom",
            "relevance_hook",
            "low-evidence signal suggests a light relevance hook without overreacting",
        )

    if engagement_like >= 0.60 or support_cue == "deepening":
        return (
            "stable_engagement_like",
            "engagement",
            "deepen_or_extend",
            "recent learning signal supports a deeper or extended explanation",
        )

    if has_reaction_context or after_answer or followup_count > 0:
        return (
            "adaptive_uncertain_or_mixed",
            "uncertain",
            "simplify_and_define_terms",
            "reaction or dialogue context is present, so use a neutral adaptive clarification instead of repeating the baseline",
        )

    return (
        "uncertain_or_mixed",
        "uncertain",
        "baseline_explanation",
        "learning signal is mixed or low confidence, so the answer should remain paper-grounded",
    )


def prompt_guidance_for_process_state(process_state: str) -> list[str]:
    state = str(process_state or "").lower()
    if state in {"frustration_like_difficulty", "continued_difficulty"}:
        return [
            "Use calm, supportive, short steps.",
            "Reduce jargon and cognitive load.",
            "Do not imply the learner has a diagnosed emotion.",
        ]
    if state in {"possible_confusion_or_curiosity"}:
        return [
            "Clarify assumptions and define key terms before adding detail.",
            "Use the user's question wording to decide between clarification and extension.",
        ]
    if state in {"clarification_needed"}:
        return [
            "Define key terms and clarify assumptions.",
            "Use a concise scaffold before adding detail.",
            "Do not imply the learner has a diagnosed emotion.",
        ]
    if state in {"possible_low_engagement"}:
        return [
            "Start with a concise takeaway.",
            "Add a relevance hook and one optional next step.",
        ]
    if state in {"adaptive_uncertain_or_mixed"}:
        return [
            "Use a concise adaptive clarification.",
            "Define key terms before adding new detail.",
            "Avoid claiming a specific learner state.",
        ]
    if state in {"engaged_deepening", "stable_engagement_like"}:
        return [
            "Offer a deeper explanation with connections to the paper.",
            "Extend only where the provided paper context supports it.",
        ]
    return ["Use a baseline paper-grounded explanation and avoid overfitting to weak signals."]


def package_from_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, LearningSignalPackage):
        return value.to_dict()
    if not isinstance(value, dict):
        return {}
    clean = _jsonable(value)
    _backfill_package_layers(clean)
    diagnostic = clean.get("diagnostic_only_direct_4class")
    if isinstance(diagnostic, dict):
        diagnostic["used_for_strategy"] = False
    return clean


def canonical_support_cue(value: Any, *, dominant_academic_state: str | None = None) -> str:
    cue = str(value or "").strip().lower()
    if cue in SUPPORT_CUE_ALIASES:
        return SUPPORT_CUE_ALIASES[cue]
    state = str(dominant_academic_state or "").strip().lower()
    if state in SUPPORT_CUE_BY_ACADEMIC_STATE:
        return SUPPORT_CUE_BY_ACADEMIC_STATE[state]
    return "uncertain"


def build_strategy_reason_text(
    *,
    source_turn_type: Any,
    academic_state_scores: dict[str, Any],
    dominant_academic_state: Any,
    secondary_academic_state: Any,
    support_cue: Any,
    trend: Any,
    avg_confidence: Any,
    recommended_strategy: Any,
    pedagogical_move: Any,
) -> dict[str, str]:
    scores = {state: _bounded_probability(academic_state_scores.get(state)) for state in ACADEMIC_STATES}
    dominant = _safe_academic_state(dominant_academic_state, scores)
    secondary = str(secondary_academic_state or "").strip().lower()
    if secondary not in ACADEMIC_STATES or secondary == dominant:
        ordered = [state for state in sorted(ACADEMIC_STATES, key=lambda state: scores.get(state, 0.0), reverse=True) if state != dominant]
        secondary = ordered[0] if ordered and scores.get(ordered[0], 0.0) > 0 else ""
    canonical_cue = canonical_support_cue(support_cue, dominant_academic_state=dominant)
    source_type = str(source_turn_type or "").strip().lower() or "previous_explanation"
    source_label = _source_explanation_label(source_type)
    trend_text = str(trend or "").strip().lower()
    confidence_handling = _confidence_handling(avg_confidence)
    evidence_text = _academic_state_evidence_text(scores, dominant, secondary, trend_text, confidence_handling)
    dominant_phrase = f"{dominant}-dominant" if dominant in ACADEMIC_STATES else "mixed"
    state_detail = _state_detail_text(scores, dominant, secondary)
    trend_clause = f" with a {trend_text} trend" if trend_text and trend_text != "uncertain" else ""
    confidence_clause = ", but the average confidence was low" if confidence_handling == "low_confidence" else ""
    first_sentence = (
        f"While {source_label} was being read, the reaction window was {dominant_phrase} "
        f"({state_detail}){trend_clause}{confidence_clause}."
    )
    resolving_difficulty = _is_resolving_difficulty_pattern(scores, dominant, secondary, trend_text)
    if resolving_difficulty and canonical_cue == "deepening":
        strategy_sentence = (
            "The difficulty signal was falling and engagement was close behind, suggesting the earlier difficulty may be resolving. "
            "The system therefore selected a cautious deepening strategy."
        )
    elif canonical_cue == "deepening" and dominant != "engagement":
        strategy_sentence = (
            "The signal did not show persistent difficulty or clarification needs across the full window, "
            "so the system selected a cautious continuation strategy."
        )
    else:
        strategy_sentence = _strategy_reason_sentence(canonical_cue, recommended_strategy, pedagogical_move, confidence_handling)
    return {
        "reason_text": f"{first_sentence} {strategy_sentence}".strip(),
        "academic_state_evidence_text": evidence_text,
        "confidence_handling": confidence_handling,
        "source_turn_type": source_type,
        "support_cue_label": _support_cue_display_label(canonical_cue),
    }


def top_probability_items(probabilities: dict[str, Any], *, label_key: str = "label", value_key: str = "probability", limit: int = 3) -> list[dict[str, Any]]:
    if not isinstance(probabilities, dict):
        return []
    items = sorted(
        ((str(label), _bounded_probability(value)) for label, value in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{label_key: label, value_key: round(value, 4)} for label, value in items[:limit]]


def _safe_academic_state(value: Any, scores: dict[str, float]) -> str:
    state = str(value or "").strip().lower()
    if state in ACADEMIC_STATES:
        return state
    ordered = sorted(ACADEMIC_STATES, key=lambda item: scores.get(item, 0.0), reverse=True)
    return ordered[0] if ordered and scores.get(ordered[0], 0.0) > 0 else "uncertain"


def _source_explanation_label(source_turn_type: str) -> str:
    if source_turn_type in {"baseline_explanation", "baseline"}:
        return "the baseline explanation"
    if source_turn_type in {"strategy_reexplanation", "adaptive"}:
        return "the previous adaptive explanation"
    return "the previous explanation"


def _confidence_handling(avg_confidence: Any) -> str:
    try:
        value = float(avg_confidence)
    except Exception:
        return "standard_confidence"
    return "low_confidence" if value < 0.5 else "standard_confidence"


def _percentage(value: float) -> str:
    return f"{round(_bounded_probability(value) * 100):.0f}%"


def _state_detail_text(scores: dict[str, float], dominant: str, secondary: str) -> str:
    if dominant not in ACADEMIC_STATES:
        return "mixed learning-state evidence"
    parts = [f"{dominant} {_percentage(scores.get(dominant, 0.0))}"]
    if secondary in ACADEMIC_STATES and scores.get(secondary, 0.0) > 0:
        parts.append(f"secondary {secondary} {_percentage(scores.get(secondary, 0.0))}")
    return ", ".join(parts)


def _academic_state_evidence_text(
    scores: dict[str, float],
    dominant: str,
    secondary: str,
    trend: str,
    confidence_handling: str,
) -> str:
    states = [dominant] if dominant in ACADEMIC_STATES else []
    if secondary in ACADEMIC_STATES and secondary not in states:
        states.append(secondary)
    if not states:
        states = sorted(ACADEMIC_STATES, key=lambda item: scores.get(item, 0.0), reverse=True)[:2]
    score_text = ", ".join(f"{state} {_percentage(scores.get(state, 0.0))}" for state in states if scores.get(state, 0.0) > 0)
    parts = [score_text or "mixed learning-state evidence"]
    if confidence_handling == "low_confidence":
        parts.append("low-confidence signal")
    if trend:
        parts.append(f"trend {trend}")
    return " · ".join(parts)


def _support_cue_display_label(support_cue: str) -> str:
    labels = {
        "re_engagement": "re-engagement",
        "clarification": "clarification",
        "deepening": "deepening",
        "difficulty_support": "reduce cognitive load",
        "neutral": "neutral",
        "uncertain": "mixed / low-confidence signal",
    }
    return labels.get(support_cue, "mixed / low-confidence signal")


def _is_resolving_difficulty_pattern(
    scores: dict[str, float],
    dominant: str,
    secondary: str,
    trend: str,
) -> bool:
    if dominant != "frustration" or secondary != "engagement" or str(trend or "").lower() != "falling":
        return False
    frustration = _bounded_probability(scores.get("frustration"))
    engagement = _bounded_probability(scores.get("engagement"))
    return frustration > 0 and engagement > 0 and 0 <= frustration - engagement <= 0.15


def _strategy_reason_sentence(
    support_cue: str,
    recommended_strategy: Any,
    pedagogical_move: Any,
    confidence_handling: str,
) -> str:
    strategy = str(recommended_strategy or "").strip().lower()
    move = str(pedagogical_move or "").strip().lower()
    step_like = "step" in strategy or "step" in move or "breakdown" in strategy or "decomposition" in strategy
    if support_cue == "deepening":
        return "This suggested the answer could be extended, so the system selected a deeper explanation strategy."
    if support_cue == "difficulty_support":
        if confidence_handling == "low_confidence":
            phrase = "step-by-step strategy" if step_like else "conservative support strategy"
            return f"The system therefore chose a conservative {phrase} to reduce cognitive load."
        phrase = "step-by-step support strategy" if step_like else "support strategy"
        return f"This suggested a strategy to reduce cognitive load, so the system selected a {phrase}."
    if support_cue == "clarification":
        return "This suggested clarification would help, so the system selected a clarification strategy."
    if support_cue == "re_engagement":
        return "This suggested a light re-engagement move, so the system selected a relevance-focused strategy."
    return "Because the signal was mixed or uncertain, the system selected a conservative continuation strategy."


def _effective_scores(
    mapping: AcademicStateMapping | None,
    reaction_window_summary: ReactionWindowSummary | None,
) -> dict[str, float]:
    if reaction_window_summary and reaction_window_summary.avg_mapped_scores:
        return {state: _bounded_probability(reaction_window_summary.avg_mapped_scores.get(state)) for state in ACADEMIC_STATES}
    if mapping and mapping.mapped_academic_scores:
        return {state: _bounded_probability(mapping.mapped_academic_scores.get(state)) for state in ACADEMIC_STATES}
    return {state: 0.0 for state in ACADEMIC_STATES}


def _academic_state_layer(
    scores: dict[str, Any],
    mapping: AcademicStateMapping | None = None,
) -> tuple[AcademicState, AcademicState | None]:
    normalized = {state: _bounded_probability(scores.get(state)) for state in ACADEMIC_STATES}
    ordered = sorted(ACADEMIC_STATES, key=lambda state: normalized.get(state, 0.0), reverse=True)
    if ordered and normalized.get(ordered[0], 0.0) > 0:
        dominant = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 and normalized.get(ordered[1], 0.0) > 0 else None
        return dominant, secondary  # type: ignore[return-value]
    mapped = str(mapping.mapped_academic_state if mapping else "").strip().lower()
    if mapped in ACADEMIC_STATES:
        return mapped, None  # type: ignore[return-value]
    return "uncertain", None


def _backfill_package_layers(clean: dict[str, Any]) -> None:
    mapping = clean.get("academic_state_mapping") if isinstance(clean.get("academic_state_mapping"), dict) else {}
    reaction = clean.get("reaction_window_summary") if isinstance(clean.get("reaction_window_summary"), dict) else {}
    scores = clean.get("academic_state_scores") if isinstance(clean.get("academic_state_scores"), dict) else {}
    if not scores:
        reaction_scores = reaction.get("avg_mapped_scores") if isinstance(reaction.get("avg_mapped_scores"), dict) else {}
        mapping_scores = mapping.get("mapped_academic_scores") if isinstance(mapping.get("mapped_academic_scores"), dict) else {}
        scores = reaction_scores or mapping_scores
    academic_state_scores = {state: _bounded_probability(scores.get(state)) for state in ACADEMIC_STATES} if isinstance(scores, dict) else {state: 0.0 for state in ACADEMIC_STATES}
    clean["academic_state_scores"] = academic_state_scores
    if not clean.get("dominant_academic_state"):
        ordered = sorted(ACADEMIC_STATES, key=lambda state: academic_state_scores.get(state, 0.0), reverse=True)
        dominant = ordered[0] if ordered and academic_state_scores.get(ordered[0], 0.0) > 0 else str(mapping.get("mapped_academic_state") or "uncertain")
        clean["dominant_academic_state"] = dominant if dominant in {*ACADEMIC_STATES, "uncertain"} else "uncertain"
    if "secondary_academic_state" not in clean:
        ordered = sorted(ACADEMIC_STATES, key=lambda state: academic_state_scores.get(state, 0.0), reverse=True)
        secondary = ordered[1] if len(ordered) > 1 and academic_state_scores.get(ordered[1], 0.0) > 0 else None
        clean["secondary_academic_state"] = secondary
    clean["support_cue"] = canonical_support_cue(
        clean.get("support_cue") or reaction.get("support_cue"),
        dominant_academic_state=str(clean.get("dominant_academic_state") or "uncertain"),
    )
    if not clean.get("reason_text") or not clean.get("academic_state_evidence_text"):
        process_context = clean.get("learning_process_context") if isinstance(clean.get("learning_process_context"), dict) else {}
        reason = build_strategy_reason_text(
            source_turn_type=clean.get("source_turn_type") or process_context.get("last_response_was_baseline_or_adaptive") or "previous_explanation",
            academic_state_scores=academic_state_scores,
            dominant_academic_state=clean.get("dominant_academic_state"),
            secondary_academic_state=clean.get("secondary_academic_state"),
            support_cue=clean.get("support_cue"),
            trend=reaction.get("trend"),
            avg_confidence=reaction.get("avg_confidence"),
            recommended_strategy=clean.get("recommended_strategy"),
            pedagogical_move=clean.get("recommended_strategy"),
        )
        clean.setdefault("reason_text", reason["reason_text"])
        clean.setdefault("strategy_reason", reason["reason_text"])
        clean.setdefault("academic_state_evidence_text", reason["academic_state_evidence_text"])
        clean.setdefault("confidence_handling", reason["confidence_handling"])
        clean.setdefault("source_turn_type", reason["source_turn_type"])
        clean.setdefault("support_cue_label", reason["support_cue_label"])


def _bounded_probability(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value
