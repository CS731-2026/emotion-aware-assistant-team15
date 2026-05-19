from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResponsePolicy:
    state: str
    strategy: str
    tone: str
    depth: str
    structure: list[str]
    avoid: str
    trend_adjustment: str
    passage_adjustment: str
    paragraph_length: str
    followup_buttons: list[str]
    avoid_list: list[str]
    ideal_response_shape: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


BASE_POLICIES: dict[str, ResponsePolicy] = {
    "confusion": ResponsePolicy(
        state="confusion",
        strategy="step_by_step_clarification",
        tone="calm and clarifying",
        depth="medium",
        structure=[
            "core intuition",
            "step-by-step breakdown",
            "key terms",
            "one example",
            "check-understanding question",
        ],
        avoid="dense paragraphs",
        trend_adjustment="Give intuition first when confusion is rising; slow down with glossary and example when stable.",
        passage_adjustment="",
        paragraph_length="short paragraphs with numbered steps",
        followup_buttons=["Break into steps", "Give an example", "Quiz me"],
        avoid_list=["dense prose", "undefined jargon", "long chains of reasoning without checkpoints"],
        ideal_response_shape=["Core idea", "Step-by-step", "Example", "Check"],
    ),
    "frustration": ResponsePolicy(
        state="frustration",
        strategy="supportive_simplification",
        tone="supportive and non-patronizing",
        depth="simple first, optional detail later",
        structure=["acknowledge density", "simplest version", "analogy", "one next step"],
        avoid="too many details upfront",
        trend_adjustment="Reduce information density as frustration rises; switch mode and suggest a takeaway if persistent.",
        passage_adjustment="",
        paragraph_length="very short paragraphs",
        followup_buttons=["Explain more simply", "Key takeaway", "Give an example"],
        avoid_list=["overloading detail", "patronizing tone", "too many caveats before the main idea"],
        ideal_response_shape=["Simplest version first", "Different way to think about it", "One small next step"],
    ),
    "boredom": ResponsePolicy(
        state="boredom",
        strategy="concise_reengagement",
        tone="concise and interactive",
        depth="short first",
        structure=["one-sentence takeaway", "why it matters", "quick quiz or challenge"],
        avoid="long explanations",
        trend_adjustment="Keep the response short and end with a concrete challenge.",
        passage_adjustment="",
        paragraph_length="one to three compact bullets",
        followup_buttons=["Key takeaway", "Quiz me", "Go deeper"],
        avoid_list=["long background", "generic encouragement", "multi-page exposition"],
        ideal_response_shape=["One-sentence takeaway", "Why it matters", "Quick quiz"],
    ),
    "engagement": ResponsePolicy(
        state="engagement",
        strategy="deeper_academic_expansion",
        tone="scholarly and direct",
        depth="deeper",
        structure=["direct answer", "technical explanation", "assumptions/limitations", "broader connection"],
        avoid="unnecessary simplification",
        trend_adjustment="Use deeper technical detail for stable engagement; moderate depth when engagement is recovering.",
        passage_adjustment="",
        paragraph_length="medium scholarly paragraphs",
        followup_buttons=["Go deeper", "Give an example", "Break into steps"],
        avoid_list=["hand-wavy summaries", "unsupported claims", "ignoring assumptions"],
        ideal_response_shape=["Technical explanation", "Assumptions", "Limitations", "Connection to broader methods"],
    ),
    "uncertain": ResponsePolicy(
        state="uncertain",
        strategy="neutral_adaptive_support",
        tone="neutral supportive",
        depth="medium",
        structure=["direct answer", "short explanation", "useful next step"],
        avoid="over-adapting",
        trend_adjustment="Use neutral support because the affective signal is weak or unavailable.",
        passage_adjustment="",
        paragraph_length="moderate paragraphs",
        followup_buttons=["Break into steps", "Give an example", "Key takeaway"],
        avoid_list=["claims about the user's emotion", "unsupported paper details"],
        ideal_response_shape=["Direct answer", "Grounded explanation", "Useful next step"],
    ),
}


PASSAGE_ADJUSTMENTS: dict[str, str] = {
    "formula/equation": "Explain variables, what the equation computes, intuition, and a small example.",
    "method/process/mechanism": "Break the method into inputs, steps, outputs, and why each step exists.",
    "result/claim": "Summarize the finding, evidence, significance, and limitation.",
    "definition/concept": "Define the concept, give an example, and contrast a non-example.",
    "limitation/future work": "Explain the constraint, implication, and possible future work.",
    "dataset/evaluation": "Explain evaluation setup, metrics, result interpretation, and what the evidence does or does not show.",
    "comparison/related work": "Compare methods across dimensions, trade-offs, assumptions, and practical implications.",
    "general": "Use a normal academic explanation.",
}


def get_response_policy(state: str, trend: str = "uncertain", passage_type: str = "general") -> ResponsePolicy:
    base = BASE_POLICIES.get(state, BASE_POLICIES["uncertain"])
    trend_adjustment = base.trend_adjustment
    if trend == "persistent_frustration":
        trend_adjustment = "Switch explanation mode, give a minimal example, and suggest focusing on the takeaway."
    elif trend == "stable_confusion":
        trend_adjustment = "Use a slower breakdown with glossary and example."
    elif trend == "rising_confusion":
        trend_adjustment = "Start with intuition before numbered steps."
    elif trend == "recovering_engagement":
        trend_adjustment = "Use moderate depth and avoid a sudden jump in density."
    elif trend == "stable_engagement":
        trend_adjustment = "Provide deeper technical detail and connect to broader literature."
    return ResponsePolicy(
        state=base.state,
        strategy=base.strategy,
        tone=base.tone,
        depth=base.depth,
        structure=base.structure,
        avoid=base.avoid,
        trend_adjustment=trend_adjustment,
        passage_adjustment=PASSAGE_ADJUSTMENTS.get(passage_type, PASSAGE_ADJUSTMENTS["general"]),
        paragraph_length=base.paragraph_length,
        followup_buttons=base.followup_buttons,
        avoid_list=base.avoid_list,
        ideal_response_shape=base.ideal_response_shape,
    )
