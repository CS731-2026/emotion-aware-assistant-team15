from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class PassageAnalysis:
    passage_type: str
    detected_keywords: list[str]
    difficulty_hint: str
    suggested_explanation_mode: str

    def as_dict(self) -> dict:
        return asdict(self)


PATTERNS: list[tuple[str, list[str], str]] = [
    ("formula/equation", ["=", "sum", "sigma", "theta", "lambda", "argmax", "argmin", "loss", "objective", "score", "gradient"], "Explain variables, computation, intuition, and one small example."),
    ("dataset/evaluation", ["dataset", "evaluate", "evaluation", "metric", "accuracy", "macro", "f1", "test split", "validation", "benchmark", "raf-db", "affectnet"], "Explain setup, metric, result, and interpretation."),
    ("comparison/related work", ["compared", "baseline", "prior work", "related work", "trade", "outperform", "versus", "unlike", "similar to"], "Compare dimensions, trade-offs, and what changes relative to the baseline."),
    ("method/process/mechanism", ["method", "algorithm", "procedure", "pipeline", "step", "first", "then", "finally", "input", "output", "mechanism", "update"], "Break into ordered steps with input, operation, output, and purpose."),
    ("result/claim", ["result", "finding", "shows", "indicates", "suggests", "claim", "significant", "improves", "performance"], "State the claim, evidence, implication, and limitation."),
    ("definition/concept", ["define", "definition", "refers to", "is called", "denotes", "concept", "means"], "Give definition, example, and non-example."),
    ("limitation/future work", ["limitation", "future work", "constraint", "threat", "cannot", "fails", "assumption", "risk"], "Explain constraint, consequence, and possible future work."),
]


def classify_passage(text: str) -> str:
    return analyze_passage(text).passage_type


def analyze_passage(text: str) -> PassageAnalysis:
    lowered = text.lower()
    if not lowered.strip():
        return PassageAnalysis(
            passage_type="general",
            detected_keywords=[],
            difficulty_hint="unknown",
            suggested_explanation_mode="Ask for a passage or summarize the available page context.",
        )

    best_type = "general"
    best_keywords: list[str] = []
    best_mode = "Give a concise academic explanation grounded in the provided context."
    for passage_type, keywords, mode in PATTERNS:
        matches = [keyword for keyword in keywords if _keyword_present(lowered, keyword)]
        if matches and (not best_keywords or len(matches) > len(best_keywords)):
            best_type = passage_type
            best_keywords = matches
            best_mode = mode

    return PassageAnalysis(
        passage_type=best_type,
        detected_keywords=best_keywords,
        difficulty_hint=_difficulty_hint(text, best_type, best_keywords),
        suggested_explanation_mode=best_mode,
    )


def surrounding_text(page_text: str, start: int, end: int, radius: int = 800) -> str:
    start = max(0, start)
    end = min(len(page_text), max(end, start))
    left = max(0, start - radius)
    right = min(len(page_text), end + radius)
    return page_text[left:right].strip()


def _keyword_present(lowered_text: str, keyword: str) -> bool:
    if keyword == "=":
        return "=" in lowered_text
    if " " in keyword or "-" in keyword:
        return keyword in lowered_text
    return re.search(rf"\b{re.escape(keyword)}\b", lowered_text) is not None


def _difficulty_hint(text: str, passage_type: str, keywords: list[str]) -> str:
    token_count = len(re.findall(r"[A-Za-z0-9_]+", text))
    symbol_count = sum(text.count(symbol) for symbol in ["=", "+", "-", "/", "(", ")", "[", "]"])
    if passage_type == "formula/equation" or symbol_count >= 4:
        return "high: mathematical or symbolic passage"
    if passage_type in {"method/process/mechanism", "dataset/evaluation"} and (token_count > 60 or len(keywords) >= 3):
        return "medium-high: procedural or evaluation detail"
    if token_count < 25:
        return "low-medium: short passage"
    return "moderate: conceptual academic prose"
