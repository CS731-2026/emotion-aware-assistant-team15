from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationRecord:
    relevance: int | None = None
    clarity: int | None = None
    empathy: int | None = None
    academic_usefulness: int | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "relevance": self.relevance,
            "clarity": self.clarity,
            "empathy": self.empathy,
            "academic_usefulness": self.academic_usefulness,
            "notes": self.notes,
        }
