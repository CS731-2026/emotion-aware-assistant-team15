from __future__ import annotations

from pathlib import Path


SAMPLE_TEXT = """Title: A Generic Study of Adaptive Reading Support

Abstract
This short sample paper describes an academic reading assistant that adapts explanations to a learner's current reading state. The system is intentionally generic and safe for classroom demos.

Method
The proposed method receives a selected passage, surrounding paper context, and a compact learning-state signal. First, the system chunks the document into page-aware segments. Then it retrieves the most relevant chunks for the user question. Finally, it builds a teaching prompt that separates factual context from response style.

Formula-like objective
The assistant can be described by the objective score = relevance(context, question) + clarity(policy, state) - overload(response). Here, relevance measures whether the answer is grounded in the paper, clarity measures whether the explanation matches the selected teaching policy, and overload penalizes responses that are too dense for the current state.

Results
In a demonstration setting, the same selected method paragraph produces different answers when the learning state changes. Confusion leads to a step-by-step explanation, frustration leads to a simpler supportive explanation, boredom leads to a concise challenge, and engagement leads to a deeper technical reading.

Limitations and future work
The affective signal is only a noisy proxy and should not be interpreted as a diagnosis. Future work should evaluate whether adaptive explanations improve comprehension, perceived support, and academic usefulness in controlled reading sessions.
"""


def create_sample_data(root: str | Path | None = None) -> Path:
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    sample_dir = project_root / "sample_data"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample_paper.txt"
    if not sample_path.exists() or not sample_path.read_text(encoding="utf-8").strip():
        sample_path.write_text(SAMPLE_TEXT, encoding="utf-8")
    readme = sample_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Sample Data\n\n`sample_paper.txt` is a generic demo paper used by smoke checks and presentations.\n",
            encoding="utf-8",
        )
    return sample_path


def main() -> int:
    path = create_sample_data()
    print(f"Sample data ready: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
