from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_aware_assistant.core.config import load_config
from emotion_aware_assistant.core.types import ChatRequest, PaperContext
from emotion_aware_assistant.emotion.affective_trend_tracker import AffectiveTrendTracker
from emotion_aware_assistant.emotion.dummy_emotion import DummyEmotionRecognizer
from emotion_aware_assistant.emotion.emotion_buffer import EmotionBuffer
from emotion_aware_assistant.emotion.state_mapper import map_prediction_to_learning_state
from emotion_aware_assistant.evaluation.interaction_logger import InteractionLogger
from emotion_aware_assistant.llm.dummy_llm import DummyLLMClient
from emotion_aware_assistant.llm.prompt_builder import PromptBuilder
from emotion_aware_assistant.paper.passage_analyzer import classify_passage
from emotion_aware_assistant.paper.pdf_loader import load_document
from emotion_aware_assistant.paper.retriever import ContextRetriever
from scripts.create_sample_data import create_sample_data


def main() -> int:
    config = load_config(ROOT / "config.yaml")
    sample_path = create_sample_data(ROOT)

    recognizer = DummyEmotionRecognizer("fear", confidence=0.86)
    recognizer.load()
    raw_prediction = recognizer.predict(None)
    buffer = EmotionBuffer(
        maxlen=config["emotion"]["buffer_size"],
        confidence_threshold=config["emotion"]["confidence_threshold"],
    )
    smoothed = buffer.add(raw_prediction)
    candidate_state = map_prediction_to_learning_state(raw_prediction, smoothed)
    tracker = AffectiveTrendTracker(
        window_sec=config["emotion"]["trend_window_sec"],
        hysteresis_updates=config["emotion"]["hysteresis_updates"],
        high_confidence_switch_threshold=config["emotion"]["high_confidence_switch_threshold"],
    )
    learning_state = tracker.update(candidate_state)

    document = load_document(sample_path)
    retriever = ContextRetriever(document, top_k=config["paper"]["top_k_chunks"])
    page_text = document.page(1).text
    start = page_text.lower().find("the proposed method")
    end = page_text.lower().find("formula-like objective")
    if start < 0 or end <= start:
        start, end = 0, min(600, len(page_text))
    selected = page_text[start:end].strip()
    passage_type = classify_passage(selected)
    chunks = [chunk.text for chunk in retriever.retrieve("Can you explain this method?", selected, 1)]
    paper_context = PaperContext(
        document_title=document.title,
        page_number=1,
        selected_text=selected,
        surrounding_text=page_text[max(0, start - 300) : min(len(page_text), end + 300)].strip(),
        retrieved_chunks=chunks,
        passage_type=passage_type,
    )
    request = ChatRequest(
        user_question="Can you explain this method?",
        paper_context=paper_context,
        learning_state=learning_state,
        conversation_history=[],
        followup_action=None,
        model_name="dummy",
    )
    messages = PromptBuilder().build_messages(request)
    if not messages or "academic paper reading tutor" not in messages[0]["content"]:
        raise RuntimeError("Prompt builder did not create the expected system prompt.")
    response = DummyLLMClient().chat(request)
    if not response.text.strip():
        raise RuntimeError("Dummy LLM returned an empty response.")
    logger = InteractionLogger(config["app"]["log_dir"])
    logger.log_chat(request, response)
    if not logger.path.exists() or logger.path.stat().st_size == 0:
        raise RuntimeError("JSONL logging did not write an event.")
    print("SMOKE CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
