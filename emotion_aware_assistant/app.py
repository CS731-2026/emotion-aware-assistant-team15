from __future__ import annotations

from dataclasses import replace
from typing import Any

from emotion_aware_assistant.core.types import ChatRequest, ChatResponse, LearningState, PaperContext, SystemStatus
from emotion_aware_assistant.emotion.affective_trend_tracker import AffectiveTrendTracker
from emotion_aware_assistant.emotion.dummy_emotion import DummyEmotionRecognizer
from emotion_aware_assistant.emotion.emotion_buffer import EmotionBuffer
from emotion_aware_assistant.emotion.labels import ALLOWED_EMOTIONS, LEARNING_STATES, normalize_emotion
from emotion_aware_assistant.emotion.manual_emotion import ManualEmotionRecognizer
from emotion_aware_assistant.emotion.state_mapper import (
    map_prediction_to_learning_state,
    state_to_learning_state,
    uncertain_learning_state,
)
from emotion_aware_assistant.evaluation.interaction_logger import InteractionLogger
from emotion_aware_assistant.llm.dummy_llm import DummyLLMClient
from emotion_aware_assistant.llm.model_registry import resolve_model
from emotion_aware_assistant.llm.openrouter_client import OpenRouterClient
from emotion_aware_assistant.paper.document import Document
from emotion_aware_assistant.paper.passage_analyzer import analyze_passage, classify_passage, surrounding_text
from emotion_aware_assistant.paper.pdf_loader import load_document
from emotion_aware_assistant.paper.retriever import ContextRetriever


class AssistantSession:
    """Shared coordinator used by CLI, GUI, scripts, and tests."""

    def __init__(
        self,
        config: dict[str, Any],
        force_dummy_llm: bool = False,
        logger: InteractionLogger | None = None,
    ):
        self.config = config
        emotion_cfg = config.get("emotion", {})
        self.buffer = EmotionBuffer(
            maxlen=int(emotion_cfg.get("buffer_size", 10)),
            confidence_threshold=float(emotion_cfg.get("confidence_threshold", 0.35)),
        )
        self.tracker = AffectiveTrendTracker(
            window_sec=float(emotion_cfg.get("trend_window_sec", 6)),
            hysteresis_updates=int(emotion_cfg.get("hysteresis_updates", 3)),
            high_confidence_switch_threshold=float(emotion_cfg.get("high_confidence_switch_threshold", 0.80)),
        )
        self.manual_recognizer = ManualEmotionRecognizer("neutral")
        self.dummy_recognizer = DummyEmotionRecognizer("neutral")
        self.learning_state: LearningState = uncertain_learning_state()
        self.document: Document | None = None
        self.retriever: ContextRetriever | None = None
        self.current_page_number = 1
        self.selected_range: tuple[int, int] | None = None
        self.manual_context: PaperContext | None = None
        self.conversation_history: list[dict[str, str]] = []
        self.logger = logger or InteractionLogger(config.get("app", {}).get("log_dir", "logs"))
        self.llm = self._create_llm(force_dummy_llm)

    def _create_llm(self, force_dummy_llm: bool):
        if force_dummy_llm:
            return DummyLLMClient()
        llm_cfg = self.config.get("llm", {})
        if llm_cfg.get("default_client") == "openrouter":
            client = OpenRouterClient(timeout_sec=int(llm_cfg.get("timeout_sec", 60)))
            if client.is_available:
                return client
        return DummyLLMClient()

    def set_manual_emotion(self, emotion: str) -> LearningState:
        normalized = normalize_emotion(emotion)
        if normalized not in ALLOWED_EMOTIONS:
            raise ValueError(f"Unsupported emotion: {emotion}")
        self.manual_recognizer.set_emotion(normalized)
        raw = self.manual_recognizer.predict(None)
        self.buffer.clear()
        smoothed = self.buffer.add(raw)
        candidate = map_prediction_to_learning_state(raw, smoothed, manual_override=True)
        self.learning_state = self.tracker.update(candidate)
        return self.learning_state

    def set_manual_state(self, state: str) -> LearningState:
        state = state.strip().lower()
        if state not in LEARNING_STATES:
            raise ValueError(f"Unsupported state: {state}")
        candidate = state_to_learning_state(state, manual_override=True)
        self.learning_state = self.tracker.update(candidate)
        return self.learning_state

    def set_override(self, value: str) -> LearningState:
        value = value.strip().lower()
        if value == "auto":
            self.learning_state = uncertain_learning_state()
            return self.learning_state
        if value in ALLOWED_EMOTIONS:
            return self.set_manual_emotion(value)
        if value in LEARNING_STATES:
            return self.set_manual_state(value)
        raise ValueError(f"Unsupported override: {value}")

    def load_document(self, path: str) -> Document:
        self.document = load_document(path)
        paper_cfg = self.config.get("paper", {})
        self.retriever = ContextRetriever(self.document, top_k=int(paper_cfg.get("top_k_chunks", 3)))
        self.current_page_number = 1
        self.selected_range = None
        self.manual_context = None
        return self.document

    def set_page(self, page_number: int) -> str:
        if self.document is None:
            raise RuntimeError("No document is loaded.")
        page = self.document.page(page_number)
        self.current_page_number = page_number
        self.selected_range = None
        return page.text

    def select_range(self, start: int, end: int) -> PaperContext:
        if self.document is None:
            raise RuntimeError("No document is loaded.")
        page = self.document.page(self.current_page_number)
        start = max(0, min(start, len(page.text)))
        end = max(start, min(end, len(page.text)))
        self.selected_range = (start, end)
        return self.current_paper_context()

    def set_manual_context(self, text: str, title: str = "Pasted context") -> PaperContext:
        self.manual_context = PaperContext(
            document_title=title,
            page_number=None,
            selected_text=text.strip(),
            surrounding_text=text.strip(),
            retrieved_chunks=[],
            passage_type=classify_passage(text),
            difficulty_hint=analyze_passage(text).difficulty_hint,
            passage_analysis=analyze_passage(text).as_dict(),
        )
        return self.manual_context

    def current_paper_context(self, question: str = "") -> PaperContext:
        if self.manual_context is not None:
            return self.manual_context
        if self.document is None:
            return PaperContext(
                document_title="No document loaded",
                page_number=None,
                selected_text="",
                surrounding_text="",
                retrieved_chunks=[],
                passage_type="general",
            )
        page = self.document.page(self.current_page_number)
        if self.selected_range is None:
            start, end = 0, min(len(page.text), 700)
        else:
            start, end = self.selected_range
        selected = page.text[start:end].strip()
        around = surrounding_text(page.text, start, end)
        analysis = analyze_passage(selected or around)
        chunks = []
        retrieval_debug = {}
        if self.retriever is not None:
            debug = self.retriever.retrieve_with_debug(
                query=question,
                selected_text=selected,
                page_number=self.current_page_number,
                top_k=int(self.config.get("paper", {}).get("top_k_chunks", 3)),
            )
            chunks = [chunk.text for chunk in debug["chunks"]]
            retrieval_debug = {
                key: value
                for key, value in debug.items()
                if key != "chunks"
            }
        section_hint = self.document.section_for_page(self.current_page_number)
        page_title = section_hint or self.document.title
        return PaperContext(
            document_title=self.document.title,
            page_number=self.current_page_number,
            selected_text=selected,
            surrounding_text=around,
            retrieved_chunks=chunks,
            passage_type=analysis.passage_type,
            page_title=page_title,
            section_hint=section_hint,
            difficulty_hint=analysis.difficulty_hint,
            passage_analysis=analysis.as_dict(),
            retrieval_debug=retrieval_debug,
        )

    def ask(
        self,
        question: str,
        paper_context: PaperContext | None = None,
        followup_action: str | None = None,
        model_alias: str | None = None,
    ) -> ChatResponse:
        context = paper_context or self.current_paper_context(question)
        state = self.learning_state
        if state.state == "uncertain":
            state = self.set_manual_emotion("neutral")
        model_name = resolve_model(self.config, model_alias)
        request = ChatRequest(
            user_question=question,
            paper_context=context,
            learning_state=state,
            conversation_history=list(self.conversation_history),
            followup_action=followup_action,
            model_name=model_name,
        )
        response = self.llm.chat(request)
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": response.text})
        if self.logger:
            self.logger.log_chat(request, response)
        return response

    def status(self, mode: str = "terminal") -> SystemStatus:
        webcam_available = False
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(0)
            webcam_available = bool(cap.isOpened())
            cap.release()
        except Exception:
            webcam_available = False
        return SystemStatus(
            webcam_available=webcam_available,
            emotion_model_loaded=False,
            llm_available=self.llm.is_available,
            speech_available=False,
            current_mode=mode,
        )

    def learning_snapshot_text(self) -> str:
        state = self.learning_state
        return (
            f"raw={state.raw_emotion}, smoothed={state.smoothed_emotion}, "
            f"state={state.state}, trend={state.trend}, confidence={state.confidence:.2f}, "
            f"strategy={state.strategy}"
        )
