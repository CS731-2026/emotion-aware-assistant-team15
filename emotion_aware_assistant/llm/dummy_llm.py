from __future__ import annotations

import time

from emotion_aware_assistant.core.types import ChatRequest, ChatResponse


class DummyLLMClient:
    """Offline client with deliberately different answer styles per learning state."""

    @property
    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "dummy"

    def chat(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        state = request.learning_state.state
        passage = self._compact(request.paper_context.selected_text.strip() or request.paper_context.surrounding_text.strip(), 360)
        if state == "confusion":
            text = self._confusion_answer(request, passage)
        elif state == "frustration":
            text = self._frustration_answer(request, passage)
        elif state == "boredom":
            text = self._boredom_answer(request, passage)
        elif state == "engagement":
            text = self._engagement_answer(request, passage)
        else:
            text = self._uncertain_answer(request, passage)
        return ChatResponse(
            text=text,
            model_name=self.name,
            latency_sec=time.perf_counter() - start,
            input_tokens=None,
            output_tokens=len(text.split()),
            estimated_cost=0.0,
        )

    def _confusion_answer(self, request: ChatRequest, passage: str) -> str:
        return (
            "### Core idea\n"
            f"This {request.paper_context.passage_type} passage is best read as: {passage}\n\n"
            "### Step-by-step\n"
            f"1. Locate the input or starting point. {self._section_hint(request)}\n"
            "2. Identify the operation the authors apply.\n"
            "3. Connect that operation to the output or claim the passage supports.\n"
            f"4. Use the retrieved context to check the interpretation: {self._retrieved_hint(request)}\n\n"
            "### Example\n"
            "If the passage describes a method, imagine feeding one selected paragraph into the system, retrieving related chunks, then choosing a teaching style from the learning-state snapshot.\n\n"
            "### Check\n"
            f"Can you point to the phrase that names the main operation? {self._followup_line(request)}"
        )

    def _frustration_answer(self, request: ChatRequest, passage: str) -> str:
        return (
            "### Simplest version first\n"
            "Here is the simplest version before any extra detail.\n\n"
            f"The passage is saying: {passage}\n\n"
            "### Different way to think about it\n"
            f"Treat it like a small pipeline rather than a wall of text. The passage type is `{request.paper_context.passage_type}`, so the useful move is: {request.paper_context.passage_analysis.get('suggested_explanation_mode', 'separate claim, evidence, and purpose')}.\n\n"
            "### One small next step\n"
            f"Read only the verbs in the selected text and list what happens first, next, and last. {self._followup_line(request)}"
        )

    def _boredom_answer(self, request: ChatRequest, passage: str) -> str:
        return (
            "### One-sentence takeaway\n"
            f"{self._one_sentence_takeaway(request, passage)}\n\n"
            "### Why it matters\n"
            f"This passage matters because it anchors the paper's explanation in `{request.paper_context.passage_type}` evidence rather than a generic summary.\n\n"
            "### Quick check / quiz\n"
            f"What is the single most important input, output, or claim in the selected text? {self._followup_line(request)}"
        )

    def _engagement_answer(self, request: ChatRequest, passage: str) -> str:
        return (
            "### Technical read\n"
            f"Technical explanation: {passage}\n\n"
            f"For a `{request.paper_context.passage_type}` passage, a careful academic reading should separate mechanism, evidence, and assumption. Retrieved context adds: {self._retrieved_hint(request)}\n\n"
            "### Assumptions\n"
            "The explanation assumes the selected passage is representative of the surrounding section and that the retrieved chunks are relevant support.\n\n"
            "### Limitations\n"
            "If the paper's full method, dataset, or result table is outside the provided context, this answer should be treated as a grounded reading aid, not a full paper review.\n\n"
            "### Connection to broader methods\n"
            f"The design resembles a perception-cognition-action loop: observe context and state, infer a reading need, then adapt the explanation. {self._followup_line(request)}"
        )

    def _uncertain_answer(self, request: ChatRequest, passage: str) -> str:
        return (
            "### Academic explanation\n"
            "Based on the provided context, the passage appears to explain part of the paper's argument or method.\n\n"
            f"Relevant text: {passage}\n\n"
            "Follow-up options: summarize, define terms, or explain the method step by step."
        )

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        text = " ".join(text.split())
        return text[:limit] + ("..." if len(text) > limit else "")

    @staticmethod
    def _section_hint(request: ChatRequest) -> str:
        if request.paper_context.section_hint:
            return f"Current section hint: {request.paper_context.section_hint}."
        return "No section heading was detected."

    @staticmethod
    def _retrieved_hint(request: ChatRequest) -> str:
        if request.paper_context.retrieved_chunks:
            return " ".join(request.paper_context.retrieved_chunks[0].split())[:220]
        return "no extra retrieved chunk was available."

    @staticmethod
    def _followup_line(request: ChatRequest) -> str:
        if request.followup_action:
            return f"Requested follow-up mode: {request.followup_action}."
        return "Useful follow-ups: break into steps, give an example, or key takeaway."

    @staticmethod
    def _one_sentence_takeaway(request: ChatRequest, passage: str) -> str:
        if request.paper_context.passage_type == "dataset/evaluation":
            return "The passage is mainly about how the paper evaluates its approach and what evidence supports the result."
        if request.paper_context.passage_type == "comparison/related work":
            return "The passage is mainly contrasting this approach with another method or baseline."
        if request.paper_context.passage_type == "method/process/mechanism":
            return "The passage explains the mechanism that turns input context into an adaptive reading-support response."
        return passage
