from __future__ import annotations

from emotion_aware_assistant.core.types import ChatRequest

from .response_policy import get_response_policy


class PromptBuilder:
    """Builds prompts where paper context controls facts and affect controls style."""

    def build_messages(self, request: ChatRequest) -> list[dict[str, str]]:
        policy = get_response_policy(
            request.learning_state.state,
            request.learning_state.trend,
            request.paper_context.passage_type,
        )
        system = """
You are an academic paper reading tutor for graduate students and researchers.

Safety and ethics:
- Treat the affective signal as a noisy style cue, not a diagnosis.
- Do not say the user is frustrated, confused, bored, angry, sad, or any other emotion.
- Do not mention webcam, face, facial expression, or camera unless the user explicitly asks.

Grounding and anti-hallucination:
- Use only the paper context provided in the user message for paper-specific facts.
- If the context is insufficient, say exactly what information is missing.
- Distinguish paper claims from your explanatory interpretation.
- Do not fabricate citations, equations, metrics, datasets, page numbers, or results.
- Use markdown with compact section headings.
""".strip()
        chunks = "\n\n".join(
            f"[Retrieved chunk {index + 1}]\n{chunk}"
            for index, chunk in enumerate(request.paper_context.retrieved_chunks)
        )
        history = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in request.conversation_history[-6:]
        )
        analysis = request.paper_context.passage_analysis or {"passage_type": request.paper_context.passage_type}
        retrieval_debug = request.paper_context.retrieval_debug or {}
        user = f"""
## Assistant Role
Act as an academic reading assistant. Paper context controls factual content; learning state controls teaching style only.

## Document Metadata
- title: {request.paper_context.document_title}
- page number: {request.paper_context.page_number}
- page title / section: {request.paper_context.page_title or request.paper_context.section_hint or '[unknown]'}
- difficulty hint: {request.paper_context.difficulty_hint}

## Selected Passage
{request.paper_context.selected_text or '[No explicit selection provided]'}

## Surrounding Context
{request.paper_context.surrounding_text or '[No surrounding context available]'}

## Retrieved Relevant Chunks
{chunks or '[No retrieved chunks available]'}

## Passage Analysis
- passage type: {request.paper_context.passage_type}
- detected keywords: {', '.join(analysis.get('detected_keywords', [])) or '[none]'}
- suggested explanation mode: {analysis.get('suggested_explanation_mode', '[none]')}
- retrieval method: {retrieval_debug.get('method', '[not available]')}

## User Question
{request.user_question}

## Follow-Up Action
{request.followup_action or '[none]'}

## Affective State Snapshot
- state: {request.learning_state.state}
- trend: {request.learning_state.trend}
- confidence: {request.learning_state.confidence:.2f}
- duration_sec: {request.learning_state.duration_sec:.1f}
- manual_override: {request.learning_state.manual_override}
- strategy: {request.learning_state.strategy}

## Response Policy
- tone: {policy.tone}
- depth: {policy.depth}
- paragraph length: {policy.paragraph_length}
- structure: {', '.join(policy.structure)}
- avoid: {policy.avoid}
- avoid list: {', '.join(policy.avoid_list)}
- ideal response shape: {'; '.join(policy.ideal_response_shape)}
- trend adjustment: {policy.trend_adjustment}
- passage adjustment: {policy.passage_adjustment}
- allowed follow-ups: {', '.join(policy.followup_buttons)}

## Passage-Type Instruction
{policy.passage_adjustment}

## Recent Conversation
{history or '[No previous turns]'}

## Output Requirements
- Start with the most useful answer for the current learning state.
- Keep factual claims grounded in the selected passage, surrounding context, or retrieved chunks.
- If you infer or explain beyond the paper text, label it as explanation or interpretation.
- End with two or three useful follow-up options from the allowed follow-up list.
""".strip()
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def build_text_prompt(self, request: ChatRequest) -> str:
        return "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in self.build_messages(request))
