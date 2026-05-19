from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from emotion_aware_assistant.core.llm_config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    provider_api_key_from_env,
    provider_base_url_from_env,
    role_config_from_env,
)
from emotion_aware_assistant.paper.paper_rag import is_low_value_context_block, normalize_pdf_text


ACADEMIC_READING_INSTRUCTION = (
    "You are an academic paper reading assistant. Explain only the selected PDF passage "
    "or selected visual area. Use the provided selected text, parsed Markdown, caption, "
    "nearby context, and image crop if available. Be accurate, concise, and helpful. "
    "If the selected area is a figure, table, or formula, explain what it shows and why "
    "it matters in the paper. If context is insufficient, clearly say what is uncertain."
)
GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def explain_selection(payload: dict[str, Any]) -> dict[str, Any]:
    if not os.environ.get("LLM_PROVIDER", "").strip():
        return _mock_response(payload)
    role = role_config_from_env("answer_model")
    provider = str(role.get("provider") or os.environ.get("LLM_PROVIDER", "mock")).strip().lower() or "mock"
    model = str(role.get("model") or "").strip()
    if provider == "gemini":
        api_key = provider_api_key_from_env("gemini")
        if not api_key:
            return _mock_response(payload, warning="GEMINI_API_KEY is missing; fell back to mock provider.")
        return _gemini_response(payload, api_key, model=model)
    if provider == "openrouter":
        api_key = provider_api_key_from_env("openrouter")
        if not api_key or not model:
            return _provider_config_error(payload, provider, model, "OpenRouter API key or model is not configured.")
        return _chat_completions_response(
            payload,
            provider=provider,
            api_key=api_key,
            base_url=DEFAULT_OPENROUTER_BASE_URL,
            model=model,
        )
    if provider == "openai_compatible":
        api_key = provider_api_key_from_env("openai_compatible")
        base_url = provider_base_url_from_env("openai_compatible")
        if not api_key or not base_url or not model:
            return _provider_config_error(payload, provider, model, "OpenAI-compatible API key, base URL, or model is not configured.")
        return _chat_completions_response(
            payload,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    return _mock_response(payload)


def build_gemini_request(payload: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    prompt = build_prompt(payload)
    parts: list[dict[str, Any]] = [{"text": prompt}]
    image = _image_part(payload)
    used_image = image is not None
    if image:
        parts.append(image)
    return prompt, {"contents": [{"parts": parts}]}, used_image


def build_prompt_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_prompt(payload)}]


def build_prompt(payload: dict[str, Any]) -> str:
    retrieval_context = _retrieval_context(payload)
    highlight_type = _text(payload.get("highlight_type")).lower()
    response_style = _response_style(payload)
    chat_style = response_style == "chat_conversational"
    selected_text = normalize_pdf_text(payload.get("selected_text"))
    matched_block = _matched_block(payload)
    matched_markdown = normalize_pdf_text(matched_block.get("markdown_content") or _matched_markdown(payload))
    selected_caption = _selected_caption(payload)
    caption_confidence = _text(payload.get("caption_confidence") or payload.get("selected_caption_confidence"))
    caption = normalize_pdf_text(selected_caption.get("markdown_content")) or (
        "" if highlight_type == "area" and caption_confidence in {"low", "none"} else _text(payload.get("caption"))
    )
    candidate_captions = _candidate_captions_text(payload)
    nearby_context = _block_list_text(retrieval_context.get("nearby_context") or payload.get("nearby_useful_context") or [])
    same_section_context = _block_list_text(retrieval_context.get("same_section_context") or [])
    related_blocks = _block_list_text(retrieval_context.get("related_blocks") or [])
    global_rag_context = _block_list_text(retrieval_context.get("global_rag_context") or [])
    paper_profile = _paper_profile_text(retrieval_context.get("paper_profile") or {})
    crop_attached = _image_part(payload) is not None
    selected_strategy_text = _selected_strategy_text(payload)
    explicit_user_question = normalize_pdf_text(payload.get("user_question") or payload.get("question"))
    default_task = _text(payload.get("default_task"))
    strategy_default_task = bool(
        selected_strategy_text
        and not explicit_user_question
        and default_task == "explain_current_selection_with_selected_strategy"
    )
    user_question_text = explicit_user_question or (
        "Explain the current selection using the selected pedagogical strategy."
        if strategy_default_task
        else "Can you explain this selected part of the paper?"
    )

    style_instruction = (
        "Answer as a helpful academic reading assistant in a natural conversational style. "
        "Do not repeat or quote the selected passage unless necessary. Start directly with the explanation. "
        "Avoid headings like 'Selected Part', 'Paper Context', 'Connection to Method and Argument', "
        "or 'Useful Follow-up Question' unless the user explicitly asks for a structured breakdown. "
        "Keep the answer concise but useful, usually 2-4 short paragraphs. Explain what the selected part "
        "means, why it matters in the paper, and how it connects to the paper's argument. "
        "If evidence is insufficient, say so clearly. Do not include a forced follow-up question by default."
    ) if chat_style else (
        "Write a first explanation that is paper-grounded and moderately informative. "
        "Address: what the selected part is, what it means in this paper, how it connects "
        "to the paper's method/result/argument, why it matters, and one useful follow-up question. "
        "Do not turn it into a full lecture."
    )
    sections = [
        ACADEMIC_READING_INSTRUCTION,
        style_instruction,
        "",
        f'user_question: "{user_question_text}"',
        f"response_style: {response_style}",
        f"highlight_type: {_text(payload.get('highlight_type')) or 'unknown'}",
        f"page_number: {payload.get('page_number') or ''}",
        f"recommended_llm_mode: {_mode(payload) or 'unknown'}",
        f"crop_image_attached: {'true' if crop_attached else 'false'}",
    ]
    follow_up_question = normalize_pdf_text(payload.get("follow_up_question"))
    thread_history = _thread_history_text(payload.get("thread_history"))
    if follow_up_question:
        sections.extend([
            "",
            f'follow_up_question: "{follow_up_question}"',
            "follow_up_guidance: Answer the follow-up using the same selected evidence, paper profile, and retrieved paper context. Stay grounded in the active highlight.",
        ])
    if thread_history:
        sections.extend(["", "thread_history:", thread_history])
    if strategy_default_task:
        sections.extend([
            "",
            "Task:",
            "Explain the selected paper passage using the selected pedagogical support strategy.",
            "",
            "Grounding:",
            "Use the selected text/crop, caption, matched block, nearby context, paper profile, and retrieved RAG chunks as factual grounding.",
            "",
            "Rules:",
            "- Do not invent paper facts.",
            "- Do not diagnose the user's emotion.",
            "- Do not say the user is confused or frustrated.",
            "- The learning-state signal only guides support style.",
            "- If evidence is insufficient, say what is missing.",
        ])
    if selected_strategy_text:
        sections.extend(["", "Selected pedagogical support strategy:", selected_strategy_text])
    if highlight_type == "area":
        if caption_confidence == "low":
            sections.extend([
                "",
                "area_caption_guidance: The crop image is the primary source. Candidate captions may be imperfect. If the image and captions conflict, mention the uncertainty instead of merging them silently.",
            ])
        else:
            sections.extend([
                "",
                "area_caption_guidance: Treat the crop image as primary evidence. Use the selected caption only as supporting context, and mention uncertainty if it conflicts with the image.",
            ])
    if paper_profile:
        sections.extend(["", "paper_profile:", paper_profile])
    if selected_text and bool(payload.get("text_available", True)):
        sections.extend(["", "selected_text:", selected_text])
    if matched_markdown:
        sections.extend(["", "matched_block:", matched_markdown])
    if caption:
        sections.extend(["", f"caption: {caption}", f"caption_confidence: {caption_confidence or 'unknown'}"])
    if candidate_captions:
        sections.extend(["", "candidate_captions:", candidate_captions])
    if nearby_context:
        sections.extend(["", "useful_nearby_context:", nearby_context])
    if same_section_context:
        sections.extend(["", "same_section_context:", same_section_context])
    if related_blocks:
        sections.extend(["", "related_blocks:", related_blocks])
    if global_rag_context:
        sections.extend([
            "",
            "global_rag_context:",
            f"retrieval_method: {_text(retrieval_context.get('retrieval_method')) or 'keyword'}",
            global_rag_context,
        ])
    if not chat_style:
        sections.extend(["", "answer_format:", "- selected_part\n- paper_context\n- connection_to_method_result_or_argument\n- why_it_matters\n- useful_follow_up_question"])
    return "\n".join(sections).strip()


def _gemini_response(payload: dict[str, Any], api_key: str, model: str | None = None) -> dict[str, Any]:
    model = (model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).strip() or DEFAULT_GEMINI_MODEL
    prompt, body, used_image = build_gemini_request(payload)
    request = urllib.request.Request(
        GEMINI_ENDPOINT_TEMPLATE.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        answer = _gemini_text(response_payload)
        return {
            "provider": "gemini",
            "model": model,
            "mode": _mode(payload),
            "recommended_llm_mode": _mode(payload),
            "response_style": _response_style(payload),
            "used_image": used_image,
            **_retrieval_metadata(payload),
            "prompt_preview": _prompt_preview(prompt),
            "answer": answer,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        return _gemini_error_response(payload, model, used_image, prompt, f"Gemini HTTP {exc.code}")
    except Exception as exc:
        return _gemini_error_response(payload, model, used_image, prompt, f"Gemini request failed: {type(exc).__name__}")


def _chat_completions_response(
    payload: dict[str, Any],
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    prompt = build_prompt(payload)
    used_image = _image_part(payload) is not None
    message_content: str | list[dict[str, Any]] = prompt
    if used_image:
        image_url = str(payload.get("crop_image_data_url") or payload.get("image_data_url") or "").strip()
        message_content = [{"type": "text", "text": prompt}]
        if image_url:
            message_content.append({"type": "image_url", "image_url": {"url": image_url}})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        "temperature": 0.35,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        answer = _chat_completion_text(response_payload)
        return {
            "provider": provider,
            "model": model,
            "mode": _mode(payload),
            "recommended_llm_mode": _mode(payload),
            "response_style": _response_style(payload),
            "used_image": used_image,
            **_retrieval_metadata(payload),
            "prompt_preview": _prompt_preview(prompt),
            "answer": answer,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        return _provider_config_error(payload, provider, model, f"{provider} HTTP {exc.code}", used_image=used_image, prompt=prompt)
    except Exception as exc:
        return _provider_config_error(payload, provider, model, f"{provider} request failed: {type(exc).__name__}", used_image=used_image, prompt=prompt)


def _chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict)).strip()
    return ""


def _provider_config_error(
    payload: dict[str, Any],
    provider: str,
    model: str,
    error: str,
    *,
    used_image: bool | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    prompt = prompt if prompt is not None else build_prompt(payload)
    return {
        "provider": provider,
        "model": model,
        "mode": _mode(payload),
        "recommended_llm_mode": _mode(payload),
        "response_style": _response_style(payload),
        "used_image": _image_part(payload) is not None if used_image is None else used_image,
        **_retrieval_metadata(payload),
        "prompt_preview": _prompt_preview(prompt),
        "answer": "",
        "error": error,
    }


def _gemini_error_response(
    payload: dict[str, Any],
    model: str,
    used_image: bool,
    prompt: str,
    error: str,
) -> dict[str, Any]:
    return {
        "provider": "gemini",
        "model": model,
        "mode": _mode(payload),
        "recommended_llm_mode": _mode(payload),
        "response_style": _response_style(payload),
        "used_image": used_image,
        **_retrieval_metadata(payload),
        "prompt_preview": _prompt_preview(prompt),
        "answer": "",
        "error": error,
    }


def _mock_response(payload: dict[str, Any], warning: str | None = None) -> dict[str, Any]:
    prompt = build_prompt(payload)
    highlight_type = _text(payload.get("highlight_type")) or "selection"
    selected_text = _text(payload.get("selected_text"))
    crop_available = bool(_text(payload.get("crop_image_data_url")))
    used_image = _image_part(payload) is not None
    answer = (
        f"This is a mock explanation for the selected {highlight_type}. "
        f"The system received page {payload.get('page_number') or ''}, "
        f"mode {_mode(payload)}, "
        f"selected text length {len(selected_text)}, "
        f"crop image available {'true' if crop_available else 'false'}."
    )
    if warning:
        answer = f"Warning: {warning} {answer}"
    return {
        "provider": "mock",
        "model": "mock",
        "mode": _mode(payload),
        "recommended_llm_mode": _mode(payload),
        "response_style": _response_style(payload),
        "used_image": used_image,
        **_retrieval_metadata(payload),
        "prompt_preview": _prompt_preview(prompt),
        "answer": answer,
        "error": None,
    }


def _selected_strategy_text(payload: dict[str, Any]) -> str:
    strategy = payload.get("selected_strategy")
    if not isinstance(strategy, dict):
        return ""
    family = _text(strategy.get("strategy_family") or strategy.get("strategy_id"))
    move = _text(strategy.get("pedagogical_move") or strategy.get("title"))
    focus = _text(strategy.get("context_focus"))
    title = _text(strategy.get("title"))
    why = _text(strategy.get("why_recommended") or strategy.get("short_description"))
    instruction = _text(strategy.get("prompt_instruction"))
    shape = strategy.get("expected_answer_shape")
    if isinstance(shape, list):
        shape_text = ", ".join(_text(item) for item in shape if _text(item))
    else:
        shape_text = _text(shape)
    lines = [
        f"- Strategy family: {family or '[not specified]'}",
        f"- Pedagogical move: {move or '[not specified]'}",
        f"- Context focus: {focus or '[not specified]'}",
        f"- Strategy title: {title or '[not specified]'}",
        f"- Why selected: {why or '[not specified]'}",
        f"- Instruction for answer: {instruction or '[not specified]'}",
        f"- Expected answer shape: {shape_text or '[not specified]'}",
        "",
        "Follow this selected strategy when explaining the paper passage.",
        "However, do not invent paper facts. Use selected text/crop, matched block, caption, nearby context, paper profile, and retrieved chunks as factual grounding.",
        "If evidence is insufficient, say what is missing.",
        "The selected strategy controls explanation style and structure. It must not override factual grounding.",
    ]
    return "\n".join(lines)


def _image_part(payload: dict[str, Any]) -> dict[str, Any] | None:
    if _text(payload.get("highlight_type")).lower() != "area":
        return None
    data_url = _text(payload.get("crop_image_data_url"))
    if not data_url:
        return None
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        return None
    return {
        "inline_data": {
            "mime_type": "image/png",
            "data": data_url.removeprefix(prefix),
        }
    }


def _gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        return ""
    return "\n".join(_text(part.get("text")) for part in parts if _text(part.get("text"))).strip()


def _matched_markdown(payload: dict[str, Any]) -> str:
    matched_block = payload.get("matched_block")
    if isinstance(matched_block, dict):
        return _text(matched_block.get("markdown_content"))
    return _text(payload.get("markdown_content") or payload.get("matched_markdown"))


def _retrieval_context(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("retrieval_context")
    return value if isinstance(value, dict) else {}


def _matched_block(payload: dict[str, Any]) -> dict[str, Any]:
    retrieval = _retrieval_context(payload)
    block = retrieval.get("matched_block")
    if isinstance(block, dict) and block:
        return block
    block = payload.get("matched_block")
    return block if isinstance(block, dict) else {}


def _selected_caption(payload: dict[str, Any]) -> dict[str, Any]:
    caption = payload.get("selected_caption")
    return caption if isinstance(caption, dict) else {}


def _candidate_captions_text(payload: dict[str, Any]) -> str:
    captions = payload.get("candidate_captions")
    if not isinstance(captions, list):
        return ""
    values = []
    for caption in captions[:5]:
        if not isinstance(caption, dict):
            continue
        text = normalize_pdf_text(caption.get("markdown_content"))
        if not text:
            continue
        values.append(
            "- "
            f"{caption.get('block_id') or 'caption'}: {text} "
            f"(relation={caption.get('relation') or 'unknown'}, "
            f"horizontal_overlap={caption.get('horizontal_overlap', '-')}, "
            f"vertical_distance={caption.get('vertical_distance', '-')}, "
            f"score={caption.get('score', '-')})"
        )
    return "\n".join(values)


def _block_list_text(items: Any, limit: int = 5) -> str:
    if not isinstance(items, list):
        return ""
    values = []
    for item in items[:limit]:
        if isinstance(item, dict):
            if is_low_value_context_block(item):
                continue
            text = normalize_pdf_text(item.get("markdown_content") or item.get("text"))
            if item.get("block_id"):
                text = f"{item.get('block_id')}: {text}"
        else:
            text = normalize_pdf_text(item)
        if text:
            values.append(f"- {_truncate(text, 520)}")
    return "\n".join(values)


def _paper_profile_text(profile: Any) -> str:
    if not isinstance(profile, dict) or not profile:
        return ""
    lines = []
    for key in (
        "title",
        "one_sentence_summary",
        "research_problem",
        "method_summary",
        "dataset_or_materials",
        "main_findings",
    ):
        value = normalize_pdf_text(profile.get(key))
        if value:
            lines.append(f"{key}: {_truncate(value, 420)}")
    key_terms = profile.get("key_terms")
    if isinstance(key_terms, list) and key_terms:
        lines.append("key_terms: " + ", ".join(_text(term) for term in key_terms[:12] if _text(term)))
    return "\n".join(lines)


def _thread_history_text(items: Any, limit: int = 6) -> str:
    if not isinstance(items, list):
        return ""
    lines = []
    for item in items[-limit:]:
        if not isinstance(item, dict):
            continue
        role = _text(item.get("role")) or "message"
        content = normalize_pdf_text(item.get("content"))
        if content:
            lines.append(f"- {role}: {_truncate(content, 360)}")
    return "\n".join(lines)


def _retrieval_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    retrieval = _retrieval_context(payload)
    paper_profile = retrieval.get("paper_profile") if isinstance(retrieval.get("paper_profile"), dict) else {}
    related_blocks = retrieval.get("related_blocks") if isinstance(retrieval.get("related_blocks"), list) else []
    global_blocks = retrieval.get("global_rag_context") if isinstance(retrieval.get("global_rag_context"), list) else []
    raw_nearby = retrieval.get("nearby_context") if isinstance(retrieval.get("nearby_context"), list) else []
    raw_same_section = retrieval.get("same_section_context") if isinstance(retrieval.get("same_section_context"), list) else []
    nearby = [block for block in raw_nearby if not is_low_value_context_block(block)]
    same_section = [block for block in raw_same_section if not is_low_value_context_block(block)]
    matched = retrieval.get("matched_block") if isinstance(retrieval.get("matched_block"), dict) and retrieval.get("matched_block") else None
    count = (1 if matched else 0) + len(nearby) + len(same_section) + len(related_blocks) + len(global_blocks)
    summary = normalize_pdf_text(paper_profile.get("one_sentence_summary") or paper_profile.get("title") or "")
    return {
        "paper_profile_used": bool(paper_profile and any(paper_profile.values())),
        "paper_profile": paper_profile,
        "paper_profile_summary": summary,
        "retrieved_block_count": count,
        "retrieved_blocks": related_blocks,
        "global_rag_context": global_blocks,
        "retrieval_method": _text(retrieval.get("retrieval_method")) or "keyword",
        "nearby_context": nearby,
        "same_section_context": same_section,
        "retrieval_strategy": _text(retrieval.get("retrieval_strategy")),
    }


def _mode(payload: dict[str, Any]) -> str:
    return _text(payload.get("mode") or payload.get("recommended_llm_mode"))


def _response_style(payload: dict[str, Any]) -> str:
    value = _text(payload.get("response_style")).lower()
    return value if value in {"chat_conversational", "debug_structured"} else "debug_structured"


def _prompt_preview(prompt: str) -> str:
    return prompt[:2400]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _text(value: Any) -> str:
    return str(value or "").strip()
