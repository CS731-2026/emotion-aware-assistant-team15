from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from emotion_aware_assistant.core.llm_config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    has_explicit_llm_config,
    provider_base_url,
    read_llm_values,
    resolve_provider_api_key,
    resolve_gemini_api_key,
    role_config_with_source,
    settings_revision_info,
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


def explain_selection(
    payload: dict[str, Any],
    *,
    project_root: str | os.PathLike[str] | None = None,
    profiles_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    values = read_llm_values(project_root, profiles_dir, include_env_file=False)
    if not has_explicit_llm_config(values):
        if payload.get("allow_mock_llm", True):
            return _mock_response(payload)
        return _provider_config_error(
            payload,
            "not_configured",
            "",
            "LLM settings are not configured. Open /settings to configure a provider and model.",
        )
    response_mode = _interaction_response_mode(payload)
    role_name = "strategy_planner_model" if response_mode in {"strategy_response", "proactive_support"} else "answer_model"
    role = role_config_with_source(role_name, project_root, profiles_dir, include_env_file=False)
    provider = str(role.get("provider") or os.environ.get("LLM_PROVIDER", "mock")).strip().lower() or "mock"
    model = str(role.get("model") or "").strip()
    diagnostics = settings_revision_info(project_root, profiles_dir)
    if provider == "gemini":
        key_info = resolve_gemini_api_key(project_root, profiles_dir, include_env_file=False)
        api_key = str(key_info.get("key") or "")
        if not api_key:
            return _provider_config_error(payload, provider, model, "LLM settings are missing a Gemini API key. Open /settings to configure LLM access.")
        return _gemini_response(
            payload,
            api_key,
            model=model,
            key_source=str(key_info.get("key_source") or ""),
            masked_key_suffix=str(key_info.get("masked_suffix") or ""),
            model_source=str(role.get("model_source") or ""),
            diagnostics=diagnostics,
        )
    if provider == "openrouter":
        key_info = resolve_provider_api_key("openrouter", project_root, profiles_dir, include_env_file=False, values=values)
        api_key = str(key_info.get("key") or "")
        base_url = provider_base_url("openrouter", values) or DEFAULT_OPENROUTER_BASE_URL
        if not api_key or not model:
            return _provider_config_error(payload, provider, model, "OpenRouter API key or model is not configured.")
        return _chat_completions_response(
            payload,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            key_source=str(key_info.get("key_source") or ""),
            masked_key_suffix=str(key_info.get("masked_suffix") or ""),
            model_source=str(role.get("model_source") or ""),
            diagnostics=diagnostics,
            provider_values=values,
        )
    if provider == "openai_compatible":
        key_info = resolve_provider_api_key("openai_compatible", project_root, profiles_dir, include_env_file=False, values=values)
        api_key = str(key_info.get("key") or "")
        base_url = provider_base_url("openai_compatible", values)
        if not api_key or not base_url or not model:
            return _provider_config_error(payload, provider, model, "OpenAI-compatible API key, base URL, or model is not configured.")
        return _chat_completions_response(
            payload,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            key_source=str(key_info.get("key_source") or ""),
            masked_key_suffix=str(key_info.get("masked_suffix") or ""),
            model_source=str(role.get("model_source") or ""),
            diagnostics=diagnostics,
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


def build_gemini_generate_content_body(prompt_text: str, generation_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"contents": [{"parts": [{"text": _text(prompt_text)}]}]}
    generation_config = _gemini_generation_config(generation_parameters or {})
    if generation_config:
        body["generationConfig"] = generation_config
    return body


def generate_gemini_direct(
    *,
    prompt_text: str,
    api_key: str,
    model: str,
    generation_parameters: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    body = body if isinstance(body, dict) else build_gemini_generate_content_body(prompt_text, generation_parameters)
    endpoint_url = GEMINI_ENDPOINT_TEMPLATE.format(model=model)
    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        text = _gemini_text(response_payload)
        return {
            "ok": bool(text.strip()),
            "text": text,
            "finish_reason": _gemini_finish_reason(response_payload),
            "status_code": getattr(response, "status", 200),
            "endpoint_url": endpoint_url,
            "payload_shape": "contents.parts",
            "generation_parameters_sent": body.get("generationConfig", {}),
            "error": "" if text.strip() else "Gemini returned an empty output.",
        }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "text": "",
            "finish_reason": "",
            "endpoint_url": endpoint_url,
            "payload_shape": "contents.parts",
            "generation_parameters_sent": body.get("generationConfig", {}),
            "error": f"Gemini HTTP {exc.code}",
            **_safe_gemini_http_error(exc, redact_values=[api_key]),
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "finish_reason": "",
            "endpoint_url": endpoint_url,
            "payload_shape": "contents.parts",
            "generation_parameters_sent": body.get("generationConfig", {}),
            "error": f"Gemini request failed: {type(exc).__name__}",
            "status_code": None,
            "google_error_status": "",
            "google_error_message": "",
        }


def _gemini_generation_config(generation_parameters: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for source_key, target_key in (
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("max_tokens", "maxOutputTokens"),
        ("top_k", "topK"),
    ):
        if source_key not in generation_parameters:
            continue
        value = generation_parameters.get(source_key)
        if value in (None, ""):
            continue
        config[target_key] = value
    return config


def _safe_gemini_http_error(exc: urllib.error.HTTPError, redact_values: list[str] | None = None) -> dict[str, Any]:
    raw = ""
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    status = ""
    message = ""
    if raw:
        try:
            payload = json.loads(raw)
            error = payload.get("error") if isinstance(payload, dict) else {}
            if isinstance(error, dict):
                status = _text(error.get("status"))
                message = _text(error.get("message"))
        except json.JSONDecodeError:
            message = raw
    return {
        "status_code": exc.code,
        "google_error_status": _safe_provider_message(status, redact_values),
        "google_error_message": _safe_provider_message(message or exc.reason or "", redact_values),
    }


def _safe_provider_message(value: Any, redact_values: list[str] | None = None, limit: int = 500) -> str:
    text = _truncate(_text(value), limit)
    for secret in redact_values or []:
        secret = _text(secret)
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"AIza[0-9A-Za-z_-]{8,}", "AIza...[redacted]", text)
    text = re.sub(r"or-[0-9A-Za-z_-]{8,}", "or-...[redacted]", text)
    return text


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
    response_mode = _interaction_response_mode(payload)
    selected_strategy_text = _selected_strategy_text(payload) if response_mode in {"strategy_response", "proactive_support"} else ""
    learning_signal_text = _learning_signal_text(payload)
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
    if learning_signal_text:
        label = "Soft learning-state style cue:" if response_mode == "normal_followup" else "Internal learning-support signal:"
        sections.extend(["", label, learning_signal_text])
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


def _gemini_response(
    payload: dict[str, Any],
    api_key: str,
    model: str | None = None,
    *,
    key_source: str = "",
    masked_key_suffix: str = "",
    model_source: str = "",
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = (model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).strip() or DEFAULT_GEMINI_MODEL
    prompt, body, used_image = build_gemini_request(payload)
    result = generate_gemini_direct(prompt_text=prompt, api_key=api_key, model=model, body=body, timeout=45)
    if result.get("ok"):
        return {
            "provider": "gemini",
            "model": model,
            "model_id": model,
            "key_source": key_source,
            "masked_key_suffix": masked_key_suffix,
            "model_source": model_source,
            **_safe_runtime_diagnostics(diagnostics),
            "mode": _mode(payload),
            "recommended_llm_mode": _mode(payload),
            "response_style": _response_style(payload),
            "used_image": used_image,
            **_retrieval_metadata(payload),
            "prompt_preview": _prompt_preview(prompt),
            "answer": _text(result.get("text")),
            "error": None,
        }
    return _gemini_error_response(
        payload,
        model,
        used_image,
        prompt,
        _text(result.get("error")) or "Gemini request failed.",
        key_source=key_source,
        masked_key_suffix=masked_key_suffix,
    )


def _chat_completions_response(
    payload: dict[str, Any],
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    key_source: str = "",
    masked_key_suffix: str = "",
    model_source: str = "",
    diagnostics: dict[str, Any] | None = None,
    provider_values: dict[str, str] | None = None,
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        values = provider_values or read_llm_values(include_env_file=False)
        site_url = _text(values.get("OPENROUTER_SITE_URL"))
        site_name = _text(values.get("OPENROUTER_SITE_NAME"))
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-OpenRouter-Title"] = site_name
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        answer = _chat_completion_text(response_payload)
        return {
            "provider": provider,
            "model": model,
            "model_id": model,
            "key_source": key_source,
            "masked_key_suffix": masked_key_suffix,
            "model_source": model_source,
            **_safe_runtime_diagnostics(diagnostics),
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


def _safe_runtime_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "settings_revision": diagnostics.get("settings_revision"),
        "settings_file_mtime": diagnostics.get("settings_file_mtime"),
    }


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
    *,
    key_source: str = "",
    masked_key_suffix: str = "",
) -> dict[str, Any]:
    return {
        "provider": "gemini",
        "model": model,
        "model_id": model,
        "key_source": key_source,
        "masked_key_suffix": masked_key_suffix,
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


def _interaction_response_mode(payload: dict[str, Any]) -> str:
    input_source = _text(payload.get("input_source")).lower()
    if input_source in {"text", "speech"}:
        return "normal_followup"
    if input_source == "strategy_click":
        return "strategy_response"
    if input_source == "proactive_recommendation":
        return "proactive_support"
    response_mode = _text(payload.get("response_mode")).lower()
    if response_mode in {"normal_followup", "strategy_response", "proactive_support"}:
        return response_mode
    if isinstance(payload.get("selected_strategy"), dict) and payload.get("selected_strategy"):
        return "strategy_response"
    if payload.get("follow_up_question"):
        return "normal_followup"
    return ""


def _learning_signal_text(payload: dict[str, Any]) -> str:
    if _interaction_response_mode(payload) == "normal_followup":
        return _soft_learning_state_text(payload)
    package = payload.get("learning_signal_package")
    if not isinstance(package, dict) or not package:
        learning_state = payload.get("learning_state") if isinstance(payload.get("learning_state"), dict) else {}
        package = learning_state.get("learning_signal_package") if isinstance(learning_state.get("learning_signal_package"), dict) else {}
    if not isinstance(package, dict) or not package:
        return ""
    context = package.get("learning_process_context") if isinstance(package.get("learning_process_context"), dict) else {}
    mapping = package.get("academic_state_mapping") if isinstance(package.get("academic_state_mapping"), dict) else {}
    reaction = package.get("reaction_window_summary") if isinstance(package.get("reaction_window_summary"), dict) else {}
    raw = package.get("raw_emotion_evidence") if isinstance(package.get("raw_emotion_evidence"), dict) else {}
    guidance = package.get("prompt_guidance") if isinstance(package.get("prompt_guidance"), list) else []
    mapped_scores = package.get("academic_state_scores") if isinstance(package.get("academic_state_scores"), dict) else {}
    if not mapped_scores:
        mapped_scores = mapping.get("mapped_academic_scores") if isinstance(mapping.get("mapped_academic_scores"), dict) else {}
    top_raw = raw.get("raw_top_emotions") if isinstance(raw.get("raw_top_emotions"), list) else []
    raw_evidence_text = _probability_items_text(top_raw, label_key="label", value_key="probability")
    academic_evidence_text = _score_dict_text(mapped_scores)
    dominant_academic_state = _text(package.get("dominant_academic_state") or mapping.get("mapped_academic_state")) or "uncertain"
    secondary_academic_state = _text(package.get("secondary_academic_state")) or "[none]"
    support_cue = _text(package.get("support_cue") or reaction.get("support_cue")) or "uncertain"
    lines = [
        "Use this as a lightweight learning-support signal, not as a psychological diagnosis.",
        "The academic learning-state labels are internal support signals, not diagnoses.",
        "Distinguish academic learning state from the derived support cue.",
        "Do not mention camera, face analysis, raw emotion labels, or detected emotion unless the user explicitly asks.",
        "Do not tell the user that they are confused, frustrated, bored, engaged, sad, angry, or any other emotion.",
        f"- active_source: {_text(package.get('active_source')) or 'raw_8class_process_aware'}",
        f"- dominant_academic_state: {dominant_academic_state}",
        f"- secondary_academic_state: {secondary_academic_state}",
        f"- support_cue: {support_cue}",
        f"- inferred_process_state: {_text(package.get('inferred_process_state')) or 'uncertain_or_mixed'}",
        f"- recommended_strategy: {_text(package.get('recommended_strategy')) or 'baseline_explanation'}",
        f"- strategy_reason: {_text(package.get('strategy_reason')) or '[not specified]'}",
        f"- question_intent: {_text(context.get('question_intent')) or 'general_followup'}",
        f"- followup_count_for_highlight: {_text(context.get('followup_count_for_highlight')) or '0'}",
        f"- academic_state_scores: {json.dumps(mapped_scores, ensure_ascii=False)}",
        f"Raw expression evidence: {raw_evidence_text or '[unavailable]'}",
        f"Academic-state evidence: {academic_evidence_text or '[unavailable]'}",
        f"Support cue: {support_cue}",
        f"- reaction_window_summary: {json.dumps(_compact_learning_signal_reaction(reaction), ensure_ascii=False)}",
        f"- raw_top_emotions_internal_only: {json.dumps(top_raw[:3], ensure_ascii=False)}",
    ]
    if guidance:
        lines.append("- prompt_guidance:")
        lines.extend(f"  - {_text(item)}" for item in guidance if _text(item))
    lines.extend([
        "Response-style guidance:",
        "- frustration-like difficulty: calm, supportive, short steps, reduced jargon.",
        "- confusion-like difficulty: define terms, clarify assumptions, add examples.",
        "- possible low engagement: concise takeaway, relevance hook, optional next step.",
        "- engagement-like: deeper explanation, connections, extension.",
        "- uncertain/mixed: baseline paper-grounded explanation.",
    ])
    return "\n".join(lines)


def _soft_learning_state_text(payload: dict[str, Any]) -> str:
    learning_state = payload.get("learning_state") if isinstance(payload.get("learning_state"), dict) else {}
    if not learning_state:
        return ""
    state = _text(learning_state.get("academic_state") or learning_state.get("state")).lower() or "uncertain"
    trend = _text(learning_state.get("trend")) or "unknown"
    confidence = _text(learning_state.get("confidence"))
    distribution = learning_state.get("distribution") if isinstance(learning_state.get("distribution"), dict) else {}
    state_guidance = {
        "frustration": "answer more patiently and clearly, with shorter steps and reduced jargon.",
        "confusion": "define terms, make assumptions explicit, and add a brief clarifying example when useful.",
        "boredom": "be concise, concrete, and connect quickly to why the passage matters.",
        "engagement": "answer normally, with slightly more depth if the question invites it.",
    }.get(state, "answer in a normal paper-grounded style.")
    lines = [
        "Use this only as a soft style cue for a normal follow-up answer.",
        "Do not label the response as a strategy response.",
        "Do not mention camera, face analysis, detected emotion, or internal learning-state labels unless the user explicitly asks.",
        f"- current_learning_state: {state}",
        f"- confidence: {confidence or '[not specified]'}",
        f"- trend: {trend}",
        f"- distribution: {json.dumps(distribution, ensure_ascii=False)}",
        f"- style_adjustment: {state_guidance}",
    ]
    return "\n".join(lines)


def _probability_items_text(items: Any, *, label_key: str, value_key: str) -> str:
    if not isinstance(items, list):
        return ""
    values = []
    for item in items[:3]:
        if not isinstance(item, dict):
            continue
        label = _text(item.get(label_key))
        if not label:
            continue
        try:
            probability = float(item.get(value_key))
        except (TypeError, ValueError):
            probability = 0.0
        values.append(f"{label} {probability:.2f}")
    return ", ".join(values)


def _score_dict_text(scores: Any) -> str:
    if not isinstance(scores, dict):
        return ""
    ordered_states = ["boredom", "confusion", "engagement", "frustration"]
    values = []
    for state in ordered_states:
        if state not in scores:
            continue
        try:
            value = float(scores.get(state))
        except (TypeError, ValueError):
            value = 0.0
        values.append(f"{state} {value:.2f}")
    if values:
        return ", ".join(values)
    values = []
    for key, value in scores.items():
        label = _text(key)
        if not label:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        values.append(f"{label} {number:.2f}")
    return ", ".join(values)


def _compact_learning_signal_reaction(reaction: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reaction, dict):
        return {}
    return {
        "trend": _text(reaction.get("trend")),
        "support_cue": _text(reaction.get("support_cue")),
        "evidence_count": reaction.get("evidence_count"),
        "dominant_mapped_states": reaction.get("dominant_mapped_states") if isinstance(reaction.get("dominant_mapped_states"), list) else [],
    }


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


def _gemini_finish_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") if isinstance(payload, dict) else []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    return _text(candidates[0].get("finishReason") or candidates[0].get("finish_reason"))


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
