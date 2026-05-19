from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from emotion_aware_assistant.core.llm_config import (
    DEFAULT_GEMINI_EMBEDDING_MODEL,
    provider_api_key_from_env,
    role_config_from_env,
)

PROFILE_FIELDS = {
    "title": "",
    "one_sentence_summary": "",
    "research_problem": "",
    "method_summary": "",
    "dataset_or_materials": "",
    "main_findings": "",
    "key_terms": [],
    "section_map": [],
}
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
GEMINI_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_EMBEDDING_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "during",
    "for",
    "from",
    "have",
    "into",
    "paper",
    "participants",
    "result",
    "results",
    "show",
    "shows",
    "study",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
    "were",
    "what",
    "when",
    "where",
    "which",
}


def normalize_pdf_text(text: Any) -> str:
    value = str(text or "")
    for source, replacement in LIGATURES.items():
        value = value.replace(source, replacement)
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_low_value_context_block(block: dict[str, Any] | None) -> bool:
    if not isinstance(block, dict):
        return True
    block_type = str(block.get("block_type") or "").lower()
    text = normalize_pdf_text(block.get("markdown_content") or block.get("text") or "")
    if not text:
        return True
    if block_type in {"caption", "formula", "table"}:
        return False
    if block_type == "title" and len(text) <= 120:
        return _is_known_header(text)
    if re.match(r"^(fig\.?|figure|table|equation|eq\.)\s*\d+", text, re.IGNORECASE):
        return False
    if re.match(r"^(-\s*)?\d+(\s*/\s*\d+|\s+of\s+\d+)?(\s*-)?$", text, re.IGNORECASE):
        return True
    if re.match(r"^[A-Z][A-Za-z-]+,?\s+et al\.?$", text):
        return True
    if _is_known_header(text):
        return True
    if re.match(r"^(doi:|https?:|www\.)", text, re.IGNORECASE):
        return True
    if len(text) < 28 and len(text.split()) <= 3:
        return True
    return False


def prepare_paper_memory(document_id: str, document_dir: Path, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    document_dir = Path(document_dir)
    rag_dir = document_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    cleaned_blocks = [clean_block(block) for block in blocks]
    useful = [block for block in cleaned_blocks if not is_low_value_context_block(block)]
    section_map = build_section_map(cleaned_blocks)
    keyword_index = build_keyword_index(useful)
    profile, profile_provider = generate_paper_profile(cleaned_blocks, section_map)
    embedding_status = build_embedding_index(document_id, rag_dir, useful)

    section_map_path = rag_dir / "section_map.json"
    keyword_index_path = rag_dir / "keyword_index.json"
    paper_profile_path = rag_dir / "paper_profile.json"
    prepare_status_path = rag_dir / "prepare_status.json"
    section_map_path.write_text(json.dumps(section_map, indent=2), encoding="utf-8")
    keyword_index_path.write_text(json.dumps(keyword_index, indent=2), encoding="utf-8")
    paper_profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    status = {
        "document_id": document_id,
        "status": "completed",
        "profile_provider": profile_provider,
        "block_count": len(blocks),
        "useful_block_count": len(useful),
        "prepared_at": time.time(),
        "paper_profile_path": str(paper_profile_path),
        "section_map_path": str(section_map_path),
        "keyword_index_path": str(keyword_index_path),
        "rag_prepare_status_path": str(prepare_status_path),
        **embedding_status,
    }
    prepare_status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def retrieve_context(
    document_id: str,
    document_dir: Path,
    blocks: list[dict[str, Any]],
    highlight_payload: dict[str, Any],
) -> dict[str, Any]:
    document_dir = Path(document_dir)
    rag_dir = document_dir / "rag"
    if not (rag_dir / "paper_profile.json").exists():
        prepare_paper_memory(document_id, document_dir, blocks)
    profile = _load_json(rag_dir / "paper_profile.json", dict(PROFILE_FIELDS))
    section_map = _load_json(rag_dir / "section_map.json", [])
    cleaned_blocks = [clean_block(block) for block in blocks]
    useful_blocks = [block for block in cleaned_blocks if not is_low_value_context_block(block)]
    match_result = _coordinate_match(cleaned_blocks, highlight_payload)
    matched_block = _first_useful(match_result.get("matched_blocks", [])) or _payload_matched_block(highlight_payload)
    nearby_context = _nearby_context(highlight_payload, match_result, page_number=_page_number(highlight_payload), limit=4)
    same_section_context = _same_section_context(
        useful_blocks,
        section_map if isinstance(section_map, list) else [],
        matched_block,
        page_number=_page_number(highlight_payload),
    )
    global_context = retrieve_global_context(
        document_id=document_id,
        document_dir=document_dir,
        query=_global_query_text(highlight_payload, matched_block, profile if isinstance(profile, dict) else {}),
        exclude_ids=_block_ids([matched_block] + nearby_context + same_section_context),
        top_k=3,
    )
    related_blocks = global_context.get("related_blocks") or _related_blocks(
        useful_blocks,
        highlight_payload,
        profile if isinstance(profile, dict) else {},
        exclude_ids=_block_ids([matched_block] + nearby_context + same_section_context),
    )
    return {
        "paper_profile": _complete_profile(profile if isinstance(profile, dict) else {}, section_map if isinstance(section_map, list) else []),
        "matched_block": matched_block or {},
        "nearby_context": nearby_context,
        "same_section_context": same_section_context,
        "related_blocks": related_blocks,
        "global_rag_context": related_blocks,
        "retrieval_method": global_context.get("retrieval_method") or "keyword",
        "embedding_index_status": global_context.get("embedding_index_status") or "",
        "retrieval_strategy": "coordinate_plus_keyword",
    }


def clean_block(block: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    cleaned = dict(block)
    cleaned["markdown_content"] = normalize_pdf_text(
        cleaned.get("markdown_content") or cleaned.get("text") or "",
    )
    if "coordinate_overlap_score" in cleaned and "coordinate_overlap" not in cleaned:
        cleaned["coordinate_overlap"] = _clamped_float(cleaned.get("coordinate_overlap_score"))
    if "selected_text_similarity" in cleaned and "text_bonus" not in cleaned:
        cleaned["text_bonus"] = _clamped_float(cleaned.get("selected_text_similarity"))
    if "overlap_score" in cleaned and "match_score" not in cleaned:
        cleaned["match_score"] = _clamped_float(cleaned.get("overlap_score"))
    if "match_score" in cleaned:
        cleaned["match_score"] = _clamped_float(cleaned.get("match_score"))
    return cleaned


def build_section_map(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    titles = [
        block for block in blocks
        if str(block.get("block_type") or "").lower() == "title"
        and not is_low_value_context_block(block)
    ]
    if not titles and blocks:
        titles = [blocks[0]]
    sections: list[dict[str, Any]] = []
    for index, block in enumerate(titles):
        next_block = titles[index + 1] if index + 1 < len(titles) else None
        start_order = int(float(block.get("reading_order_index", 0) or 0))
        end_order = int(float(next_block.get("reading_order_index", 10**9) or 10**9)) if next_block else 10**9
        sections.append(
            {
                "heading": normalize_pdf_text(block.get("markdown_content")),
                "page_number": _block_page_number(block),
                "block_id": str(block.get("block_id") or ""),
                "start_order": start_order,
                "end_order": end_order,
            }
        )
    return sections


def build_keyword_index(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for block in blocks:
        terms = sorted(_terms(block.get("markdown_content")))
        if not terms:
            continue
        entries.append(
            {
                "block_id": block.get("block_id") or "",
                "page_number": _block_page_number(block),
                "block_type": block.get("block_type") or "",
                "terms": terms[:40],
                "text": _truncate(block.get("markdown_content"), 360),
            }
        )
    return {"blocks": entries}


def build_embedding_index(document_id: str, rag_dir: Path, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    embeddings_path = Path(rag_dir) / "embeddings.json"
    role = role_config_from_env("embedding_model")
    provider = str(role.get("provider") or "gemini").strip().lower() or "gemini"
    model = str(role.get("model") or DEFAULT_GEMINI_EMBEDDING_MODEL).strip() or DEFAULT_GEMINI_EMBEDDING_MODEL
    api_key = provider_api_key_from_env("gemini") if provider == "gemini" else ""
    unavailable_message = (
        f"Embedding provider {provider} is not supported by the current embedding path; keyword retrieval will be used."
        if provider != "gemini"
        else "GEMINI_API_KEY is not configured; keyword retrieval will be used."
    )
    base_payload: dict[str, Any] = {
        "document_id": document_id,
        "provider": provider,
        "model": model,
        "status": "unavailable",
        "message": unavailable_message,
        "embeddings": [],
    }
    if provider != "gemini" or not api_key:
        embeddings_path.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
        return {
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_index_status": "unavailable",
            "embedding_message": base_payload["message"],
            "embeddings_path": str(embeddings_path),
        }

    embeddings = []
    try:
        for block in blocks[:120]:
            text = _truncate(block.get("markdown_content"), 1200)
            if not text:
                continue
            vector = _gemini_embedding(text, api_key=api_key, model=model, task_type="RETRIEVAL_DOCUMENT")
            if not vector:
                continue
            embeddings.append(
                {
                    "block_id": block.get("block_id") or "",
                    "page_number": _block_page_number(block),
                    "block_type": block.get("block_type") or "",
                    "markdown_content": _truncate(text, 520),
                    "embedding": vector,
                }
            )
        status = "completed" if embeddings else "failed"
        message = "Gemini embedding index prepared." if embeddings else "Gemini returned no usable embeddings; keyword retrieval will be used."
    except Exception as exc:
        embeddings = []
        status = "failed"
        message = f"Gemini embedding preparation failed: {type(exc).__name__}; keyword retrieval will be used."
    payload = {
        "document_id": document_id,
        "provider": provider,
        "model": model,
        "status": status,
        "message": message,
        "embeddings": embeddings,
    }
    embeddings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_index_status": status,
        "embedding_message": message,
        "embeddings_path": str(embeddings_path),
    }


def retrieve_global_context(
    document_id: str,
    document_dir: Path,
    query: str,
    exclude_ids: set[str] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    exclude_ids = exclude_ids or set()
    rag_dir = Path(document_dir) / "rag"
    embeddings_payload = _load_json(rag_dir / "embeddings.json", {})
    status = embeddings_payload.get("status") if isinstance(embeddings_payload, dict) else ""
    if status == "completed" and os.environ.get("GEMINI_API_KEY", "").strip():
        query_vector = _gemini_embedding(
            query,
            api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
            model=str(embeddings_payload.get("model") or DEFAULT_GEMINI_EMBEDDING_MODEL),
            task_type="RETRIEVAL_QUERY",
        )
        if query_vector:
            scored = []
            for entry in embeddings_payload.get("embeddings", []):
                if not isinstance(entry, dict) or str(entry.get("block_id") or "") in exclude_ids:
                    continue
                score = _cosine_similarity(query_vector, entry.get("embedding") or [])
                scored.append((score, entry))
            scored.sort(key=lambda item: -item[0])
            return {
                "document_id": document_id,
                "retrieval_method": "embedding",
                "embedding_index_status": status,
                "related_blocks": [
                    _related_entry_from_embedding(entry, score)
                    for score, entry in scored[:top_k]
                    if score > 0
                ],
            }
    return {
        "document_id": document_id,
        "retrieval_method": "keyword",
        "embedding_index_status": status or "unavailable",
        "related_blocks": _keyword_index_related_blocks(rag_dir / "keyword_index.json", query, exclude_ids, top_k),
    }


def generate_paper_profile(blocks: list[dict[str, Any]], section_map: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    profile_input = _profile_input_text(blocks)
    if os.environ.get("LLM_PROVIDER", "mock").strip().lower() == "gemini" and os.environ.get("GEMINI_API_KEY", "").strip():
        generated = _gemini_profile(profile_input, section_map)
        if generated:
            return _complete_profile(generated, section_map), "gemini"
    return _rule_based_profile(blocks, section_map), "rule_based"


def _rule_based_profile(blocks: list[dict[str, Any]], section_map: list[dict[str, Any]]) -> dict[str, Any]:
    useful = [block for block in blocks if not is_low_value_context_block(block)]
    title_block = next((block for block in useful if str(block.get("block_type") or "").lower() == "title"), None)
    title = normalize_pdf_text(title_block.get("markdown_content")) if title_block else ""
    if not title and useful:
        title = _truncate(useful[0].get("markdown_content"), 180)

    abstract = _find_text(useful, r"\babstract\b") or _first_body_text(useful)
    problem = _find_text(useful, r"\b(problem|ask|aim|objective|challenge|difficulty)\b") or abstract
    method = _find_section_text(useful, section_map, "method") or _find_text(useful, r"\b(method|participant|dataset|audio|recording|material)\b")
    materials = _find_text(useful, r"\b(dataset|data|participant|material|audio|recording|corpus)\b") or method
    findings = _find_section_text(useful, section_map, "result") or _find_text(useful, r"\b(result|finding|found|show|conclusion)\b")
    key_terms = _top_terms(" ".join(str(block.get("markdown_content") or "") for block in useful), limit=12)
    summary_source = _first_sentence(abstract) or (f"This paper examines {title.lower()}." if title else "")
    return _complete_profile(
        {
            "title": title,
            "one_sentence_summary": summary_source,
            "research_problem": _first_sentence(problem),
            "method_summary": _first_sentence(method),
            "dataset_or_materials": _first_sentence(materials),
            "main_findings": _first_sentence(findings),
            "key_terms": key_terms,
        },
        section_map,
    )


def _gemini_profile(profile_input: str, section_map: list[dict[str, Any]]) -> dict[str, Any] | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    prompt = (
        "Create a concise academic paper profile as strict JSON with keys: "
        "title, one_sentence_summary, research_problem, method_summary, "
        "dataset_or_materials, main_findings, key_terms. Use only this parsed text.\n\n"
        f"{_truncate(profile_input, 6000)}"
    )
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    request = urllib.request.Request(
        GEMINI_ENDPOINT_TEMPLATE.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = _gemini_text(payload)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group(0) if match else text)
        parsed["section_map"] = section_map
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _coordinate_match(blocks: list[dict[str, Any]], highlight_payload: dict[str, Any]) -> dict[str, Any]:
    from emotion_aware_assistant.paper.pdf_parse_pipeline import match_blocks_for_rects

    rects = (
        highlight_payload.get("normalized_rects")
        or highlight_payload.get("parser_rects_1000")
        or highlight_payload.get("viewport_rects")
        or []
    )
    if not rects:
        return {"matched_blocks": [], "previous_blocks": [], "next_blocks": []}
    return match_blocks_for_rects(
        blocks,
        page_number=_page_number(highlight_payload),
        rects=rects,
        selected_text=normalize_pdf_text(highlight_payload.get("selected_text")),
    )


def _payload_matched_block(highlight_payload: dict[str, Any]) -> dict[str, Any]:
    block = highlight_payload.get("matched_block")
    if isinstance(block, dict) and normalize_pdf_text(block.get("markdown_content")):
        return clean_block(block)
    return {}


def _nearby_context(
    highlight_payload: dict[str, Any],
    match_result: dict[str, Any],
    page_number: int,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for key in ("previous_blocks", "next_blocks"):
        candidates.extend(match_result.get(key) or [])
    for key in ("nearby_useful_context", "previous_block", "next_block"):
        value = highlight_payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, dict):
            candidates.append(value)
    return _dedupe_useful(candidates, page_number=page_number, limit=limit)


def _same_section_context(
    blocks: list[dict[str, Any]],
    section_map: list[dict[str, Any]],
    matched_block: dict[str, Any],
    page_number: int,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not blocks:
        return []
    order = int(float(matched_block.get("reading_order_index", -1) or -1)) if matched_block else -1
    section = _section_for_order(section_map, order, page_number)
    candidates = []
    if section:
        start_order = int(float(section.get("start_order", 0) or 0))
        end_order = int(float(section.get("end_order", 10**9) or 10**9))
        candidates = [
            block for block in blocks
            if start_order <= int(float(block.get("reading_order_index", 0) or 0)) < end_order
        ]
    if not candidates:
        candidates = [block for block in blocks if _block_page_number(block) == page_number]
    return _dedupe_useful(candidates, limit=limit)


def _related_blocks(
    blocks: list[dict[str, Any]],
    highlight_payload: dict[str, Any],
    profile: dict[str, Any],
    exclude_ids: set[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    query_text = " ".join(
        [
            normalize_pdf_text(highlight_payload.get("selected_text")),
            normalize_pdf_text(highlight_payload.get("caption")),
            normalize_pdf_text((highlight_payload.get("matched_block") or {}).get("markdown_content") if isinstance(highlight_payload.get("matched_block"), dict) else ""),
            " ".join(str(term) for term in profile.get("key_terms", []) if isinstance(profile.get("key_terms"), list)),
        ]
    )
    query_terms = _terms(query_text)
    if not query_terms:
        query_terms = set(profile.get("key_terms") or [])
    scored = []
    for block in blocks:
        block_id = str(block.get("block_id") or "")
        if block_id in exclude_ids:
            continue
        terms = _terms(block.get("markdown_content"))
        if not terms:
            continue
        score = len(query_terms & terms) / max(len(query_terms), 1)
        if score > 0:
            enriched = clean_block(block)
            enriched["keyword_score"] = round(min(score, 1.0), 4)
            scored.append((score, enriched))
    scored.sort(key=lambda item: (-item[0], int(float(item[1].get("reading_order_index", 0) or 0))))
    return [_compact_block(block) for _, block in scored[:limit]]


def _global_query_text(highlight_payload: dict[str, Any], matched_block: dict[str, Any], profile: dict[str, Any]) -> str:
    return " ".join(
        value
        for value in [
            normalize_pdf_text(highlight_payload.get("selected_text")),
            normalize_pdf_text(highlight_payload.get("caption")),
            normalize_pdf_text(matched_block.get("markdown_content")),
            normalize_pdf_text(profile.get("one_sentence_summary")),
            " ".join(str(term) for term in profile.get("key_terms", []) if isinstance(profile.get("key_terms"), list)),
        ]
        if value
    )


def _keyword_index_related_blocks(path: Path, query: str, exclude_ids: set[str], top_k: int) -> list[dict[str, Any]]:
    payload = _load_json(path, {})
    entries = payload.get("blocks") if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        return []
    query_terms = _terms(query)
    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        block_id = str(entry.get("block_id") or "")
        if block_id in exclude_ids:
            continue
        terms = set(entry.get("terms") or [])
        if not terms:
            continue
        score = len(query_terms & terms) / max(len(query_terms), 1)
        if score <= 0:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], int(float(item[1].get("page_number", 0) or 0))))
    return [
        {
            "block_id": entry.get("block_id") or "",
            "page_number": entry.get("page_number") or "",
            "block_type": entry.get("block_type") or "",
            "markdown_content": _truncate(entry.get("text"), 520),
            "score": round(min(float(score), 1.0), 4),
        }
        for score, entry in scored[:top_k]
    ]


def _related_entry_from_embedding(entry: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "block_id": entry.get("block_id") or "",
        "page_number": entry.get("page_number") or "",
        "block_type": entry.get("block_type") or "",
        "markdown_content": _truncate(entry.get("markdown_content"), 520),
        "score": round(min(max(float(score), 0.0), 1.0), 4),
    }


def _gemini_embedding(text: str, api_key: str, model: str, task_type: str) -> list[float]:
    body = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": _truncate(text, 3000)}]},
        "taskType": task_type,
    }
    request = urllib.request.Request(
        GEMINI_EMBEDDING_ENDPOINT_TEMPLATE.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    values = payload.get("embedding", {}).get("values", [])
    return [float(value) for value in values if isinstance(value, (int, float))]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def _dedupe_useful(
    blocks: list[dict[str, Any]],
    page_number: int | None = None,
    limit: int = 4,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    results = []
    for block in blocks:
        cleaned = clean_block(block)
        if not cleaned or is_low_value_context_block(cleaned):
            continue
        if page_number and cleaned.get("page_number") is not None and _block_page_number(cleaned) != page_number:
            continue
        key = str(cleaned.get("block_id") or cleaned.get("markdown_content") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(_compact_block(cleaned))
        if len(results) >= limit:
            break
    return results


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    compact = clean_block(block)
    keep = {
        "block_id",
        "page_number",
        "page_idx",
        "block_type",
        "markdown_content",
        "reading_order_index",
        "coordinate_overlap",
        "text_bonus",
        "match_score",
        "keyword_score",
    }
    result = {key: value for key, value in compact.items() if key in keep and value not in (None, "")}
    if "markdown_content" in result:
        result["markdown_content"] = _truncate(result["markdown_content"], 520)
    return result


def _first_useful(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    for block in blocks:
        cleaned = clean_block(block)
        if not is_low_value_context_block(cleaned):
            return _compact_block(cleaned)
    return {}


def _complete_profile(profile: dict[str, Any], section_map: list[dict[str, Any]]) -> dict[str, Any]:
    complete = dict(PROFILE_FIELDS)
    for key in complete:
        if key in profile and profile[key] not in (None, ""):
            complete[key] = profile[key]
    if not isinstance(complete.get("key_terms"), list):
        complete["key_terms"] = []
    complete["key_terms"] = [normalize_pdf_text(term).lower() for term in complete["key_terms"] if normalize_pdf_text(term)][:12]
    complete["section_map"] = section_map
    return complete


def _profile_input_text(blocks: list[dict[str, Any]]) -> str:
    useful = [block for block in blocks if not is_low_value_context_block(block)]
    return "\n\n".join(_truncate(block.get("markdown_content"), 800) for block in useful[:40])


def _find_text(blocks: list[dict[str, Any]], pattern: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for block in blocks:
        text = normalize_pdf_text(block.get("markdown_content"))
        if regex.search(text):
            return text
    return ""


def _find_section_text(blocks: list[dict[str, Any]], section_map: list[dict[str, Any]], name: str) -> str:
    section = next((section for section in section_map if name.lower() in str(section.get("heading") or "").lower()), None)
    if not section:
        return ""
    start_order = int(float(section.get("start_order", 0) or 0))
    end_order = int(float(section.get("end_order", 10**9) or 10**9))
    for block in blocks:
        order = int(float(block.get("reading_order_index", 0) or 0))
        if start_order < order < end_order and str(block.get("block_type") or "").lower() != "title":
            return normalize_pdf_text(block.get("markdown_content"))
    return ""


def _first_body_text(blocks: list[dict[str, Any]]) -> str:
    for block in blocks:
        if str(block.get("block_type") or "").lower() != "title":
            return normalize_pdf_text(block.get("markdown_content"))
    return ""


def _first_sentence(text: Any, limit: int = 260) -> str:
    value = normalize_pdf_text(text)
    if not value:
        return ""
    match = re.search(r"(.+?[.!?])\s", value)
    return _truncate(match.group(1) if match else value, limit)


def _top_terms(text: str, limit: int) -> list[str]:
    counts = Counter(
        word for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", normalize_pdf_text(text).lower())
        if word not in STOPWORDS
    )
    return [term for term, _ in counts.most_common(limit)]


def _terms(text: Any) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", normalize_pdf_text(text).lower())
    return {word for word in words if word not in STOPWORDS}


def _section_for_order(section_map: list[dict[str, Any]], order: int, page_number: int) -> dict[str, Any]:
    for section in section_map:
        start = int(float(section.get("start_order", 0) or 0))
        end = int(float(section.get("end_order", 10**9) or 10**9))
        if order >= 0 and start <= order < end:
            return section
    return next((section for section in section_map if int(float(section.get("page_number", 0) or 0)) == page_number), {})


def _block_ids(blocks: list[dict[str, Any]]) -> set[str]:
    return {str(block.get("block_id") or "") for block in blocks if isinstance(block, dict) and block.get("block_id")}


def _page_number(payload: dict[str, Any]) -> int:
    try:
        return max(1, int(float(payload.get("page_number") or 1)))
    except (TypeError, ValueError):
        return 1


def _block_page_number(block: dict[str, Any]) -> int:
    if block.get("page_number") is not None:
        try:
            return max(1, int(float(block.get("page_number"))))
        except (TypeError, ValueError):
            return 1
    if block.get("page_idx") is not None:
        try:
            return max(1, int(float(block.get("page_idx"))) + 1)
        except (TypeError, ValueError):
            return 1
    return 1


def _clamped_float(value: Any) -> float:
    try:
        return round(max(0.0, min(float(value), 1.0)), 4)
    except (TypeError, ValueError):
        return 0.0


def _truncate(text: Any, limit: int) -> str:
    value = normalize_pdf_text(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    if not isinstance(parts, list):
        return ""
    return "\n".join(str(part.get("text") or "").strip() for part in parts if str(part.get("text") or "").strip())


def _is_known_header(text: str) -> bool:
    return bool(
        re.match(r"^trends in hearing\s+\d+\(.*\)$", text, re.IGNORECASE)
        or re.match(r"^[A-Z][A-Za-z0-9& .-]*\s*['\u2019]\d{2},\s+[A-Z][a-z]+\s+\d{1,2}[-\u2013]\d{1,2},\s+\d{4},", text)
        or re.match(r"^(research article|original article|article)$", text, re.IGNORECASE)
        or re.match(r"^[A-Za-z ]+\s+\d+\(\d+\)$", text)
    )
