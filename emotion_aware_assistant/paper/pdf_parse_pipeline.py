from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from emotion_aware_assistant.paper.paper_rag import normalize_pdf_text, prepare_paper_memory


BLOCK_TYPES = {"text", "title", "table", "formula", "image", "caption", "list", "footnote"}


def parse_pdf_to_blocks(document_id: str, pdf_path: Path, documents_root: Path) -> dict[str, Any]:
    document_dir = documents_root / document_id
    parsed_dir = document_dir / "parsed"
    images_dir = parsed_dir / "images"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    mineru_status = _run_mineru_if_available(pdf_path, parsed_dir)
    blocks: list[dict[str, Any]] = []
    parser = "mineru"
    parser_status = mineru_status["status"]
    parser_message = mineru_status["message"]
    if mineru_status["status"] == "completed":
        blocks = _load_mineru_blocks(parsed_dir, document_id)
        if not blocks:
            parser_status = "fallback"
            parser_message = "MinerU completed but no usable blocks were found; used pdftotext fallback."

    if not blocks:
        parser = "pdftotext_fallback"
        blocks = _parse_with_pdftotext(pdf_path, document_id)
        if not blocks:
            parser_status = "failed"
            parser_message = "No parsed blocks were produced by MinerU or pdftotext fallback."
        elif mineru_status["status"] != "completed":
            parser_status = "completed"
            parser_message = f"{mineru_status['message']} Used pdftotext fallback."

    page_count = _page_count_for_pdf(pdf_path, blocks)
    markdown = "\n\n".join(block["markdown_content"] for block in blocks if block.get("markdown_content")).strip()
    (parsed_dir / "document.md").write_text(markdown + ("\n" if markdown else ""), encoding="utf-8")
    content_list = [_content_item_from_block(block) for block in blocks]
    (parsed_dir / "content_list.json").write_text(json.dumps(content_list, indent=2), encoding="utf-8")
    rag_status = prepare_paper_memory(document_id, document_dir, blocks)
    blocks_payload = {
        "document_id": document_id,
        "parser": parser,
        "status": parser_status,
        "message": parser_message,
        "page_count": page_count,
        "blocks": blocks,
    }
    (parsed_dir / "blocks_index.json").write_text(json.dumps(blocks_payload, indent=2), encoding="utf-8")
    return {
        "document_id": document_id,
        "status": parser_status,
        "parser": parser,
        "message": parser_message,
        "parsed_dir": str(parsed_dir),
        "document_md_path": str(parsed_dir / "document.md"),
        "content_list_path": str(parsed_dir / "content_list.json"),
        "blocks_index_path": str(parsed_dir / "blocks_index.json"),
        "images_dir": str(images_dir),
        "page_count": page_count,
        "block_count": len(blocks),
        "duration_sec": round(time.time() - started, 3),
        **rag_status,
    }


def load_blocks(blocks_index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(blocks_index_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("blocks"), list):
        return payload["blocks"]
    return []


def match_blocks_for_rects(
    blocks: list[dict[str, Any]],
    page_number: int,
    rects: list[dict[str, Any]],
    selected_text: str = "",
    min_overlap: float = 0.08,
) -> dict[str, Any]:
    highlight_page_number = _coerce_page_number(page_number)
    normalized_rects = [_normalized_rect(rect) for rect in rects]
    normalized_rects = [rect for rect in normalized_rects if rect]
    page_blocks = [
        block for block in blocks
        if _block_page_number(block) == highlight_page_number
    ]
    caption_result = _caption_candidates_for_rects(page_blocks, normalized_rects, highlight_page_number)
    scored: list[tuple[float, dict[str, Any]]] = []
    for block in page_blocks:
        bbox = _normalized_rect(block.get("bbox") or {})
        if not bbox:
            continue
        coordinate_overlap = max((_overlap_score(bbox, rect) for rect in normalized_rects), default=0.0)
        text_similarity = _text_similarity_bonus(str(block.get("markdown_content") or ""), selected_text) if selected_text else 0.0
        text_bonus = min(text_similarity, 0.05) if coordinate_overlap > 0 else 0.0
        score = min(1.0, coordinate_overlap + text_bonus) if coordinate_overlap > 0 else 0.0
        if score > 0:
            enriched = _block_with_explicit_page(block)
            enriched["coordinate_overlap"] = round(min(coordinate_overlap, 1.0), 4)
            enriched["text_bonus"] = round(min(text_bonus, 1.0), 4)
            enriched["match_score"] = round(score, 4)
            enriched["overlap_score"] = round(score, 4)
            enriched["coordinate_overlap_score"] = round(min(coordinate_overlap, 1.0), 4)
            enriched["selected_text_similarity"] = round(text_similarity, 4)
            scored.append((score, enriched))
    scored.sort(key=lambda item: (-item[0], int(item[1].get("reading_order_index", 0))))
    matched = [block for score, block in scored if score >= min_overlap][:5]
    page_ordered = sorted(page_blocks, key=_block_order_key)
    if matched:
        previous_blocks, next_blocks = _neighbor_blocks_for_anchor(page_ordered, matched[0])
    else:
        previous_blocks, next_blocks = _nearest_same_page_context(page_ordered, normalized_rects)
    return {
        "matched_blocks": matched,
        "previous_blocks": previous_blocks,
        "next_blocks": next_blocks,
        "fallback_required": not matched or float(matched[0].get("match_score", 0.0)) < min_overlap,
        "match_strategy": "page_bbox_overlap",
        "context_page_policy": "same_page_only",
        "highlight_page_number": highlight_page_number,
        **caption_result,
    }


def _run_mineru_if_available(pdf_path: Path, parsed_dir: Path) -> dict[str, str]:
    candidates = []
    mineru = shutil.which("mineru")
    if mineru:
        candidates.append([mineru, "-p", str(pdf_path), "-o", str(parsed_dir)])
    magic_pdf = shutil.which("magic-pdf")
    if magic_pdf:
        candidates.append([magic_pdf, "-p", str(pdf_path), "-o", str(parsed_dir), "-m", "auto"])
    marker_single = shutil.which("marker_single")
    if marker_single:
        candidates.append([marker_single, str(pdf_path), "--output_dir", str(parsed_dir)])
    marker = shutil.which("marker")
    if marker and marker != marker_single:
        candidates.append([marker, str(pdf_path), "--output_dir", str(parsed_dir)])
    if not candidates:
        return {"status": "missing", "message": "MinerU/Marker command not found."}
    messages = []
    for command in candidates:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        except Exception as exc:
            messages.append(f"{Path(command[0]).name}: {exc}")
            continue
        if result.returncode == 0:
            return {"status": "completed", "message": f"External parser command succeeded: {Path(command[0]).name}"}
        messages.append(f"{Path(command[0]).name}: exit {result.returncode}: {(result.stderr or result.stdout)[:500]}")
    return {"status": "failed", "message": "External PDF parser failed. " + " | ".join(messages)}


def _load_mineru_blocks(parsed_dir: Path, document_id: str) -> list[dict[str, Any]]:
    content_path = _first_existing(parsed_dir, ["content_list.json", "*content_list*.json", "*.json"])
    if not content_path:
        return []
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload if isinstance(payload, list) else payload.get("content") or payload.get("blocks") or []
    blocks: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        bbox = _bbox_from_any(item.get("bbox") or item.get("poly") or item.get("position") or {})
        if not bbox:
            continue
        bbox = _scale_0_1000_bbox(bbox)
        page_number = _parser_item_page_number(item)
        block_type = _normalize_block_type(str(item.get("type") or item.get("block_type") or "text"))
        text = _mineru_markdown_content(item)
        asset_path = item.get("img_path") or item.get("image_path") or item.get("asset_path")
        blocks.append(_block(document_id, index, page_number, bbox, block_type, text, asset_path))
    return blocks


def _parse_with_pdftotext(pdf_path: Path, document_id: str) -> list[dict[str, Any]]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return []
    result = subprocess.run(
        [pdftotext, "-bbox-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    root = ET.fromstring(result.stdout)
    blocks: list[dict[str, Any]] = []
    order = 0
    for page_idx, page in enumerate(_children_named(root, "page")):
        page_width = _float(page.attrib.get("width"), 1.0)
        page_height = _float(page.attrib.get("height"), 1.0)
        for raw_block in page.iter():
            if not _tag_name(raw_block.tag) == "block":
                continue
            words = [
                (word.text or "").strip()
                for word in raw_block.iter()
                if _tag_name(word.tag) == "word" and (word.text or "").strip()
            ]
            text = " ".join(words).strip()
            if not text:
                continue
            bbox = {
                "x1": _float(raw_block.attrib.get("xMin"), 0.0) / page_width,
                "y1": _float(raw_block.attrib.get("yMin"), 0.0) / page_height,
                "x2": _float(raw_block.attrib.get("xMax"), 0.0) / page_width,
                "y2": _float(raw_block.attrib.get("yMax"), 0.0) / page_height,
            }
            block_type = "title" if page_idx == 0 and order < 2 and len(text) < 180 else _heuristic_block_type(text)
            blocks.append(_block(document_id, order, page_idx + 1, bbox, block_type, text, None))
            order += 1
    return blocks


def _page_count_for_pdf(pdf_path: Path, blocks: list[dict[str, Any]]) -> int:
    candidates = [_page_count_from_blocks(blocks), _page_count_from_pdfinfo(pdf_path)]
    return max((value for value in candidates if value > 0), default=0)


def _page_count_from_blocks(blocks: list[dict[str, Any]]) -> int:
    return max((_block_page_number(block) for block in blocks), default=0)


def _page_count_from_pdfinfo(pdf_path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return 0
    try:
        result = subprocess.run(
            [pdfinfo, str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else 0


def _children_named(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _tag_name(element.tag) == name]


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _block(
    document_id: str,
    index: int,
    page_number: int,
    bbox: dict[str, float],
    block_type: str,
    markdown_content: str,
    asset_path: str | None,
) -> dict[str, Any]:
    page_idx = page_number - 1
    return {
        "block_id": f"{document_id}-p{page_number}-b{index}",
        "page_idx": page_idx,
        "page_number": page_number,
        "bbox": {key: round(float(value), 6) for key, value in bbox.items()},
        "block_type": _normalize_block_type(block_type),
        "markdown_content": normalize_pdf_text(markdown_content),
        "asset_path": str(asset_path) if asset_path else None,
        "reading_order_index": index,
    }


def _content_item_from_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": block["block_type"],
        "page_idx": block["page_idx"],
        "bbox": block["bbox"],
        "text": block["markdown_content"],
        "asset_path": block.get("asset_path"),
    }


def _coerce_page_number(value: Any) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def _block_page_number(block: dict[str, Any]) -> int:
    if block.get("page_number") is not None:
        return _coerce_page_number(block.get("page_number"))
    if block.get("page_idx") is not None:
        try:
            return max(1, int(float(block.get("page_idx"))) + 1)
        except (TypeError, ValueError):
            return 1
    return 1


def _block_with_explicit_page(block: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(block)
    page_number = _block_page_number(block)
    enriched["page_number"] = page_number
    enriched["page_idx"] = page_number - 1
    return enriched


def _parser_item_page_number(item: dict[str, Any]) -> int:
    if item.get("page_idx") is not None:
        try:
            return max(1, int(float(item.get("page_idx"))) + 1)
        except (TypeError, ValueError):
            return 1
    if item.get("page_number") is not None:
        return _coerce_page_number(item.get("page_number"))
    if item.get("page") is not None:
        try:
            page = int(float(item.get("page")))
        except (TypeError, ValueError):
            return 1
        return 1 if page <= 0 else page
    return 1


def _block_order_key(block: dict[str, Any]) -> tuple[int, float, str]:
    return (
        int(float(block.get("reading_order_index", 0) or 0)),
        _block_center_y(block),
        str(block.get("block_id") or ""),
    )


def _neighbor_blocks_for_anchor(
    page_ordered: list[dict[str, Any]],
    anchor: dict[str, Any],
    limit: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_id = str(anchor.get("block_id") or "")
    anchor_order = int(float(anchor.get("reading_order_index", -1) or -1))
    anchor_index = -1
    for index, block in enumerate(page_ordered):
        if anchor_id and str(block.get("block_id") or "") == anchor_id:
            anchor_index = index
            break
        if int(float(block.get("reading_order_index", -2) or -2)) == anchor_order:
            anchor_index = index
            break
    if anchor_index < 0:
        return _nearest_same_page_context(page_ordered, [])
    previous_blocks = [_block_with_explicit_page(block) for block in page_ordered[max(0, anchor_index - limit):anchor_index]]
    next_blocks = [_block_with_explicit_page(block) for block in page_ordered[anchor_index + 1:anchor_index + 1 + limit]]
    return previous_blocks, next_blocks


def _nearest_same_page_context(
    page_ordered: list[dict[str, Any]],
    normalized_rects: list[dict[str, float]],
    limit: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not page_ordered:
        return [], []
    if not normalized_rects:
        return [], [_block_with_explicit_page(block) for block in page_ordered[:limit]]

    selection_center_y = _rect_center_y(_union_rect(normalized_rects))
    previous: list[dict[str, Any]] = []
    next_blocks: list[dict[str, Any]] = []
    for block in page_ordered:
        center_y = _block_center_y(block)
        if center_y <= selection_center_y:
            previous.append(block)
        else:
            next_blocks.append(block)
    return (
        [_block_with_explicit_page(block) for block in previous[-limit:]],
        [_block_with_explicit_page(block) for block in next_blocks[:limit]],
    )


def _caption_candidates_for_rects(
    page_blocks: list[dict[str, Any]],
    normalized_rects: list[dict[str, float]],
    page_number: int,
    limit: int = 5,
) -> dict[str, Any]:
    if not normalized_rects:
        return {
            "selected_caption": {},
            "caption_confidence": "none",
            "selected_caption_confidence": "none",
            "candidate_captions": [],
        }
    area = _union_rect(normalized_rects)
    candidates = []
    for block in page_blocks:
        if not _is_caption_block(block):
            continue
        bbox = _normalized_rect(block.get("bbox") or {})
        if not bbox:
            continue
        horizontal_overlap = _horizontal_overlap(area, bbox)
        relation, vertical_distance = _vertical_relation(area, bbox)
        if horizontal_overlap < 0.10 or vertical_distance > 0.35:
            continue
        score = _caption_score(area, bbox, horizontal_overlap, vertical_distance, relation)
        candidates.append(
            {
                "block_id": str(block.get("block_id") or ""),
                "page_number": page_number,
                "markdown_content": str(block.get("markdown_content") or ""),
                "horizontal_overlap": round(horizontal_overlap, 4),
                "vertical_distance": round(vertical_distance, 4),
                "relation": relation,
                "score": round(score, 4),
                "reading_order_index": int(float(block.get("reading_order_index", 0) or 0)),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["vertical_distance"]), item["reading_order_index"]))
    candidates = candidates[:limit]
    confidence = _caption_confidence(candidates)
    selected_caption = _caption_public(candidates[0]) if confidence in {"medium", "high"} and candidates else {}
    return {
        "selected_caption": selected_caption,
        "caption_confidence": confidence,
        "selected_caption_confidence": confidence,
        "candidate_captions": [_caption_public(candidate) for candidate in candidates],
    }


def _caption_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_id": candidate.get("block_id") or "",
        "page_number": candidate.get("page_number") or "",
        "markdown_content": candidate.get("markdown_content") or "",
        "horizontal_overlap": candidate.get("horizontal_overlap", 0.0),
        "vertical_distance": candidate.get("vertical_distance", 0.0),
        "relation": candidate.get("relation") or "",
        "score": candidate.get("score", 0.0),
    }


def _caption_confidence(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "none"
    top = float(candidates[0].get("score", 0.0) or 0.0)
    second = float(candidates[1].get("score", 0.0) or 0.0) if len(candidates) > 1 else 0.0
    margin = top - second
    if top >= 0.78 and (len(candidates) == 1 or margin >= 0.12):
        return "high"
    if top >= 0.80 and margin >= 0.05:
        return "medium"
    if top >= 0.58 and (len(candidates) == 1 or margin >= 0.07):
        return "medium"
    return "low"


def _is_caption_block(block: dict[str, Any]) -> bool:
    block_type = str(block.get("block_type") or "").lower()
    text = str(block.get("markdown_content") or "").strip()
    return block_type == "caption" or bool(re.match(r"^(fig\.?|figure|table)\s*\d+", text, re.IGNORECASE))


def _horizontal_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(float(a["x1"]), float(b["x1"]))
    right = min(float(a["x2"]), float(b["x2"]))
    if right <= left:
        return 0.0
    width_a = max(float(a["x2"]) - float(a["x1"]), 1e-9)
    width_b = max(float(b["x2"]) - float(b["x1"]), 1e-9)
    return max(0.0, min((right - left) / min(width_a, width_b), 1.0))


def _vertical_relation(area: dict[str, float], caption: dict[str, float]) -> tuple[str, float]:
    if float(caption["y1"]) >= float(area["y2"]):
        return "below", float(caption["y1"]) - float(area["y2"])
    if float(caption["y2"]) <= float(area["y1"]):
        return "above", float(area["y1"]) - float(caption["y2"])
    return "overlapping", 0.0


def _caption_score(
    area: dict[str, float],
    caption: dict[str, float],
    horizontal_overlap: float,
    vertical_distance: float,
    relation: str,
) -> float:
    proximity = max(0.0, 1.0 - vertical_distance / 0.35)
    area_center = (float(area["x1"]) + float(area["x2"])) / 2.0
    caption_center = (float(caption["x1"]) + float(caption["x2"])) / 2.0
    region_width = max(float(area["x2"]) - float(area["x1"]), float(caption["x2"]) - float(caption["x1"]), 1e-9)
    same_region = max(0.0, 1.0 - abs(area_center - caption_center) / region_width)
    relation_bonus = {"below": 0.05, "overlapping": 0.03, "above": 0.0}.get(relation, 0.0)
    return min(1.0, 0.55 * horizontal_overlap + 0.25 * proximity + 0.15 * same_region + relation_bonus)


def _block_center_y(block: dict[str, Any]) -> float:
    bbox = _normalized_rect(block.get("bbox") or {})
    if not bbox:
        return 0.0
    return _rect_center_y(bbox)


def _rect_center_y(rect: dict[str, float]) -> float:
    return (float(rect["y1"]) + float(rect["y2"])) / 2.0


def _union_rect(rects: list[dict[str, float]]) -> dict[str, float]:
    return {
        "x1": min(float(rect["x1"]) for rect in rects),
        "y1": min(float(rect["y1"]) for rect in rects),
        "x2": max(float(rect["x2"]) for rect in rects),
        "y2": max(float(rect["y2"]) for rect in rects),
    }


def _mineru_markdown_content(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, dict):
        for key in ("text", "title_content", "table_body", "html", "latex", "code_body"):
            if content.get(key):
                return str(content[key]).strip()
        for key in ("caption", "img_caption", "table_caption", "chart_caption"):
            if isinstance(content.get(key), list):
                return " ".join(str(value) for value in content[key]).strip()
            if content.get(key):
                return str(content[key]).strip()
    if isinstance(content, str) and content.strip():
        return content.strip()
    for key in (
        "md",
        "markdown",
        "text",
        "html",
        "latex",
        "table_body",
        "table_caption",
        "img_caption",
        "chart_caption",
        "code_body",
    ):
        value = item.get(key)
        if isinstance(value, list):
            value = " ".join(str(entry) for entry in value)
        if value:
            return str(value).strip()
    return ""


def _first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        direct = root / pattern
        if "*" not in pattern and direct.exists():
            return direct
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _bbox_from_any(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        return _normalized_rect(value)
    if isinstance(value, list) and len(value) >= 4:
        if all(isinstance(item, (int, float)) for item in value[:4]):
            x_values = [float(value[0]), float(value[2])]
            y_values = [float(value[1]), float(value[3])]
            return _normalize_bbox_values(min(x_values), min(y_values), max(x_values), max(y_values))
        if all(isinstance(item, list) and len(item) >= 2 for item in value):
            xs = [float(item[0]) for item in value]
            ys = [float(item[1]) for item in value]
            return _normalize_bbox_values(min(xs), min(ys), max(xs), max(ys))
    return None


def _normalized_rect(rect: dict[str, Any]) -> dict[str, float] | None:
    if not isinstance(rect, dict):
        return None
    if {"x1", "y1", "x2", "y2"}.issubset(rect):
        return _normalize_rect_values(rect["x1"], rect["y1"], rect["x2"], rect["y2"], rect)
    if {"left", "top", "width", "height"}.issubset(rect):
        left = _float(rect["left"], 0.0)
        top = _float(rect["top"], 0.0)
        return _normalize_rect_values(left, top, left + _float(rect["width"], 0.0), top + _float(rect["height"], 0.0), rect)
    return None


def _normalize_rect_values(x1: Any, y1: Any, x2: Any, y2: Any, source: dict[str, Any]) -> dict[str, float]:
    left = min(float(x1), float(x2))
    top = min(float(y1), float(y2))
    right = max(float(x1), float(x2))
    bottom = max(float(y1), float(y2))
    if right <= 1.0 and bottom <= 1.0:
        return _normalize_bbox_values(left, top, right, bottom)

    page_width = _float(source.get("width"), 0.0)
    page_height = _float(source.get("height"), 0.0)
    if page_width > 1.0 and page_height > 1.0 and right <= page_width * 1.1 and bottom <= page_height * 1.1:
        return _normalize_bbox_values(left / page_width, top / page_height, right / page_width, bottom / page_height)

    max_value = max(right, bottom)
    if max_value <= 1000.0:
        return _normalize_bbox_values(left / 1000.0, top / 1000.0, right / 1000.0, bottom / 1000.0)

    return _normalize_bbox_values(left, top, right, bottom)


def _normalize_bbox_values(x1: Any, y1: Any, x2: Any, y2: Any) -> dict[str, float]:
    left = max(0.0, min(float(x1), float(x2)))
    top = max(0.0, min(float(y1), float(y2)))
    right = max(left, max(float(x1), float(x2)))
    bottom = max(top, max(float(y1), float(y2)))
    if right > 1.0 or bottom > 1.0:
        return {"x1": left, "y1": top, "x2": right, "y2": bottom}
    return {"x1": left, "y1": top, "x2": min(right, 1.0), "y2": min(bottom, 1.0)}


def _scale_0_1000_bbox(bbox: dict[str, float]) -> dict[str, float]:
    if max(float(value) for value in bbox.values()) <= 1.0:
        return bbox
    return {key: min(max(round(float(value) / 1000.0, 6), 0.0), 1.0) for key, value in bbox.items()}


def _overlap_score(a: dict[str, float], b: dict[str, float]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    a_area = max((a["x2"] - a["x1"]) * (a["y2"] - a["y1"]), 1e-9)
    b_area = max((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), 1e-9)
    return intersection / min(a_area, b_area)


def _text_similarity_bonus(markdown_content: str, selected_text: str) -> float:
    words = {word.lower() for word in selected_text.split() if len(word) > 3}
    if not words:
        return 0.0
    block_words = {word.lower() for word in markdown_content.split()}
    return min(0.5, len(words & block_words) / max(len(words), 1))


def _heuristic_block_type(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith(("figure ", "fig. ")):
        return "caption"
    if lowered.startswith("table "):
        return "table"
    if stripped.startswith(("-", "•")):
        return "list"
    return "text"


def _normalize_block_type(value: str) -> str:
    lowered = value.lower().replace("_", "-")
    if "title" in lowered or lowered in {"heading", "header"}:
        return "title"
    if "table" in lowered:
        return "table"
    if "formula" in lowered or "equation" in lowered:
        return "formula"
    if "image" in lowered or "figure" in lowered or "chart" in lowered:
        return "image"
    if "caption" in lowered:
        return "caption"
    if "list" in lowered:
        return "list"
    if "footnote" in lowered:
        return "footnote"
    return "text"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
