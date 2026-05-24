from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def learning_state_payload(state) -> dict[str, Any]:
    payload = jsonable(state)
    payload["source_mode"] = "manual" if state.manual_override else "auto"
    return payload


def document_payload(document, current_page_text: str, current_page: int) -> dict[str, Any]:
    return {
        "title": document.title,
        "source_path": str(document.source_path),
        "page_count": document.page_count,
        "current_page": current_page,
        "current_page_text": current_page_text,
        "pages": [{"page_number": page.page_number, "text_length": len(page.text)} for page in document.pages],
    }


def error_payload(message: str, status: int = 400) -> dict[str, Any]:
    return {"status": status, "json": {"error": message}}

