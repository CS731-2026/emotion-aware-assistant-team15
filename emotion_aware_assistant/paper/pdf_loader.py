from __future__ import annotations

import re
from pathlib import Path

from emotion_aware_assistant.core.errors import DocumentLoadError, MissingOptionalDependency

from .document import Document, Page
from .text_chunker import chunk_document


_HEADING_RE = re.compile(r"^\s*(title|abstract|introduction|background|related work|method|methods|methodology|formula|results?|discussion|limitations?|future work|conclusion|evaluation|dataset)\b.*", re.IGNORECASE)


def _load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _extract_txt_title(text, path)
    pages = _txt_pages(text)
    section_hints = [page.heading for page in pages if page.heading]
    document = Document(
        title=title,
        source_path=path,
        pages=pages,
        metadata={"format": "txt", "source_name": path.name},
        section_hints=section_hints,
    )
    chunk_document(document)
    return document


def _load_pdf(path: Path) -> Document:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise MissingOptionalDependency(
            "PDF loading requires PyMuPDF. Install with `pip install pymupdf`, "
            "or use TXT mode."
        ) from exc

    pages: list[Page] = []
    with fitz.open(path) as pdf:
        metadata = pdf.metadata or {}
        title = (metadata.get("title") or path.stem).strip() or path.stem
        offset = 0
        for index, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            heading = _first_heading(page_text)
            pages.append(Page(page_number=index, text=page_text, heading=heading, start_char=offset, end_char=offset + len(page_text)))
            offset += len(page_text) + 2
    document = Document(
        title=title,
        source_path=path,
        pages=pages,
        metadata={str(key): str(value) for key, value in metadata.items() if value},
        section_hints=[page.heading for page in pages if page.heading],
    )
    chunk_document(document)
    return document


def load_document(path: str | Path) -> Document:
    doc_path = Path(path).expanduser().resolve()
    if not doc_path.exists():
        raise DocumentLoadError(f"Document not found: {doc_path}")
    suffix = doc_path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(doc_path)
    if suffix == ".pdf":
        return _load_pdf(doc_path)
    raise DocumentLoadError("Only PDF and TXT documents are supported.")


def _extract_txt_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("title:"):
            return stripped.split(":", 1)[1].strip() or path.stem
        if len(stripped) < 120:
            return stripped
    return path.stem.replace("_", " ").strip() or path.name


def _txt_pages(text: str, max_chars: int = 3200) -> list[Page]:
    headings = _section_headings(text)
    if len(text) <= max_chars:
        heading = _first_heading(text)
        return [Page(page_number=1, text=text, heading=heading, start_char=0, end_char=len(text))]

    pages: list[Page] = []
    start = 0
    page_number = 1
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        page_text = text[start:end].strip()
        heading = _heading_for_offset(headings, start) or _first_heading(page_text)
        pages.append(Page(page_number=page_number, text=page_text, heading=heading, start_char=start, end_char=end))
        start = end
        page_number += 1
    return pages or [Page(page_number=1, text=text, heading=_first_heading(text), start_char=0, end_char=len(text))]


def _section_headings(text: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and (_HEADING_RE.match(stripped) or stripped.lower().startswith("title:")):
            headings.append((offset, stripped.replace("Title:", "").strip()))
        offset += len(line)
    return headings


def _heading_for_offset(headings: list[tuple[int, str]], offset: int) -> str | None:
    current = None
    for position, heading in headings:
        if position <= offset:
            current = heading
        else:
            break
    return current


def _first_heading(text: str) -> str | None:
    for line in text.splitlines()[:20]:
        stripped = line.strip()
        if stripped and (_HEADING_RE.match(stripped) or stripped.lower().startswith("title:")):
            return stripped.replace("Title:", "").strip()
    return None
