from __future__ import annotations

from .document import Chunk, Document


def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for page in document.pages:
        text = page.text.strip()
        if not text:
            continue
        start = 0
        local_index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"p{page.page_number}-{local_index}",
                        page_number=page.page_number,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        section_hint=page.heading,
                    )
                )
            if end == len(text):
                break
            start += step
            local_index += 1
    document.chunks = chunks
    return chunks
