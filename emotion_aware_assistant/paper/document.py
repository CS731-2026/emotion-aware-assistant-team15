from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Page:
    page_number: int
    text: str
    heading: str | None = None
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    section_hint: str | None = None


@dataclass
class Document:
    title: str
    source_path: Path
    pages: list[Page]
    chunks: list[Chunk] = field(default_factory=list)
    current_page: int = 1
    metadata: dict[str, str] = field(default_factory=dict)
    section_hints: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, page_number: int) -> Page:
        if page_number < 1 or page_number > len(self.pages):
            raise IndexError(f"Page {page_number} is outside 1..{len(self.pages)}")
        return self.pages[page_number - 1]

    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    def section_for_page(self, page_number: int) -> str | None:
        try:
            return self.page(page_number).heading
        except IndexError:
            return None
