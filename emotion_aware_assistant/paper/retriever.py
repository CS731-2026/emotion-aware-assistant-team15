from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, asdict

from .document import Chunk, Document


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class ContextRetriever:
    def __init__(self, document: Document, top_k: int = 3):
        self.document = document
        self.top_k = top_k
        self._vectorizer = None
        self._matrix = None
        self._fit_tfidf()

    def _fit_tfidf(self) -> None:
        if not self.document.chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception:
            return
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.document.chunks])

    def retrieve(
        self,
        query: str,
        selected_text: str = "",
        page_number: int | None = None,
        top_k: int | None = None,
    ) -> list[Chunk]:
        if not self.document.chunks:
            return []
        limit = top_k or self.top_k
        search_text = "\n".join(part for part in [selected_text, query] if part).strip()
        if not search_text:
            search_text = selected_text or query or self.document.title

        if self._vectorizer is not None and self._matrix is not None:
            return self._retrieve_tfidf(search_text, page_number, limit)
        return self._retrieve_keyword(search_text, page_number, limit)

    def retrieve_with_debug(
        self,
        query: str,
        selected_text: str = "",
        page_number: int | None = None,
        top_k: int | None = None,
    ) -> dict:
        limit = top_k or self.top_k
        search_text = "\n".join(part for part in [selected_text, query] if part).strip()
        if not search_text:
            search_text = selected_text or query or self.document.title
        if not self.document.chunks:
            return {
                "chunks": [],
                "ranked_chunks": [],
                "method": "none",
                "included_sources": self._included_sources(selected_text, query, page_number),
            }
        if self._vectorizer is not None and self._matrix is not None:
            ranked = self._rank_tfidf(search_text, page_number)
        else:
            ranked = self._rank_keyword(search_text, page_number)
        selected = [item.chunk for item in ranked[:limit] if item.score > 0] or self.document.chunks[:limit]
        return {
            "chunks": selected,
            "ranked_chunks": [item.as_debug_dict() for item in ranked[: max(limit, 5)]],
            "method": ranked[0].method if ranked else "none",
            "included_sources": self._included_sources(selected_text, query, page_number),
        }

    def _retrieve_tfidf(self, search_text: str, page_number: int | None, limit: int) -> list[Chunk]:
        ranked = self._rank_tfidf(search_text, page_number)
        return [item.chunk for item in ranked[:limit] if item.score > 0][:limit] or self.document.chunks[:limit]

    def _retrieve_keyword(self, search_text: str, page_number: int | None, limit: int) -> list[Chunk]:
        query_counts = Counter(_tokens(search_text))
        if not query_counts:
            return self.document.chunks[:limit]
        ranked: list[tuple[float, Chunk]] = []
        for chunk in self.document.chunks:
            chunk_counts = Counter(_tokens(chunk.text))
            overlap = sum(min(count, chunk_counts[token]) for token, count in query_counts.items())
            norm = math.sqrt(sum(v * v for v in chunk_counts.values())) or 1.0
            score = overlap / norm + self._page_bonus(chunk, page_number)
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in ranked[:limit] if score > 0][:limit] or self.document.chunks[:limit]

    def _rank_tfidf(self, search_text: str, page_number: int | None) -> list[RetrievalScore]:
        query_vec = self._vectorizer.transform([search_text])
        scores = (self._matrix @ query_vec.T).toarray().ravel()
        query_terms = set(_tokens(search_text))
        ranked: list[RetrievalScore] = []
        for index, base in enumerate(scores):
            chunk = self.document.chunks[index]
            bonus = self._page_bonus(chunk, page_number)
            matched = sorted(query_terms.intersection(_tokens(chunk.text)))[:12]
            ranked.append(RetrievalScore(chunk, float(base) + bonus, float(base), bonus, "tfidf", matched))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _rank_keyword(self, search_text: str, page_number: int | None) -> list[RetrievalScore]:
        query_counts = Counter(_tokens(search_text))
        ranked: list[RetrievalScore] = []
        for chunk in self.document.chunks:
            chunk_counts = Counter(_tokens(chunk.text))
            matched = sorted(token for token in query_counts if token in chunk_counts)[:12]
            overlap = sum(min(count, chunk_counts[token]) for token, count in query_counts.items())
            norm = math.sqrt(sum(v * v for v in chunk_counts.values())) or 1.0
            base = overlap / norm
            bonus = self._page_bonus(chunk, page_number)
            ranked.append(RetrievalScore(chunk, base + bonus, base, bonus, "keyword_overlap", matched))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    @staticmethod
    def _page_bonus(chunk: Chunk, page_number: int | None) -> float:
        return 0.15 if page_number is not None and chunk.page_number == page_number else 0.0

    @staticmethod
    def _included_sources(selected_text: str, query: str, page_number: int | None) -> list[str]:
        sources = []
        if selected_text:
            sources.append("selected_text")
        if query:
            sources.append("user_question")
        if page_number is not None:
            sources.append("page_bonus")
        return sources or ["document_title"]
@dataclass(frozen=True)
class RetrievalScore:
    chunk: Chunk
    score: float
    base_score: float
    page_bonus: float
    method: str
    matched_terms: list[str]

    def as_debug_dict(self) -> dict:
        data = asdict(self)
        data["chunk"] = {
            "chunk_id": self.chunk.chunk_id,
            "page_number": self.chunk.page_number,
            "section_hint": self.chunk.section_hint,
            "text_preview": self.chunk.text[:220],
        }
        return data
