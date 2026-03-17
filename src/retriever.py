"""
Lightweight lexical retrieval module for RAG over document chunks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
import re
from typing import Dict, List


_TOKEN_PATTERN = re.compile(r"\b\w+\b")


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str


class ChunkRetriever:
    """
    A simple BM25-like lexical retriever over chunked text.
    """

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.doc_tokens = [self._tokenize(chunk) for chunk in chunks]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        )
        self.doc_freqs = self._compute_document_frequencies(self.doc_term_freqs)

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_PATTERN.findall(text.lower())

    def _compute_document_frequencies(self, term_freqs: List[Counter]) -> Dict[str, int]:
        doc_freqs: Dict[str, int] = {}
        for freqs in term_freqs:
            for term in freqs.keys():
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        return doc_freqs

    def _bm25_score(
        self,
        query_tokens: List[str],
        term_freqs: Counter,
        doc_length: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if doc_length == 0 or self.avg_doc_length == 0:
            return 0.0

        score = 0.0
        num_docs = len(self.doc_term_freqs)

        for token in query_tokens:
            term_frequency = term_freqs.get(token, 0)
            if term_frequency == 0:
                continue

            document_frequency = self.doc_freqs.get(token, 0)
            idf = log((num_docs - document_frequency + 0.5) / (document_frequency + 0.5) + 1)

            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve top-k chunks relevant to the query.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored_chunks: List[RetrievedChunk] = []
        for index, (term_freqs, doc_length) in enumerate(
            zip(self.doc_term_freqs, self.doc_lengths)
        ):
            score = self._bm25_score(
                query_tokens=query_tokens, term_freqs=term_freqs, doc_length=doc_length
            )
            if score > 0:
                scored_chunks.append(
                    RetrievedChunk(chunk_id=index, score=score, text=self.chunks[index])
                )

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[: max(1, top_k)]
