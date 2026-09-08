from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache


@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def features(text: str) -> Counter[str]:
    text = normalize(text)
    terms = Counter("w:" + word for word in text.split())
    # Whole-text character features also work for scripts without word spaces.
    terms.update("c:" + text[i:i + 3] for i in range(max(0, len(text) - 2)))
    return terms


class LexicalIndex:
    def __init__(self, documents: tuple[tuple[str, ...], ...]) -> None:
        self.documents = documents
        counts = [features(" ".join(parts)) for parts in documents]
        frequency = Counter(term for document in counts for term in document)
        self.idf = {term: math.log(1 + len(counts) / count) for term, count in frequency.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for index, document in enumerate(counts):
            weights = self._weights(document)
            for term, weight in weights.items():
                self.postings[term].append((index, weight))

    def _weights(self, counts: Counter[str]) -> dict[str, float]:
        weights = {
            term: (1 + math.log(count)) * self.idf[term] * (1 if term.startswith("w:") else 0.35)
            for term, count in counts.items() if term in self.idf
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        return {term: weight / norm for term, weight in weights.items()} if norm else {}

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self.documents)
        for term, weight in self._weights(features(query)).items():
            for index, document_weight in self.postings[term]:
                scores[index] += weight * document_weight
        return scores


@lru_cache(maxsize=8)
def lexical_index(documents: tuple[tuple[str, ...], ...]) -> LexicalIndex:
    """Rebuild only when searchable metadata changes; retain no query history."""
    return LexicalIndex(documents)
