from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from custom_components.ha_ragent.src.const import CANONICAL_NAME_SPLIT_PATTERN

MatchFeatures = tuple[str, frozenset[str], frozenset[str]]

@lru_cache(maxsize=8)
def lexical_index(documents: tuple[tuple[str, ...], ...]) -> LexicalIndex:
    """Rebuild only when searchable metadata changes; retain no query history."""
    return LexicalIndex(documents)

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


@lru_cache(maxsize=32768)
def canonical_normalize(text: str) -> str:
    return normalize(re.sub(CANONICAL_NAME_SPLIT_PATTERN, " ", text))


@lru_cache(maxsize=32768)
def match_features(text: str, size: int = 3) -> MatchFeatures:
    normalized = canonical_normalize(text)
    compact = normalized.replace(" ", "")
    grams = frozenset(compact[i:i + size] for i in range(max(1, len(compact) - size + 1))) if compact else frozenset()
    return normalized, frozenset(normalized.split()), grams

def match_score(query_features: MatchFeatures, value_features: MatchFeatures) -> tuple[float, float]:
    query, query_tokens, query_grams = query_features
    value, value_tokens, value_grams = value_features
    if not value or not query:
        return 0.0, 0.0
    if value == query:
        exact = 1.0
    elif f" {value} " in f" {query} ":
        exact = 0.9
    else:
        exact = len(query_tokens & value_tokens) / len(value_tokens) if value_tokens else 0.0
    denominator = len(query_grams) + len(value_grams)
    fuzzy = 2.0 * len(query_grams & value_grams) / denominator if denominator else 0.0
    return exact, fuzzy


class FieldIndex:
    def __init__(self, documents: tuple[tuple[str, ...], ...], size: int = 3) -> None:
        self.size = size
        self.fields: list[MatchFeatures] = []
        self.documents: list[list[int]] = []
        self.postings: dict[str, set[int]] = defaultdict(set)
        field_ids: dict[str, int] = {}
        for document_id, parts in enumerate(documents):
            for part in set(parts):
                if part not in field_ids:
                    field_id = len(self.fields)
                    field_ids[part] = field_id
                    prepared = match_features(part, size)
                    self.fields.append(prepared)
                    self.documents.append([])
                    for term in prepared[1]:
                        self.postings["w:" + term].add(field_id)
                    for gram in prepared[2]:
                        self.postings["c:" + gram].add(field_id)
                self.documents[field_ids[part]].append(document_id)

    def scores(self, query: str) -> dict[int, tuple[float, float]]:
        prepared = match_features(query, self.size)
        matching: set[int] = set()
        for term in prepared[1]:
            matching.update(self.postings.get("w:" + term, ()))
        for gram in prepared[2]:
            matching.update(self.postings.get("c:" + gram, ()))
        scores: dict[int, tuple[float, float]] = {}
        for field_id in matching:
            exact, fuzzy = match_score(prepared, self.fields[field_id])
            for document_id in self.documents[field_id]:
                previous = scores.get(document_id, (0.0, 0.0))
                scores[document_id] = max(previous[0], exact), max(previous[1], fuzzy)
        return scores


class LexicalIndex:
    def __init__(self, documents: tuple[tuple[str, ...], ...]) -> None:
        self.documents = documents
        self._field_indexes: dict[int, FieldIndex] = {3: FieldIndex(documents)}
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
        size = min(3, len(canonical_normalize(query).replace(" ", "")))
        if 0 < size < 3:
            # Short substrings are weak evidence, independent of language/script.
            for index, (exact, fuzzy) in self.match_scores(query).items():
                scores[index] = max(exact, 0.35 * fuzzy)
            return scores
        for term, weight in self._weights(features(query)).items():
            for index, document_weight in self.postings[term]:
                scores[index] += weight * document_weight
        return scores

    def match_scores(self, query: str) -> dict[int, tuple[float, float]]:
        size = min(3, max(1, len(canonical_normalize(query).replace(" ", ""))))
        if size not in self._field_indexes:
            self._field_indexes[size] = FieldIndex(self.documents, size)
        return self._field_indexes[size].scores(query)