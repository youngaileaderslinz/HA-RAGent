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
    """Return word and character TF-IDF features without a language lexicon."""
    terms: Counter[str] = Counter()
    for word in canonical_normalize(text).split():
        terms[f"w:{word}"] += 1
        padded = f" {word} "
        for size in range(3, 6):
            terms.update(
                f"c{size}:" + padded[index:index + size]
                for index in range(max(0, len(padded) - size + 1))
            )
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
        # A name, location, description, and schema value are separate
        # fields.  Flattening them made long schemas reduce the score of a
        # short exact identity match.  Per-field normalization preserves the
        # strongest evidence without a language-specific vocabulary.
        self.counts = [[features(part) for part in parts if part] for parts in documents]
        frequency = Counter(
            term for document in self.counts for field in document for term in field
        )
        field_count = sum(len(document) for document in self.counts)
        self.idf = {
            term: math.log(1 + field_count / count)
            for term, count in frequency.items()
        }
        # Posting lists keep long collections from scanning every field for a
        # query.  They are built with the collection and reused by its cache.
        self.postings: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        self.field_weights: list[list[dict[str, float]]] = [
            [self._weights(field) for field in document]
            for document in self.counts
        ]
        for document_id, document in enumerate(self.field_weights):
            for field_id, weights in enumerate(document):
                for term, weight in weights.items():
                    self.postings[term].append((document_id, field_id, weight))

    def _weights(self, counts: Counter[str]) -> dict[str, float]:
        weights = {
            term: (1 + math.log(count)) * self.idf[term]
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
        query_weights = self._weights(features(query))
        field_scores: dict[tuple[int, int], float] = defaultdict(float)
        for term, query_weight in query_weights.items():
            for document_id, field_id, field_weight in self.postings.get(term, ()):
                field_scores[(document_id, field_id)] += query_weight * field_weight
        by_document: dict[int, list[float]] = defaultdict(list)
        for (document_id, _field_id), score in field_scores.items():
            by_document[document_id].append(score)
        for document_id, values in by_document.items():
            values.sort(reverse=True)
            # A short exact identifier remains dominant, but corroborating
            # identity, room, and description fields can lift a candidate.
            scores[document_id] = min(
                1.0,
                values[0] + sum(value * 0.25 for value in values[1:3]),
            )
        return scores

    def match_scores(self, query: str) -> dict[int, tuple[float, float]]:
        size = min(3, max(1, len(canonical_normalize(query).replace(" ", ""))))
        if size not in self._field_indexes:
            self._field_indexes[size] = FieldIndex(self.documents, size)
        return self._field_indexes[size].scores(query)
