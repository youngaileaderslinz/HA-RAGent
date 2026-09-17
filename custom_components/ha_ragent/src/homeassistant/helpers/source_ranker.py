"""Rank candidates using one explicitly selected retrieval source."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from custom_components.ha_ragent.src.const import RETRIEVAL_METHOD_VECTOR
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.models.retrieval.confidence_profile import ConfidenceProfile
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


T = TypeVar("T")


class SourceRanker:
    """Keep vector and lexical ranking independent of hybrid evidence."""

    @staticmethod
    def rank(
        vector_results: Iterable[ScoredResult[T]],
        lexical_items: Iterable[T],
        query: str,
        key: Callable[[T], str],
        text_parts: Callable[[T], Iterable[object]],
        limit: int,
        method: str,
        ranking_evidence: dict[str, dict[str, float]] | None = None,
    ) -> list[T]:
        candidates: dict[str, T] = {}
        scores: dict[str, float] = {}
        if method == RETRIEVAL_METHOD_VECTOR:
            for result in sorted(vector_results, key=lambda result: (-result.score, result.rank)):
                name = key(result.item)
                if name not in candidates:
                    candidates[name] = result.item
                    scores[name] = result.score
            ordered = list(candidates)
        else:
            for item in lexical_items:
                candidates.setdefault(key(item), item)
            names = sorted(candidates)
            documents = tuple(
                tuple(str(part) for part in text_parts(candidates[name]) if part)
                for name in names
            )
            index = lexical_index(documents)
            tfidf = index.scores(query)
            matches = index.match_scores(query)
            for position, name in enumerate(names):
                exact, fuzzy = matches.get(position, (0.0, 0.0))
                scores[name] = max(tfidf[position], exact, 0.5 * fuzzy)
            ordered = sorted(names, key=lambda name: (-scores[name], name))
        if ranking_evidence is not None:
            ranking_evidence[method] = scores
        return [candidates[name] for name in ordered[:limit]]

    @staticmethod
    def confidence(
        keys: Iterable[str],
        method: str,
        ranking_evidence: dict[str, dict[str, float]],
        profile: ConfidenceProfile,
        kind: str,
    ) -> ConfidenceAssessment:
        return RetrievalConfidence.assess_distribution_confidence(
            keys,
            {method: ranking_evidence.get(method, {})},
            profile=profile,
            minimum_independent_signals=1,
            kind=kind,
        )
