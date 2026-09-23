"""Shared source ranking, text evidence, and rank fusion."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from custom_components.ha_ragent.src.const import (
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.models.embedding.tool_metadata import normalize_canonical_text
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.models.retrieval.lexical_index import (
    lexical_index,
    match_features,
    match_score,
)
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

T = TypeVar("T")

_logger = BaseLogger(__name__)


class SourceRanker:
    """Shared source ranking, text evidence, and rank fusion."""

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
        near_tie_margin: float,
        kind: str,
    ) -> ConfidenceAssessment:
        return RetrievalConfidence.assess_distribution_confidence(
            keys,
            {method: ranking_evidence.get(method, {})},
            near_tie_margin=near_tie_margin,
            minimum_independent_signals=1,
            kind=kind,
        )

    @staticmethod
    def local_candidates_confident(query: str, items: Iterable[Any]) -> bool:
        """Skip semantic recall only for a unique, complete local name match.

        Additional words may express another target or capability, so a partial
        match is insufficient to skip semantic retrieval.
        """
        normalized = SourceRanker.normalize(query)
        if not normalized:
            return False
        matches = 0
        for item in items:
            values = SourceRanker.candidate_identity_values(item)
            if normalized in {SourceRanker.normalize(value) for value in values if value}:
                matches += 1
        return matches == 1

    @staticmethod
    def normalize(text: object) -> str:
        return normalize_canonical_text(text)

    @staticmethod
    def canonical_search_signature(query: object) -> str:
        """Normalize spelling representation without conflating distinct intents."""
        return " ".join(unicodedata.normalize("NFC", str(query or "")).casefold().split())

    @staticmethod
    def candidate_identity_values(device: Any) -> tuple[object, ...]:
        aliases = SourceRanker.candidate_value(device, "aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return (
            SourceRanker.candidate_value(device, "id", ""),
            SourceRanker.candidate_value(device, "name", ""),
            SourceRanker.candidate_value(device, "friendly_name", ""),
            *aliases,
        )

    @staticmethod
    def adaptive_candidate_limit(limit: int | float) -> int:
        """Return an integer bounded internal pool size for hybrid retrieval."""
        try:
            requested = int(limit)
        except (TypeError, ValueError, OverflowError):
            return 0
        return min(64, max(requested * 6, requested + 12)) if requested > 0 else 0

    @staticmethod
    def match_scores(query: str, values: Iterable[object]) -> tuple[float, float]:
        query_text = SourceRanker.normalize(query)
        size = min(3, max(1, len(query_text.replace(" ", ""))))
        prepared_query = match_features(query, size)
        exact_score = 0.0
        fuzzy_score = 0.0
        for value in values:
            exact, fuzzy = match_score(prepared_query, match_features(str(value or ""), size))
            exact_score = max(exact_score, exact)
            fuzzy_score = max(fuzzy_score, fuzzy)
        return exact_score, fuzzy_score

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_keys: Iterable[Iterable[str] | dict[str, float]],
        rank_constant: int = 60,
    ) -> dict[str, float]:
        """Fuse rankings, assigning equal evidence to equal-scored items."""
        scores: dict[str, float] = {}
        for ranking in ranked_keys:
            if isinstance(ranking, dict):
                # A score map is evidence only when it establishes positive
                # relevance. In particular, zero/negative compatibility and
                # similarity values must not create RRF votes.
                ordered = sorted(
                    (
                        (key, float(value))
                        for key, value in ranking.items()
                        if math.isfinite(float(value)) and float(value) > 0.0
                    ),
                    key=lambda pair: (-pair[1], pair[0]),
                )
                previous_score: float | None = None
                rank = 0
                for position, (key, value) in enumerate(ordered, start=1):
                    if previous_score is None or value < previous_score:
                        # Competition ranking: ties receive the same vote and
                        # the next distinct score reflects their positions.
                        rank = position
                    scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
                    previous_score = value
            else:
                for rank, key in enumerate(ranking, start=1):
                    scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
        return scores

    @staticmethod
    def rank_positive_scores(scores: dict[str, float], minimum: float) -> list[str]:
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [item_key for item_key, score in ranked if score > minimum]

    @staticmethod
    def rank_only_if_separated(scores: dict[str, float], minimum: float) -> list[str]:
        """Return a ranking only when the signal itself distinguishes items."""
        positive = [score for score in scores.values() if score > minimum]
        if len(set(positive)) <= 1:
            return []
        return SourceRanker.rank_positive_scores(scores, minimum)

    @staticmethod
    def _has_strong_current_match(
        match_scores: dict[str, tuple[float, float]],
        vector_ranking: list[str],
        exact_ranking: list[str],
        fuzzy_ranking: list[str],
        query: str = "",
    ) -> bool:
        numeric_query = SourceRanker.has_numeric_token(query)
        if any(
            exact >= 0.9 or (fuzzy >= 0.9 and not numeric_query)
            for exact, fuzzy in match_scores.values()
        ):
            return True
        if not vector_ranking:
            return False

        best_vector = vector_ranking[0]
        if exact_ranking and best_vector == exact_ranking[0]:
            return True
        return (
            bool(fuzzy_ranking)
            and best_vector == fuzzy_ranking[0]
            and match_scores[best_vector][1] >= 0.7
            and not numeric_query
        )

    @staticmethod
    def has_numeric_token(value: object) -> bool:
        """Return whether normalized text contains a token with a digit."""
        return any(
            any(character.isdigit() for character in token)
            for token in SourceRanker.normalize(value).split()
        )

    @staticmethod
    def has_textual_overlap(query: object, value: object) -> bool:
        """Require non-numeric token overlap for numeric locations."""
        query_tokens = {
            token for token in SourceRanker.normalize(query).split()
            if not any(character.isdigit() for character in token)
        }
        value_tokens = {
            token for token in SourceRanker.normalize(value).split()
            if not any(character.isdigit() for character in token)
        }
        return bool(query_tokens & value_tokens)

    @staticmethod
    def _trim_to_confident_keys(
        ordered_keys: list[str],
        match_scores: dict[str, tuple[float, float]],
        vector_positions: dict[str, int],
        limit: int,
        query: str = "",
    ) -> list[str]:
        if not ordered_keys:
            return ordered_keys

        top_key = ordered_keys[0]
        top_exact, top_fuzzy = match_scores[top_key]
        top_vector_rank = vector_positions.get(top_key)
        top_is_confident = (
            top_exact >= 0.9
            or (top_fuzzy >= 0.9 and not SourceRanker.has_numeric_token(query))
            or (
                top_vector_rank == 1
                and (top_exact >= 0.5 or (
                    top_fuzzy >= 0.7
                    and not SourceRanker.has_numeric_token(query)
                ))
            )
        )
        if not top_is_confident:
            return ordered_keys

        confident = [
            item_key
            for item_key in ordered_keys[:limit]
            if match_scores[item_key][0] >= 0.5
            or (
                match_scores[item_key][1] >= 0.7
                and not SourceRanker.has_numeric_token(query)
            )
            or vector_positions.get(item_key, limit + 1) <= 2
        ]
        return confident or ordered_keys

    @staticmethod
    def _merge_preserved_keys(selected_keys: list[str], preserved_keys: list[str], limit: int) -> list[str]:
        missing = [key for key in preserved_keys if key not in selected_keys]
        if not missing:
            return selected_keys
        retained = [
            key for key in selected_keys if key not in preserved_keys
        ][:max(0, limit - len(preserved_keys))]
        return [*retained, *preserved_keys]

    @staticmethod
    def rank_scored_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_items: Iterable[T],
        query: str,
        key: Callable[[T], str],
        text_parts: Callable[[T], Iterable[object]],
        limit: int,
        metadata_score: Callable[[T], float] | None = None,
        continuity_score: Callable[[T], float] | None = None,
        preserve_score: Callable[[T], float] | None = None,
        trim_confident: bool = True,
        ranking_evidence: dict[str, dict[str, float]] | None = None,
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> list[T]:
        """Fuse rank-based signals and suppress stale continuity on strong matches."""
        if limit <= 0:
            return []
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            return SourceRanker.rank(
                vector_results, lexical_items, query, key, text_parts,
                limit, retrieval_method, ranking_evidence,
            )

        vector_results = list(vector_results)
        lexical_items = list(lexical_items)
        candidates: dict[str, T] = {
            key(result.item): result.item for result in vector_results
        }
        for item in lexical_items:
            candidates.setdefault(key(item), item)

        vector_ranking = [
            key(result.item)
            for result in sorted(vector_results, key=lambda result: result.rank)
        ]
        vector_positions = {
            item_key: position
            for position, item_key in enumerate(vector_ranking, start=1)
        }
        lexical_keys = sorted(candidates)
        documents = tuple(
            tuple(str(value) for value in text_parts(candidates[item_key]) if value)
            for item_key in lexical_keys
        )
        index = lexical_index(documents)
        lexical_scores = dict(zip(lexical_keys, index.scores(query)))
        field_scores = index.match_scores(query)
        match_scores = {
            item_key: field_scores.get(position, (0.0, 0.0))
            for position, item_key in enumerate(lexical_keys)
        }
        exact_ranking = SourceRanker.rank_positive_scores(
            {item_key: scores[0] for item_key, scores in match_scores.items()},
            0.75,
        )
        fuzzy_ranking = SourceRanker.rank_positive_scores(
            {item_key: scores[1] for item_key, scores in match_scores.items()},
            0.2,
        )
        metadata_scores = {
            item_key: metadata_score(item)
            for item_key, item in candidates.items()
        } if metadata_score else {}
        has_strong_current_match = SourceRanker._has_strong_current_match(
            match_scores,
            vector_ranking,
            exact_ranking,
            fuzzy_ranking,
            query,
        )

        # Always expose the diagnostic channel.  A strong current match may
        # disable continuity's influence on fusion, but hiding the scores made
        # it impossible to tell whether continuity was absent or intentionally
        # suppressed.
        continuity_scores = {
            item_key: continuity_score(item)
            for item_key, item in candidates.items()
        } if continuity_score else {}
        continuity_ranking = SourceRanker.rank_positive_scores(
            {
                key: score for key, score in continuity_scores.items()
                if score > 0
            } if not has_strong_current_match else {},
            0.0,
        )

        fused = SourceRanker.reciprocal_rank_fusion(
            (
                {key(result.item): result.score for result in vector_results},
                {
                    item_key: max(lexical_scores.get(item_key, 0.0), exact, 0.5 * fuzzy)
                    for item_key, (exact, fuzzy) in match_scores.items()
                },
                metadata_scores,
            )
        )
        for rank, item_key in enumerate(continuity_ranking, start=1):
            fused[item_key] = fused.get(item_key, 0.0) + 0.25 / (60 + rank)
        for item_key, (exact_score, fuzzy_score) in match_scores.items():
            fused[item_key] = (
                fused.get(item_key, 0.0)
                + (0.03 * max(lexical_scores.get(item_key, 0.0), exact_score, 0.5 * fuzzy_score))
            )
        ordered_keys = sorted(
            candidates,
            key=lambda item_key: (
                -fused.get(item_key, 0.0),
                vector_positions.get(item_key, len(vector_ranking) + 1),
            ),
        )

        # Strong agreement permits a smaller, precise result. Weak confidence
        # retains the configured limit so downstream semantic search can recover.
        if trim_confident:
            ordered_keys = SourceRanker._trim_to_confident_keys(
                ordered_keys,
                match_scores,
                vector_positions,
                limit,
                query,
            )

        selected_keys = ordered_keys[:limit]
        # A literal device identity is stronger evidence than agreement between
        # two approximate retrievers.  In particular, a vector-only candidate
        # must not displace an exact full name (or alias) merely because it is
        # also present in the lexical corpus.  Reserve these identities before
        # filling the remaining budget from fusion.
        identity_keys = [
            item_key
            for item_key in ordered_keys
            if (
                match_scores[item_key][0] >= 0.9
                or (
                    match_scores[item_key][1] >= 0.9
                    and not SourceRanker.has_numeric_token(query)
                )
            )
        ]
        if identity_keys:
            selected_keys = [
                *identity_keys[:limit],
                *(item_key for item_key in ordered_keys if item_key not in identity_keys),
            ][:limit]
        if preserve_score and not has_strong_current_match:
            preserved_keys = [
                item_key
                for item_key, score in sorted(
                    (
                        (item_key, preserve_score(item))
                        for item_key, item in candidates.items()
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                if score > 0
            ][:limit]
            selected_keys = SourceRanker._merge_preserved_keys(
                selected_keys,
                preserved_keys,
                limit,
            )

        result = [candidates[item_key] for item_key in selected_keys]
        if ranking_evidence is not None:
            ranking_evidence["lexical"] = lexical_scores
            ranking_evidence["fused"] = fused
        _logger.log_payload("retrieval.device_ranking", query=query, limit=limit,
            vector_results=vector_results, lexical_items=lexical_items,
            candidates=candidates, vector_ranking=vector_ranking,
            lexical_scores=lexical_scores, match_scores=match_scores,
            metadata_scores=metadata_scores, continuity_scores=continuity_scores,
            fused_scores=fused, ordered_keys=ordered_keys,
            selected_keys=selected_keys, selected=result,
        )
        return result

    @staticmethod
    def candidate_value(device: Any, name: str, default: Any = None) -> Any:
        if isinstance(device, dict):
            return device.get(name, default)
        return getattr(device, name, default)
