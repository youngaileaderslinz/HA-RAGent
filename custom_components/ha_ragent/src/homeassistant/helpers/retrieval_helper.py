from __future__ import annotations

import logging
import math
import time
import json
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
    RETRIEVAL_TOOL_SIGNAL_WEIGHTS,
)
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext
from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    normalize_canonical_text,
)

from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.utils import get_setting_value

T = TypeVar("T")
_logger = logging.getLogger(__name__)

class RetrievalHelper:
    """Stateless helpers for building and reranking retrieval queries."""

    @staticmethod
    def retrieval_method(options: dict) -> str:
        method = str(get_setting_value(CONF_RETRIEVAL_METHOD, options)).strip().lower()
        return method if method in {
            RETRIEVAL_METHOD_AUTOMATIC, RETRIEVAL_METHOD_LEXICAL, RETRIEVAL_METHOD_VECTOR,
        } else RETRIEVAL_METHOD_AUTOMATIC

    @staticmethod
    async def async_retrieve_sources(
        backend: Any, object_type: type, options: dict, collection: str,
        embedding: list[float] | QueryEmbedding, limit: int, query: str = "",
    ) -> tuple[list, list]:
        """Try local retrieval first and share a lazy vector on weak evidence."""
        if limit <= 0:
            return [], []
        method = RetrievalHelper.retrieval_method(options)
        lexical = []
        if method != RETRIEVAL_METHOD_VECTOR:
            try:
                lexical = await backend.async_get_lexical_objects(object_type, options, collection)
            except Exception as err:
                _logger.warning("Lexical retrieval failed for %s: %s", collection, err)
        if method == RETRIEVAL_METHOD_LEXICAL:
            return [], lexical
        if isinstance(embedding, QueryEmbedding):
            if method == RETRIEVAL_METHOD_AUTOMATIC and RetrievalHelper.local_candidates_confident(query, lexical):
                return [], lexical
            try:
                embedding = await embedding.get()
            except Exception as err:
                _logger.warning("Query embedding failed for %s: %s", collection, err)
                return [], lexical
        if not embedding:
            return [], lexical
        try:
            vector = await backend.async_retrieve_scored_objects(
                object_type, options, collection, embedding, limit,
            )
            return vector, lexical
        except Exception as err:
            _logger.warning("Vector retrieval failed for %s: %s", collection, err)
            return [], lexical

    @staticmethod
    def local_candidates_confident(query: str, items: Iterable[Any]) -> bool:
        """Skip semantic recall only for a unique, complete local name match.

        Additional words may express another target or capability, so a partial
        match is insufficient to skip semantic retrieval.
        """
        normalized = RetrievalHelper._normalize(query)
        if not normalized:
            return False
        matches = 0
        for item in items:
            values = RetrievalHelper._candidate_identity_values(item)
            if normalized in {RetrievalHelper._normalize(value) for value in values if value}:
                matches += 1
        return matches == 1

    @staticmethod
    def _normalize(text: object) -> str:
        return normalize_canonical_text(text)

    @staticmethod
    def canonical_search_signature(query: object) -> str:
        """Normalize spelling representation without conflating distinct intents."""
        return " ".join(unicodedata.normalize("NFC", str(query or "")).casefold().split())

    @staticmethod
    def build_tool_search_query(
        trusted_query: str,
        fallback_query: str,
        devices: Iterable[Any],
    ) -> str:
        """Preserve the request and corrective intent without generated aliases."""
        query = trusted_query or fallback_query
        if trusted_query and fallback_query and fallback_query != trusted_query:
            query += f"\nSearch intent: {fallback_query}"
        return query

    @staticmethod
    def _tool_query_text(query: str) -> str:
        return query

    @staticmethod
    def continuity_prompt(continuity: ContinuityContext) -> str:
        """Supply bounded historical data for the existing LLM to interpret."""
        groups = [
            {
                "entities": list(group.entities[:12]),
                "areas": list(group.areas[:4]),
                "floors": list(group.floors[:4]),
                "domains": list(group.domains[:4]),
                "tool": group.tool,
                "action": group.action,
            }
            for group, _ in continuity.target_groups[:2]
        ]
        if not groups:
            return ""
        return (
            "\nRecent successful targets (historical data, not a new request; "
            "resolve references from the current message and conversation):\n"
            + json.dumps(groups, ensure_ascii=False)
        )

    @staticmethod
    def _candidate_identity_values(device: Any) -> tuple[object, ...]:
        aliases = RetrievalHelper._device_value(device, "aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return (
            RetrievalHelper._device_value(device, "id", ""),
            RetrievalHelper._device_value(device, "name", ""),
            RetrievalHelper._device_value(device, "friendly_name", ""),
            *aliases,
        )

    @staticmethod
    def _candidate_location_values(device: Any) -> tuple[object, ...]:
        return (
            RetrievalHelper._device_value(device, "area_name", ""),
            RetrievalHelper._device_value(device, "area", ""),
            RetrievalHelper._device_value(device, "floor_name", ""),
            RetrievalHelper._device_value(device, "floor", ""),
            *(RetrievalHelper._device_value(device, "area_aliases", []) or []),
            *(RetrievalHelper._device_value(device, "floor_aliases", []) or []),
        )

    @staticmethod
    def device_resolution(query: str, devices: Iterable[Any]) -> tuple[str, tuple[str, ...]]:
        """Resolve literal identities only; leave command scope to the LLM."""
        devices = list(devices)
        normalized = RetrievalHelper._normalize(query)
        exact = [
            str(RetrievalHelper._device_value(device, "id", "") or RetrievalHelper._device_value(device, "name", ""))
            for device in devices
            if normalized and normalized in {
                RetrievalHelper._normalize(value)
                for value in RetrievalHelper._candidate_identity_values(device) if value
            }
        ]
        if len(exact) == 1:
            return "high", tuple(exact)
        names = tuple(
            str(RetrievalHelper._device_value(device, "id", "") or RetrievalHelper._device_value(device, "name", ""))
            for device in devices
        )
        return ("ambiguous" if len(devices) > 1 else "weak"), names

    @staticmethod
    def reduce_confident_devices(query: str, devices: Iterable[T]) -> list[T]:
        """Expose only independently resolved devices when confidence is high."""
        devices = list(devices)
        status, names = RetrievalHelper.device_resolution(query, devices)
        if status != "high":
            return devices
        selected = set(names)
        return [
            device
            for device in devices
            if str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ) in selected
        ]

    @staticmethod
    def select_device_candidates(query: str, devices: Iterable[T], limit: int) -> list[T]:
        """Apply ambiguity-aware exposure after broad device retrieval."""
        if limit <= 0:
            return []
        devices = list(devices)
        status, _ = RetrievalHelper.device_resolution(query, devices)
        if status == "high":
            return RetrievalHelper.reduce_confident_devices(query, devices)[:1]
        if status == "ambiguous":
            return devices[:max(2, limit)]
        return devices[:limit]

    @staticmethod
    def build_retrieval_text(current_request: str) -> str:
        """Build a language-neutral query from only the current request."""
        return " ".join(current_request.split())

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        """Return cosine similarity for two embedding vectors."""
        left = list(left)
        right = list(right)
        if len(left) != len(right) or not left:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def select_history_contexts(
        contexts: Iterable[TurnContext],
        vectors: dict[str, list[float]],
        current_vector: list[float],
        max_age_seconds: float = 300.0,
        limit: int = 3,
        now: float | None = None,
    ) -> list[tuple[TurnContext, float]]:
        """Select semantic history with decay plus a small short-term signal."""
        now = time.time() if now is None else now
        contexts = list(contexts)
        selected: dict[str, tuple[TurnContext, float]] = {}
        for index, context in enumerate(contexts):
            fallback_age = float((len(contexts) - index - 1) * 30)
            age = max(0.0, now - context.created_at) if context.created_at is not None else fallback_age
            if age > max_age_seconds:
                continue
            decay = 0.5 ** (age / 120.0)
            similarity = max(0.0, RetrievalHelper.cosine_similarity(
                current_vector,
                vectors.get(context.key, []),
            ))
            relevance = similarity * decay
            if context.has_canonical_context:
                relevance += 0.1 * decay
            if similarity >= 0.2:
                selected[context.key] = (context, relevance)

            # Keep the last two retained turns, including clarification questions
            # with no successful tool result. The existing LLM needs their text
            # to resolve an unfinished request without a language parser.
            if index >= len(contexts) - 2:
                short_term_weight = 0.15 * decay
                previous = selected.get(context.key)
                if previous is None or short_term_weight > previous[1]:
                    selected[context.key] = (context, short_term_weight)

        return sorted(selected.values(), key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def build_continuity_context(selected_contexts: Iterable[tuple[TurnContext, float]]) -> ContinuityContext:
        """Aggregate selected structured turns into weighted continuity maps."""
        continuity = ContinuityContext()
        selected_contexts = list(selected_contexts)
        for context, weight in selected_contexts:
            continuity.selected_turn_keys.add(context.key)
            for attribute in (
                "entities",
                "tools",
                "areas",
                "domains",
                "device_classes",
                "actions",
                "ambiguous_entities",
            ):
                values = getattr(context, attribute)
                target = getattr(continuity, attribute)
                for value in values:
                    normalized = str(value).casefold()
                    target[normalized] = max(target.get(normalized, 0.0), weight)
        recent_contexts = sorted(
            selected_contexts,
            key=lambda item: item[0].created_at or 0.0,
            reverse=True,
        )
        continuity.target_groups = [
            (group, weight)
            for context, weight in recent_contexts
            for group in context.target_groups
        ]
        return continuity

    @staticmethod
    def adaptive_candidate_limit(limit: int) -> int:
        """Return a bounded internal pool size for hybrid retrieval."""
        return min(64, max(limit * 6, limit + 12)) if limit > 0 else 0

    @staticmethod
    def expanded_tool_limit(limit: int) -> int:
        """Expand the exposed tool set for a confidently resolved target."""
        return min(20, limit * 3) if limit > 0 else 0

    @staticmethod
    def expanded_device_limit(limit: int, continuity: ContinuityContext) -> int:
        """Keep all recent successful entity targets within a bounded shortlist."""
        successful_entities = {
            entity.casefold()
            for group, _ in continuity.target_groups
            for entity in group.entities
        }
        return min(12, max(limit, len(successful_entities))) if limit > 0 else 0

    @staticmethod
    def _character_ngrams(text: str, size: int = 3) -> set[str]:
        compact = text.replace(" ", "")
        if len(compact) <= size:
            return {compact} if compact else set()
        return {compact[index:index + size] for index in range(len(compact) - size + 1)}

    @staticmethod
    def _match_scores(query: str, values: Iterable[object]) -> tuple[float, float]:
        query_text = RetrievalHelper._normalize(query)
        query_tokens = set(query_text.split())
        query_ngrams = RetrievalHelper._character_ngrams(query_text)
        exact_score = 0.0
        fuzzy_score = 0.0
        for value in values:
            normalized = RetrievalHelper._normalize(value)
            if not normalized:
                continue
            value_tokens = set(normalized.split())
            if normalized == query_text:
                exact_score = max(exact_score, 1.0)
            elif f" {normalized} " in f" {query_text} ":
                exact_score = max(exact_score, 0.9)
            elif value_tokens:
                exact_score = max(
                    exact_score,
                    len(query_tokens & value_tokens) / len(value_tokens),
                )

            value_ngrams = RetrievalHelper._character_ngrams(normalized)
            denominator = len(query_ngrams) + len(value_ngrams)
            if denominator:
                fuzzy_score = max(
                    fuzzy_score,
                    2.0 * len(query_ngrams & value_ngrams) / denominator,
                )
        return exact_score, fuzzy_score

    @staticmethod
    def field_match_score(query: str, values: Iterable[object]) -> float:
        """Score structured metadata using language-neutral matching."""
        exact, fuzzy = RetrievalHelper._match_scores(query, values)
        return exact + (0.5 * fuzzy)

    @staticmethod
    def device_target_score(query: str, device: Any) -> float:
        """Score target identity and capability above location-only similarity."""
        aliases = RetrievalHelper._device_value(device, "aliases", []) or []
        domains = RetrievalHelper._device_value(device, "domain", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(domains, str):
            domains = [domains]
        return RetrievalHelper.field_match_score(query, (
            RetrievalHelper._device_value(device, "id", ""),
            RetrievalHelper._device_value(device, "friendly_name", ""),
            *aliases,
            *domains,
            RetrievalHelper._device_value(device, "device_class", ""),
        ))

    @staticmethod
    def trusted_location_score(device: Any, area: str = "", floor: str = "") -> float:
        """Prefer the requesting device's location as a bounded ranking signal."""
        device_area = str(getattr(device, "area_name", "") or "").casefold()
        device_floor = str(getattr(device, "floor_name", "") or "").casefold()
        score = 0.0
        if area and device_area == area.casefold():
            score += 1.0
        if floor and device_floor == floor.casefold():
            score += 0.5
        return score

    @staticmethod
    def reciprocal_rank_fusion(ranked_keys: Iterable[Iterable[str]], rank_constant: int = 60) -> dict[str, float]:
        """Fuse independent rankings using reciprocal rank fusion."""
        scores: dict[str, float] = {}
        for ranking in ranked_keys:
            for rank, key in enumerate(ranking, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
        return scores

    @staticmethod
    def _rank_positive_scores(scores: dict[str, float], minimum: float) -> list[str]:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [item_key for item_key, score in ranked if score >= minimum]

    @staticmethod
    def _has_strong_current_match(
        match_scores: dict[str, tuple[float, float]],
        vector_ranking: list[str],
        exact_ranking: list[str],
        fuzzy_ranking: list[str],
    ) -> bool:
        if any(
            exact >= 0.9 or fuzzy >= 0.9
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
        )

    @staticmethod
    def _trim_to_confident_keys(
        ordered_keys: list[str],
        match_scores: dict[str, tuple[float, float]],
        vector_positions: dict[str, int],
        limit: int,
    ) -> list[str]:
        if not ordered_keys:
            return ordered_keys

        top_key = ordered_keys[0]
        top_exact, top_fuzzy = match_scores[top_key]
        top_vector_rank = vector_positions.get(top_key)
        top_is_confident = (
            top_exact >= 0.9
            or top_fuzzy >= 0.9
            or (
                top_vector_rank == 1
                and (top_exact >= 0.5 or top_fuzzy >= 0.7)
            )
        )
        if not top_is_confident:
            return ordered_keys

        confident = [
            item_key
            for item_key in ordered_keys[:limit]
            if match_scores[item_key][0] >= 0.5
            or match_scores[item_key][1] >= 0.7
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
    ) -> list[T]:
        """Fuse rank-based signals and suppress stale continuity on strong matches."""
        if limit <= 0:
            return []

        vector_results = list(vector_results)
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
        lexical_scores = dict(zip(lexical_keys, lexical_index(documents).scores(query)))
        match_scores = {
            item_key: RetrievalHelper._match_scores(query, parts)
            for item_key, parts in zip(lexical_keys, documents)
        }
        lexical_ranking = RetrievalHelper._rank_positive_scores(lexical_scores, 0.01)
        exact_ranking = RetrievalHelper._rank_positive_scores(
            {item_key: scores[0] for item_key, scores in match_scores.items()},
            0.75,
        )
        fuzzy_ranking = RetrievalHelper._rank_positive_scores(
            {item_key: scores[1] for item_key, scores in match_scores.items()},
            0.2,
        )
        metadata_scores = {
            item_key: metadata_score(item)
            for item_key, item in candidates.items()
        } if metadata_score else {}
        metadata_ranking = RetrievalHelper._rank_positive_scores(
            {key: score for key, score in metadata_scores.items() if score > 0},
            0.0,
        )
        has_strong_current_match = RetrievalHelper._has_strong_current_match(
            match_scores,
            vector_ranking,
            exact_ranking,
            fuzzy_ranking,
        )

        continuity_scores = {
            item_key: continuity_score(item)
            for item_key, item in candidates.items()
        } if continuity_score and not has_strong_current_match else {}
        continuity_ranking = RetrievalHelper._rank_positive_scores(
            {key: score for key, score in continuity_scores.items() if score > 0},
            0.0,
        )

        fused = RetrievalHelper.reciprocal_rank_fusion(
            (vector_ranking, lexical_ranking, exact_ranking, fuzzy_ranking, metadata_ranking)
        )
        for rank, item_key in enumerate(continuity_ranking, start=1):
            fused[item_key] = fused.get(item_key, 0.0) + 0.25 / (60 + rank)
        for item_key, (exact_score, fuzzy_score) in match_scores.items():
            fused[item_key] = (
                fused.get(item_key, 0.0)
                + (0.03 * lexical_scores.get(item_key, 0.0))
                + (0.01 * exact_score)
                + (0.01 * fuzzy_score)
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
            ordered_keys = RetrievalHelper._trim_to_confident_keys(
                ordered_keys,
                match_scores,
                vector_positions,
                limit,
            )

        selected_keys = ordered_keys[:limit]
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
            selected_keys = RetrievalHelper._merge_preserved_keys(
                selected_keys,
                preserved_keys,
                limit,
            )

        return [candidates[item_key] for item_key in selected_keys]

    @staticmethod
    def target_is_confident(query: str, devices: Iterable[Any], continuity: ContinuityContext) -> bool:
        """Return whether the current or successful prior target is resolved."""
        status, _ = RetrievalHelper.device_resolution(query, devices)
        return status == "high"

    @staticmethod
    def _schema_values(schema: object) -> set[str]:
        values: set[str] = set()
        if isinstance(schema, dict):
            for name, value in schema.items():
                if name == "const" and isinstance(value, (str, int, float)):
                    values.add(str(value).casefold())
                elif name == "enum" and isinstance(value, list):
                    values.update(str(item).casefold() for item in value)
                else:
                    values.update(RetrievalHelper._schema_values(value))
        elif isinstance(schema, list):
            for value in schema:
                values.update(RetrievalHelper._schema_values(value))
        return values

    @staticmethod
    def _device_value(device: Any, name: str, default: Any = None) -> Any:
        if isinstance(device, dict):
            return device.get(name, default)
        return getattr(device, name, default)

    @staticmethod
    def _device_domains(devices: Iterable[Any]) -> set[str]:
        domains: set[str] = set()
        for device in devices:
            values = RetrievalHelper._device_value(device, "domain", []) or []
            if isinstance(values, str):
                values = [values]
            domains.update(str(value).casefold() for value in values)
        return domains

    @staticmethod
    def _device_classes(devices: Iterable[Any]) -> set[str]:
        return {
            str(value).casefold()
            for device in devices
            if (value := RetrievalHelper._device_value(device, "device_class"))
        }

    @staticmethod
    def _metadata_value(tool: Any, name: str, default: Any = False) -> Any:
        metadata = getattr(tool, "metadata", None)
        if isinstance(metadata, dict):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    @staticmethod
    def _tool_declared_domains(tool: Any) -> set[str]:
        properties = (getattr(tool, "parameters", None) or {}).get("properties") or {}
        domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        domains.update(RetrievalHelper._metadata_value(tool, "supported_domains", ()) or ())
        return domains

    @staticmethod
    def _tool_domain_signal(tool: Any, requested_domains: set[str]) -> float:
        if not requested_domains:
            return 0.0
        declared_domains = RetrievalHelper._tool_declared_domains(tool)
        if declared_domains:
            return 1.0 if declared_domains & requested_domains else -0.35
        return 0.0

    @staticmethod
    def tool_ranking_signals(
        tool: Any,
        query: str,
        devices: Iterable[Any] = (),
        semantic_rank: int | None = None,
        semantic_score: float | None = None,
        continuity: float = 0.0,
    ) -> dict[str, float]:
        """Return independent, extensible signals used to rank a tool."""
        devices = list(devices)
        query = RetrievalHelper._tool_query_text(query)
        requested_domains = RetrievalHelper._device_domains(devices)
        exact, fuzzy = RetrievalHelper._match_scores(
            query,
            getattr(tool, "canonical_search_parts", ()) or (
                getattr(tool, "name", ""),
                getattr(tool, "description", ""),
            ),
        )
        compatibility = RetrievalHelper.tool_device_compatibility(tool, devices)
        return {
            "semantic_rank": 1.0 / semantic_rank if semantic_rank else 0.0,
            "semantic_similarity": max(0.0, min(1.0, semantic_score or 0.0)),
            "lexical_exact": exact,
            "lexical_fuzzy": fuzzy,
            "lexical_action": float(bool(RetrievalHelper._matched_action_phrases(tool, query))),
            "domain": RetrievalHelper._tool_domain_signal(tool, requested_domains),
            "device_metadata": max(-1.0, min(1.0, compatibility / 2.0)),
            "continuity": max(0.0, continuity),
        }

    @staticmethod
    def _matched_action_phrases(tool: Any, query: str) -> set[str]:
        """Match ordered live capability names without interpreting user intent.

        Keep complete phrases distinct: overlapping words do not make two
        capabilities interchangeable. This is retrieval evidence only, including
        when a capability occurs in a negated request.
        """
        phrases = (
            getattr(tool, "canonical_action", ""),
            " ".join(getattr(tool, "canonical_action_keywords", ()) or ()),
        )
        normalized_query = f" {RetrievalHelper._normalize(query)} "
        return {
            normalized
            for phrase in phrases
            if (normalized := RetrievalHelper._normalize(phrase))
            and f" {normalized} " in normalized_query
        }

    @staticmethod
    def tool_signal_score(signals: dict[str, float]) -> float:
        """Combine named tool-ranking signals using centralized weights."""
        return sum(
            RETRIEVAL_TOOL_SIGNAL_WEIGHTS.get(name, 0.0) * value
            for name, value in signals.items()
        )

    @staticmethod
    def rank_tool_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_tools: Iterable[T],
        query: str,
        devices: Iterable[Any],
        limit: int,
        continuity_score: Callable[[T], float] | None = None,
    ) -> list[T]:
        """Rank a broad tool pool without discarding uncertain candidates."""
        if limit <= 0:
            return []
        devices = list(devices)
        vector_results = list(vector_results)
        candidate_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in lexical_tools
        }
        semantic_ranks: dict[str, int] = {}
        semantic_scores: dict[str, float] = {}
        for result in vector_results:
            name = str(getattr(result.item, "name", ""))
            candidate_by_name.setdefault(name, result.item)
            semantic_ranks[name] = result.rank
            semantic_scores[name] = result.score

        corpus_names = sorted(candidate_by_name)
        corpus_tools = [candidate_by_name[name] for name in corpus_names]
        documents = tuple(
            tuple(str(value) for value in (getattr(tool, "canonical_search_parts", ()) or (
                getattr(tool, "name", ""), getattr(tool, "description", ""),
            )) if value)
            for tool in corpus_tools
        )
        corpus_scores = dict(zip(corpus_names, lexical_index(documents).scores(query)))
        scored: list[tuple[float, int, str, T]] = []
        for name, tool in candidate_by_name.items():
            continuity = continuity_score(tool) if continuity_score else 0.0
            signals = RetrievalHelper.tool_ranking_signals(
                tool,
                query,
                devices,
                semantic_rank=semantic_ranks.get(name),
                semantic_score=semantic_scores.get(name),
                continuity=continuity,
            )
            signals["lexical_corpus"] = corpus_scores[name]
            scored.append(
                (
                    RetrievalHelper.tool_signal_score(signals),
                    semantic_ranks.get(name, len(semantic_ranks) + 1),
                    name,
                    tool,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        # Reward coverage of distinct request terms so several power tools do
        # not crowd out a color setter or a user-defined mode tool. Derive this
        # from live names, descriptions and schemas, never a tool-name allowlist.
        query_tokens = set(RetrievalHelper._normalize(
            RetrievalHelper._tool_query_text(query)
        ).split())
        matched_terms = {}
        for _, _, name, tool in scored:
            text = " ".join((
                name,
                str(getattr(tool, "description", "") or ""),
                *getattr(tool, "canonical_schema_parts", ()),
            ))
            matched_terms[name] = query_tokens & set(RetrievalHelper._normalize(text).split())
        term_counts = Counter(term for terms in matched_terms.values() for term in terms)
        term_weights = {
            term: math.log(1.0 + len(scored) / count)
            for term, count in term_counts.items()
        }
        tool_weights = {
            name: sum(term_weights[term] for term in terms)
            for name, terms in matched_terms.items()
        }
        covered: set[str] = set()
        matched_actions = {
            name: RetrievalHelper._matched_action_phrases(tool, query)
            for _, _, name, tool in scored
        }
        covered_actions: set[str] = set()
        selected: list[T] = []
        while scored and len(selected) < limit:
            def marginal_score(index: int) -> float:
                score, _, name, _ = scored[index]
                action_weight = RETRIEVAL_TOOL_SIGNAL_WEIGHTS["lexical_action"]
                # Reserve the phrase bonus for a capability not yet represented.
                # Descriptions mentioning the opposite action cannot cover it.
                action_bonus = action_weight if matched_actions[name] - covered_actions else 0.0
                score -= action_weight if matched_actions[name] else 0.0
                weight = tool_weights[name]
                if not weight:
                    return score + action_bonus
                uncovered = sum(
                    term_weights[term]
                    for term in matched_terms[name] - covered
                ) / weight
                # Keep a residual relevance score for alternatives, but spend
                # the limited slots on capabilities not represented yet.
                return score - 0.8 * abs(score) * (1.0 - uncovered) + action_bonus

            best = max(range(len(scored)), key=marginal_score)
            _, _, name, tool = scored.pop(best)
            selected.append(tool)
            covered.update(matched_terms[name])
            covered_actions.update(matched_actions[name])
        return selected

    @staticmethod
    def tool_search_confidence(tools: Iterable[Any], query: str, devices: Iterable[Any]) -> str:
        """Classify the top result without turning uncertainty into a hard failure."""
        tools = list(tools)
        if not tools:
            return "none"
        query = RetrievalHelper._tool_query_text(query)
        top_signals = RetrievalHelper.tool_ranking_signals(tools[0], query, devices)
        # Lexical similarity and metadata are evidence of relevance, never
        # proof of an action, negation, group scope, or a complete compound task.
        if RetrievalHelper.local_candidates_confident(query, tools):
            return "high"
        if top_signals["lexical_exact"] >= 0.5 or top_signals["lexical_fuzzy"] >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def rank_tools_for_query(tools: Iterable[T], query: str, devices: Iterable[Any] = ()) -> list[T]:
        tools = list(tools)
        return RetrievalHelper.rank_tool_candidates([], tools, query, devices, len(tools))

    @staticmethod
    def build_tool_candidate_pool(
        vector_results: Iterable[ScoredResult[T]],
        lexical_tools: Iterable[T],
        query: str,
        devices: Iterable[Any] = (),
    ) -> tuple[list[ScoredResult[T]], list[T]]:
        """Build a complete tool pool while preserving vector-rank metadata."""
        vector_results = list(vector_results)
        candidate_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in lexical_tools
        }
        for result in vector_results:
            candidate_by_name.setdefault(str(getattr(result.item, "name", "")), result.item)
        candidate_tools = RetrievalHelper.rank_tools_for_query(
            candidate_by_name.values(),
            query,
            devices,
        )
        return vector_results, candidate_tools

    @staticmethod
    def tool_device_compatibility(tool: Any, devices: Iterable[Any]) -> float:
        """Score whether a tool schema can target the retrieved devices."""
        devices = list(devices)
        if not devices:
            return 0.0
        properties = (tool.parameters or {}).get("properties") or {}
        domains = RetrievalHelper._device_domains(devices)
        device_classes = RetrievalHelper._device_classes(devices)
        score = 0.0
        allowed_domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        if allowed_domains:
            score += 2.0 if domains & allowed_domains else -2.0
        if allowed_classes:
            score += 2.0 if device_classes & allowed_classes else -2.0

        if RetrievalHelper._metadata_value(tool, "is_domain_aware"):
            score += 0.5
        if RetrievalHelper._metadata_value(tool, "is_device_class_aware") and device_classes:
            score += 0.5
        has_area = any(
            RetrievalHelper._device_value(device, "area_name")
            or RetrievalHelper._device_value(device, "area")
            for device in devices
        )
        if RetrievalHelper._metadata_value(tool, "is_area_aware") and has_area:
            score += 0.25
        return score

    @staticmethod
    def rerank_tools_for_devices(tools: Iterable[T], devices: Iterable[Any], limit: int) -> list[T]:
        """Softly prefer device-compatible tools without erasing alternatives."""
        devices = list(devices)
        tools = list(tools)
        ranked = sorted(
            enumerate(tools),
            key=lambda pair: (
                -RetrievalHelper.tool_device_compatibility(pair[1], devices),
                pair[0],
            ),
        )
        return [tool for _, tool in ranked[:limit]]
