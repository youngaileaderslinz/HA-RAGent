from __future__ import annotations

import logging
import math
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
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

from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index, match_features, match_score
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.debug import log_debug_payload
from custom_components.ha_ragent.src.utils import get_setting_value

T = TypeVar("T")
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceProfile:
    """Distribution thresholds kept separate for devices and tools."""

    high_margin: float
    high_ratio: float
    medium_margin: float
    medium_ratio: float
    near_tie_margin: float


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Explainable confidence derived from a local candidate distribution."""

    level: str
    top_score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    ratio: float = 0.0
    agreeing_signals: tuple[str, ...] = ()
    disagreeing_signals: tuple[str, ...] = ()
    reason: str = "no candidates"
    candidate_scores: tuple[tuple[str, float], ...] = ()
    candidate_support: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# These profiles are deliberately separate and dimensionless. Scores are
# normalized against each request's local distribution. The emitted diagnostics
# make these initial values calibratable from real deployments without tying
# confidence to backend-specific raw score scales.
DEVICE_CONFIDENCE_PROFILE = ConfidenceProfile(0.30, 1.55, 0.10, 1.18, 0.06)
TOOL_CONFIDENCE_PROFILE = ConfidenceProfile(0.24, 1.40, 0.08, 1.14, 0.05)

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
        """Retrieve exactly the sources selected by the configured method."""
        if limit <= 0:
            return [], []
        method = RetrievalHelper.retrieval_method(options)
        log_debug_payload(
            _logger, "retrieval.sources.request", collection=collection,
            object_type=getattr(object_type, "__name__", str(object_type)),
            method=method, query=query, limit=limit,
            embedding_deferred=isinstance(embedding, QueryEmbedding),
        )
        lexical = []
        if method != RETRIEVAL_METHOD_VECTOR:
            try:
                lexical = await backend.async_get_lexical_objects(
                    object_type, options, collection,
                )
            except Exception as err:
                _logger.warning("Lexical retrieval failed for %s: %s", collection, err)
        if method == RETRIEVAL_METHOD_LEXICAL:
            log_debug_payload(
                _logger, "retrieval.sources.result", collection=collection,
                method=method, vector=[], lexical=lexical,
            )
            return [], lexical
        if isinstance(embedding, QueryEmbedding):
            try:
                embedding = await embedding.get()
            except Exception as err:
                _logger.warning("Query embedding failed for %s: %s", collection, err)
                log_debug_payload(
                    _logger, "retrieval.sources.result", collection=collection,
                    method=method, vector=[], lexical=lexical,
                    failure={"stage": "embedding", "error": repr(err)},
                )
                return [], lexical
        if not embedding:
            log_debug_payload(
                _logger, "retrieval.sources.result", collection=collection,
                method=method, vector=[], lexical=lexical,
                failure={"stage": "embedding", "error": "empty embedding"},
            )
            return [], lexical
        try:
            vector = await backend.async_retrieve_scored_objects(
                object_type, options, collection, embedding, limit,
            )
            log_debug_payload(
                _logger, "retrieval.sources.result", collection=collection,
                method=method, embedding=embedding, vector=vector, lexical=lexical,
            )
            return vector, lexical
        except Exception as err:
            _logger.warning("Vector retrieval failed for %s: %s", collection, err)
            log_debug_payload(
                _logger, "retrieval.sources.result", collection=collection,
                method=method, embedding=embedding, vector=[], lexical=lexical,
                failure={"stage": "vector", "error": repr(err)},
            )
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
    def confidence_limit(level: str, minimum: int, maximum: int) -> int:
        """Map high/medium/low confidence gradually onto an exposure range."""
        minimum = max(0, int(minimum))
        maximum = max(minimum, int(maximum))
        if level == "high":
            return minimum
        if level == "medium":
            return minimum + ((maximum - minimum + 1) // 2)
        return maximum

    @staticmethod
    def _normalize_confidence_signal(values: dict[str, float]) -> dict[str, float]:
        """Normalize one signal against its local candidate distribution."""
        finite = {
            key: max(0.0, float(value))
            for key, value in values.items()
            if math.isfinite(float(value))
        }
        maximum = max(finite.values(), default=0.0)
        if maximum <= 0:
            return {}
        return {key: value / maximum for key, value in finite.items()}

    @staticmethod
    def assess_distribution_confidence(
        ordered_keys: Iterable[str],
        signal_scores: dict[str, dict[str, float]],
        *,
        profile: ConfidenceProfile,
        weak_signals: set[str] | None = None,
        confirmed_keys: set[str] | None = None,
        kind: str = "candidate",
    ) -> ConfidenceAssessment:
        """Classify confidence from separation and independent signal agreement."""
        ordered_keys = list(dict.fromkeys(ordered_keys))
        if not ordered_keys:
            return ConfidenceAssessment(level="none")

        weak_signals = weak_signals or set()
        confirmed_keys = confirmed_keys or set()
        normalized = {
            name: values
            for name, raw_values in signal_scores.items()
            if (values := RetrievalHelper._normalize_confidence_signal(raw_values))
        }
        weights = {
            name: (0.2 if name in weak_signals else 1.0)
            for name in normalized
        }
        totals = {key: 0.0 for key in ordered_keys}
        total_weight = sum(weights.values()) or 1.0
        for name, values in normalized.items():
            for key in ordered_keys:
                totals[key] += weights[name] * values.get(key, 0.0)
        totals = {key: value / total_weight for key, value in totals.items()}

        top_key = ordered_keys[0]
        top_score = totals.get(top_key, 0.0)
        runner_scores = sorted(
            (totals.get(key, 0.0) for key in ordered_keys[1:]), reverse=True,
        )
        second_score = runner_scores[0] if runner_scores else 0.0
        margin = max(0.0, top_score - second_score)
        ratio = top_score / second_score if second_score > 1e-9 else (
            float("inf") if top_score > 0 else 0.0
        )

        agreeing: list[str] = []
        disagreeing: list[str] = []
        for name, values in normalized.items():
            if name in weak_signals:
                continue
            ranked = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            if not ranked:
                continue
            source_margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            if source_margin <= profile.near_tie_margin:
                continue
            if ranked[0][0] == top_key:
                agreeing.append(name)
            else:
                disagreeing.append(name)

        confirmed_top = top_key in confirmed_keys and not any(
            key in confirmed_keys for key in ordered_keys[1:]
        )
        if confirmed_top:
            level = "high"
            reason = "unique confirmed continuity target"
        elif len(ordered_keys) == 1:
            if len(agreeing) >= 2:
                level = "high"
                reason = "the only candidate is independently supported by multiple signals"
            else:
                level = "low"
                reason = "the only candidate lacks independent corroboration"
        elif margin <= profile.near_tie_margin or ratio < profile.medium_ratio:
            level = "low"
            reason = "top candidates are near-tied"
        elif disagreeing and len(disagreeing) >= len(agreeing):
            level = "low"
            reason = "strong ranking signals disagree"
        elif (
            margin >= profile.high_margin
            and ratio >= profile.high_ratio
            and len(agreeing) >= 2
        ) or (len(agreeing) >= 3 and margin >= profile.medium_margin):
            level = "high"
            reason = "clear winner supported by independent signals"
        elif (
            margin >= profile.medium_margin
            and ratio >= profile.medium_ratio
            and agreeing
        ):
            level = "medium"
            reason = "moderately separated winner"
        else:
            level = "low"
            reason = "winner depends on insufficient or weak evidence"

        assessment = ConfidenceAssessment(
            level=level,
            top_score=round(top_score, 6),
            second_score=round(second_score, 6),
            margin=round(margin, 6),
            ratio=round(ratio, 6) if math.isfinite(ratio) else ratio,
            agreeing_signals=tuple(sorted(agreeing)),
            disagreeing_signals=tuple(sorted(disagreeing)),
            reason=reason,
            candidate_scores=tuple(
                (key, round(totals.get(key, 0.0), 6)) for key in ordered_keys
            ),
            candidate_support=tuple(
                (
                    key,
                    tuple(sorted(
                        name
                        for name, values in normalized.items()
                        if name not in weak_signals and values.get(key, 0.0) >= 0.5
                    )),
                )
                for key in ordered_keys
            ),
        )
        log_debug_payload(
            _logger, f"retrieval.{kind}_confidence",
            top_candidate=top_key,
            top_score=assessment.top_score,
            second_score=assessment.second_score,
            margin=assessment.margin,
            ratio=assessment.ratio,
            confidence=assessment.level,
            confidence_reason=assessment.reason,
            agreeing_signals=assessment.agreeing_signals,
            disagreeing_signals=assessment.disagreeing_signals,
            normalized_signals=normalized,
        )
        return assessment

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
    def continuity_groups(continuity: ContinuityContext) -> list[dict[str, object]]:
        """Return bounded historical target data for prompt rendering."""
        return [
            {
                "entities": list(group.entities[:12]),
                "areas": list(group.areas[:4]),
                "floors": list(group.floors[:4]),
                "domains": list(group.domains[:4]),
                "device_classes": list(group.device_classes[:4]),
                "tool": group.tool,
                "action": group.action,
            }
            for group, _ in continuity.target_groups[:2]
        ]

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
    def device_search_confidence(
        devices: Iterable[T],
        vector_results: Iterable[ScoredResult[T]] = (),
        *,
        query: str = "",
        key: Callable[[T], str] = lambda item: str(getattr(item, "id", "")),
        text_parts: Callable[[T], Iterable[object]] | None = None,
        metadata_score: Callable[[T], float] | None = None,
        structured_scores: dict[str, Callable[[T], float]] | None = None,
        continuity_score: Callable[[T], float] | None = None,
        confirmed_score: Callable[[T], float] | None = None,
    ) -> ConfidenceAssessment:
        """Measure per-target confidence from its local candidate distribution."""
        devices = list(devices)
        keys = [key(device) for device in devices]
        vector = {key(result.item): result.score for result in vector_results}
        signals: dict[str, dict[str, float]] = {"vector": vector}
        if metadata_score:
            signals["metadata"] = {
                key(device): metadata_score(device) for device in devices
            }
        for name, score in (structured_scores or {}).items():
            signals[name] = {key(device): score(device) for device in devices}
        if continuity_score:
            signals["continuity"] = {
                key(device): continuity_score(device) for device in devices
            }
        confirmed_keys: set[str] = set()
        if confirmed_score:
            confirmed = {key(device): confirmed_score(device) for device in devices}
            signals["confirmed_continuity"] = confirmed
            confirmed_keys = {item_key for item_key, score in confirmed.items() if score > 0}
        if text_parts:
            signals["lexical"] = {
                key(device): RetrievalHelper.field_match_score(query, text_parts(device))
                for device in devices
            }
        return RetrievalHelper.assess_distribution_confidence(
            keys,
            signals,
            profile=DEVICE_CONFIDENCE_PROFILE,
            weak_signals={"lexical"},
            confirmed_keys=confirmed_keys,
            kind="device",
        )

    @staticmethod
    def select_device_candidates(
        query: str,
        devices: Iterable[T],
        min_limit: int,
        max_limit: int | None = None,
        confidence: ConfidenceAssessment | str | None = None,
    ) -> list[T]:
        """Expose the plausible ambiguity cluster within the configured ceiling."""
        if min_limit <= 0 and (max_limit is None or max_limit <= 0):
            return []
        devices = list(devices)
        min_limit = max(0, min_limit)
        ceiling = min_limit if max_limit is None else max(0, int(max_limit))
        if ceiling <= 0 or not devices:
            return []
        if not isinstance(confidence, ConfidenceAssessment) or not confidence.candidate_scores:
            # Runtime retrieval supplies score diagnostics. Keep third-party
            # compatibility callers bounded when those diagnostics are absent.
            return devices[:ceiling]

        scores = dict(confidence.candidate_scores)
        support = dict(confidence.candidate_support)
        top_key = confidence.candidate_scores[0][0]
        top_score = scores.get(top_key, 0.0)
        margin_threshold = DEVICE_CONFIDENCE_PROFILE.medium_margin
        ratio_threshold = DEVICE_CONFIDENCE_PROFILE.medium_ratio
        near_tie_threshold = DEVICE_CONFIDENCE_PROFILE.near_tie_margin
        selected: list[T] = []
        decisions: list[dict[str, object]] = []
        previous_score = top_score

        for index, device in enumerate(devices):
            candidate_key = str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            score = scores.get(candidate_key, 0.0)
            candidate_margin = max(0.0, top_score - score)
            candidate_ratio = top_score / score if score > 1e-9 else (
                float("inf") if top_score > 0 else 1.0
            )
            step_drop = max(0.0, previous_score - score)
            supporting_signals = support.get(candidate_key, ())
            near_tied = (
                candidate_margin <= near_tie_threshold
                or candidate_ratio < ratio_threshold
            )
            within_cluster = (
                candidate_margin <= margin_threshold
                and candidate_ratio <= ratio_threshold
            )
            independently_supported = len(supporting_signals) >= 2

            if index == 0:
                include = True
                reason = "top-ranked candidate anchors the ambiguity cluster"
            elif not within_cluster:
                include = False
                reason = "excluded at meaningful top-score drop-off"
            elif not near_tied and not independently_supported:
                include = False
                reason = "excluded because ranking evidence lacks corroboration"
            else:
                include = True
                reason = (
                    "included because it is near-tied with the top candidate"
                    if near_tied
                    else "included because independent structured signals support it"
                )

            if include and len(selected) < ceiling:
                selected.append(device)
            elif include:
                include = False
                reason = "excluded by configured maximum ceiling"
            decisions.append({
                "candidate": candidate_key,
                "included": include,
                "reason": reason,
                "score": round(score, 6),
                "top_margin": round(candidate_margin, 6),
                "top_ratio": (
                    round(candidate_ratio, 6)
                    if math.isfinite(candidate_ratio) else candidate_ratio
                ),
                "step_drop": round(step_drop, 6),
                "supporting_signals": supporting_signals,
            })
            previous_score = score
            if index > 0 and not within_cluster:
                for excluded in devices[index + 1:]:
                    excluded_key = str(
                        RetrievalHelper._device_value(excluded, "id", "")
                        or RetrievalHelper._device_value(excluded, "name", "")
                    )
                    decisions.append({
                        "candidate": excluded_key,
                        "included": False,
                        "reason": "excluded after earlier meaningful score drop-off",
                        "score": round(scores.get(excluded_key, 0.0), 6),
                        "supporting_signals": support.get(excluded_key, ()),
                    })
                break

        log_debug_payload(
            _logger, "retrieval.device_ambiguity_cluster",
            confidence=confidence.level,
            confidence_reason=confidence.reason,
            top_score=confidence.top_score,
            second_score=confidence.second_score,
            margin=confidence.margin,
            ratio=confidence.ratio,
            configured_minimum=min_limit,
            configured_maximum=ceiling,
            score_dropoff_margin_threshold=margin_threshold,
            score_dropoff_ratio_threshold=ratio_threshold,
            near_tie_margin_threshold=near_tie_threshold,
            selected_candidate_count=len(selected),
            selected_candidates=[
                str(RetrievalHelper._device_value(device, "id", ""))
                for device in selected
            ],
            candidate_decisions=decisions,
        )
        return selected

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
            if context.entities or context.target_groups:
                relevance += 0.3 * decay
            elif context.has_canonical_context:
                relevance += 0.05 * decay
            if similarity >= 0.2:
                selected[context.key] = (context, relevance)

            # Keep the configured number of recent retained turns, including
            # clarification questions
            # with no successful tool result. The existing LLM needs their text
            # to resolve an unfinished request without a language parser.
            if index >= len(contexts) - limit:
                short_term_weight = 0.15 * decay
                previous = selected.get(context.key)
                if previous is None or short_term_weight > previous[1]:
                    selected[context.key] = (context, short_term_weight)

        result = sorted(selected.values(), key=lambda item: item[1], reverse=True)[:limit]
        log_debug_payload(
            _logger, "continuity.history_selection", contexts=contexts,
            vectors=vectors, current_vector=current_vector,
            max_age_seconds=max_age_seconds, limit=limit, now=now, selected=result,
        )
        return result

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
                "floors",
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
        log_debug_payload(
            _logger, "continuity.built", selected_contexts=selected_contexts,
            continuity=continuity,
        )
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
    def _character_ngrams(text: str, size: int = 3) -> set[str]:
        compact = text.replace(" ", "")
        if len(compact) <= size:
            return {compact} if compact else set()
        return {compact[index:index + size] for index in range(len(compact) - size + 1)}

    @staticmethod
    def _match_scores(query: str, values: Iterable[object]) -> tuple[float, float]:
        query_text = RetrievalHelper._normalize(query)
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

        result = [candidates[item_key] for item_key in selected_keys]
        log_debug_payload(
            _logger, "retrieval.device_ranking", query=query, limit=limit,
            vector_results=vector_results, lexical_items=lexical_items,
            candidates=candidates, vector_ranking=vector_ranking,
            lexical_scores=lexical_scores, match_scores=match_scores,
            metadata_scores=metadata_scores, continuity_scores=continuity_scores,
            fused_scores=fused, ordered_keys=ordered_keys,
            selected_keys=selected_keys, selected=result,
        )
        return result

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
        schema_domains = getattr(tool, "schema_domains", None)
        domains = set(schema_domains) if schema_domains is not None else RetrievalHelper._schema_values(properties.get("domain", {}))
        domains.update(RetrievalHelper._metadata_value(tool, "supported_domains", ()) or ())
        return {str(domain).casefold() for domain in domains if domain}

    @staticmethod
    def _tool_domain_signal(tool: Any, requested_domains: set[str]) -> float:
        if not requested_domains:
            return 0.0
        declared_domains = RetrievalHelper._tool_declared_domains(tool)
        if declared_domains:
            return 1.0 if declared_domains & requested_domains else 0.0
        return 0.0

    @staticmethod
    def normalize_requested_capability(capability: object) -> dict[str, object]:
        """Normalize model-provided structured capability metadata."""
        if not isinstance(capability, dict):
            return {}
        action = str(capability.get("action", "") or "").strip().casefold()
        domains = capability.get("domains", capability.get("domain", ())) or ()
        if isinstance(domains, str):
            domains = (domains,)
        return {
            "action": action,
            "domains": tuple(sorted({str(value).strip().casefold() for value in domains if value})),
        }

    @staticmethod
    def tool_capability_compatibility(tool: Any, capability: object) -> float:
        """Deterministically compare requested and declared tool capabilities.

        A negative value is an explicit contradiction. Missing metadata remains
        neutral so incomplete device or tool metadata cannot suppress recovery.
        """
        requested = RetrievalHelper.normalize_requested_capability(capability)
        if not requested:
            return 0.0
        requested_action = str(requested.get("action", "") or "")
        requested_domains = set(requested.get("domains", ()) or ())
        tool_action = str(getattr(tool, "canonical_action", "") or "").casefold()
        tool_domains = RetrievalHelper._tool_declared_domains(tool)
        if requested_action and tool_action and requested_action != tool_action:
            return -1.0
        if requested_domains and tool_domains and not requested_domains & tool_domains:
            return -1.0
        score = 0.0
        if requested_action and tool_action == requested_action:
            score += 1.0
        if requested_domains and tool_domains & requested_domains:
            score += 0.5
        return score

    @staticmethod
    def tool_ranking_signals(
        tool: Any,
        query: str,
        devices: Iterable[Any] = (),
        semantic_rank: int | None = None,
        semantic_score: float | None = None,
        continuity: float = 0.0,
        lexical_match: tuple[float, float] | None = None,
        requested_capability: object = None,
    ) -> dict[str, float]:
        """Return independent, extensible signals used to rank a tool."""
        devices = list(devices)
        query = RetrievalHelper._tool_query_text(query)
        requested_domains = RetrievalHelper._device_domains(devices)
        exact, fuzzy = lexical_match if lexical_match is not None else RetrievalHelper._match_scores(
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
            "capability": RetrievalHelper.tool_capability_compatibility(tool, requested_capability),
            "domain": RetrievalHelper._tool_domain_signal(tool, requested_domains),
            "device_metadata": max(-1.0, min(1.0, compatibility / 2.0)),
            "continuity": max(0.0, continuity),
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
        requested_capability: object = None,
    ) -> list[T]:
        """Rank a broad tool pool without discarding uncertain candidates."""
        if limit <= 0:
            return []
        devices = list(devices)
        vector_results = list(vector_results)
        lexical_tools = list(lexical_tools)
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
        index = lexical_index(documents)
        corpus_scores = dict(zip(corpus_names, index.scores(query)))
        field_scores = index.match_scores(query)
        matches_by_name = {
            name: field_scores.get(position, (0.0, 0.0))
            for position, name in enumerate(corpus_names)
        }
        scored: list[tuple[float, int, str, T]] = []
        signals_by_name: dict[str, dict[str, float]] = {}
        for name, tool in candidate_by_name.items():
            continuity = continuity_score(tool) if continuity_score else 0.0
            signals = RetrievalHelper.tool_ranking_signals(
                tool,
                query,
                devices,
                semantic_rank=semantic_ranks.get(name),
                semantic_score=semantic_scores.get(name),
                continuity=continuity,
                lexical_match=matches_by_name[name],
                requested_capability=requested_capability,
            )
            signals["lexical_corpus"] = corpus_scores[name]
            signals_by_name[name] = signals
            scored.append(
                (
                    RetrievalHelper.tool_signal_score(signals),
                    semantic_ranks.get(name, len(semantic_ranks) + 1),
                    name,
                    tool,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if requested_capability and any(
            signals["capability"] >= 0 for signals in signals_by_name.values()
        ):
            scored = [
                item for item in scored
                if signals_by_name[item[2]]["capability"] >= 0
            ]
        # Near-identical tools must not monopolize a small result window. Keep
        # the best member of each declared/observable capability first, then
        # use duplicates only to fill remaining slots.
        diverse: list[tuple[float, int, str, T]] = []
        duplicates: list[tuple[float, int, str, T]] = []
        seen_signatures: set[tuple[object, ...]] = set()
        for item in scored:
            tool = item[3]
            signature = (
                str(getattr(tool, "canonical_action", "") or "").casefold(),
                tuple(getattr(tool, "canonical_supported_domains", ()) or ()),
                RetrievalHelper._normalize(getattr(tool, "description", "")),
                tuple(getattr(tool, "canonical_schema_parts", ()) or ()),
            )
            if not any(signature):
                signature = (*signature, str(getattr(tool, "name", "")))
            if signature in seen_signatures:
                duplicates.append(item)
            else:
                seen_signatures.add(signature)
                diverse.append(item)
        selected = [*diverse[:limit]]
        if len(selected) < limit:
            selected.extend(duplicates[:limit - len(selected)])
        result = [tool for _, _, _, tool in selected]
        if _logger.isEnabledFor(logging.DEBUG):
            log_debug_payload(
                _logger, "retrieval.tool_ranking", query=query, limit=limit,
                requested_capability=requested_capability, devices=devices,
                vector_results=vector_results, lexical_tools=lexical_tools,
                candidates=candidate_by_name, semantic_ranks=semantic_ranks,
                semantic_scores=semantic_scores, corpus_scores=corpus_scores,
                lexical_matches=matches_by_name, signals=signals_by_name,
                scored=[{"score": score, "semantic_rank": rank, "name": name}
                        for score, rank, name, _ in scored],
                selected=result,
            )
        return result

    @staticmethod
    def tool_search_confidence_details(
        tools: Iterable[Any],
        query: str,
        devices: Iterable[Any],
        requested_capability: object = None,
        vector_results: Iterable[ScoredResult[Any]] = (),
        continuity_score: Callable[[Any], float] | None = None,
    ) -> ConfidenceAssessment:
        """Measure per-capability confidence from separation and schema evidence."""
        tools = list(tools)
        if not tools:
            return ConfidenceAssessment(level="none")
        query = RetrievalHelper._tool_query_text(query)
        devices = list(devices)
        names = [str(getattr(tool, "name", "")) for tool in tools]
        semantic = {
            str(getattr(result.item, "name", "")): result.score
            for result in vector_results
        }
        requested = RetrievalHelper.normalize_requested_capability(requested_capability)
        requested_action = str(requested.get("action", "") or "")
        requested_domains = set(requested.get("domains", ()) or ())
        signals: dict[str, dict[str, float]] = {
            "vector": semantic,
            "action_schema": {
                str(getattr(tool, "name", "")): float(
                    bool(requested_action)
                    and str(getattr(tool, "canonical_action", "") or "").casefold() == requested_action
                )
                for tool in tools
            },
            "domain_schema": {
                str(getattr(tool, "name", "")): float(
                    bool(requested_domains)
                    and bool(RetrievalHelper._tool_declared_domains(tool) & requested_domains)
                )
                for tool in tools
            },
            "device_compatibility": {
                str(getattr(tool, "name", "")): max(
                    0.0, RetrievalHelper.tool_device_compatibility(tool, devices),
                )
                for tool in tools
            },
            "lexical": {
                str(getattr(tool, "name", "")): RetrievalHelper.field_match_score(
                    query, getattr(tool, "canonical_search_parts", ()) or (),
                )
                for tool in tools
            },
        }
        if continuity_score:
            signals["continuity"] = {
                str(getattr(tool, "name", "")): continuity_score(tool)
                for tool in tools
            }
        return RetrievalHelper.assess_distribution_confidence(
            names,
            signals,
            profile=TOOL_CONFIDENCE_PROFILE,
            weak_signals={"lexical"},
            kind="tool",
        )

    @staticmethod
    def tool_search_confidence(
        tools: Iterable[Any], query: str, devices: Iterable[Any], requested_capability: object = None,
        vector_results: Iterable[ScoredResult[Any]] = (),
        continuity_score: Callable[[Any], float] | None = None,
    ) -> str:
        """Return the explainable distribution-confidence level."""
        return RetrievalHelper.tool_search_confidence_details(
            tools,
            query,
            devices,
            requested_capability,
            vector_results,
            continuity_score,
        ).level

    @staticmethod
    def rank_tools_for_query(
        tools: Iterable[T], query: str, devices: Iterable[Any] = (), requested_capability: object = None,
    ) -> list[T]:
        tools = list(tools)
        return RetrievalHelper.rank_tool_candidates(
            [], tools, query, devices, len(tools), requested_capability=requested_capability,
        )

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
        allowed_domains = getattr(tool, "schema_domains", None)
        if allowed_domains is None:
            allowed_domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        allowed_classes = getattr(tool, "schema_device_classes", None)
        if allowed_classes is None:
            allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        if allowed_domains:
            score += 2.0 if domains & allowed_domains else 0.0
        if allowed_classes:
            score += 2.0 if device_classes & allowed_classes else 0.0

        if (
            RetrievalHelper._metadata_value(tool, "is_domain_aware")
            and (not allowed_domains or domains & allowed_domains)
        ):
            score += 0.5
        if (
            RetrievalHelper._metadata_value(tool, "is_device_class_aware")
            and device_classes
            and (not allowed_classes or device_classes & allowed_classes)
        ):
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
        """Jointly rerank with a bounded positive device-compatibility boost."""
        devices = list(devices)
        tools = list(tools)
        ranked = sorted(
            enumerate(tools),
            key=lambda pair: (
                -(
                    1.0 / (pair[0] + 1)
                    + 0.3 * min(
                        2.0,
                        max(
                            0.0,
                            RetrievalHelper.tool_device_compatibility(pair[1], devices),
                        ),
                    )
                ),
                pair[0],
            ),
        )
        return [tool for _, tool in ranked[:limit]]
