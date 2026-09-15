"""History selection and continuity aggregation for retrieval."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable

from custom_components.ha_ragent.src.logging import log_debug_payload
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext


_logger = logging.getLogger(__name__)


class HistoryRetriever:
    """Keep conversational continuity separate from source retrieval/ranking."""

    @staticmethod
    def build_retrieval_text(current_request: str) -> str:
        return " ".join(current_request.split())

    @staticmethod
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        left, right = list(left), list(right)
        if len(left) != len(right) or not left:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot_product / (left_norm * right_norm) if left_norm and right_norm else 0.0

    @classmethod
    def select_history_contexts(
        cls, contexts: Iterable[TurnContext], vectors: dict[str, list[float]],
        current_vector: list[float], max_age_seconds: float = 300.0,
        limit: int = 3, now: float | None = None,
    ) -> list[tuple[TurnContext, float]]:
        now = time.time() if now is None else now
        contexts = list(contexts)
        selected: dict[str, tuple[TurnContext, float]] = {}
        for index, context in enumerate(contexts):
            fallback_age = float((len(contexts) - index - 1) * 30)
            age = max(0.0, now - context.created_at) if context.created_at is not None else fallback_age
            if age > max_age_seconds:
                continue
            decay = 0.5 ** (age / 120.0)
            similarity = max(0.0, cls.cosine_similarity(current_vector, vectors.get(context.key, [])))
            relevance = similarity * decay
            if context.entities or context.target_groups:
                relevance += 0.3 * decay
            elif context.has_canonical_context:
                relevance += 0.05 * decay
            if similarity >= 0.2:
                selected[context.key] = (context, relevance)
            if index >= len(contexts) - limit:
                short_term_weight = 0.15 * decay
                previous = selected.get(context.key)
                if previous is None or short_term_weight > previous[1]:
                    selected[context.key] = (context, short_term_weight)
        result = sorted(selected.values(), key=lambda item: item[1], reverse=True)[:limit]
        log_debug_payload(
            _logger, "continuity.history_selection",
            contexts=contexts, vectors=vectors, current_vector=current_vector,
            max_age_seconds=max_age_seconds, limit=limit, now=now, selected=result,
        )
        return result

    @staticmethod
    def build_continuity_context(
        selected_contexts: Iterable[tuple[TurnContext, float]],
    ) -> ContinuityContext:
        continuity = ContinuityContext()
        selected_contexts = list(selected_contexts)
        for context, weight in selected_contexts:
            continuity.selected_turn_keys.add(context.key)
            for attribute in (
                "entities", "tools", "areas", "floors", "domains", "device_classes",
                "actions", "ambiguous_entities",
            ):
                target = getattr(continuity, attribute)
                for value in getattr(context, attribute):
                    normalized = str(value).casefold()
                    target[normalized] = max(target.get(normalized, 0.0), weight)
        recent_contexts = sorted(selected_contexts, key=lambda item: item[0].created_at or 0.0, reverse=True)
        continuity.target_groups = [
            (group, weight) for context, weight in recent_contexts for group in context.target_groups
        ]
        log_debug_payload(
            _logger, "continuity.built",
            selected_contexts=selected_contexts, continuity=continuity,
        )
        return continuity
