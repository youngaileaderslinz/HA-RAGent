from __future__ import annotations

import logging
import math
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, TypeVar

from custom_components.ha_ragent.src.const import (
    DEVICE_CONFIDENCE_NEAR_TIE_MARGIN,
    TOOL_CONFIDENCE_NEAR_TIE_MARGIN,
    DEVICE_SELECTION_ABSOLUTE_FLOOR,
    DEVICE_SELECTION_RELATIVE_FLOOR,
    DEVICE_SELECTION_GAP_THRESHOLD,
    DEVICE_CONTINUITY_MAX_BOOST,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import (
    ConfidenceAssessment,
)
from custom_components.ha_ragent.src.models.retrieval.confidence_profile import (
    ConfidenceProfile,
)
from custom_components.ha_ragent.src.models.embedding.schema_constraints import root_property_values
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext
from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    normalize_canonical_text,
)

from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index, match_features, match_score
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.homeassistant.helpers.source_retriever import SourceRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.history_retriever import HistoryRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.logging.helper import log_debug_payload

T = TypeVar("T")
_logger = logging.getLogger(__name__)


DEVICE_CONFIDENCE_PROFILE = ConfidenceProfile(
    DEVICE_CONFIDENCE_NEAR_TIE_MARGIN,
)
TOOL_CONFIDENCE_PROFILE = ConfidenceProfile(
    TOOL_CONFIDENCE_NEAR_TIE_MARGIN,
)

class RetrievalHelper(RetrievalConfidence):
    """Stateless helpers for building and reranking retrieval queries."""

    @staticmethod
    async def async_retrieve_sources(
        backend: Any, object_type: type, options: dict, collection: str,
        embedding: list[float] | QueryEmbedding, limit: int, query: str = "",
    ) -> tuple[list, list]:
        """Retrieve and combine sources through the specialized retrievers."""
        return await SourceRetriever.async_retrieve_sources(
            backend, object_type, options, collection, embedding, limit, query,
        )

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
        capability: object = None,
    ) -> str:
        """Combine the supplied query with declared action/domain evidence.

        Retain the original query for tools whose description or schema is
        their only capability information. Add no language-specific labels.
        """
        requested = RetrievalHelper.normalize_requested_capability(capability)
        if requested:
            action = str(requested.get("action", "") or "")
            domains = tuple(requested.get("domains", ()) or ())
            # Only declared intent domains belong in the retrieval query.
            all_domains = domains
            return "\n".join(dict.fromkeys(
                part for part in (trusted_query, fallback_query, action, *all_domains) if part
            ))

        query = trusted_query or fallback_query
        if trusted_query and fallback_query and fallback_query != trusted_query:
            query += f"\n{fallback_query}"
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
        ranking_evidence: dict[str, dict[str, float]] | None = None,
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> ConfidenceAssessment:
        """Measure per-target confidence from its local candidate distribution."""
        devices = list(devices)
        keys = [key(device) for device in devices]
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            evidence = ranking_evidence if ranking_evidence is not None else {}
            if retrieval_method not in evidence:
                SourceRanker.rank(
                    vector_results, devices, query, key, text_parts or (lambda item: ()),
                    len(devices), retrieval_method, evidence,
                )
            return SourceRanker.confidence(
                keys, retrieval_method, evidence, DEVICE_CONFIDENCE_PROFILE, "device",
            )
        vector = {key(result.item): result.score for result in vector_results}
        signals: dict[str, dict[str, float]] = {"vector": vector}
        if metadata_score:
            signals["metadata"] = {
                key(device): metadata_score(device) for device in devices
            }
        for name, score in (structured_scores or {}).items():
            signals[name] = {key(device): score(device) for device in devices}
        # Current-request evidence establishes confidence and eligibility.
        # Historical context is applied afterwards as a capped prior so it
        # cannot make an old target outrank a stronger current match.
        continuity_raw = {
            key(device): continuity_score(device) for device in devices
        } if continuity_score else {}
        if confirmed_score:
            for device in devices:
                item_key = key(device)
                continuity_raw[item_key] = max(
                    continuity_raw.get(item_key, 0.0), confirmed_score(device),
                )
        if ranking_evidence and "lexical" in ranking_evidence:
            signals["lexical"] = {
                item_key: ranking_evidence["lexical"].get(item_key, 0.0)
                for item_key in keys
            }
        elif text_parts:
            # Match device confidence to the lexical TF-IDF signal used by
            # device ranking; field matching remains an identity feature,
            # not an independent confidence distribution.
            documents = tuple(
                tuple(str(value) for value in text_parts(device) if value)
                for device in devices
            )
            signals["lexical"] = dict(zip(
                keys, lexical_index(documents).scores(query),
            ))
        assessment = RetrievalHelper.assess_distribution_confidence(
            keys,
            signals,
            profile=DEVICE_CONFIDENCE_PROFILE,
            weak_signals={"lexical"},
            kind="device",
        )
        # Keep raw fusion strictly as an ordering diagnostic.  Eligibility is
        # based on the calibrated, locally normalized confidence values above;
        # RRF scores are normally around 0.01--0.05 and are not thresholds.
        if ranking_evidence and "fused" in ranking_evidence:
            fused_scores = ranking_evidence["fused"]
            fused_order = sorted(
                keys, key=lambda item_key: (-fused_scores.get(item_key, 0.0), item_key),
            )
            assessment = replace(
                assessment,
                final_candidate_scores=tuple(
                    (item_key, fused_scores.get(item_key, 0.0))
                    for item_key in fused_order
                ),
            )
        current_scores = dict(assessment.candidate_scores)
        continuity_boosts = {
            item_key: min(
                DEVICE_CONTINUITY_MAX_BOOST,
                max(0.0, float(raw_score)),
            )
            for item_key, raw_score in continuity_raw.items()
        }
        final_scores = {
            item_key: current_scores.get(item_key, 0.0)
            + continuity_boosts.get(item_key, 0.0)
            for item_key in current_scores
        }
        original_positions = {
            item_key: index for index, (item_key, _score)
            in enumerate(assessment.final_candidate_scores or assessment.candidate_scores)
        }
        final_order = sorted(
            final_scores,
            key=lambda item_key: (-final_scores[item_key], original_positions[item_key]),
        )
        assessment = replace(
            assessment,
            final_candidate_scores=tuple(
                (item_key, round(final_scores[item_key], 6))
                for item_key in final_order
            ),
            continuity_boosts=tuple(
                (item_key, round(continuity_boosts.get(item_key, 0.0), 6))
                for item_key in final_order
            ),
        )
        if _logger.isEnabledFor(logging.DEBUG):
            hybrid_scores = dict(assessment.candidate_scores)
            _logger.debug(
                "RAGent device confidence: %s",
                [
                    {
                        "candidate": candidate_key,
                        "current_score": round(hybrid_scores.get(candidate_key, 0.0), 3),
                        "final_score": round(final_scores.get(candidate_key, 0.0), 3),
                        "continuity_boost": round(continuity_boosts.get(candidate_key, 0.0), 3),
                        "vector": round(vector.get(candidate_key, 0.0), 3),
                        "metadata": round(signals.get("metadata", {}).get(candidate_key, 0.0), 3),
                        "lexical": round(signals.get("lexical", {}).get(candidate_key, 0.0), 3),
                        "support": assessment.candidate_support[index][1],
                    }
                    for index, candidate_key in enumerate(
                        key for key, _score in assessment.candidate_scores[:12]
                    )
                ],
            )
        return assessment

    @staticmethod
    def select_device_candidates(
        query: str,
        devices: Iterable[T],
        min_limit: int,
        max_limit: int | None = None,
        confidence: ConfidenceAssessment | str | None = None,
        preferred_domains: Iterable[str] = (),
        preferred_areas: Iterable[str] = (),
        preserve_score: Callable[[T], float] | None = None,
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> list[T]:
        """Expose the plausible ambiguity cluster within the configured ceiling."""
        if min_limit <= 0 and (max_limit is None or max_limit <= 0):
            return []
        devices = list(devices)
        min_limit = max(0, min_limit)
        ceiling = min_limit if max_limit is None else max(0, int(max_limit))
        if ceiling <= 0 or not devices:
            return []
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            return devices[:ceiling]
        if not isinstance(confidence, ConfidenceAssessment) or not confidence.candidate_scores:
            # Runtime retrieval supplies score diagnostics. Keep third-party
            # compatibility callers precise when those diagnostics are absent.
            return devices[:min(ceiling, max(1, min_limit))]

        # Confidence thresholds use current-turn evidence. Final ordering may
        # include a small, bounded continuity boost, but that prior must never
        # set the relevance baseline for the current request.
        scores = dict(confidence.candidate_scores)
        final_scores = dict(confidence.final_candidate_scores) or dict(scores)
        support = dict(confidence.candidate_support)
        top_key, top_score = max(scores.items(), key=lambda item: item[1])
        devices_by_key = {
            str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ): device
            for device in devices
        }
        scored_devices = [
            devices_by_key[candidate_key]
            for candidate_key, _score in (
                confidence.final_candidate_scores or confidence.candidate_scores
            )
            if candidate_key in devices_by_key
        ]
        scored_keys = {
            str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            for device in scored_devices
        }
        scored_devices.extend(
            device
            for device in devices
            if str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ) not in scored_keys
        )
        current_scored_devices = sorted(
            scored_devices,
            key=lambda device: scores.get(
                str(
                    RetrievalHelper._device_value(device, "id", "")
                    or RetrievalHelper._device_value(device, "name", "")
                ),
                0.0,
            ),
            reverse=True,
        )
        near_tie_threshold = DEVICE_CONFIDENCE_PROFILE.near_tie_margin
        preferred_domains = {
            str(domain).casefold() for domain in preferred_domains if domain
        }
        preferred_areas = {
            str(area).casefold() for area in preferred_areas if area
        }
        preferred_domain_candidates = {
            str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            for device in scored_devices
            if (
                {
                    str(value).casefold()
                    for value in (
                        ([RetrievalHelper._device_value(device, "domain", "")]
                         if isinstance(RetrievalHelper._device_value(device, "domain", ""), str)
                         else RetrievalHelper._device_value(device, "domain", ()) or ())
                    )
                }
                & preferred_domains
            )
        }
        preferred_area_candidates = {
            str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            for device in scored_devices
            if str(
                RetrievalHelper._device_value(device, "area_name", "")
                or RetrievalHelper._device_value(device, "area", "")
                or ""
            ).casefold() in preferred_areas
        }
        # ``minimum`` is a safety setting, not a desired exposure count. Prune
        # using the final hybrid confidence distribution, retaining explicit
        # identity evidence even when its vector score is weak.
        absolute_floor = DEVICE_SELECTION_ABSOLUTE_FLOOR
        relative_floor = DEVICE_SELECTION_RELATIVE_FLOOR
        gap_threshold = DEVICE_SELECTION_GAP_THRESHOLD
        explicit_identity_keys = {
            candidate_key
            for candidate_key, _score in confidence.candidate_scores
            if "identity_metadata" in support.get(candidate_key, ())
        }
        eligible_keys: set[str] = set()
        previous_score = top_score
        for index, device in enumerate(current_scored_devices):
            candidate_key = str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            score = scores.get(candidate_key, 0.0)
            supporting_signals = support.get(candidate_key, ())
            explicit_identity = candidate_key in explicit_identity_keys
            if not explicit_identity and (
                score < absolute_floor or score < top_score * relative_floor
            ):
                continue
            if index and previous_score - score >= gap_threshold and not explicit_identity:
                continue
            eligible_keys.add(candidate_key)
            previous_score = score
        if not eligible_keys and scored_devices:
            # Weak/flat distributions retain one uncertain candidate; semantic
            # search remains available to recover a missed target.
            eligible_keys.add(str(
                RetrievalHelper._device_value(scored_devices[0], "id", "")
                or RetrievalHelper._device_value(scored_devices[0], "name", "")
            ))
        selected: list[T] = []
        decisions: list[dict[str, object]] = []
        selected_keys: set[str] = set()
        previous_score = top_score
        candidate_keys = eligible_keys
        # A direct, exact identity mention is stronger current-turn evidence
        # than continuity. Preserve historical targets for ambiguous language,
        # including false-high distributions, but never replace an explicitly
        # requested new target such as "Current lamp".
        has_explicit_current_target = any(
            RetrievalHelper.device_target_score(query, device) >= 0.9
            for device in devices
        )
        preserved_keys = {
            candidate_key
            for candidate_key, device in devices_by_key.items()
            if (
                preserve_score is not None
                and not has_explicit_current_target
                and preserve_score(device) > 0
            )
        }
        # Successful historical targets are continuity evidence, not current
        # identity evidence. Carry them through this final gate independently
        # of the current score thresholds.
        candidate_keys |= preserved_keys
        target_count = min(ceiling, len(candidate_keys))

        for index, device in enumerate(scored_devices):
            candidate_key = str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            current_score = scores.get(candidate_key, 0.0)
            score = final_scores.get(candidate_key, current_score)
            candidate_margin = max(0.0, top_score - current_score)
            step_drop = max(0.0, previous_score - score)
            supporting_signals = support.get(candidate_key, ())
            raw_candidate_domains = RetrievalHelper._device_value(device, "domain", ()) or ()
            if isinstance(raw_candidate_domains, str):
                raw_candidate_domains = (raw_candidate_domains,)
            candidate_domains = {
                str(value).casefold() for value in raw_candidate_domains
            }
            preferred = bool(preferred_domains & candidate_domains)
            candidate_area = str(
                RetrievalHelper._device_value(device, "area_name", "")
                or RetrievalHelper._device_value(device, "area", "")
                or ""
            ).casefold()
            area_compatible = not preferred_areas or candidate_area in preferred_areas
            plausible = candidate_key in candidate_keys
            preserved = candidate_key in preserved_keys
            if preferred_domain_candidates and not preferred and not preserved:
                plausible = False
            if preferred_area_candidates and not area_compatible and not preserved:
                plausible = False
            include = (
                plausible
                and len(selected) < ceiling
                and (preserved or len(selected) < target_count)
            )
            reason = (
                "preserved successful target"
                if preserved else
                "top-ranked candidate anchors recall"
                if index == 0 else
                "included from the confidence band"
                if include else
                "excluded as non-plausible or beyond configured recall budget"
            )
            if include:
                selected.append(device)
                selected_keys.add(candidate_key)
            decisions.append({
                "candidate": candidate_key,
                "included": include,
                "reason": reason,
                "score": round(score, 6),
                "current_score": round(current_score, 6),
                "continuity_boost": round(score - current_score, 6),
                "top_margin": round(candidate_margin, 6),
                "step_drop": round(max(0.0, previous_score - score), 6),
                "supporting_signals": supporting_signals,
                "preserved_successful_target": preserved,
            })
            previous_score = score

        # A tight ceiling can otherwise fill before a preserved target is
        # reached in the current-score ordering. Replace the weakest selected
        # item so successful target continuity survives the final gate.
        missing_preserved = [
            device for device in scored_devices
            if str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ) in preserved_keys and str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ) not in selected_keys
        ]
        for device in missing_preserved:
            if len(selected) < ceiling:
                selected.append(device)
                selected_keys.add(str(
                    RetrievalHelper._device_value(device, "id", "")
                    or RetrievalHelper._device_value(device, "name", "")
                ))
                continue
            if not selected:
                break
            replaced = selected.pop()
            replaced_key = str(
                RetrievalHelper._device_value(replaced, "id", "")
                or RetrievalHelper._device_value(replaced, "name", "")
            )
            selected_keys.discard(replaced_key)
            selected.append(device)
            selected_keys.add(str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ))

        # Fill only from the confidence band if ordering or area filtering
        # left the initial pass short.
        if len(selected) < target_count:
            for device in scored_devices:
                candidate_key = str(
                    RetrievalHelper._device_value(device, "id", "")
                    or RetrievalHelper._device_value(device, "name", "")
                )
                if candidate_key in selected_keys or candidate_key not in candidate_keys:
                    continue
                selected.append(device)
                selected_keys.add(candidate_key)
                decision = next(
                    (item for item in decisions if item["candidate"] == candidate_key),
                    None,
                )
                if decision is None:
                    decisions.append({
                        "candidate": candidate_key,
                        "included": True,
                        "reason": "included from the confidence band",
                        "score": round(scores.get(candidate_key, 0.0), 6),
                        "supporting_signals": support.get(candidate_key, ()),
                    })
                else:
                    decision["included"] = True
                    decision["reason"] = "included from the confidence band"
                if len(selected) >= target_count:
                    break

        # The configured minimum is a safety floor after confidence pruning,
        # never a desired count. Preserve the highest-ranked remaining items
        # only when the confidence band is smaller than that floor.
        safety_target = min(ceiling, min_limit, len(scored_devices))
        if len(selected) < safety_target:
            for device in scored_devices:
                candidate_key = str(
                    RetrievalHelper._device_value(device, "id", "")
                    or RetrievalHelper._device_value(device, "name", "")
                )
                if candidate_key in selected_keys:
                    continue
                selected.append(device)
                selected_keys.add(candidate_key)
                decision = next(
                    (item for item in decisions if item["candidate"] == candidate_key),
                    None,
                )
                if decision is None:
                    decisions.append({
                        "candidate": candidate_key,
                        "included": True,
                        "reason": "included to satisfy configured safety minimum",
                        "score": round(scores.get(candidate_key, 0.0), 6),
                        "supporting_signals": support.get(candidate_key, ()),
                    })
                else:
                    decision["included"] = True
                    decision["reason"] = "included to satisfy configured safety minimum"
                if len(selected) >= safety_target:
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
            near_tie_margin_threshold=near_tie_threshold,
            absolute_floor=absolute_floor,
            relative_floor=relative_floor,
            gap_threshold=gap_threshold,
            continuity_max_boost=DEVICE_CONTINUITY_MAX_BOOST,
            explicit_identity_candidates=sorted(explicit_identity_keys),
            normal_confidence_candidates=sorted(eligible_keys),
            preferred_domains=sorted(preferred_domains),
            preferred_areas=sorted(preferred_areas),
            selected_candidate_count=len(selected),
            selected_candidates=[
                str(RetrievalHelper._device_value(device, "id", ""))
                for device in selected
            ],
            candidate_decisions=decisions,
        )
        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("RAGent device pruning: %s", decisions[:12])
        return selected

    @staticmethod
    def build_retrieval_text(current_request: str) -> str:
        return HistoryRetriever.build_retrieval_text(current_request)


    @staticmethod
    def select_history_contexts(
        contexts: Iterable[TurnContext],
        vectors: dict[str, list[float]],
        current_vector: list[float],
        max_age_seconds: float = 300.0,
        limit: int = 3,
        now: float | None = None,
    ) -> list[tuple[TurnContext, float]]:
        return HistoryRetriever.select_history_contexts(
            contexts, vectors, current_vector, max_age_seconds, limit, now,
        )

    @staticmethod
    def build_continuity_context(selected_contexts: Iterable[tuple[TurnContext, float]]) -> ContinuityContext:
        return HistoryRetriever.build_continuity_context(selected_contexts)

    @staticmethod
    def adaptive_candidate_limit(limit: int) -> int:
        """Return a bounded internal pool size for hybrid retrieval."""
        return min(64, max(limit * 6, limit + 12)) if limit > 0 else 0

    @staticmethod
    def expanded_tool_limit(limit: int) -> int:
        """Expand the exposed tool set for a confidently resolved target."""
        return min(20, limit * 3) if limit > 0 else 0

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
        """Score entity identity without saturating every same-type candidate."""
        aliases = RetrievalHelper._device_value(device, "aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        values = (
            RetrievalHelper._device_value(device, "id", ""),
            RetrievalHelper._device_value(device, "friendly_name", ""),
            *aliases,
        )
        exact, fuzzy = RetrievalHelper._match_scores(query, values)
        # Numeric fragments are not stable identity evidence: a requested
        # value such as 5 can be a substring or typo-match for 50. Preserve
        # exact lexical identity, but do not let fuzzy numeric overlap boost
        # the identity signal.
        return exact + (0.5 * fuzzy if not RetrievalHelper._has_numeric_token(query) else 0.0)

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
    def _rank_positive_scores(scores: dict[str, float], minimum: float) -> list[str]:
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [item_key for item_key, score in ranked if score > minimum]

    @staticmethod
    def _rank_only_if_separated(scores: dict[str, float], minimum: float) -> list[str]:
        """Return a ranking only when the signal itself distinguishes items."""
        positive = [score for score in scores.values() if score > minimum]
        if len(set(positive)) <= 1:
            return []
        return RetrievalHelper._rank_positive_scores(scores, minimum)

    @staticmethod
    def _has_strong_current_match(
        match_scores: dict[str, tuple[float, float]],
        vector_ranking: list[str],
        exact_ranking: list[str],
        fuzzy_ranking: list[str],
        query: str = "",
    ) -> bool:
        numeric_query = RetrievalHelper._has_numeric_token(query)
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
    def _has_numeric_token(value: object) -> bool:
        """Return whether normalized text contains a token with a digit."""
        return any(
            any(character.isdigit() for character in token)
            for token in RetrievalHelper._normalize(value).split()
        )

    @staticmethod
    def _has_textual_overlap(query: object, value: object) -> bool:
        """Require non-numeric token overlap for numeric locations."""
        query_tokens = {
            token for token in RetrievalHelper._normalize(query).split()
            if not any(character.isdigit() for character in token)
        }
        value_tokens = {
            token for token in RetrievalHelper._normalize(value).split()
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
            or (top_fuzzy >= 0.9 and not RetrievalHelper._has_numeric_token(query))
            or (
                top_vector_rank == 1
                and (top_exact >= 0.5 or (
                    top_fuzzy >= 0.7
                    and not RetrievalHelper._has_numeric_token(query)
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
                and not RetrievalHelper._has_numeric_token(query)
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
        continuity_ranking = RetrievalHelper._rank_positive_scores(
            {
                key: score for key, score in continuity_scores.items()
                if score > 0
            } if not has_strong_current_match else {},
            0.0,
        )

        fused = RetrievalHelper.reciprocal_rank_fusion(
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
            ordered_keys = RetrievalHelper._trim_to_confident_keys(
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
                    and not RetrievalHelper._has_numeric_token(query)
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
            selected_keys = RetrievalHelper._merge_preserved_keys(
                selected_keys,
                preserved_keys,
                limit,
            )

        result = [candidates[item_key] for item_key in selected_keys]
        if ranking_evidence is not None:
            ranking_evidence["lexical"] = lexical_scores
            ranking_evidence["fused"] = fused
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
    def target_is_confident(query: str, devices: Iterable[Any]) -> bool:
        """Return whether the current target is resolved."""
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
    def _schema_target_values(schema: object, field: str) -> set[str]:
        return root_property_values(schema, field)

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
        parameters = getattr(tool, "parameters", None) or {}
        schema_domains = getattr(tool, "schema_domains", None)
        domains = set(schema_domains) if schema_domains is not None else RetrievalHelper._schema_target_values(parameters, "domain")
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
        """Normalize optional model-provided capability hints."""
        if not isinstance(capability, dict):
            return {}
        action = str(capability.get("action", "") or "").strip().casefold()
        domains = capability.get("domains", capability.get("domain", ())) or ()
        if isinstance(domains, str):
            domains = (domains,)
        if not isinstance(domains, (list, tuple, set, frozenset)):
            domains = ()
        domains = tuple(sorted({
            value.strip().casefold() for value in domains
            if isinstance(value, str) and value.strip()
        }))
        normalized = {
            "action": action,
            "domains": domains,
        }
        return normalized if action or domains else {}

    @staticmethod
    def tool_capability_compatibility(tool: Any, capability: object) -> float:
        """Deterministically compare requested and declared tool capabilities.

        Missing or inferred metadata remains neutral so incomplete device or
        tool metadata cannot suppress recovery.
        """
        requested = RetrievalHelper.normalize_requested_capability(capability)
        if not requested:
            return 0.0
        requested_action = str(requested.get("action", "") or "")
        requested_domains = set(requested.get("domains", ()) or ())
        tool_action = str(getattr(tool, "canonical_action", "") or "").casefold()
        tool_domains = RetrievalHelper._tool_declared_domains(tool)
        # Action IDs are integration-local declarations, not a universal
        # ontology. A different ID is neutral rather than incompatibility.
        # An inferred domain field is compatibility evidence, not a safe
        # exclusion for arbitrary/custom integrations. Domain agreement can
        # still provide a bounded positive signal below.
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
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> dict[str, float]:
        """Return independent, extensible signals used to rank a tool."""
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            exact, fuzzy = (0.0, 0.0)
            if retrieval_method == RETRIEVAL_METHOD_LEXICAL:
                exact, fuzzy = lexical_match if lexical_match is not None else RetrievalHelper._match_scores(
                    query, getattr(tool, "canonical_search_parts", ()) or (
                        getattr(tool, "name", ""), getattr(tool, "description", ""),
                    ),
                )
            vector = retrieval_method == RETRIEVAL_METHOD_VECTOR
            return {
                "semantic_rank": 1.0 / semantic_rank if vector and semantic_rank else 0.0,
                "semantic_similarity": max(0.0, min(1.0, semantic_score or 0.0)) if vector else 0.0,
                "lexical_exact": exact,
                "lexical_fuzzy": fuzzy,
                "capability": 0.0,
                "device_relevance": 0.0,
                "continuity": 0.0,
            }
        devices = list(devices)
        exact, fuzzy = lexical_match if lexical_match is not None else RetrievalHelper._match_scores(
            query,
            getattr(tool, "canonical_search_parts", ()) or (
                getattr(tool, "name", ""),
                getattr(tool, "description", ""),
            ),
        )
        compatibility = RetrievalHelper.tool_device_compatibility(tool, devices)
        device_aware = any(
            RetrievalHelper._metadata_value(tool, name)
            for name in ("is_domain_aware", "is_device_class_aware", "is_area_aware")
        )
        return {
            "semantic_rank": 1.0 / semantic_rank if semantic_rank else 0.0,
            "semantic_similarity": max(0.0, min(1.0, semantic_score or 0.0)),
            "lexical_exact": exact,
            "lexical_fuzzy": fuzzy,
            "capability": RetrievalHelper.tool_capability_compatibility(tool, requested_capability),
            "device_relevance": (
                max(0.0, min(1.0, compatibility / 2.0))
                if devices and device_aware else 0.0
            ),
            "continuity": max(0.0, continuity),
        }

    @staticmethod
    def tool_signal_score(signals: dict[str, float]) -> float:
        """Return semantic similarity for retrieval diagnostics."""
        return max(0.0, signals.get("semantic_similarity", 0.0))

    @staticmethod
    def tool_lexical_scores(tools: Iterable[Any], query: str) -> tuple[dict[str, float], dict[str, float]]:
        """Return operation and description/schema TF-IDF scores."""
        tools = list(tools)
        names = [str(getattr(tool, "name", "")) for tool in tools]
        action_documents = tuple(
            (str(
                getattr(tool, "canonical_action_document", "")
                or getattr(tool, "canonical_action", "")
                or getattr(tool, "name", ""),
            ),)
            for tool in tools
        )
        broad_documents = tuple(
            tuple(str(part) for part in (
                getattr(tool, "description", ""),
                *(getattr(tool, "canonical_schema_parts", ()) or ()),
            ) if part) or (str(getattr(tool, "name", "")),)
            for tool in tools
        )
        action_scores = dict(zip(names, lexical_index(action_documents).scores(query)))
        broad_scores = dict(zip(names, lexical_index(broad_documents).scores(query)))
        return action_scores, broad_scores

    @staticmethod
    def rank_tool_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_tools: Iterable[T],
        query: str,
        devices: Iterable[Any],
        limit: int,
        requested_capability: object = None,
        ranking_evidence: dict[str, dict[str, float]] | None = None,
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> list[T]:
        """Rank a broad tool pool without discarding uncertain candidates."""
        if limit <= 0:
            return []
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            return SourceRanker.rank(
                vector_results, lexical_tools, query,
                lambda tool: str(getattr(tool, "name", "")),
                lambda tool: getattr(tool, "canonical_search_parts", ()) or (
                    getattr(tool, "name", ""), getattr(tool, "description", ""),
                ),
                limit, retrieval_method, ranking_evidence,
            )
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
        action_scores, description_scores = RetrievalHelper.tool_lexical_scores(corpus_tools, query)
        # Correlated lexical fields share one vote; trace overlap is neutral.
        lexical_minimum = 0.05
        corpus_scores = {
            name: max(action_scores[name], description_scores[name])
            for name in corpus_names
        }
        calibrated_lexical_scores = {
            name: score if score >= lexical_minimum else 0.0
            for name, score in corpus_scores.items()
        }
        lexical_ranking = RetrievalHelper._rank_positive_scores(corpus_scores, 0.01)
        compatibility_scores = {
            name: RetrievalHelper.tool_capability_compatibility(tool, requested_capability)
            for name, tool in candidate_by_name.items()
        }
        compatibility_ranking = RetrievalHelper._rank_only_if_separated(
            {name: score for name, score in compatibility_scores.items() if score > 0}, 0.0,
        )
        device_scores = {
            name: RetrievalHelper.tool_device_compatibility(tool, devices)
            for name, tool in candidate_by_name.items()
        }
        # Query relevance is the baseline.  Device compatibility is neither a
        # fourth equal RRF vote nor an eligibility filter: selected devices can
        # be incomplete or ambiguous.  It can only break close query ties when
        # a device identifier/alias is strongly present in the request.
        device_ranking = RetrievalHelper._rank_only_if_separated(device_scores, 0.0)
        fused = RetrievalHelper.reciprocal_rank_fusion((
            # Preserve score ties from the vector backend. Converting these
            # to a positional list made equal scores order-dependent.
            semantic_scores,
            calibrated_lexical_scores,
            compatibility_scores,
        ))
        device_reliability = max(
            (RetrievalHelper.device_target_score(query, device) for device in devices),
            default=0.0,
        )
        if device_reliability >= 0.9 and device_ranking:
            maximum_device_score = max(device_scores.values(), default=0.0)
            if maximum_device_score > 0:
                # At most one fifth of the smallest ordinary RRF vote: enough
                # to settle close alternatives, never enough to displace a
                # clearly better query-only candidate.
                adjustment = 0.2 / (60 + len(candidate_by_name))
                for name, score in device_scores.items():
                    fused[name] = fused.get(name, 0.0) + adjustment * score / maximum_device_score
        scored: list[tuple[float, int, str, T]] = []
        signals_by_name: dict[str, dict[str, float]] = {}
        for name, tool in candidate_by_name.items():
            signals_by_name[name] = {
                "semantic_rank": float(semantic_ranks.get(name, 0)),
                "action_tfidf": action_scores[name],
                "description_schema_tfidf": description_scores[name],
                "capability": compatibility_scores[name],
                "device_compatibility": device_scores[name],
                "rrf": fused.get(name, 0.0),
            }
            scored.append(
                (
                    fused.get(name, 0.0),
                    semantic_ranks.get(name, len(semantic_ranks) + 1),
                    name,
                    tool,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if ranking_evidence is not None:
            ranking_evidence["lexical"] = dict(corpus_scores)
            ranking_evidence["calibrated_lexical"] = dict(calibrated_lexical_scores)
            ranking_evidence["action_lexical"] = dict(action_scores)
            ranking_evidence["description_schema_lexical"] = dict(description_scores)
            ranking_evidence["fused"] = dict(fused)
            ranking_evidence["vector"] = dict(semantic_scores)
        if requested_capability and any(
            score >= 0 for score in compatibility_scores.values()
        ):
            scored = [
                item for item in scored
                if compatibility_scores[item[2]] >= 0
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
                semantic_scores=semantic_scores, action_scores=action_scores,
                description_scores=description_scores, corpus_scores=corpus_scores,
                lexical_ranking=lexical_ranking, compatibility_ranking=compatibility_ranking,
                device_ranking=device_ranking, signals=signals_by_name, fused_scores=fused,
                device_reliability=device_reliability,
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
        ranking_evidence: dict[str, dict[str, float]] | None = None,
        retrieval_method: str = RETRIEVAL_METHOD_AUTOMATIC,
    ) -> ConfidenceAssessment:
        """Measure per-capability confidence from separation and schema evidence."""
        tools = list(tools)
        if not tools:
            return ConfidenceAssessment(level="none")
        names = [str(getattr(tool, "name", "")) for tool in tools]
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            evidence = ranking_evidence if ranking_evidence is not None else {}
            if retrieval_method not in evidence:
                RetrievalHelper.rank_tool_candidates(
                    vector_results, tools, query, (), len(tools),
                    ranking_evidence=evidence, retrieval_method=retrieval_method,
                )
            return SourceRanker.confidence(
                names, retrieval_method, evidence, TOOL_CONFIDENCE_PROFILE, "tool",
            )
        semantic = {
            str(getattr(result.item, "name", "")): result.score
            for result in vector_results
        }
        requested = RetrievalHelper.normalize_requested_capability(requested_capability)
        requested_action = str(requested.get("action", "") or "")
        requested_domains = set(requested.get("domains", ()) or ())
        signals: dict[str, dict[str, float]] = {
            "vector": (
                dict(ranking_evidence.get("vector", {}))
                if ranking_evidence else semantic
            ),
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
            "lexical": {},
        }
        # Confidence pruning uses the exact combined action/broad corpus that
        # determines tool ranking, rather than a separate approximation.
        _action_scores, recomputed_lexical = RetrievalHelper.tool_lexical_scores(tools, query)
        signals["lexical"] = (
            {name: ranking_evidence["lexical"].get(name, 0.0) for name in names}
            if ranking_evidence and "lexical" in ranking_evidence
            else recomputed_lexical
        )
        assessment = RetrievalHelper.assess_distribution_confidence(
            names,
            signals,
            profile=TOOL_CONFIDENCE_PROFILE,
            signal_families={
                "action_schema": "schema",
                "domain_schema": "schema",
            },
            # A binary schema match is compatibility evidence, not a measure
            # of how strongly the request matches a tool.
            strength_signals={"vector", "lexical"},
            kind="tool",
        )
        if ranking_evidence and "fused" in ranking_evidence:
            fused = ranking_evidence["fused"]
            order = sorted(names, key=lambda name: (-fused.get(name, 0.0), name))
            assessment = replace(
                assessment,
                # Fusion is a rank-only score.  Do not overwrite normalized
                # confidence: consumers use that scale for pruning.
                final_candidate_scores=tuple((name, fused.get(name, 0.0)) for name in order),
            )
        return assessment

    @staticmethod
    def tool_search_confidence(
        tools: Iterable[Any], query: str, devices: Iterable[Any], requested_capability: object = None,
        vector_results: Iterable[ScoredResult[Any]] = (),
    ) -> str:
        """Return the explainable distribution-confidence level."""
        return RetrievalHelper.tool_search_confidence_details(
            tools,
            query,
            devices,
            requested_capability,
            vector_results,
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
    def tool_device_compatibility(tool: Any, devices: Iterable[Any]) -> float:
        """Score whether a tool schema can target the retrieved devices."""
        devices = list(devices)
        if not devices:
            return 0.0
        parameters = tool.parameters or {}
        domains = RetrievalHelper._device_domains(devices)
        device_classes = RetrievalHelper._device_classes(devices)
        score = 0.0
        allowed_domains = RetrievalHelper._tool_declared_domains(tool)
        allowed_classes = getattr(tool, "schema_device_classes", None)
        if allowed_classes is None:
            allowed_classes = RetrievalHelper._schema_target_values(parameters, "device_class")
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
    def _tool_service_names(tool: Any) -> set[str]:
        """Map canonical tool actions to Home Assistant service names."""
        action = str(getattr(tool, "canonical_action", "") or "").casefold()
        aliases = {
            "light_set": {"turn_on"},
            "light_set_brightness": {"turn_on"},
            "light_set_color": {"turn_on"},
            "fan_set_speed": {"set_percentage", "turn_on"},
        }
        return aliases.get(action, {action} if action else set())

    @staticmethod
    def tool_device_service_compatibility(tool: Any, device: Any) -> float:
        """Return whether a tool can invoke a compatible service on one device."""
        device_domains = RetrievalHelper._device_domains((device,))
        declared_domains = RetrievalHelper._tool_declared_domains(tool)
        # An empty declaration is deliberately unrestricted, not incompatible.
        if declared_domains and not declared_domains & device_domains:
            return 0.0
        properties = (getattr(tool, "parameters", None) or {}).get("properties") or {}
        allowed_classes = getattr(tool, "schema_device_classes", None)
        if allowed_classes is None:
            allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        device_class = str(RetrievalHelper._device_value(device, "device_class", "") or "").casefold()
        if allowed_classes and device_class and device_class not in allowed_classes:
            return 0.0
        services = {
            str(service).casefold()
            for service in (RetrievalHelper._device_value(device, "services", ()) or ())
        }
        required_services = RetrievalHelper._tool_service_names(tool)
        if services and required_services and not services & required_services:
            return 0.0
        return 1.0
