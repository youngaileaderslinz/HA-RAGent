"""Device identity, location, confidence, and candidate selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, TypeVar

from custom_components.ha_ragent.src.const import (
    DEVICE_CONFIDENCE_NEAR_TIE_MARGIN,
    DEVICE_CONTINUITY_MAX_BOOST,
    DEVICE_SELECTION_ABSOLUTE_FLOOR,
    DEVICE_SELECTION_GAP_THRESHOLD,
    DEVICE_SELECTION_RELATIVE_FLOOR,
    RETRIEVAL_METHOD_AUTOMATIC,
)
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

T = TypeVar("T")

_logger = BaseLogger(__name__)


class DeviceRanker:
    """Device identity, location, confidence, and candidate selection."""

    @staticmethod
    def text_parts(device: Any) -> tuple[object, ...]:
        """Return the device fields shared by initial and corrective retrieval."""
        return (
            device.id, device.friendly_name, *(device.aliases or []),
            device.area_name, device.floor_name, *(device.area_aliases or []),
            *(device.floor_aliases or []), *(device.domain or []),
            device.device_class, *(device.device_labels or []),
        )

    @staticmethod
    def _candidate_location_values(device: Any) -> tuple[object, ...]:
        return (
            SourceRanker.candidate_value(device, "area_name", ""),
            SourceRanker.candidate_value(device, "area", ""),
            SourceRanker.candidate_value(device, "floor_name", ""),
            SourceRanker.candidate_value(device, "floor", ""),
            *(SourceRanker.candidate_value(device, "area_aliases", []) or []),
            *(SourceRanker.candidate_value(device, "floor_aliases", []) or []),
        )

    @staticmethod
    def device_resolution(query: str, devices: Iterable[Any]) -> tuple[str, tuple[str, ...]]:
        """Resolve literal identities only; leave command scope to the LLM."""
        devices = list(devices)
        normalized = SourceRanker.normalize(query)
        exact = [
            str(SourceRanker.candidate_value(device, "id", "") or SourceRanker.candidate_value(device, "name", ""))
            for device in devices
            if normalized and normalized in {
                SourceRanker.normalize(value)
                for value in SourceRanker.candidate_identity_values(device) if value
            }
        ]
        if len(exact) == 1:
            return "high", tuple(exact)
        names = tuple(
            str(SourceRanker.candidate_value(device, "id", "") or SourceRanker.candidate_value(device, "name", ""))
            for device in devices
        )
        return ("ambiguous" if len(devices) > 1 else "weak"), names

    @staticmethod
    def reduce_confident_devices(query: str, devices: Iterable[T]) -> list[T]:
        """Expose only independently resolved devices when confidence is high."""
        devices = list(devices)
        status, names = DeviceRanker.device_resolution(query, devices)
        if status != "high":
            return devices
        selected = set(names)
        return [
            device
            for device in devices
            if str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
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
                keys, retrieval_method, evidence, DEVICE_CONFIDENCE_NEAR_TIE_MARGIN, "device",
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
        assessment = RetrievalConfidence.assess_distribution_confidence(
            keys,
            signals,
            near_tie_margin=DEVICE_CONFIDENCE_NEAR_TIE_MARGIN,
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
        _logger.log_payload(
            "retrieval.device_confidence", assessment=assessment,
            vector_scores=vector, signals=signals,
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
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
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
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            )
            for device in scored_devices
        }
        scored_devices.extend(
            device
            for device in devices
            if str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            ) not in scored_keys
        )
        current_scored_devices = sorted(
            scored_devices,
            key=lambda device: scores.get(
                str(
                    SourceRanker.candidate_value(device, "id", "")
                    or SourceRanker.candidate_value(device, "name", "")
                ),
                0.0,
            ),
            reverse=True,
        )
        near_tie_threshold = DEVICE_CONFIDENCE_NEAR_TIE_MARGIN
        preferred_domains = {
            str(domain).casefold() for domain in preferred_domains if domain
        }
        preferred_areas = {
            str(area).casefold() for area in preferred_areas if area
        }
        preferred_domain_candidates = {
            str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            )
            for device in scored_devices
            if (
                {
                    str(value).casefold()
                    for value in (
                        ([SourceRanker.candidate_value(device, "domain", "")]
                         if isinstance(SourceRanker.candidate_value(device, "domain", ""), str)
                         else SourceRanker.candidate_value(device, "domain", ()) or ())
                    )
                }
                & preferred_domains
            )
        }
        preferred_area_candidates = {
            str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            )
            for device in scored_devices
            if str(
                SourceRanker.candidate_value(device, "area_name", "")
                or SourceRanker.candidate_value(device, "area", "")
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
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
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
                SourceRanker.candidate_value(scored_devices[0], "id", "")
                or SourceRanker.candidate_value(scored_devices[0], "name", "")
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
            DeviceRanker.device_target_score(query, device) >= 0.9
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
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            )
            current_score = scores.get(candidate_key, 0.0)
            score = final_scores.get(candidate_key, current_score)
            candidate_margin = max(0.0, top_score - current_score)
            supporting_signals = support.get(candidate_key, ())
            raw_candidate_domains = SourceRanker.candidate_value(device, "domain", ()) or ()
            if isinstance(raw_candidate_domains, str):
                raw_candidate_domains = (raw_candidate_domains,)
            candidate_domains = {
                str(value).casefold() for value in raw_candidate_domains
            }
            preferred = bool(preferred_domains & candidate_domains)
            candidate_area = str(
                SourceRanker.candidate_value(device, "area_name", "")
                or SourceRanker.candidate_value(device, "area", "")
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
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            ) in preserved_keys and str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            ) not in selected_keys
        ]
        for device in missing_preserved:
            if len(selected) < ceiling:
                selected.append(device)
                selected_keys.add(str(
                    SourceRanker.candidate_value(device, "id", "")
                    or SourceRanker.candidate_value(device, "name", "")
                ))
                continue
            if not selected:
                break
            replaced = selected.pop()
            replaced_key = str(
                SourceRanker.candidate_value(replaced, "id", "")
                or SourceRanker.candidate_value(replaced, "name", "")
            )
            selected_keys.discard(replaced_key)
            selected.append(device)
            selected_keys.add(str(
                SourceRanker.candidate_value(device, "id", "")
                or SourceRanker.candidate_value(device, "name", "")
            ))

        # Fill only from the confidence band if ordering or area filtering
        # left the initial pass short.
        if len(selected) < target_count:
            for device in scored_devices:
                candidate_key = str(
                    SourceRanker.candidate_value(device, "id", "")
                    or SourceRanker.candidate_value(device, "name", "")
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
                    SourceRanker.candidate_value(device, "id", "")
                    or SourceRanker.candidate_value(device, "name", "")
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

        _logger.log_payload("retrieval.device_ambiguity_cluster",
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
                str(SourceRanker.candidate_value(device, "id", ""))
                for device in selected
            ],
            candidate_decisions=decisions,
        )
        return selected

    @staticmethod
    def device_target_score(query: str, device: Any) -> float:
        """Score entity identity without saturating every same-type candidate."""
        aliases = SourceRanker.candidate_value(device, "aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        values = (
            SourceRanker.candidate_value(device, "id", ""),
            SourceRanker.candidate_value(device, "friendly_name", ""),
            *aliases,
        )
        exact, fuzzy = SourceRanker.match_scores(query, values)
        # Numeric fragments are not stable identity evidence: a requested
        # value such as 5 can be a substring or typo-match for 50. Preserve
        # exact lexical identity, but do not let fuzzy numeric overlap boost
        # the identity signal.
        return exact + (0.5 * fuzzy if not SourceRanker.has_numeric_token(query) else 0.0)

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
    def target_is_confident(query: str, devices: Iterable[Any]) -> bool:
        """Return whether the current target is resolved."""
        status, _ = DeviceRanker.device_resolution(query, devices)
        return status == "high"

    @staticmethod
    def device_domains(devices: Iterable[Any]) -> set[str]:
        domains: set[str] = set()
        for device in devices:
            values = SourceRanker.candidate_value(device, "domain", []) or []
            if isinstance(values, str):
                values = [values]
            domains.update(str(value).casefold() for value in values)
        return domains

    @staticmethod
    def device_classes(devices: Iterable[Any]) -> set[str]:
        return {
            str(value).casefold()
            for device in devices
            if (value := SourceRanker.candidate_value(device, "device_class"))
        }
