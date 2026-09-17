"""Distribution confidence and candidate exposure thresholds."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

from custom_components.ha_ragent.src.logging import log_debug_payload
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.models.retrieval.confidence_profile import ConfidenceProfile


_logger = logging.getLogger("custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper")


class RetrievalConfidence:
    """Assess independent retrieval evidence and select confidence bands."""

    @staticmethod
    def prune_confidence_band(
        confidence: ConfidenceAssessment,
        maximum: int,
        *,
        absolute_floor: float,
        relative_floor: float,
        gap_threshold: float,
        preserve_signals: set[str] | None = None,
    ) -> list[str]:
        """Select a confidence band from final hybrid scores, not raw vectors."""
        ranked = list(confidence.candidate_scores)
        if maximum <= 0 or not ranked:
            return []
        support = dict(confidence.candidate_support)
        preserve_signals = preserve_signals or set()
        best = ranked[0][1]
        selected: list[str] = []
        previous = best
        for index, (key, score) in enumerate(ranked):
            preserved = bool(set(support.get(key, ())) & preserve_signals)
            if index and previous - score >= gap_threshold and not preserved:
                break
            if not preserved and (score < absolute_floor or score < best * relative_floor):
                break
            selected.append(key)
            previous = score
            if len(selected) >= maximum:
                break
        return selected or [ranked[0][0]]

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
        signal_families: dict[str, str] | None = None,
        strength_signals: set[str] | None = None,
        confirmed_keys: set[str] | None = None,
        minimum_independent_signals: int = 2,
        kind: str = "candidate",
    ) -> ConfidenceAssessment:
        """Classify confidence from separation and independent signal agreement."""
        ordered_keys = list(dict.fromkeys(ordered_keys))
        if not ordered_keys:
            return ConfidenceAssessment(level="none")

        weak_signals = weak_signals or set()
        signal_families = signal_families or {}
        # Schema fields describe the same source of evidence. They can help
        # rank candidates, but must not count as independent observations.
        strength_signals = strength_signals or set(signal_scores)
        confirmed_keys = confirmed_keys or set()
        normalized = {
            name: values
            for name, raw_values in signal_scores.items()
            if (values := RetrievalConfidence._normalize_confidence_signal(raw_values))
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

        original_positions = {key: index for index, key in enumerate(ordered_keys)}
        score_order = sorted(
            ordered_keys,
            key=lambda key: (-totals.get(key, 0.0), original_positions[key]),
        )
        top_key = score_order[0]
        top_score = totals.get(top_key, 0.0)
        second_score = totals.get(score_order[1], 0.0) if len(score_order) > 1 else 0.0
        margin = max(0.0, top_score - second_score)
        ratio = top_score / second_score if second_score > 1e-9 else (
            float("inf") if top_score > 0 else 0.0
        )

        agreeing: list[str] = []
        disagreeing: list[str] = []
        absolute_strength = max(
            (max(0.0, float(signal_scores[name].get(top_key, 0.0)))
             for name in strength_signals if name in signal_scores),
            default=0.0,
        )
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

        independent_agreement = {
            signal_families.get(name, name) for name in agreeing
        }
        confirmed_top = top_key in confirmed_keys and not any(
            key in confirmed_keys for key in score_order[1:]
        )
        if confirmed_top:
            level = "high"
            reason = "unique confirmed continuity target"
        elif len(ordered_keys) == 1:
            if len(independent_agreement) >= minimum_independent_signals and absolute_strength > 0.2:
                level = "high"
                reason = (
                    "the only candidate is supported by the selected source"
                    if minimum_independent_signals == 1
                    else "the only candidate is independently supported by multiple signals"
                )
            else:
                level = "low"
                reason = "the only candidate lacks independent corroboration"
        elif margin <= profile.near_tie_margin:
            level = "low"
            reason = "top candidates are near-tied"
        elif disagreeing and len(disagreeing) >= len(agreeing):
            level = "low"
            reason = "strong ranking signals disagree"
        elif (
            len(independent_agreement) >= minimum_independent_signals
            and margin > profile.near_tie_margin
            and top_score >= 0.35
            and absolute_strength > 0.2
        ):
            level = "high"
            reason = (
                "clear winner supported by the selected source"
                if minimum_independent_signals == 1
                else "clear winner supported by independent signals"
            )
        else:
            level = "low"
            reason = "winner lacks sufficient independent separation"

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
                (key, round(totals.get(key, 0.0), 6)) for key in score_order
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
                for key in score_order
            ),
            absolute_strength=round(min(1.0, absolute_strength), 6),
            independent_signal_count=len(independent_agreement),
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
            raw_signals=signal_scores,
            normalized_signals=normalized,
            absolute_strength=assessment.absolute_strength,
            independent_signal_count=assessment.independent_signal_count,
        )
        return assessment
