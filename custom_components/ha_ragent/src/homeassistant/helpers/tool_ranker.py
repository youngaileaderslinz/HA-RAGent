from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any, TypeVar

from custom_components.ha_ragent.src.const import (
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
    TOOL_CONFIDENCE_NEAR_TIE_MARGIN,
)
from custom_components.ha_ragent.src.homeassistant.helpers.device_ranker import DeviceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.models.embedding.schema_constraints import root_property_values
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

T = TypeVar("T")

_logger = BaseLogger(__name__)


class ToolRanker:
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
        requested = ToolRanker.normalize_requested_capability(capability)
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
    def expanded_tool_limit(limit: int | float) -> int:
        """Return an integer expanded tool candidate limit."""
        try:
            requested = int(limit)
        except (TypeError, ValueError, OverflowError):
            return 0
        return min(20, requested * 3) if requested > 0 else 0

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
                    values.update(ToolRanker._schema_values(value))
        elif isinstance(schema, list):
            for value in schema:
                values.update(ToolRanker._schema_values(value))
        return values

    @staticmethod
    def _schema_target_values(schema: object, field: str) -> set[str]:
        return root_property_values(schema, field)

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
        domains = set(schema_domains) if schema_domains is not None else ToolRanker._schema_target_values(parameters, "domain")
        domains.update(ToolRanker._metadata_value(tool, "supported_domains", ()) or ())
        return {str(domain).casefold() for domain in domains if domain}

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
        requested = ToolRanker.normalize_requested_capability(capability)
        if not requested:
            return 0.0
        requested_action = str(requested.get("action", "") or "")
        requested_domains = set(requested.get("domains", ()) or ())
        tool_action = str(getattr(tool, "canonical_action", "") or "").casefold()
        tool_domains = ToolRanker._tool_declared_domains(tool)
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
                exact, fuzzy = lexical_match if lexical_match is not None else SourceRanker.match_scores(
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
        exact, fuzzy = lexical_match if lexical_match is not None else SourceRanker.match_scores(
            query,
            getattr(tool, "canonical_search_parts", ()) or (
                getattr(tool, "name", ""),
                getattr(tool, "description", ""),
            ),
        )
        compatibility = ToolRanker.tool_device_compatibility(tool, devices)
        device_aware = any(
            ToolRanker._metadata_value(tool, name)
            for name in ("is_domain_aware", "is_device_class_aware", "is_area_aware")
        )
        return {
            "semantic_rank": 1.0 / semantic_rank if semantic_rank else 0.0,
            "semantic_similarity": max(0.0, min(1.0, semantic_score or 0.0)),
            "lexical_exact": exact,
            "lexical_fuzzy": fuzzy,
            "capability": ToolRanker.tool_capability_compatibility(tool, requested_capability),
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
        action_scores, description_scores = ToolRanker.tool_lexical_scores(corpus_tools, query)
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
        lexical_ranking = SourceRanker.rank_positive_scores(corpus_scores, 0.01)
        compatibility_scores = {
            name: ToolRanker.tool_capability_compatibility(tool, requested_capability)
            for name, tool in candidate_by_name.items()
        }
        compatibility_ranking = SourceRanker.rank_only_if_separated(
            {name: score for name, score in compatibility_scores.items() if score > 0}, 0.0,
        )
        device_scores = {
            name: ToolRanker.tool_device_compatibility(tool, devices)
            for name, tool in candidate_by_name.items()
        }
        # Query relevance is the baseline.  Device compatibility is neither a
        # fourth equal RRF vote nor an eligibility filter: selected devices can
        # be incomplete or ambiguous.  It can only break close query ties when
        # a device identifier/alias is strongly present in the request.
        device_ranking = SourceRanker.rank_only_if_separated(device_scores, 0.0)
        fused = SourceRanker.reciprocal_rank_fusion((
            # Preserve score ties from the vector backend. Converting these
            # to a positional list made equal scores order-dependent.
            semantic_scores,
            calibrated_lexical_scores,
            compatibility_scores,
        ))
        device_reliability = max(
            (DeviceRanker.device_target_score(query, device) for device in devices),
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
                SourceRanker.normalize(getattr(tool, "description", "")),
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
        _logger.log_payload("retrieval.tool_ranking", query=query, limit=limit,
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
                ToolRanker.rank_tool_candidates(
                    vector_results, tools, query, (), len(tools),
                    ranking_evidence=evidence, retrieval_method=retrieval_method,
                )
            return SourceRanker.confidence(
                names, retrieval_method, evidence, TOOL_CONFIDENCE_NEAR_TIE_MARGIN, "tool",
            )
        semantic = {
            str(getattr(result.item, "name", "")): result.score
            for result in vector_results
        }
        requested = ToolRanker.normalize_requested_capability(requested_capability)
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
                    and bool(ToolRanker._tool_declared_domains(tool) & requested_domains)
                )
                for tool in tools
            },
            "lexical": {},
        }
        # Confidence pruning uses the exact combined action/broad corpus that
        # determines tool ranking, rather than a separate approximation.
        _action_scores, recomputed_lexical = ToolRanker.tool_lexical_scores(tools, query)
        signals["lexical"] = (
            {name: ranking_evidence["lexical"].get(name, 0.0) for name in names}
            if ranking_evidence and "lexical" in ranking_evidence
            else recomputed_lexical
        )
        assessment = RetrievalConfidence.assess_distribution_confidence(
            names,
            signals,
            near_tie_margin=TOOL_CONFIDENCE_NEAR_TIE_MARGIN,
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
        return ToolRanker.tool_search_confidence_details(
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
        return ToolRanker.rank_tool_candidates(
            [], tools, query, devices, len(tools), requested_capability=requested_capability,
        )

    @staticmethod
    def tool_device_compatibility(tool: Any, devices: Iterable[Any]) -> float:
        """Score whether a tool schema can target the retrieved devices."""
        devices = list(devices)
        if not devices:
            return 0.0
        parameters = tool.parameters or {}
        domains = DeviceRanker.device_domains(devices)
        device_classes = DeviceRanker.device_classes(devices)
        score = 0.0
        allowed_domains = ToolRanker._tool_declared_domains(tool)
        allowed_classes = getattr(tool, "schema_device_classes", None)
        if allowed_classes is None:
            allowed_classes = ToolRanker._schema_target_values(parameters, "device_class")
        if allowed_domains:
            score += 2.0 if domains & allowed_domains else 0.0
        if allowed_classes:
            score += 2.0 if device_classes & allowed_classes else 0.0

        if (
            ToolRanker._metadata_value(tool, "is_domain_aware")
            and (not allowed_domains or domains & allowed_domains)
        ):
            score += 0.5
        if (
            ToolRanker._metadata_value(tool, "is_device_class_aware")
            and device_classes
            and (not allowed_classes or device_classes & allowed_classes)
        ):
            score += 0.5
        has_area = any(
            SourceRanker.candidate_value(device, "area_name")
            or SourceRanker.candidate_value(device, "area")
            for device in devices
        )
        if ToolRanker._metadata_value(tool, "is_area_aware") and has_area:
            score += 0.25
        return score
