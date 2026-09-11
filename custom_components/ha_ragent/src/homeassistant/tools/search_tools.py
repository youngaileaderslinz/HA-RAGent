from __future__ import annotations

import logging
import asyncio
from typing import Any

import voluptuous as vol

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MIN_TOOLS_TO_EXTRACT,
    CONF_MAX_TOOLS_TO_EXTRACT,
    DOMAIN,
    RAGENT_MAX_SEARCH_QUERY_CHARS,
    RAGENT_MAX_SEARCH_QUERIES,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
    RETRIEVAL_METHOD_LEXICAL,
    TRANSLATION_ERROR_SEARCH_QUERY_EMPTY,
    TRANSLATION_ERROR_SEARCH_QUERIES_TOO_MANY,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.translation import RAGentTranslations
from custom_components.ha_ragent.src.debug import log_debug_payload
from custom_components.ha_ragent.src.utils import get_setting_value

_logger = logging.getLogger(__name__)


class RAGentSemanticSearchTool(llm.Tool):
    name = RAGENT_SEMANTIC_SEARCH_TOOL_NAME
    parameters = vol.Schema(
        {
            vol.Required("search_queries"): vol.All(
                [str], vol.Length(min=1, max=RAGENT_MAX_SEARCH_QUERIES)
            ),
            vol.Optional("capabilities"): vol.All(
                [{
                    vol.Required("action"): str,
                    vol.Optional("domain"): vol.Any(str, [str]),
                }],
                vol.Length(min=1, max=RAGENT_MAX_SEARCH_QUERIES),
            ),
            vol.Optional("scope", default="devices_and_tools"): vol.In(["devices", "tools", "devices_and_tools"]),
        }
    )

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.subentry_id = subentry_id
        self.translations = RAGentTranslations(language or "en")
        self.description = self.translations.tool(RAGENT_SEMANTIC_SEARCH_TOOL_NAME)
        self._latest_request = ""
        self._contextual_query = ""
        self._candidate_context: list[dict[str, object]] = []
        self._requested_capabilities: list[dict[str, object]] = []

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _candidate_summary(cls, candidate: dict[str, object]) -> str:
        """Return compact searchable text for a current device candidate."""
        domains = candidate.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        values = (
            candidate.get("name"),
            candidate.get("friendly_name"),
            candidate.get("area"),
            candidate.get("floor"),
            *domains,
            candidate.get("device_class"),
            candidate.get("state"),
            candidate.get("unit_of_measurement"),
        )
        return " | ".join(cls._clean(value) for value in values if cls._clean(value))

    @classmethod
    def _build_search_query(
        cls,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> str:
        """Build a bounded fallback query from user requests and trusted location."""
        sections: list[str] = []
        latest = cls._clean(latest_request)
        if latest:
            sections.append(f"Current request: {latest}")

        current_area = cls._clean(area)
        current_floor = cls._clean(floor)
        if current_area:
            sections.append(f"Default area when the request has no explicit location: {current_area}")
        if current_floor:
            sections.append(f"Default floor when the request has no explicit location: {current_floor}")

        for candidate in candidates or []:
            summary = cls._candidate_summary(candidate)
            if summary:
                sections.append(f"Current candidate: {summary}")

        return "\n".join(sections)[:RAGENT_MAX_SEARCH_QUERY_CHARS].strip()

    def set_search_context(
        self,
        *,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> None:
        """Bind trusted context for the current conversation turn."""
        self._latest_request = self._clean(latest_request)
        self._contextual_query = self._build_search_query(
            latest_request=latest_request,
            area=area,
            floor=floor,
            candidates=candidates,
        )
        self._completed_candidate_names: set[str] = set()
        self._candidate_context = list(candidates or [])
        self._requested_capabilities = []
        log_debug_payload(
            _logger, "search.context_set", entry_id=getattr(self, "entry_id", ""),
            subentry_id=getattr(self, "subentry_id", ""), latest_request=self._latest_request,
            contextual_query=self._contextual_query,
            candidate_context=self._candidate_context,
            completed_candidate_names=self._completed_candidate_names,
        )

    @property
    def requested_capabilities(self) -> list[dict[str, object]]:
        """Return structured operations supplied during the latest search."""
        return list(self._requested_capabilities)

    def refresh_candidates(self, candidates: list[dict[str, object]]) -> None:
        """Replace candidate context after a corrective search."""
        completed = getattr(self, "_completed_candidate_names", set())
        self._candidate_context = [
            candidate
            for candidate in candidates
            if str(candidate.get("name", "")).casefold() not in completed
        ]
        log_debug_payload(
            _logger, "search.context_refreshed", candidates=candidates,
            completed_candidate_names=completed,
            candidate_context=self._candidate_context,
        )

    def prune_candidates(self, completed_names: set[str]) -> None:
        """Remove completed targets from later corrective searches."""
        normalized = {name.casefold() for name in completed_names}
        self._completed_candidate_names = getattr(self, "_completed_candidate_names", set()) | normalized
        self._candidate_context = [
            candidate
            for candidate in self._candidate_context
            if str(candidate.get("name", "")).casefold() not in self._completed_candidate_names
        ]
        log_debug_payload(
            _logger, "search.context_pruned", completed_names=completed_names,
            all_completed_names=self._completed_candidate_names,
            candidate_context=self._candidate_context,
        )

    @staticmethod
    def _get_effective_ranges(entry: Any, subentry: Any) -> tuple[int, int, int, int]:
        """Return normalized minimum and maximum exposure limits."""
        entry_options = getattr(entry, "options", {}) or {}
        subentry_data = getattr(subentry, "data", {}) or {}
        runtime_options = {**entry_options, **subentry_data}
        requested_min_devices = max(0, int(get_setting_value(CONF_MIN_DEVICES_TO_EXTRACT, runtime_options)))
        requested_min_tools = max(0, int(get_setting_value(CONF_MIN_TOOLS_TO_EXTRACT, runtime_options)))
        max_devices = max(0, int(get_setting_value(CONF_MAX_DEVICES_TO_EXTRACT, runtime_options)))
        max_tools = max(0, int(get_setting_value(CONF_MAX_TOOLS_TO_EXTRACT, runtime_options)))
        return (
            requested_min_devices,
            max_devices,
            requested_min_tools,
            max_tools,
        )

    def _model_search_queries(self, tool_input: llm.ToolInput) -> list[str]:
        """Return the distinct focused search intents supplied by the model."""
        raw_queries = tool_input.tool_args.get("search_queries")
        if raw_queries is None:
            raw_queries = [tool_input.tool_args.get("search_query")]
        elif isinstance(raw_queries, str):
            raw_queries = [raw_queries]

        queries: list[str] = []
        seen: set[str] = set()
        for raw_query in raw_queries if isinstance(raw_queries, list) else []:
            query = self._clean(raw_query)
            normalized = query.casefold()
            if query and normalized not in seen:
                queries.append(query)
                seen.add(normalized)
        return queries

    def _model_capabilities(self, tool_input: llm.ToolInput) -> list[dict[str, object]]:
        """Return structured requested operations without parsing natural language."""
        raw = tool_input.tool_args.get("capabilities", [])
        if isinstance(raw, dict):
            raw = [raw]
        return [
            RetrievalHelper.normalize_requested_capability(capability)
            for capability in raw
            if isinstance(capability, dict)
        ]

    async def _validate_query(self, tool_input: llm.ToolInput) -> str | None:
        """Prefer the model's explicit search intent, with user context as fallback."""
        model_search_query = next(iter(self._model_search_queries(tool_input)), "")
        if model_search_query:
            sections = [f"Search intent: {model_search_query}"]
            if self._contextual_query:
                sections.append(self._contextual_query)
            return "\n".join(sections)[:RAGENT_MAX_SEARCH_QUERY_CHARS].strip()
        return self._contextual_query or None

    async def _validate_queries(self, tool_input: llm.ToolInput) -> list[str]:
        """Keep explicit intents isolated; retain legacy single-query context."""
        queries = self._model_search_queries(tool_input)
        if "search_queries" in tool_input.tool_args:
            return [query[:RAGENT_MAX_SEARCH_QUERY_CHARS] for query in queries]
        if not queries:
            return [self._contextual_query] if self._contextual_query else []
        return [
            "\n".join(
                section
                for section in (f"Search intent: {query}", self._contextual_query)
                if section
            )[:RAGENT_MAX_SEARCH_QUERY_CHARS].strip()
            for query in queries
        ]

    @staticmethod
    def _merge_query_candidates(
        candidate_batches: list[list[dict[str, object]]],
        limit: int,
    ) -> list[dict[str, object]]:
        """Take each query's next unseen candidate in turn within the limit."""
        merged: list[dict[str, object]] = []
        seen_names: set[str] = set()
        batches = [iter(candidates) for candidates in candidate_batches]
        while len(merged) < limit:
            added = False
            for candidates in batches:
                for candidate in candidates:
                    name = str(candidate["name"])
                    if name in seen_names:
                        continue
                    merged.append(candidate)
                    seen_names.add(name)
                    added = True
                    break
                if len(merged) >= limit:
                    break
            if not added:
                break
        return merged

    def _iter_searchable_entries(self):
        """Yield only the active entry and subentry."""
        domain_data = self.hass.data.get(DOMAIN, {})
        entry = domain_data.get(self.entry_id)
        if not entry or not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
            return
        subentry = entry.subentries.get(self.subentry_id)
        if not subentry or subentry.data.get(CONF_LLM_HASS_API) == "none":
            return
        limits = self._get_effective_ranges(entry, subentry)
        yield entry, self.subentry_id, subentry, *limits

    async def _embed_query_for_subentry(self, entry: Any, subentry: Any, query: str) -> list[float]:
        """Embed a search query for a specific subentry."""
        options = {**(getattr(entry, "options", {}) or {}), **subentry.data}
        if RetrievalHelper.retrieval_method(options) == RETRIEVAL_METHOD_LEXICAL:
            return []
        try:
            return await entry.embedder_backend.async_embed_text(dict(subentry.data), query) or []
        except Exception as err:
            _logger.warning("Search embedding failed: %s", err)
            return []

    def _device_search_query(
        self, model_search_query: str, fallback_query: str, *, focused: bool = False,
    ) -> str:
        """Search a focused target without adding unrelated tasks."""
        if focused:
            return model_search_query[:RAGENT_MAX_SEARCH_QUERY_CHARS]
        return self._latest_request or model_search_query or fallback_query

    def _tool_search_query(
        self,
        model_search_query: str,
        devices: list[Device | dict[str, object]],
        *,
        focused: bool = False,
    ) -> str:
        """Keep each explicit query independent of the compound request."""
        if focused:
            return model_search_query[:RAGENT_MAX_SEARCH_QUERY_CHARS]
        return RetrievalHelper.build_tool_search_query(
            self._latest_request,
            model_search_query,
            devices,
        )

    @staticmethod
    def _tool_search_feedback(
        search_tools: bool,
        confidence: str,
        tools: list[dict[str, object]],
    ) -> tuple[str, bool, str]:
        """Return an explicit status without discarding uncertain candidates."""
        if not search_tools:
            return "not_requested", False, ""
        if not tools:
            return (
                "no_tools_found",
                True,
                "No tools were retrieved. Do not invent a tool name; "
                "broaden retrieval or report that no action tool is available.",
            )
        if confidence in {"none", "low"}:
            return (
                "weak_candidates",
                True,
                "Candidate confidence is weak. Reuse these candidates, "
                "broaden retrieval if needed, and do not invent a tool.",
            )
        return "candidates_found", False, ""

    def _device_candidate(self, device: Device) -> dict[str, object]:
        """Return enough current device data to answer without another search."""
        state = self.hass.states.get(device.id)
        return {
            "name": device.id,
            "friendly_name": device.friendly_name,
            "area": device.area_name,
            "floor": device.floor_name,
            "area_aliases": device.area_aliases or [],
            "floor_aliases": device.floor_aliases or [],
            "domain": device.domain,
            "device_class": device.device_class,
            "aliases": device.aliases or [],
            "state": state.state if state else None,
            "attributes": Device.clean_attributes(state.attributes) if state else {},
            "unit_of_measurement": (
                state.attributes.get("unit_of_measurement")
                if state
                else device.unit_of_measurement
            ),
        }

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        model_search_queries = self._model_search_queries(tool_input)
        requested_capabilities = self._model_capabilities(tool_input)
        self._requested_capabilities = requested_capabilities
        raw_queries = tool_input.tool_args.get("search_queries", model_search_queries)
        if len(raw_queries) > RAGENT_MAX_SEARCH_QUERIES:
            return {
                "error": self.translations.error(
                    TRANSLATION_ERROR_SEARCH_QUERIES_TOO_MANY,
                    max_queries=RAGENT_MAX_SEARCH_QUERIES,
                )
            }
        queries = await self._validate_queries(tool_input)
        if not queries:
            return {"error": self.translations.error(TRANSLATION_ERROR_SEARCH_QUERY_EMPTY)}
        query = queries[0]
        log_debug_payload(
            _logger, "search.request", tool_arguments=tool_input.tool_args,
            model_search_queries=model_search_queries, validated_queries=queries,
            requested_capabilities=requested_capabilities,
            latest_request=self._latest_request,
            contextual_query=self._contextual_query,
            candidate_context=self._candidate_context,
        )
        _logger.debug(
            "Semantic search model queries=%r",
            model_search_queries,
        )

        scope = (
            str(tool_input.tool_args.get("scope", "devices_and_tools")).strip().lower()
            or "devices_and_tools"
        )
        search_devices = scope in {"devices", "devices_and_tools"}
        search_tools = scope in {"tools", "devices_and_tools"}

        devices: list[dict[str, object]] = []
        tools: list[dict[str, object]] = []
        errors: list[str] = []
        focused = "search_queries" in tool_input.tool_args
        device_queries: list[str] = []
        device_candidate_batches: list[list[dict[str, object]]] = [[] for _ in queries]
        result_tool_limit = 0
        tool_confidences: list[str] = []
        device_limit = 0
        tool_limit = 0
        tool_query = ""
        tool_queries: list[str] = []
        tool_candidate_batches: list[list[dict[str, object]]] = [
            [] for _ in queries
        ]
        tool_confidence = "not_requested"

        for searchable_entry in self._iter_searchable_entries():
            entry, subentry_id, subentry, min_devices, max_devices, min_tools, max_tools = searchable_entry
            device_limit = max_devices
            tool_limit = max_tools
            result_tool_limit = max(result_tool_limit, max_tools)
            for query_index, model_query in enumerate(model_search_queries or [""]):
                try:
                    requested_capability = (
                        requested_capabilities[query_index]
                        if query_index < len(requested_capabilities)
                        else (requested_capabilities[0] if len(requested_capabilities) == 1 else {})
                    )
                    options = {**(getattr(entry, "options", {}) or {}), **subentry.data}
                    device_query = self._device_search_query(model_query, queries[query_index], focused=focused)
                    if search_devices:
                        device_queries.append(device_query)
                    tool_query = self._tool_search_query(model_query, [], focused=focused)
                    device_embedding = QueryEmbedding(
                        lambda query=device_query: self._embed_query_for_subentry(entry, subentry, query)
                    )
                    tool_embedding = device_embedding if tool_query == device_query else QueryEmbedding(
                        lambda query=tool_query: self._embed_query_for_subentry(entry, subentry, query)
                    )
                    if search_devices and device_limit > 0:
                        collection_name = f"devices_{subentry_id}"
                        candidate_limit = RetrievalHelper.adaptive_candidate_limit(device_limit)
                        scored_devices, all_devices = await RetrievalHelper.async_retrieve_sources(
                            entry.vector_db_backend, DeviceEmbedding, options,
                            collection_name, device_embedding, candidate_limit, query=device_query,
                        )
                        def device_text_parts(device: Device) -> tuple[object, ...]:
                            return (
                                device.id,
                                device.friendly_name,
                                *(device.aliases or []),
                                device.area_name,
                                device.floor_name,
                                *(device.area_aliases or []),
                                *(device.floor_aliases or []),
                                *(device.domain or []),
                                device.device_class,
                                *(device.device_labels or []),
                            )

                        def device_identity_score(device: Device) -> float:
                            return 2.0 * RetrievalHelper.device_target_score(
                                device_query, device,
                            )

                        def device_taxonomy_score(device: Device) -> float:
                            return RetrievalHelper.field_match_score(
                                device_query,
                                (*(device.domain or ()), device.device_class),
                            )

                        requested_domains = set(
                            RetrievalHelper.normalize_requested_capability(
                                requested_capability,
                            ).get("domains", ())
                        )

                        def requested_domain_score(device: Device) -> float:
                            return float(bool(
                                requested_domains
                                & {str(value).casefold() for value in (device.domain or ())}
                            ))

                        def device_metadata_score(device: Device) -> float:
                            return device_identity_score(device)

                        retrieved_devices = await asyncio.to_thread(
                            RetrievalHelper.rank_scored_candidates,
                            scored_devices,
                            all_devices,
                            device_query,
                            lambda device: device.id,
                            device_text_parts,
                            candidate_limit,
                            metadata_score=device_metadata_score,
                            trim_confident=False,
                        )
                        device_confidence = RetrievalHelper.device_search_confidence(
                            retrieved_devices,
                            scored_devices,
                            query=device_query,
                            key=lambda device: device.id,
                            text_parts=device_text_parts,
                            structured_scores={
                                "identity_metadata": device_identity_score,
                                "domain_device_class": device_taxonomy_score,
                                "requested_domain": requested_domain_score,
                            },
                        )
                        retrieved_devices = RetrievalHelper.select_device_candidates(
                            device_query,
                            retrieved_devices,
                            min_devices,
                            max_devices,
                            device_confidence,
                        )
                        device_candidate_batches[query_index].extend(
                            self._device_candidate(device)
                            for device in retrieved_devices
                            if isinstance(device, Device)
                        )

                    if search_tools and tool_limit > 0:
                        tool_queries.append(tool_query)
                        collection_name = f"tools_{subentry_id}"
                        candidate_limit = RetrievalHelper.adaptive_candidate_limit(tool_limit)
                        scored_tools, all_tools = await RetrievalHelper.async_retrieve_sources(
                            entry.vector_db_backend, LlmToolEmbedding, options,
                            collection_name, tool_embedding, candidate_limit, query=tool_query,
                        )
                        semantic_ranks = {
                            result.item.name: result.rank
                            for result in scored_tools
                        }
                        semantic_scores = {
                            result.item.name: result.score
                            for result in scored_tools
                        }
                        retrieved_tools = await asyncio.to_thread(
                            RetrievalHelper.rank_tool_candidates,
                            scored_tools,
                            all_tools,
                            tool_query,
                            [],
                            max(tool_limit, RetrievalHelper.expanded_tool_limit(tool_limit)),
                            requested_capability=requested_capability,
                        )
                        confidence = RetrievalHelper.tool_search_confidence_details(
                            retrieved_tools[:max_tools],
                            tool_query,
                            [],
                            requested_capability,
                            scored_tools,
                        )
                        query_confidence = confidence.level
                        tool_confidences.append(query_confidence)
                        query_tool_limit = RetrievalHelper.confidence_limit(
                            query_confidence, min_tools, max_tools,
                        )
                        log_debug_payload(
                            _logger, "search.tool_exposure",
                            query_index=query_index,
                            query=tool_query,
                            requested_capability=requested_capability,
                            confidence=query_confidence,
                            confidence_reason=confidence.reason,
                            top_score=confidence.top_score,
                            second_score=confidence.second_score,
                            margin=confidence.margin,
                            ratio=confidence.ratio,
                            selected_candidate_count=min(
                                query_tool_limit, len(retrieved_tools),
                            ),
                        )
                        seen_query_tool_names: set[str] = set()
                        for tool in retrieved_tools:
                            if not isinstance(tool, LlmTool) or tool.name in seen_query_tool_names:
                                continue
                            seen_query_tool_names.add(tool.name)
                            ranking_signals = RetrievalHelper.tool_ranking_signals(
                                tool,
                                tool_query,
                                [],
                                semantic_rank=semantic_ranks.get(tool.name),
                                semantic_score=semantic_scores.get(tool.name),
                                requested_capability=requested_capability,
                            )
                            tool_candidate_batches[query_index].append(
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.parameters or {},
                                    "metadata": tool.metadata.to_dict() if tool.metadata else None,
                                    "action": tool.canonical_action,
                                    "domains": tool.canonical_supported_domains,
                                    "expected_states": (
                                        tool.metadata.expected_states if tool.metadata else ()
                                    ),
                                    "canonical_action": tool.canonical_action,
                                    "supported_domains": tool.canonical_supported_domains,
                                    "ranking_signals": {
                                        name: round(value, 4)
                                        for name, value in ranking_signals.items()
                                    },
                                    "retrieval_score": round(
                                        RetrievalHelper.tool_signal_score(ranking_signals),
                                        4,
                                    ),
                                }
                            )
                            if len(tool_candidate_batches[query_index]) >= query_tool_limit:
                                break
                except Exception as err:
                    errors.append(f"Failed to search subentry {subentry.title}: {err}")
                    if search_tools:
                        tool_confidences.append("none")

        if search_tools:
            tools = self._merge_query_candidates(tool_candidate_batches, result_tool_limit)
            # A strong result for one task does not establish coverage of another.
            confidence_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
            tool_confidence = min(tool_confidences, key=confidence_rank.get) if tool_confidences else "none"
        devices = self._merge_query_candidates(device_candidate_batches, device_limit)
        if devices:
            self.refresh_candidates(devices)

        # Device selection already applied the configured mode above.
        returned_devices = devices
        if not returned_devices and search_devices:
            returned_devices = list(self._candidate_context[:device_limit])
        tool_status, fallback_required, tool_message = self._tool_search_feedback(
            search_tools,
            tool_confidence,
            tools,
        )
        result = {
            "result_type": "candidate_search",
            "candidate_notice": "Candidates only; no action has been performed.",
            "candidate_data_notice": (
                "Use the included state and location data directly when it answers the request."
            ),
            "search_query": query,
            "search_queries": queries,
            "requested_capabilities": requested_capabilities,
            "device_search_query": device_queries[0] if device_queries else "",
            "device_search_queries": device_queries,
            "tool_search_query": tool_queries[0] if search_tools and tool_queries else "",
            "tool_search_queries": tool_queries if search_tools else [],
            "scope": scope,
            "candidate_devices": returned_devices,
            "candidate_tools": tools if search_tools else [],
            "tool_search_status": tool_status,
            "tool_search_confidence": tool_confidence,
            "fallback_required": fallback_required,
            "tool_search_message": tool_message,
            "error": errors,
        }
        log_debug_payload(
            _logger, "search.result", result=result,
            device_candidate_batches=device_candidate_batches,
            tool_candidate_batches=tool_candidate_batches,
            tool_confidences=tool_confidences,
        )
        return result
