from __future__ import annotations

import asyncio
import logging
from typing import Any

from probatio import to_openapi

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    TOOL_SELECTION_ABSOLUTE_FLOOR,
    TOOL_SELECTION_GAP_THRESHOLD,
    TOOL_SELECTION_RELATIVE_FLOOR,
)
from custom_components.ha_ragent.src.homeassistant.helpers.device_ranker import DeviceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.source_retriever import SourceRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

_logger = BaseLogger(__name__)


class ConversationRetriever:
    """Orchestrate source access and domain ranking for one conversation."""

    def __init__(self, hass: Any, entry: Any, entry_id: str, subentry_id: str, subentry: Any) -> None:
        self.hass = hass
        self.entry = entry
        self.entry_id = entry_id
        self.subentry_id = subentry_id
        self.subentry = subentry

    def _options(self, retrieval_method: str) -> dict:
        """Make the mode explicit at the retrieval boundary."""
        return {
            **self.entry.options,
            **self.subentry.data,
            CONF_RETRIEVAL_METHOD: retrieval_method,
        }

    async def async_retrieve_devices(
        self, embedding: list[float] | QueryEmbedding, query: str, *,
        minimum: int, maximum: int, continuity: ContinuityContext,
        retrieval_method: str, current_area: str = "", current_floor: str = "",
    ) -> list[Device]:
        """Return device candidates in the supplied retrieval mode."""
        try:
            minimum = max(0, int(minimum))
            maximum = max(0, int(maximum))
        except (TypeError, ValueError, OverflowError):
            return []
        limit = max(minimum, maximum)
        if limit <= 0:
            return []
        options = self._options(retrieval_method)
        candidate_limit = SourceRanker.adaptive_candidate_limit(limit)
        try:
            scored, lexical = await SourceRetriever.async_retrieve_sources(
                self.entry.vector_db_backend, DeviceEmbedding, options,
                f"devices_{self.subentry_id}", embedding, candidate_limit, query=query,
            )
        except Exception as err:
            _logger.log_string(level=logging.ERROR, message=f"Error retrieving devices from vector DB: {err}")
            return []

        def identity(device: Device) -> float:
            return 2.0 * DeviceRanker.device_target_score(query, device)

        def location(device: Device) -> float:
            return DeviceRanker.trusted_location_score(device, current_area, current_floor)

        def continuity_score(device: Device) -> float:
            return (continuity.entity_score(device) + continuity.area_score(device)
                    + continuity.ambiguous_entity_score(device))

        evidence: dict[str, dict[str, float]] = {}
        ranked = await asyncio.to_thread(
            SourceRanker.rank_scored_candidates, scored, lexical, query,
            lambda device: device.id, DeviceRanker.text_parts, candidate_limit,
            metadata_score=lambda device: identity(device) + 0.5 * location(device),
            continuity_score=continuity_score,
            preserve_score=continuity.successful_target_score,
            trim_confident=False, ranking_evidence=evidence,
            retrieval_method=retrieval_method,
        )
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            return ranked[:limit]
        confidence = DeviceRanker.device_search_confidence(
            ranked, scored, query=query, key=lambda device: device.id,
            text_parts=DeviceRanker.text_parts,
            structured_scores={"identity_metadata": identity, "area_floor": location},
            continuity_score=continuity_score,
            confirmed_score=continuity.successful_target_score,
            ranking_evidence=evidence,
        )
        query_areas = {
            str(device.area_name).casefold() for device in ranked if device.area_name
            and (SourceRanker.normalize(device.area_name) in SourceRanker.normalize(query)
                 if not SourceRanker.has_numeric_token(device.area_name)
                 else SourceRanker.has_textual_overlap(query, device.area_name))
        }
        return DeviceRanker.select_device_candidates(
            query, ranked, minimum, maximum, confidence, preferred_areas=query_areas,
            preserve_score=continuity.successful_target_score,
        )

    async def async_retrieve_search_devices(
        self, embedding: QueryEmbedding, query: str, *, minimum: int, maximum: int,
        retrieval_method: str,
    ) -> list[Device]:
        """Retrieve device candidates for the semantic-search tool."""
        candidate_limit = SourceRanker.adaptive_candidate_limit(maximum)
        options = self._options(retrieval_method)
        scored_devices, all_devices = await SourceRetriever.async_retrieve_sources(
            self.entry.vector_db_backend, DeviceEmbedding, options,
            f"devices_{self.subentry_id}", embedding, candidate_limit, query=query,
        )

        def identity_score(device: Device) -> float:
            return 2.0 * DeviceRanker.device_target_score(query, device)

        evidence: dict[str, dict[str, float]] = {}
        devices = await asyncio.to_thread(
            SourceRanker.rank_scored_candidates,
            scored_devices, all_devices, query, lambda device: device.id, DeviceRanker.text_parts,
            candidate_limit, metadata_score=identity_score, trim_confident=False,
            ranking_evidence=evidence, retrieval_method=retrieval_method,
        )
        confidence = DeviceRanker.device_search_confidence(
            devices, scored_devices, query=query, key=lambda device: device.id,
            text_parts=DeviceRanker.text_parts, structured_scores={"identity_metadata": identity_score},
            ranking_evidence=evidence, retrieval_method=retrieval_method,
        )
        selected = DeviceRanker.select_device_candidates(
            query, devices, minimum, maximum, confidence, retrieval_method=retrieval_method,
        )
        return [device for device in selected if isinstance(device, Device)]

    async def async_retrieve_search_tools(
        self, embedding: QueryEmbedding, query: str, *,
        devices: list[Device | dict[str, object]], maximum: int,
        requested_capability: object = None, retrieval_method: str,
    ) -> tuple[list[LlmTool], list[ScoredResult[LlmTool]], dict[str, dict[str, float]]]:
        """Retrieve tool candidates and diagnostics for the semantic-search tool."""
        candidate_limit = SourceRanker.adaptive_candidate_limit(maximum)
        options = self._options(retrieval_method)
        scored_tools, all_tools = await SourceRetriever.async_retrieve_sources(
            self.entry.vector_db_backend, LlmToolEmbedding, options,
            f"tools_{self.subentry_id}", embedding, candidate_limit, query=query,
        )
        ranking_evidence: dict[str, dict[str, float]] = {}
        tools = await asyncio.to_thread(
            ToolRanker.rank_tool_candidates,
            scored_tools, all_tools, query, devices,
            max(maximum, ToolRanker.expanded_tool_limit(maximum)),
            requested_capability=requested_capability,
            ranking_evidence=ranking_evidence, retrieval_method=retrieval_method,
        )
        required_names = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        tools = [tool for tool in tools if tool.name not in required_names]
        if requested_capability and retrieval_method == RETRIEVAL_METHOD_AUTOMATIC:
            compatible = [
                tool for tool in tools
                if ToolRanker.tool_capability_compatibility(tool, requested_capability) >= 0
            ]
            if compatible:
                tools = compatible
        return tools, scored_tools, ranking_evidence
    @staticmethod
    def _required_tools(llm_api: Any) -> list[LlmTool]:
        """Build required tools from the live API; they are not embedded."""
        # Import lazily: ToolExtractor loads the LLM API, which imports the semantic search tool.
        from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import (
            ToolExtractor,
        )

        api_tools = {
            tool.name: tool
            for tool in getattr(llm_api, "tools", ()) or ()
            if getattr(tool, "name", None) in RAGENT_PREFIXED_REQUIRED_TOOL_NAMES
        }
        result: list[LlmTool] = []
        for name in RAGENT_PREFIXED_REQUIRED_TOOL_NAMES:
            tool = api_tools.get(name)
            if tool is None:
                continue
            try:
                parameters = to_openapi(
                    getattr(tool, "parameters", {}) or {},
                    custom_serializer=getattr(llm_api, "custom_serializer", None),
                )
            except Exception:
                parameters = {}
            parameters = parameters if isinstance(parameters, dict) else {}
            result.append(LlmTool(
                name=tool.name,
                description=getattr(tool, "description", ""),
                parameters=parameters,
                metadata=ToolExtractor.extract_tool_metadata(tool, parameters),
            ))
        return result
    async def async_retrieve_tools(
        self,
        embedding: list[float] | QueryEmbedding,
        query: str,
        *,
        minimum: int,
        maximum: int,
        retrieval_method: str,
        llm_api: Any | None = None,
    ) -> list[LlmTool]:
        """Return required tools first, then retrieve optional tools independently."""
        required_tools = self._required_tools(llm_api) if llm_api is not None else []
        try:
            minimum = max(0, int(minimum))
            maximum = max(0, int(maximum))
        except (TypeError, ValueError, OverflowError):
            return required_tools
        if maximum <= 0:
            return required_tools
        candidate_limit = SourceRanker.adaptive_candidate_limit(max(minimum, maximum))
        options = self._options(retrieval_method)
        try:
            scored, lexical = await SourceRetriever.async_retrieve_sources(
                self.entry.vector_db_backend,
                LlmToolEmbedding,
                options,
                f"tools_{self.subentry_id}",
                embedding,
                candidate_limit,
                query=query,
            )
        except Exception as err:
            _logger.log_string(level=logging.ERROR, message=f"Error retrieving tools from vector DB: {err}")
            return required_tools

        required = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        scored = [result for result in scored if result.item.name not in required]
        lexical = [tool for tool in lexical if tool.name not in required]
        evidence: dict[str, dict[str, float]] = {}
        ranked = await asyncio.to_thread(
            ToolRanker.rank_tool_candidates,
            scored,
            lexical,
            query,
            [],
            max(maximum, ToolRanker.expanded_tool_limit(maximum)),
            ranking_evidence=evidence,
            retrieval_method=retrieval_method,
        )
        if retrieval_method != RETRIEVAL_METHOD_AUTOMATIC:
            return [*required_tools, *ranked[:maximum]]
        confidence = ToolRanker.tool_search_confidence_details(
            ranked,
            query,
            [],
            vector_results=scored,
            ranking_evidence=evidence,
        )
        if confidence.level in {"low", "none"}:
            selected_names = {tool.name for tool in ranked[:maximum]}
        else:
            selected_names = set(RetrievalConfidence.prune_confidence_band(
                confidence,
                maximum,
                absolute_floor=TOOL_SELECTION_ABSOLUTE_FLOOR,
                relative_floor=TOOL_SELECTION_RELATIVE_FLOOR,
                gap_threshold=TOOL_SELECTION_GAP_THRESHOLD,
                preserve_signals=set(),
            ))
        selected = [tool for tool in ranked if tool.name in selected_names][:maximum]
        known = {tool.name for tool in selected}
        for tool in ranked:
            if len(selected) >= min(maximum, minimum):
                break
            if tool.name not in known:
                selected.append(tool)
                known.add(tool.name)
        _logger.log_payload(
            "retrieval.tool_exposure",
            confidence=confidence.level,
            configured_minimum=minimum,
            configured_maximum=maximum,
            selected_tools=[tool.name for tool in selected],
        )
        return [*required_tools, *selected]
    async def async_retrieve_memories(
        self,
        embedding: list[float] | QueryEmbedding,
        *,
        minimum: int,
        maximum: int,
    ) -> list[Memory]:
        """Recall persistent memory using confidence-adaptive exposure."""
        if maximum <= 0:
            return []
        try:
            return await MemoryManager(self.hass, self.entry_id, self.subentry_id).async_recall(
                embedding, minimum, maximum,
            )
        except Exception as err:
            _logger.log_string(level=logging.ERROR, message=f"Error retrieving memories from vector DB: {err}")
            return []
