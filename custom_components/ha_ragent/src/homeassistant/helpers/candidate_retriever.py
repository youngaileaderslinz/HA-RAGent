"""Retrieve and rank candidates for semantic search tools."""

from __future__ import annotations

import asyncio
from typing import Any

from custom_components.ha_ragent.src.const import (
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    RETRIEVAL_METHOD_AUTOMATIC,
)
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


class CandidateRetriever:
    """Keep source retrieval and ranking separate from search orchestration."""

    @staticmethod
    async def async_retrieve_devices(
        backend: Any,
        options: dict,
        collection: str,
        embedding: QueryEmbedding,
        query: str,
        minimum: int,
        maximum: int,
    ) -> list[Device]:
        method = RetrievalHelper.retrieval_method(options)
        candidate_limit = RetrievalHelper.adaptive_candidate_limit(maximum)
        scored_devices, all_devices = await RetrievalHelper.async_retrieve_sources(
            backend, DeviceEmbedding, options, collection, embedding,
            candidate_limit, query=query,
        )

        def text_parts(device: Device) -> tuple[object, ...]:
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

        def identity_score(device: Device) -> float:
            return 2.0 * RetrievalHelper.device_target_score(query, device)

        ranking_evidence: dict[str, dict[str, float]] = {}
        devices = await asyncio.to_thread(
            RetrievalHelper.rank_scored_candidates,
            scored_devices,
            all_devices,
            query,
            lambda device: device.id,
            text_parts,
            candidate_limit,
            metadata_score=identity_score,
            trim_confident=False,
            ranking_evidence=ranking_evidence,
            retrieval_method=method,
        )
        confidence = RetrievalHelper.device_search_confidence(
            devices,
            scored_devices,
            query=query,
            key=lambda device: device.id,
            text_parts=text_parts,
            structured_scores={"identity_metadata": identity_score},
            ranking_evidence=ranking_evidence,
            retrieval_method=method,
        )
        selected = RetrievalHelper.select_device_candidates(
            query, devices, minimum, maximum, confidence, retrieval_method=method,
        )
        return [device for device in selected if isinstance(device, Device)]

    @staticmethod
    async def async_retrieve_tools(
        backend: Any,
        options: dict,
        collection: str,
        embedding: QueryEmbedding,
        query: str,
        devices: list[Device | dict[str, object]],
        maximum: int,
        requested_capability: object = None,
    ) -> tuple[list[LlmTool], list[ScoredResult[LlmTool]], dict[str, dict[str, float]]]:
        method = RetrievalHelper.retrieval_method(options)
        candidate_limit = RetrievalHelper.adaptive_candidate_limit(maximum)
        scored_tools, all_tools = await RetrievalHelper.async_retrieve_sources(
            backend, LlmToolEmbedding, options, collection, embedding,
            candidate_limit, query=query,
        )
        ranking_evidence: dict[str, dict[str, float]] = {}
        tools = await asyncio.to_thread(
            RetrievalHelper.rank_tool_candidates,
            scored_tools,
            all_tools,
            query,
            devices,
            max(maximum, RetrievalHelper.expanded_tool_limit(maximum)),
            requested_capability=requested_capability,
            ranking_evidence=ranking_evidence,
            retrieval_method=method,
        )
        required_names = set(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES)
        tools = [tool for tool in tools if tool.name not in required_names]
        if requested_capability and method == RETRIEVAL_METHOD_AUTOMATIC:
            compatible = [
                tool for tool in tools
                if RetrievalHelper.tool_capability_compatibility(
                    tool, requested_capability,
                ) >= 0
            ]
            if compatible:
                tools = compatible
        return tools, scored_tools, ranking_evidence
