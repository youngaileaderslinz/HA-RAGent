from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    RAGENT_LLM_API_ID,
    RAGENT_LLM_API_NAME,
    RAGENT_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding

_LOGGER = logging.getLogger(__name__)

SEARCH_TYPE_ALL = "all"
SEARCH_TYPE_DEVICES = "devices"
SEARCH_TYPE_TOOLS = "tools"
SEARCH_TYPE_OPTIONS = [
    SEARCH_TYPE_ALL,
    SEARCH_TYPE_DEVICES,
    SEARCH_TYPE_TOOLS,
]


class RAGentSemanticSearchTool(llm.Tool):
    """Semantic search tool over embedded devices and tools."""

    name = RAGENT_SEARCH_TOOL_NAME
    description = (
        "Search Home Assistant devices and available tools using semantic similarity. "
        "Use this when you need to find matching entities, areas, domains, or tool names from a natural-language query."
    )
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
            vol.Optional("search_type", default=SEARCH_TYPE_ALL): vol.In(SEARCH_TYPE_OPTIONS),
            vol.Optional("limit", default=5): vol.All(vol.Coerce(int), vol.Range(min=1, max=25)),
        }
    )

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def async_call(self, tool_input: llm.ToolInput, *args: Any, **kwargs: Any) -> dict[str, Any]:
        query = str(tool_input.tool_args.get("query", "")).strip()
        search_type = tool_input.tool_args.get("search_type", SEARCH_TYPE_ALL)
        limit = int(tool_input.tool_args.get("limit", 5))

        if not query:
            return {"error": "query must not be empty"}

        include_devices = search_type in (SEARCH_TYPE_ALL, SEARCH_TYPE_DEVICES)
        include_tools = search_type in (SEARCH_TYPE_ALL, SEARCH_TYPE_TOOLS)

        results: dict[str, list[dict[str, Any]]] = {"devices": [], "tools": []}
        errors: list[str] = []
        seen_device_ids: set[str] = set()
        seen_tool_names: set[str] = set()

        domain_data = self.hass.data.get(DOMAIN, {})
        for _, entry in domain_data.items():
            if not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
                continue

            for subentry_id, subentry in entry.subentries.items():
                if subentry.data.get(CONF_LLM_HASS_API) == "none":
                    continue

                try:
                    query_embedding = await entry.embedder_backend.async_embed_text(
                        dict(subentry.data), query
                    )
                except Exception as err:
                    errors.append(f"Failed to embed query for subentry {subentry.title}: {err}")
                    continue

                try:
                    if include_devices and len(results["devices"]) < limit:
                        devices = await entry.vector_db_backend.async_retrieve_objects(
                            object_type=DeviceEmbedding,
                            config_subentry=dict(subentry.data),
                            collection_name=f"devices_{subentry_id}",
                            query_embedding=query_embedding,
                            top_k=limit,
                        )
                        for device in devices:
                            if not isinstance(device, Device) or device.id in seen_device_ids:
                                continue
                            seen_device_ids.add(device.id)
                            state = self.hass.states.get(device.id)
                            results["devices"].append(
                                {
                                    "entity_id": device.id,
                                    "name": device.name,
                                    "area": device.area_name,
                                    "domain": device.domain,
                                    "aliases": device.aliases or [],
                                    "services": device.services or [],
                                    "state": state.state if state else None,
                                    "subentry": subentry.title,
                                }
                            )
                            if len(results["devices"]) >= limit:
                                break

                    if include_tools and len(results["tools"]) < limit:
                        tools = await entry.vector_db_backend.async_retrieve_objects(
                            object_type=LlmToolEmbedding,
                            config_subentry=dict(subentry.data),
                            collection_name=f"tools_{subentry_id}",
                            query_embedding=query_embedding,
                            top_k=limit,
                        )
                        for tool in tools:
                            if not isinstance(tool, LlmTool) or tool.name in seen_tool_names:
                                continue
                            seen_tool_names.add(tool.name)
                            results["tools"].append(
                                {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.parameters or {},
                                    "metadata": tool.metadata or {},
                                    "subentry": subentry.title,
                                }
                            )
                            if len(results["tools"]) >= limit:
                                break
                except Exception as err:
                    errors.append(f"Failed to search subentry {subentry.title}: {err}")

        return {
            "query": query,
            "search_type": search_type,
            "devices": results["devices"][:limit],
            "tools": results["tools"][:limit],
            "errors": errors,
        }


class RAGentLLMAPIInstance(llm.APIInstance):
    """Concrete API instance for the HA RAGent tool API."""

    def __init__(self, hass: HomeAssistant, api: "RAGentLLMAPI", llm_context: llm.LLMContext) -> None:
        self.hass = hass
        self.api = api
        self.llm_context = llm_context
        self.id = api.id
        self.name = api.name
        self.prompt = ""
        self.tools = [RAGentSemanticSearchTool(hass)]
        self.custom_serializer = None

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        for tool in self.tools:
            if tool.name == tool_input.tool_name:
                return await tool.async_call(tool_input)
        raise ValueError(f"Unknown tool: {tool_input.tool_name}")


class RAGentLLMAPI(llm.API):
    """Provider for the HA RAGent tool API."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.id = RAGENT_LLM_API_ID
        self.name = RAGENT_LLM_API_NAME

    async def async_get_api_instance(
        self,
        llm_context: llm.LLMContext,
        *args: Any,
        **kwargs: Any,
    ) -> llm.APIInstance:
        return RAGentLLMAPIInstance(self.hass, self, llm_context)

