from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    RAGENT_LLM_API_ID,
    RAGENT_LLM_API_NAME,
)

from .tools.search_devices import (
    RAGentSemanticSearchDevicesTool,
)
from .tools.search_tools import (
    RAGentSemanticSearchToolsTool,
)


class RAGentSearchAugmentedAPIInstance(llm.APIInstance):
    """Wrap an existing HA LLM API instance and append the RAGent search tool."""

    def __init__(self, hass: HomeAssistant, wrapped_api: llm.APIInstance) -> None:
        self.hass = hass
        self._wrapped_api = wrapped_api
        self.prompt = getattr(wrapped_api, "prompt", "")
        self.custom_serializer = getattr(wrapped_api, "custom_serializer", None)

        wrapped_tools = list(getattr(wrapped_api, "tools", []) or [])
        missing_tools = []
        if not any(getattr(tool, "name", None) == RAGentSemanticSearchDevicesTool.name for tool in wrapped_tools):
            missing_tools.append(RAGentSemanticSearchDevicesTool(hass))
        if not any(getattr(tool, "name", None) == RAGentSemanticSearchToolsTool.name for tool in wrapped_tools):
            missing_tools.append(RAGentSemanticSearchToolsTool(hass))
        self.tools = [*wrapped_tools, *missing_tools]

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped API instance."""
        return getattr(self._wrapped_api, name)

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        if tool_input.tool_name in {
            RAGentSemanticSearchDevicesTool.name,
            RAGentSemanticSearchToolsTool.name,
        }:
            for tool in self.tools:
                if tool.name == tool_input.tool_name:
                    return await tool.async_call(tool_input)

        return await self._wrapped_api.async_call_tool(tool_input)


def augment_api_with_search_tool(hass: HomeAssistant, llm_api: llm.APIInstance | None) -> llm.APIInstance | None:
    """Expose the semantic search tool on top of an existing HA LLM API instance."""
    if llm_api is None:
        return None

    tool_names = {getattr(tool, "name", None) for tool in getattr(llm_api, "tools", []) or []}
    if {
        RAGentSemanticSearchDevicesTool.name,
        RAGentSemanticSearchToolsTool.name,
    }.issubset(tool_names):
        return llm_api

    return RAGentSearchAugmentedAPIInstance(hass, llm_api)


class RAGentLLMAPIInstance(llm.APIInstance):
    """Concrete API instance for the HA RAGent tool API."""

    def __init__(self, hass: HomeAssistant, api: "RAGentLLMAPI", llm_context: llm.LLMContext) -> None:
        self.hass = hass
        self.api = api
        self.llm_context = llm_context
        self.id = api.id
        self.name = api.name
        self.prompt = ""
        self.tools = [RAGentSemanticSearchDevicesTool(hass), RAGentSemanticSearchToolsTool(hass)]
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
