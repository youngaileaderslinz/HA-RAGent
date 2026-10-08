from __future__ import annotations

from abc import abstractmethod

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper


class RAGentTool(llm.Tool):
    """Expose scoped tools through Home Assistant's native tool interface."""

    integration = DOMAIN

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext,
    ) -> llm.ToolResult | dict[str, object]:
        validated_input = llm.ToolInput(
            tool_name=tool_input.tool_name,
            tool_args=self.parameters(tool_input.tool_args),
            id=tool_input.id,
            external=tool_input.external,
        )
        data = await self._async_call(validated_input)
        result_type = getattr(llm, "ToolResult", None)
        if result_type is None:
            return data
        return result_type(data=data, error=not MessageHelper.tool_result_succeeded(data))

    @abstractmethod
    async def _async_call(self, tool_input: llm.ToolInput) -> dict[str, object]:
        """Perform the operation using this tool's bound agent scope."""
        raise NotImplementedError
