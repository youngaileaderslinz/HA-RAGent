import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import probatio
import pytest
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import DOMAIN, RAGENT_PREFIXED_TOOL_NAMES_BY_NAME
from custom_components.ha_ragent.src.homeassistant.ragent_api import RAGentAugmentedAPIInstance
from custom_components.ha_ragent.src.homeassistant.tools.list_planned_actions import RAGentListPlannedActionsTool
from custom_components.ha_ragent.src.homeassistant.tools.remember_fact import RAGentRememberTool


@dataclass
class ToolResult:
    data: dict
    error: bool = False


def test_native_tool_call_returns_structured_result(monkeypatch):
    monkeypatch.setattr(llm, "ToolResult", ToolResult, raising=False)
    hass = SimpleNamespace(data={})
    tool = RAGentListPlannedActionsTool(hass, "subentry")
    tool_input = llm.ToolInput(tool_name=tool.name, tool_args={})

    result = asyncio.run(tool.async_call(hass, tool_input, None))

    assert result.data == {"success": True, "actions": [], "count": 0}
    assert result.error is False
    assert tool.integration == DOMAIN


def test_home_assistant_api_instance_can_dispatch_native_scoped_tool():
    hass = SimpleNamespace(data={})
    tool = RAGentListPlannedActionsTool(hass, "subentry")
    api = llm.APIInstance(
        api=SimpleNamespace(hass=hass), api_prompt="", llm_context=None, tools=[tool],
    )

    result = asyncio.run(api.async_call_tool(llm.ToolInput(tool_name=tool.name, tool_args={})))

    data = getattr(result, "data", result)
    assert data == {"success": True, "actions": [], "count": 0}


def test_native_tool_call_validates_before_side_effects():
    hass = SimpleNamespace()
    tool = RAGentRememberTool(hass, "entry", "subentry")
    tool._async_call = AsyncMock()
    tool_input = llm.ToolInput(tool_name=tool.name, tool_args={"memory": "x" * 1001})

    with pytest.raises(probatio.Invalid):
        asyncio.run(tool.async_call(hass, tool_input, None))
    tool._async_call.assert_not_awaited()


def test_native_tool_failure_sets_error_flag(monkeypatch):
    monkeypatch.setattr(llm, "ToolResult", ToolResult, raising=False)
    hass = SimpleNamespace()
    tool = RAGentRememberTool(hass, "entry", "subentry")
    tool_input = llm.ToolInput(tool_name=tool.name, tool_args={"memory": " "})

    result = asyncio.run(tool.async_call(hass, tool_input, None))

    assert result.error is True
    assert result.data["success"] is False


def test_scoped_api_dispatches_with_native_home_assistant_signature(monkeypatch):
    monkeypatch.setattr(llm, "ToolResult", ToolResult, raising=False)
    hass = SimpleNamespace(data={})
    context = SimpleNamespace(language="en", context=None, device_id=None)
    wrapped = SimpleNamespace(tools=[], async_call_tool=AsyncMock())
    api = RAGentAugmentedAPIInstance(hass, wrapped, "entry", "subentry", context, "agent")
    tool = next(item for item in api.tools if isinstance(item, RAGentListPlannedActionsTool))
    assert tool.name == RAGENT_PREFIXED_TOOL_NAMES_BY_NAME[RAGentListPlannedActionsTool.name]

    result = asyncio.run(api.async_call_tool(llm.ToolInput(tool_name=tool.name, tool_args={})))

    assert result.error is False
    assert result.data["count"] == 0
    wrapped.async_call_tool.assert_not_awaited()
