import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.ha_ragent.src.const import (
    CONF_LLM_HASS_API,
    CONF_MAX_TOOL_CALL_ITERATIONS,
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MAX_MEMORIES_TO_EXTRACT,
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_LEXICAL,
    RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES,
)
from custom_components.ha_ragent.src.homeassistant import ragent as ragent_module
from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import HistoryManager
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.translation import RAGentTranslations


@pytest.mark.parametrize("selected_api", ["none", None])
def test_no_control_still_answers_without_loading_or_retrieving_tools(monkeypatch, selected_api):
    chat_log = SimpleNamespace(llm_api=None)
    history = Mock(message_history=[])
    get_api = AsyncMock(side_effect=AssertionError("No Control loaded an API"))
    retriever = SimpleNamespace(async_retrieve_tools=AsyncMock())
    monkeypatch.setattr(ragent_module.llm, "async_get_api", get_api)
    monkeypatch.setattr(ragent_module.chat_session, "async_get_chat_session", lambda *_: nullcontext(object()))
    monkeypatch.setattr(ragent_module.conversation, "async_get_chat_log", lambda *_: nullcontext(chat_log))
    monkeypatch.setattr(ragent_module, "HistoryManager", lambda **_: history)
    monkeypatch.setattr(ragent_module, "ConversationRetriever", lambda *_: retriever)
    result = object()
    agent = SimpleNamespace(
        hass=SimpleNamespace(), entry=SimpleNamespace(), entry_id="entry", subentry_id="subentry",
        subentry=SimpleNamespace(), runtime_options={
            CONF_LLM_HASS_API: selected_api,
            CONF_RETRIEVAL_METHOD: RETRIEVAL_METHOD_LEXICAL,
            CONF_MAX_DEVICES_TO_EXTRACT: 0, CONF_MAX_MEMORIES_TO_EXTRACT: 0,
        },
        _get_current_device_location=lambda *_: (None, None),
        _async_build_continuity_context=AsyncMock(return_value=ContinuityContext()),
        _exclude_prohibited_scheduled_request_tools=lambda tools, _: tools,
        _candidate_context_from_devices=lambda *_: [],
        _async_render_prompts=AsyncMock(return_value=("rules", "state")),
        _async_prompt_model=AsyncMock(return_value=result),
    )
    user_input = SimpleNamespace(
        text="Hello", agent_id="agent", conversation_id="conversation", language="en",
        as_llm_context=lambda _: None,
    )

    assert asyncio.run(ragent_module.RAGent.async_process(agent, user_input)) is result
    get_api.assert_not_awaited()
    retriever.async_retrieve_tools.assert_not_awaited()
    assert agent._async_prompt_model.await_args.args[0] is None
    assert agent._async_prompt_model.await_args.args[2] == []


@pytest.mark.parametrize("available,scheduled", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("scope", [None, "devices", "tools", "devices_and_tools"])
def test_corrective_search_exposes_tools_only_when_explicitly_requested(available, scheduled, scope):
    search_name = "ha_ragent__HassSemanticSearch"
    action_name = (
        RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES[0]
        if scheduled else "VendorAction"
    )
    action = SimpleNamespace(
        name=action_name, description="Perform the action",
        parameters={"type": "object", "properties": {}},
    )
    api = SimpleNamespace(
        tools=[action] if available else [], custom_serializer=None,
        async_call_tool=AsyncMock(side_effect=[
            {"candidate_tools": [{"name": action_name}], "candidate_devices": [], "error": []},
            {"success": True},
        ]),
    )
    requests = []
    should_add_tools = available and not scheduled and scope in {"tools", "devices_and_tools"}

    async def send(_config, _messages, tools):
        requests.append([tool.name for tool in tools])
        if len(requests) == 1:
            tool_name, arguments = search_name, {"search_queries": ["perform the action"]}
            if scope is not None:
                arguments["scope"] = scope
        elif len(requests) == 2:
            tool_name, arguments = action_name, {}
        else:
            yield "Done."
            return
        yield "```homeassistant\n" + json.dumps({"tool": tool_name, "arguments": arguments}) + "\n```"

    agent = SimpleNamespace(
        hass=SimpleNamespace(),
        entry=SimpleNamespace(llm_backend=SimpleNamespace(async_send_chat_request=send),
                              translations=RAGentTranslations.default("en")),
        subentry=SimpleNamespace(data={}), runtime_options={CONF_MAX_TOOL_CALL_ITERATIONS: 3},
        _exclude_prohibited_scheduled_request_tools=ragent_module.RAGent._exclude_prohibited_scheduled_request_tools,
    )
    user_input = SimpleNamespace(agent_id="agent", language="en", conversation_id="conversation")
    history = HistoryManager({})
    history.append_message(ragent_module.conversation.UserContent(content="Perform the action"))
    initial_tools = [
        LlmTool(search_name, "Search", parameters={"type": "object", "properties": {}}),
        LlmTool("ExistingAction", "Initially selected action", parameters={"type": "object", "properties": {}}),
    ]

    result = asyncio.run(ragent_module.RAGent._async_prompt_model(
        agent, api, user_input, initial_tools, SimpleNamespace(content=[]), history,
        [], "Perform the action", ContinuityContext(), scheduled_request=scheduled,
    ))

    assert result.response.error_code is None
    assert requests[0] == [search_name, "ExistingAction"]
    assert requests[1] == [search_name, "ExistingAction", *([action_name] if should_add_tools else [])]
    assert api.async_call_tool.await_count == (2 if should_add_tools else 1)
    assert [tool.name for tool in initial_tools] == [search_name, "ExistingAction"]
