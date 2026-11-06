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
    CONF_MAX_TOOLS_TO_EXTRACT,
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_MIN_MEMORIES_TO_EXTRACT,
    CONF_MIN_TOOLS_TO_EXTRACT,
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
    RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES,
)
from custom_components.ha_ragent.src.homeassistant import ragent as ragent_module
from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import HistoryManager
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.translation import RAGentTranslations


@pytest.mark.parametrize("invalid_arguments", [
    None,
    {"area": "Other", "filters": {"area": "Office"}, "scope": "devices"},
    {"action": "turn_on", "scope": "devices"},
    {"scope": "tools", "filters": {"scope": "devices"}},
])
def test_device_search_recovers_argument_errors_without_detail(compact_search, invalid_arguments):
    from custom_components.ha_ragent.src.homeassistant.ragent_api import RAGentAugmentedAPIInstance

    search, _ = compact_search
    api = RAGentAugmentedAPIInstance.__new__(RAGentAugmentedAPIInstance)
    api.hass = search.hass
    api.tools = [search]
    api.llm_context = None
    api._wrapped_api = SimpleNamespace()
    api.async_call_tool = AsyncMock(wraps=api.async_call_tool)
    requests = []
    correcting = invalid_arguments is not None

    async def send(_config, messages, _tools):
        requests.append(messages[:])
        iteration = len(requests)
        if correcting and iteration == 1:
            arguments = invalid_arguments
        elif iteration == 1 + correcting:
            if correcting:
                error = [message["content"] for message in messages if message["role"] == "tool"][-1]
                assert error["status"] == "validation_error"
                assert "inside filters" in error["error"]
                assert "candidate_devices" not in error
            arguments = {
                "search_queries": ["office lights"], "scope": "devices", "filters": {"area": "Office"},
            }
        else:
            results = [message["content"] for message in messages if message["role"] == "tool"][-1]
            assert all(item["entity_id"].startswith("light.office_") for item in results["results"])
            assert all("off" in item["summary"] for item in results["results"])
            yield "The office lights are off."
            return
        yield "```homeassistant\n" + json.dumps({"tool": search.name, "arguments": arguments}) + "\n```"

    agent = SimpleNamespace(
        hass=search.hass,
        entry=SimpleNamespace(llm_backend=SimpleNamespace(async_send_chat_request=send),
                              translations=RAGentTranslations.default("en")),
        subentry=SimpleNamespace(data={}), runtime_options={CONF_MAX_TOOL_CALL_ITERATIONS: 2 + correcting},
        _exclude_prohibited_scheduled_request_tools=ragent_module.RAGent._exclude_prohibited_scheduled_request_tools,
    )
    result = asyncio.run(ragent_module.RAGent._async_prompt_model(
        agent, api, SimpleNamespace(agent_id="agent", language="en", conversation_id="turn"),
        [LlmTool(search.name, search.description, parameters={"type": "object"})],
        SimpleNamespace(content=[]), HistoryManager({}), [], "Are the office lights on?", ContinuityContext(),
    ))

    assert result.response.error_code is None
    assert len(requests) == 2 + correcting
    assert api.async_call_tool.await_count == 1 + correcting
    assert search._details == {}
    assert {candidate["name"] for candidate in search._candidate_context} == {
        f"light.office_{index}" for index in range(4)
    }


@pytest.mark.parametrize("available,scheduled", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("discover_device", [False, True])
def test_compact_search_defers_live_schema_until_detail_and_executes(available, scheduled, discover_device, compact_search):
    from custom_components.ha_ragent.src.homeassistant.ragent_api import RAGentAugmentedAPIInstance
    search, actions = compact_search
    action = actions[0]
    if scheduled:
        action.name = RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES[0]
    live_action = SimpleNamespace(name=action.name, description="Live action", parameters={
        "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
    })
    api = RAGentAugmentedAPIInstance.__new__(RAGentAugmentedAPIInstance)
    api.hass = search.hass
    api.tools = [search, *([live_action] if available else [])]
    api.custom_serializer = None
    api._wrapped_api = SimpleNamespace()
    exposed = []
    requests = []
    selected = {}
    schemas = []

    async def execute(call):
        if call.tool_name.endswith("HassSemanticSearch"):
            result = await search._async_call(call)
            if "results" in result:
                for candidate in result["results"]:
                    selected.setdefault(candidate["type"] + "s", candidate["id"])
            return result
        exposed.append(call.tool_name)
        return {"success": True}

    api.async_call_tool = execute

    async def send(_config, messages, tools):
        requests.append(([t.name for t in tools], [dict(m) for m in messages]))
        schemas.append({t.name: t.parameters for t in tools})
        n = len(requests)
        if n == 1:
            if discover_device:
                yield "```homeassistant\n" + json.dumps({"tool": "ha_ragent__HassSemanticSearch", "arguments": {
                    "search_queries": ["office lights"], "scope": "devices"}}) + "\n```\n"
            args = {"search_queries": ["office lights"], "scope": "tools"}
            name = "ha_ragent__HassSemanticSearch"
        elif n == 2:
            if "devices" in selected:
                yield "```homeassistant\n" + json.dumps({"tool": "ha_ragent__HassSemanticSearch", "arguments": {
                    "operation": "detail", "result_id": selected["devices"], "scope": "devices"}}) + "\n```\n"
            args = {"operation": "detail", "result_id": selected["tools"], "scope": "tools"}
            name = "ha_ragent__HassSemanticSearch"
        elif n == 3 and available and not scheduled:
            args = {"name": "light.office_0"}
            name = action.name
        else:
            yield "Done."
            return
        yield "```homeassistant\n" + json.dumps({"tool": name, "arguments": args}) + "\n```"

    agent = SimpleNamespace(
        hass=api.hass, entry=SimpleNamespace(llm_backend=SimpleNamespace(async_send_chat_request=send),
                                          translations=RAGentTranslations.default("en")),
        subentry=SimpleNamespace(data={}), runtime_options={CONF_MAX_TOOL_CALL_ITERATIONS: 4},
        _exclude_prohibited_scheduled_request_tools=ragent_module.RAGent._exclude_prohibited_scheduled_request_tools)
    history = HistoryManager({})
    initial = [LlmTool("ha_ragent__HassSemanticSearch", "Search", parameters={"type": "object"})]
    result = asyncio.run(ragent_module.RAGent._async_prompt_model(
        agent, api, SimpleNamespace(agent_id="agent", language="en", conversation_id="turn"), initial,
        SimpleNamespace(content=[]), history, [], "Control office lights", ContinuityContext(), scheduled_request=scheduled))
    assert result.response.error_code is None
    assert requests[0][0] == requests[1][0] == [initial[0].name]
    assert (action.name in requests[2][0]) == (available and not scheduled)
    assert exposed == ([action.name] if available and not scheduled else [])
    assert "_candidate_entities" not in json.dumps([messages for _, messages in requests])
    if discover_device:
        from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
        stored_candidates = [MessageHelper.tool_result_data(message).get("_candidate_entities")
                             for message in history.message_history
                             if isinstance(message, ragent_module.conversation.ToolResultContent)]
        identities = next(mapping for mapping in stored_candidates if mapping)
        assert set(identities.values()) == {f"light.office_{n}" for n in range(4)}
    detail = [m["content"] for m in requests[2][1] if m["role"] == "tool"][-1]
    if available and not scheduled:
        assert detail["detail"]["name"] == action.name
        assert "parameters" not in detail["detail"]  # exactly one copy in tools
        assert schemas[2][action.name] == live_action.parameters
    else:
        assert detail["status"] == "detail_error"


@pytest.mark.parametrize("selected_api", ["none", None])
@pytest.mark.parametrize("context", [None, SimpleNamespace(user_id="user")])
@pytest.mark.parametrize("supports_context", [False, True])
def test_no_control_still_answers_without_loading_or_retrieving_tools(monkeypatch, selected_api, context, supports_context):
    chat_log = SimpleNamespace(llm_api=None)
    history = Mock(message_history=[])
    get_api = AsyncMock(side_effect=AssertionError("No Control loaded an API"))
    retriever = SimpleNamespace(async_retrieve_tools=AsyncMock())
    monkeypatch.setattr(ragent_module.llm, "async_get_api", get_api)
    get_chat_session = Mock(return_value=nullcontext(object()))
    monkeypatch.setattr(ragent_module.chat_session, "async_get_chat_session", get_chat_session)
    monkeypatch.setattr(ragent_module, "_CHAT_SESSION_SUPPORTS_CONTEXT", supports_context)
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
        context=context,
        as_llm_context=lambda _: None,
    )

    assert asyncio.run(ragent_module.RAGent.async_process(agent, user_input)) is result
    expected_kwargs = {"context": context} if supports_context else {}
    get_chat_session.assert_called_once_with(agent.hass, user_input.conversation_id, **expected_kwargs)
    get_api.assert_not_awaited()
    retriever.async_retrieve_tools.assert_not_awaited()
    assert agent._async_prompt_model.await_args.args[0] is None
    assert agent._async_prompt_model.await_args.args[2] == []


@pytest.mark.parametrize("mode", [
    RETRIEVAL_METHOD_AUTOMATIC, RETRIEVAL_METHOD_LEXICAL, RETRIEVAL_METHOD_VECTOR,
])
@pytest.mark.parametrize("minimum,maximum", [(3, 5), (5, 2), (3, 0)])
def test_conversation_normalizes_all_exposure_ranges(monkeypatch, mode, minimum, maximum):
    chat_log = SimpleNamespace(llm_api=None)
    retriever = SimpleNamespace(
        async_retrieve_devices=AsyncMock(return_value=[]),
        async_retrieve_tools=AsyncMock(return_value=[]),
        async_retrieve_memories=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(ragent_module.llm, "async_get_api", AsyncMock(return_value=object()))
    monkeypatch.setattr(ragent_module.chat_session, "async_get_chat_session", lambda *_, **kwargs: nullcontext(object()))
    monkeypatch.setattr(ragent_module.conversation, "async_get_chat_log", lambda *_: nullcontext(chat_log))
    monkeypatch.setattr(ragent_module, "HistoryManager", lambda **_: Mock(message_history=[]))
    monkeypatch.setattr(ragent_module, "ConversationRetriever", lambda *_: retriever)
    result = object()
    agent = SimpleNamespace(
        hass=SimpleNamespace(), entry=SimpleNamespace(), entry_id="entry", subentry_id="subentry",
        subentry=SimpleNamespace(), runtime_options={
            CONF_LLM_HASS_API: "test_api", CONF_RETRIEVAL_METHOD: mode,
            CONF_MIN_DEVICES_TO_EXTRACT: minimum, CONF_MAX_DEVICES_TO_EXTRACT: maximum,
            CONF_MIN_TOOLS_TO_EXTRACT: minimum, CONF_MAX_TOOLS_TO_EXTRACT: maximum,
            CONF_MIN_MEMORIES_TO_EXTRACT: minimum, CONF_MAX_MEMORIES_TO_EXTRACT: maximum,
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
        context=SimpleNamespace(user_id="user"),
        as_llm_context=lambda _: None,
    )

    assert asyncio.run(ragent_module.RAGent.async_process(agent, user_input)) is result
    for retrieve in (retriever.async_retrieve_devices, retriever.async_retrieve_memories, retriever.async_retrieve_tools):
        if maximum == 0 and retrieve is not retriever.async_retrieve_tools:
            retrieve.assert_not_awaited()
        else:
            retrieve.assert_awaited_once()
            assert retrieve.await_args.kwargs["minimum"] == min(minimum, maximum)
            assert retrieve.await_args.kwargs["maximum"] == maximum


@pytest.mark.parametrize("available,scheduled", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("scope", ["devices", "tools"])
def test_corrective_search_exposes_tools_only_when_explicitly_requested(available, scheduled, scope):
    search_name = "ha_ragent__HassSemanticSearch"
    action_name = (
        RAGENT_PREFIXED_SCHEDULED_REQUEST_PROHIBITED_TOOL_NAMES[0]
        if scheduled else "VendorAction"
    )
    action = SimpleNamespace(
        name=action_name, description="Perform the action",
        parameters={
            "type": "object",
            "properties": {"level": {"type": "integer", "description": "Requested level"}},
            "required": ["level"],
            "additionalProperties": False,
        },
    )
    api = SimpleNamespace(
        tools=[action] if available else [], custom_serializer=None,
        async_call_tool=AsyncMock(side_effect=[
            {
                "candidate_tools": [{
                    "name": action_name, "description": "Cached candidate description",
                    "parameters": {"cached_schema": True}, "action": "cached_action",
                    "domains": ["cached_domain"],
                }],
                "candidate_devices": [{"name": "light.desk", "state": "on"}], "error": [],
            },
            {"success": True},
        ]),
    )
    requests = []
    tool_definitions = []
    request_messages = []
    should_add_tools = available and not scheduled and scope == "tools"

    async def send(_config, _messages, tools):
        requests.append([tool.name for tool in tools])
        tool_definitions.append([tool.to_tool_dict() for tool in tools])
        request_messages.append([dict(message) for message in _messages])
        if len(requests) == 1:
            tool_name, arguments = search_name, {"search_queries": ["perform the action"]}
            if scope is not None:
                arguments["scope"] = scope
        elif len(requests) == 2:
            tool_name, arguments = action_name, {"level": 1}
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
    if should_add_tools:
        assert tool_definitions[1][-1]["function"]["parameters"] == action.parameters
        assert tool_definitions[1][-1]["function"]["description"] == action.description
    search_result = next(
        message["content"] for message in request_messages[1]
        if message["role"] == "tool" and message["tool_name"] == search_name
    )
    assert search_result["available_tools"] == ([action_name] if should_add_tools else [])
    assert search_result["candidate_devices"] == [{"name": "light.desk", "state": "on"}]
    assert "candidate_tools" not in search_result
    assert "Cached candidate description" not in json.dumps(search_result)
    assert "cached_schema" not in json.dumps(search_result)
    if scope == "tools" and not should_add_tools:
        assert search_result["tool_search_status"] == "no_tools_available"
        assert search_result["fallback_required"] is True


@pytest.mark.parametrize("exposed", [True, False])
def test_streaming_preserves_tool_validation_and_stores_each_message_once(exposed):
    deltas = []
    tool_name = "VendorAction"
    api = SimpleNamespace(async_call_tool=AsyncMock(return_value={"success": True}))

    class StreamingChatLog:
        def __init__(self):
            self.content = [ragent_module.conversation.UserContent(content="Perform the action")]

        async def async_add_delta_content_stream(self, agent_id, stream):
            parts = []
            async for delta in stream:
                assert "tool_calls" not in delta
                deltas.append(delta)
                parts.append(delta["content"])
            if parts:
                message = ragent_module.conversation.AssistantContent(
                    agent_id=agent_id, content="".join(parts),
                )
                self.content.append(message)
                yield message

    requests = 0

    async def send(_config, _messages, _tools):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield "```home"
            assert api.async_call_tool.await_count == 0
            yield 'assistant\n{"tool":"VendorAction","arguments":{}}\n```'
        else:
            yield "Done" if exposed else "Unavailable"
            assert deltas, "Speech was buffered until generation finished"
            yield "."

    agent = SimpleNamespace(
        hass=SimpleNamespace(),
        entry=SimpleNamespace(
            llm_backend=SimpleNamespace(async_send_chat_request=send),
            translations=RAGentTranslations.default("en"),
        ),
        subentry=SimpleNamespace(data={}), runtime_options={CONF_MAX_TOOL_CALL_ITERATIONS: 2},
    )
    user_input = SimpleNamespace(
        text="Perform the action", agent_id="agent", language="en", conversation_id="conversation",
    )
    chat_log = StreamingChatLog()
    history = HistoryManager({})
    history.build_prompt_history(chat_log, user_input, "rules", "state")
    tools = [LlmTool(tool_name, "Perform the action", parameters={})] if exposed else []

    result = asyncio.run(ragent_module.RAGent._async_prompt_model(
        agent, api, user_input, tools, chat_log, history,
        [], user_input.text, ContinuityContext(),
    ))

    assert result.response.error_code is None
    assert requests == 2
    assert api.async_call_tool.await_count == int(exposed)
    assert "".join(delta["content"] for delta in deltas) == ("Done." if exposed else "Unavailable.")
    assert len(chat_log.content) == 4
    assert len([message for message in chat_log.content if message.role == "assistant"]) == 2
    assert chat_log.content[-1].content == ("Done." if exposed else "Unavailable.")
