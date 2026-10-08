import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.ha_ragent.src.const import (
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MAX_MEMORIES_TO_EXTRACT,
    CONF_MAX_TOOLS_TO_EXTRACT,
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_MIN_MEMORIES_TO_EXTRACT,
    CONF_MIN_TOOLS_TO_EXTRACT,
    CONF_RETRIEVAL_METHOD,
    DOMAIN,
    RAGENT_SCHEDULED_CONTEXT_PREFIX,
    RAGENT_SCHEDULED_EXECUTION_CONTEXTS,
    RAGENT_SCHEDULED_REQUEST_PREFIX,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant import ragent as ragent_module
from custom_components.ha_ragent.src.homeassistant.ragent import RAGent
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.scheduled_context import ScheduledContext
from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup


@pytest.mark.parametrize("mode", [RETRIEVAL_METHOD_LEXICAL, RETRIEVAL_METHOD_VECTOR])
def test_scheduled_request_passes_snapshot_continuity_through_retrieval_and_history(monkeypatch, mode):
    original_continuity = ContinuityContext(
        selected_turn_keys={"original-turn"},
        entities={"fan.office": 0.9},
        target_groups=[(TargetGroup(entities=("fan.office",)), 0.9)],
    )
    snapshot = ScheduledContext.capture(
        subentry_id="subentry", agent_id="agent", request="turn on the fan",
        candidates=[{"name": "fan.office", "friendly_name": "Office fan"}],
        continuity=original_continuity,
    )
    hass = SimpleNamespace(data={DOMAIN: {RAGENT_SCHEDULED_EXECUTION_CONTEXTS: {"token": snapshot}}}, states=Mock())
    chat_log = SimpleNamespace(llm_api=None)
    history_manager = Mock()
    history_manager.message_history = []
    monkeypatch.setattr(ragent_module.chat_session, "async_get_chat_session", lambda *args: nullcontext(object()))
    monkeypatch.setattr(ragent_module.conversation, "async_get_chat_log", lambda *args: nullcontext(chat_log))
    monkeypatch.setattr(ragent_module.llm, "async_get_api", AsyncMock(return_value=object()))
    monkeypatch.setattr(ragent_module, "HistoryManager", lambda **kwargs: history_manager)
    required_tool = LlmTool("ha_ragent__semantic_search", "Search")
    retriever = SimpleNamespace(
        async_retrieve_devices=AsyncMock(return_value=[]),
        async_retrieve_tools=AsyncMock(return_value=[required_tool]),
    )
    monkeypatch.setattr(ragent_module, "ConversationRetriever", lambda *args: retriever)

    result = object()
    agent = SimpleNamespace(
        hass=hass, entry=SimpleNamespace(), entry_id="entry", subentry_id="subentry",
        subentry=SimpleNamespace(), runtime_options={
            CONF_RETRIEVAL_METHOD: mode,
            CONF_MIN_MEMORIES_TO_EXTRACT: 0, CONF_MAX_MEMORIES_TO_EXTRACT: 0,
            CONF_MIN_DEVICES_TO_EXTRACT: 1, CONF_MAX_DEVICES_TO_EXTRACT: 1,
            CONF_MIN_TOOLS_TO_EXTRACT: 0, CONF_MAX_TOOLS_TO_EXTRACT: 0,
        },
        _get_current_device_location=lambda *_: (None, None),
        _async_build_continuity_context=AsyncMock(side_effect=AssertionError("Scheduled request read live history")),
        _exclude_prohibited_scheduled_request_tools=lambda tools, scheduled: tools,
        _candidate_context_from_devices=lambda *_: [],
        _async_render_prompts=AsyncMock(return_value=("rules", "scheduled state")),
        _async_prompt_model=AsyncMock(return_value=result),
    )
    user_input = SimpleNamespace(
        text=f"{RAGENT_SCHEDULED_REQUEST_PREFIX}{RAGENT_SCHEDULED_CONTEXT_PREFIX}token] turn it on",
        agent_id="agent", conversation_id=None, language="en",
        as_llm_context=lambda domain: None,
    )

    assert asyncio.run(RAGent.async_process(agent, user_input)) is result
    passed_continuity = retriever.async_retrieve_devices.call_args.kwargs["continuity"]
    assert isinstance(passed_continuity, ContinuityContext)
    assert passed_continuity.selected_turn_keys == set()
    assert passed_continuity.entities == {"fan.office": 0.9}
    assert passed_continuity.successful_target_score(SimpleNamespace(id="fan.office")) == 0.9
    assert snapshot.continuity.selected_turn_keys == {"original-turn"}
    original_continuity.entities.clear()
    assert snapshot.continuity.entities == {"fan.office": 0.9}
    assert history_manager.build_prompt_history.call_args.kwargs["relevant_turn_keys"] == set()
    assert retriever.async_retrieve_devices.call_args.args[1].startswith("turn it on\nScheduled target context:")
    agent._async_build_continuity_context.assert_not_awaited()
    retriever.async_retrieve_tools.assert_awaited_once()
    assert retriever.async_retrieve_tools.call_args.kwargs["maximum"] == 0
    assert retriever.async_retrieve_devices.call_args.kwargs["retrieval_method"] == mode
    assert retriever.async_retrieve_tools.call_args.kwargs["retrieval_method"] == mode
    assert agent._async_prompt_model.await_args.args[2] == [required_tool]


def test_scheduled_action_freezes_continuity_when_timer_is_created(monkeypatch):
    from custom_components.ha_ragent.src.homeassistant.ragent_api import RAGentAugmentedAPIInstance
    from custom_components.ha_ragent.src.homeassistant.tools import planned_action as planned_action_module
    from custom_components.ha_ragent.src.homeassistant.tools.planned_action import RAGentPlannedActionTool

    hass = SimpleNamespace(data={})
    tool = RAGentPlannedActionTool.__new__(RAGentPlannedActionTool)
    tool.hass = hass
    tool.subentry_id = "subentry"
    tool.agent_id = "agent"
    tool._async_execute = AsyncMock()
    api = RAGentAugmentedAPIInstance.__new__(RAGentAugmentedAPIInstance)
    api.tools = [tool]
    api._scheduling_area = "Office"
    api._scheduling_floor = "First"
    continuity = ContinuityContext(entities={"fan.office": 0.9})
    api.set_scheduling_context("turn on the fan", [], [], continuity)

    callbacks = []
    monkeypatch.setattr(planned_action_module, "async_call_later", lambda hass, delay, callback: callbacks.append(callback) or (lambda: None))
    scheduled = asyncio.run(tool._async_call(SimpleNamespace(tool_args={"action_request": "turn it on", "minutes": 1})))
    assert scheduled["success"] is True
    continuity.entities.clear()
    tool._scheduling_context.continuity.entities.clear()
    asyncio.run(callbacks[0](None))

    snapshot = tool._async_execute.await_args.args[2]
    assert snapshot.continuity.entities == {"fan.office": 0.9}
    assert snapshot.area == "Office"
    assert snapshot.floor == "First"
