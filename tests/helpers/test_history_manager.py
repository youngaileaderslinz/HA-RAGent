from types import SimpleNamespace

from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.homeassistant.helpers.history_retriever import HistoryRetriever
from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import (
    HistoryManager,
    conversation,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


def test_empty_history_produces_no_structured_or_prompt_history() -> None:
    manager = HistoryManager({})
    chat_log = SimpleNamespace(content=[])

    assert manager.structured_turn_contexts(chat_log) == []
    assert manager.filter_prompt_history(chat_log) == []


def test_structured_context_prefers_canonical_tool_calls_and_results() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(
            content="first request",
        ),
        conversation.AssistantContent(
            agent_id="agent",
            content="",
            tool_calls=[SimpleNamespace(
                tool_name="HassTurnOn",
                tool_args={
                    "name": "light.kitchen",
                    "area": "Kitchen",
                    "domain": ["light"],
                    "device_class": ["light"],
                },
            )],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="call",
            tool_name="HassTurnOn",
            tool_result={
                "success": ["light.kitchen"],
                "execution_status": {
                    "executed_capability": "on",
                    "fulfillment_status": "unverified",
                },
            },
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="search",
            tool_name="ha_ragent__HassSemanticSearch",
            tool_result={
                "devices": [
                    {"name": "light.kitchen", "area": "Kitchen", "domain": ["light"]},
                    {"name": "light.dining", "area": "Dining", "domain": ["light"]},
                ],
                "tools": [{"name": "HassTurnOn"}],
            },
        ),
        conversation.UserContent(content="current request"),
    ])

    context = manager.structured_turn_contexts(chat_log)[0]

    assert context.entities == ("light.kitchen",)
    assert context.tools == ("HassTurnOn",)
    assert context.actions == ("on",)
    assert context.areas == ("Kitchen",)
    assert context.domains == ("light",)
    assert context.device_classes == ("light",)
    assert context.ambiguous_entities == ("light.dining", "light.kitchen")
    assert context.target_groups[0].entities == ("light.kitchen",)
    assert context.target_groups[0].action == "on"


def test_success_entity_and_capability_survive_the_next_related_turn() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="stop playback"),
        conversation.AssistantContent(
            agent_id="agent",
            content="",
            tool_calls=[SimpleNamespace(
                id="media-call",
                tool_name="VendorMediaStop",
                tool_args={"name": "media_player.living", "domain": ["media_player"]},
            )],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="media-call",
            tool_name="VendorMediaStop",
            tool_result={
                "success": ["media_player.living"],
                "execution_status": {
                    "executed_capability": "stop",
                    "fulfillment_status": "verified",
                },
            },
        ),
        conversation.UserContent(content="do that again"),
    ])
    context = manager.structured_turn_contexts(chat_log)[0]
    continuity = HistoryRetriever.build_continuity_context([(context, 1.0)])
    confirmed = Device("media_player.living", "Living room player", "Living room", "")
    wrong = Device("timer.living", "Living room timer", "Living room", "")

    devices = SourceRanker.rank_scored_candidates(
        [ScoredResult(wrong, 0.99, 1)],
        [wrong, confirmed],
        "do that again",
        lambda device: device.id,
        lambda device: (device.id, device.friendly_name),
        1,
        preserve_score=continuity.successful_target_score,
    )
    stop = LlmTool(
        "VendorMediaStop", "",
        metadata=ToolMetadata(canonical_action="stop", supported_domains=("media_player",)),
    )
    timer = LlmTool(
        "VendorTimerStop", "",
        metadata=ToolMetadata(canonical_action="stop", supported_domains=("timer",)),
    )
    tools = ToolRanker.rank_tool_candidates(
        [], [timer, stop], "", devices, 2,
        requested_capability={"action": "stop", "domain": "media_player"},
    )

    assert devices == [confirmed]
    assert tools == [stop, timer]


def test_prompt_history_uses_selected_semantic_turns() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(
            content="turn on the kitchen light",
        ),
        conversation.AssistantContent(
            agent_id="agent",
            content="Done",
            tool_calls=[],
        ),
        conversation.UserContent(
            content="what is the weather",
        ),
        conversation.AssistantContent(
            agent_id="agent",
            content="Sunny",
            tool_calls=[],
        ),
        conversation.UserContent(content="current request"),
    ])
    contexts = manager.structured_turn_contexts(chat_log)

    retained = manager.filter_prompt_history(chat_log, {contexts[0].key})

    assert [message.content for message in retained] == [
        "turn on the kitchen light",
        "Done",
    ]


def test_failed_tool_calls_are_excluded_from_structured_history() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(
            content="turn on the bedroom lamp",
        ),
        conversation.AssistantContent(
            agent_id="agent",
            content="",
            tool_calls=[SimpleNamespace(
                id="failed-call",
                tool_name="HassTurnOn",
                tool_args={"name": "light.bedroom", "area": "Bedroom"},
            )],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="failed-call",
            tool_name="HassTurnOn",
            tool_result={"success": False, "error": "not available"},
        ),
        conversation.UserContent(content="current request"),
    ])

    context = manager.structured_turn_contexts(chat_log)[0]

    assert context.entities == ()
    assert context.tools == ()
    assert context.areas == ()
    assert context.target_groups == ()
    retained = manager.filter_prompt_history(chat_log)
    assert len(retained) == 1
    assert isinstance(retained[0], conversation.UserContent)


def test_successful_tool_calls_are_retained_in_prompt_history() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    call = SimpleNamespace(
        id="successful-call",
        tool_name="HassTurnOn",
        tool_args={"name": "light.bedroom"},
    )
    result = conversation.ToolResultContent(
        agent_id="agent",
        tool_call_id="successful-call",
        tool_name="HassTurnOn",
        tool_result={"success": ["light.bedroom"]},
    )
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="turn on the bedroom lamp"),
        conversation.AssistantContent(agent_id="agent", content="", tool_calls=[call]),
        result,
        conversation.UserContent(content="current request"),
    ])

    retained = manager.filter_prompt_history(chat_log)

    assert retained[0].content == "turn on the bedroom lamp"
    assert retained[1].tool_calls == [call]
    assert retained[2] is result


def test_custom_script_wrapper_preserves_successful_target() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="start music"),
        conversation.AssistantContent(
            agent_id="agent",
            content="",
            tool_calls=[SimpleNamespace(
                id="script-call",
                tool_name="script__hass_start_music_playback",
                tool_args={},
            )],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="script-call",
            tool_name="script__hass_start_music_playback",
            tool_result={
                "success": True,
                "result": {"success": ["media_player.living_room"]},
            },
        ),
        conversation.UserContent(content="pause it"),
    ])

    context = manager.structured_turn_contexts(chat_log)[0]

    assert context.entities == ("media_player.living_room",)
    assert context.actions == ()
    assert context.target_groups[0].entities == ("media_player.living_room",)
    assert context.target_groups[0].action == ""


def test_unresolved_targets_and_requested_action_survive_structurally() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="任意の依頼"),
        conversation.AssistantContent(
            agent_id="agent",
            content="",
            tool_calls=[SimpleNamespace(
                id="call", tool_name="intent__HassTurnOn",
                tool_args={"name": "switch.living_room_plug"},
            )],
        ),
        conversation.ToolResultContent(
            agent_id="agent",
            tool_call_id="call",
            tool_name="intent__HassTurnOn",
            tool_result={
                "success": ["switch.living_room_plug"],
                "execution_status": {
                    "requested_capabilities": [{
                        "action": "turn_on", "domains": ["switch"],
                    }],
                    "executed_capability": "turn_on",
                    "unresolved_targets": [{
                        "name": "switch.bedroom_heater",
                        "area": "Bedroom",
                        "domain": ["switch"],
                    }],
                },
            },
        ),
        conversation.UserContent(content="完全に言語非依存の次の要求"),
    ])

    context = manager.structured_turn_contexts(chat_log)[0]
    continuity = HistoryRetriever.build_continuity_context([(context, 1.0)])
    unresolved = Device(
        "switch.bedroom_heater", "Heater", "Bedroom", "", domain=["switch"],
    )

    assert context.ambiguous_entities == ("switch.bedroom_heater",)
    assert context.actions == ("turn_on",)
    assert continuity.entity_score(unresolved) == 0
    assert continuity.ambiguous_entity_score(unresolved) > 0


def test_replace_system_prompt_removes_stale_candidate_context() -> None:
    manager = HistoryManager({})
    manager._message_history = [
        conversation.SystemContent(content="candidate light.kitchen"),
        conversation.UserContent(content="turn it on"),
    ]

    manager.replace_system_prompt("candidate-free prompt")

    assert manager.message_history[0].content == "candidate-free prompt"
    assert manager.message_history[1].content == "turn it on"


def test_persist_keeps_successful_tool_protocol_in_prompt_and_chat_log() -> None:
    manager = HistoryManager({
        CONF_REMEMBER_CONVERSATION_TIME_MINUTES: 10,
        CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS: 10,
    })
    current_user = conversation.UserContent(content="turn it off")
    tool_call = conversation.AssistantContent(
        agent_id="agent",
        content="",
        tool_calls=[SimpleNamespace(
            id="call-1", tool_name="HassTurnOn", tool_args={"name": "light.kitchen"},
        )],
    )
    tool_result = conversation.ToolResultContent(
        agent_id="agent",
        tool_call_id="call-1",
        tool_name="HassTurnOn",
        tool_result={"success": ["light.kitchen"]},
    )
    chat_log = SimpleNamespace(content=[
        conversation.UserContent(content="turn it on"),
        tool_call,
        tool_result,
        current_user,
    ])

    prompt = manager.build_prompt_history(
        chat_log,
        SimpleNamespace(text="turn it off"),
        "system prompt",
    )
    manager.append_message(conversation.AssistantContent(agent_id="agent", content="Done"))
    manager.persist_chat_history(chat_log)

    assert any(isinstance(message, conversation.ToolResultContent) for message in prompt)
    assert any(
        isinstance(message, conversation.AssistantContent)
        and message.tool_calls
        for message in prompt
    )
    assert tool_call in chat_log.content
    assert tool_result in chat_log.content
