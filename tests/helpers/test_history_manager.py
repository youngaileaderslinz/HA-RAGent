from types import SimpleNamespace

from custom_components.ha_ragent.src.const import (
    CONF_REMEMBER_CONVERSATION_NUM_INTERACTIONS,
    CONF_REMEMBER_CONVERSATION_TIME_MINUTES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.history_manager import (
    HistoryManager,
    conversation,
)


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
            tool_result={"success": ["light.kitchen"]},
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
    assert context.actions == ("HassTurnOn",)
    assert context.areas == ("Kitchen",)
    assert context.domains == ("light",)
    assert context.device_classes == ("light",)
    assert context.ambiguous_entities == ("light.dining", "light.kitchen")
    assert context.target_groups[0].entities == ("light.kitchen",)


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


def test_replace_system_prompt_removes_stale_candidate_context() -> None:
    manager = HistoryManager({})
    manager._message_history = [
        conversation.SystemContent(content="candidate light.kitchen"),
        conversation.UserContent(content="turn it on"),
    ]

    manager.replace_system_prompt("candidate-free prompt")

    assert manager.message_history[0].content == "candidate-free prompt"
    assert manager.message_history[1].content == "turn it on"


def test_persist_keeps_tool_protocol_out_of_prompt_but_in_chat_log() -> None:
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

    assert not any(isinstance(message, conversation.ToolResultContent) for message in prompt)
    assert tool_call in chat_log.content
    assert tool_result in chat_log.content
