from types import SimpleNamespace

from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import (
    MessageHelper,
    conversation,
)

ToolInput = SimpleNamespace

def test_create_tool_failure_message() -> None:
    message = MessageHelper.create_tool_failure_message(
        "agent", "call-1", "HassTurnOn", RuntimeError("failed"),
    )

    assert message.tool_result == {
        "success": False,
        "tool": "HassTurnOn",
        "error": "failed",
    }

def test_compact_tool_result_labels_semantic_search_candidates() -> None:
    search = conversation.ToolResultContent(
        agent_id="agent", tool_call_id="call-1",
        tool_name="ha_ragent__HassSemanticSearch",
        tool_result={"devices": ["large"]},
    )
    other = conversation.ToolResultContent(
        agent_id="agent", tool_call_id="call-2",
        tool_name="HassTurnOn", tool_result={"success": False},
    )

    compact_search = MessageHelper.compact_tool_result(search).tool_result

    assert compact_search["result_type"] == "candidate_search"
    assert compact_search["candidate_devices"] == ["large"]
    assert "no action has been performed" in compact_search["candidate_notice"]
    assert MessageHelper.compact_tool_result(other).tool_result == {"success": False}

def test_message_to_retrieval_text() -> None:
    assert MessageHelper.message_to_retrieval_text(
        conversation.UserContent(content="  turn on light  ")
    ) == "turn on light"
    assert MessageHelper.message_to_retrieval_text(
        conversation.ToolResultContent(
            agent_id="agent", tool_call_id="call", tool_name="HassTurnOn",
            tool_result={"success": True},
        )
    ) == "HassTurnOn {'success': True}"
    assert MessageHelper.message_to_retrieval_text(object()) == ""

def test_message_to_chat_messages() -> None:
    messages = MessageHelper.message_to_chat_messages([
        conversation.SystemContent(content="system"),
        conversation.UserContent(content="user"),
        conversation.AssistantContent(
            agent_id="agent", content="assistant",
            tool_calls=[ToolInput(id="call", tool_name="HassTurnOn", tool_args={"name": "light.x"})],
        ),
        conversation.ToolResultContent(
            agent_id="agent", tool_call_id="call", tool_name="HassTurnOn",
            tool_result={"success": True},
        ),
    ])

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == "HassTurnOn"
    assert messages[3]["tool_call_id"] == "call"

def test_clean_assistant_content() -> None:
    content = "Done\n```homeassistant\n{\"tool\": \"HassTurnOn\"}\n```"

    assert MessageHelper.clean_assistant_content(content, True) == "Done"
    assert MessageHelper.clean_assistant_content(content, False) == content

def test_repeated_success_result_contains_only_success_fields() -> None:
    message = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-1", "HassTurnOn",
        {"success": True, "description": "light", "error": "stale"},
    )

    assert message.tool_result == {
        "success": True,
        "already_executed": True,
    }

def test_repeated_failure_preserves_error() -> None:
    message = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-2", "HassTurnOn",
        {"success": False, "error": "target not found", "details": "ignored"},
    )

    assert message.tool_result == {
        "success": False,
        "already_executed": True,
        "error": "target not found",
    }

def test_repeated_list_and_search_results_remain_successful() -> None:
    list_result = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-3", "HassTurnOff", {"success": ["light.bedroom"]},
    )
    search_result = MessageHelper.create_repeated_tool_result_message(
        "agent", "call-4", "ha_ragent__HassSemanticSearch", {"devices": [], "error": []},
    )

    assert list_result.tool_result["success"] is True
    assert search_result.tool_result["result_type"] == "candidate_search"
    assert search_result.tool_result["reused"] is True


def test_repeated_search_reuses_compact_candidate_names() -> None:
    message = MessageHelper.create_repeated_tool_result_message(
        "agent",
        "call-5",
        "ha_ragent__HassSemanticSearch",
        {
            "candidate_devices": [{"name": "light.kitchen"}],
            "candidate_tools": [{"name": "HassTurnOn"}],
            "error": [],
        },
    )

    assert message.tool_result["candidate_devices"] == [{"name": "light.kitchen"}]
    assert message.tool_result["candidate_tools"] == [{"name": "HassTurnOn"}]
    assert message.tool_result["reused"] is True


def test_compact_search_preserves_device_state_and_location() -> None:
    result = MessageHelper.compact_tool_result_value(
        "ha_ragent__HassSemanticSearch",
        {
            "candidate_devices": [{
                "name": "sensor.kitchen_temperature",
                "friendly_name": "Kitchen temperature",
                "state": "21.5",
                "unit_of_measurement": "°C",
                "area": "Kitchen",
                "floor": "Ground floor",
                "domain": ["sensor"],
                "aliases": ["kitchen thermometer"],
                "attributes": {"temperature": 21.5},
            }],
            "error": [],
        },
    )

    candidate = result["candidate_devices"][0]
    assert candidate == {
        "name": "sensor.kitchen_temperature",
        "friendly_name": "Kitchen temperature",
        "state": "21.5",
        "unit_of_measurement": "°C",
        "area": "Kitchen",
        "floor": "Ground floor",
        "domain": ["sensor"],
        "aliases": ["kitchen thermometer"],
        "attributes": {"temperature": 21.5},
    }
    assert "directly" in result["candidate_data_notice"]


def test_compact_search_preserves_zero_tool_fallback_signal() -> None:
    result = MessageHelper.compact_tool_result_value(
        "ha_ragent__HassSemanticSearch",
        {
            "candidate_devices": [{"name": "switch.bathroom_heater"}],
            "candidate_tools": [],
            "tool_search_status": "no_tools_found",
            "fallback_required": True,
            "tool_search_message": "Do not invent a tool name.",
            "error": [],
        },
    )

    assert result["tool_search_status"] == "no_tools_found"
    assert result["fallback_required"] is True
    assert result["tool_search_message"] == "Do not invent a tool name."


def test_compact_search_keeps_capabilities_without_detailed_ranking_signals() -> None:
    result = MessageHelper.compact_tool_result_value(
        "ha_ragent__HassSemanticSearch",
        {
            "candidate_tools": [{
                "name": "HassTurnOn",
                "description": "Turn on a device",
                "canonical_action": "on",
                "supported_domains": ["light", "switch"],
                "retrieval_score": 6.25,
                "ranking_signals": {
                    "semantic_similarity": 0.8,
                    "action_intent": 1.0,
                    "domain": 0.35,
                },
                "parameters": {"omitted": True},
            }],
            "tool_search_confidence": "high",
            "error": [],
        },
    )

    candidate = result["candidate_tools"][0]
    assert candidate["name"] == "HassTurnOn"
    assert candidate["canonical_action"] == "on"
    assert candidate["retrieval_score"] == 6.25
    assert "ranking_signals" not in candidate
    assert "parameters" not in candidate
    assert result["tool_search_confidence"] == "high"


def test_long_success_result_is_bounded() -> None:
    result = MessageHelper.compact_tool_result_value(
        "HassTurnOn",
        {"success": [f"light.room_{index}" for index in range(30)]},
    )

    assert len(result["success"]) == MessageHelper._MAX_RESULT_ITEMS


def test_tool_result_success_rejects_failed_and_error_results() -> None:
    assert MessageHelper.tool_result_succeeded({"success": ["light.kitchen"]})
    assert MessageHelper.tool_result_succeeded({"devices": [], "error": []})
    assert not MessageHelper.tool_result_succeeded({"success": False})
    assert not MessageHelper.tool_result_succeeded({"success": []})
    assert not MessageHelper.tool_result_succeeded({"failed": ["light.kitchen"]})
    assert not MessageHelper.tool_result_succeeded({"error": "unavailable"})


def test_custom_result_status_does_not_hide_payload_or_recovery_details():
    results = [
        {"success": True, "temperature": 21, "unit": "C"},
        {"success": True, "memory_id": "abc", "content": "Thermostat target is 21 C"},
        {"success": True, "minutes": 10, "execute_at": "2026-09-05T12:10:00"},
        {"success": False, "error": "invalid profile", "valid_profiles": ["quiet", "boost"]},
    ]
    for result in results:
        compact = MessageHelper.compact_tool_result_value("VendorTool", result)
        assert compact == result
        assert MessageHelper.compact_tool_result_value("VendorTool", compact) == compact


def test_custom_nested_result_is_bounded_and_keeps_status():
    result = {
        "rows": [{"text": "x" * 10000} for _ in range(100)],
        "details": {str(index): "x" * 10000 for index in range(100)},
        "deep": {"a": {"b": {"c": {"d": {"e": 1}}}}},
        "success": True,
    }
    compact = MessageHelper.compact_tool_result_value("VendorTool", result)
    assert compact["success"] is True
    assert len(compact["rows"]) == MessageHelper._MAX_RESULT_ITEMS
    assert len(compact["rows"][0]["text"]) == MessageHelper._MAX_RESULT_TEXT
    assert len(compact["details"]) == MessageHelper._MAX_RESULT_ITEMS
    assert compact["deep"]["a"]["b"]["c"]["d"] == "[truncated]"
