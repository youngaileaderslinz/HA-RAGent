from copy import deepcopy
import json
from types import SimpleNamespace

from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.const import RAGENT_PREFIXED_REQUIRED_TOOL_NAMES
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool


def test_tool_serialization_is_stable_when_retrieval_ranking_changes():
    tools = [
        LlmTool("ZAction", "Last action", parameters={"type": "object"}),
        LlmTool(RAGENT_PREFIXED_REQUIRED_TOOL_NAMES[0], "Search", parameters={"type": "object"}),
        LlmTool("AAction", "First action", parameters={"type": "object"}),
    ]
    original = list(tools)

    first = ALlmBaseBackend.convert_tools_to_model_format(tools)
    second = ALlmBaseBackend.convert_tools_to_model_format(list(reversed(tools)))

    assert json.dumps(first) == json.dumps(second)
    assert [tool["function"]["name"] for tool in first] == [
        RAGENT_PREFIXED_REQUIRED_TOOL_NAMES[0], "AAction", "ZAction",
    ]
    assert tools == original
    assert first[-1] == tools[0].to_tool_dict()


def test_openai_history_uses_compact_unicode_json_without_changing_data():
    backend = OpenAiLlmBackend(SimpleNamespace(), {})
    arguments = {"name": "Küche", "temperature": 0}
    result = {"success": True, "temperature": 0, "unit": "°C", "enabled": False}
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "SetTemperature", "arguments": arguments,
            },
        }]},
        {"role": "tool", "tool_call_id": "call-1", "tool_name": "SetTemperature", "content": result},
    ]
    original = deepcopy(messages)

    prepared = backend.format_messages_for_backend(messages)
    encoded_arguments = prepared[0]["tool_calls"][0]["function"]["arguments"]
    encoded_result = prepared[1]["content"]

    assert json.loads(encoded_arguments) == arguments
    assert json.loads(encoded_result) == result
    assert "Küche" in encoded_arguments
    assert "°C" in encoded_result
    assert len(encoded_result) < len(json.dumps(result))
    assert prepared[1]["tool_call_id"] == "call-1"
    assert messages == original


def test_openai_preserves_already_serialized_history():
    backend = OpenAiLlmBackend(SimpleNamespace(), {})
    content = '{ "success": true }'
    arguments = '{ "name": "light.desk" }'
    prepared = backend.format_messages_for_backend([
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "TurnOn", "arguments": arguments,
            },
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": content},
    ])

    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert prepared[1]["content"] == content
