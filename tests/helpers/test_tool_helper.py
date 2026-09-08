import json
from unittest.mock import Mock


from custom_components.ha_ragent.src.homeassistant.helpers import (
    tool_helper as tool_helper_module,
)
from custom_components.ha_ragent.src.homeassistant.helpers.tool_helper import ToolHelper
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata

def test_parse_tool_call_preserves_friendly_name() -> None:
    """A friendly-name target must survive nested argument parsing."""
    hass = Mock()
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": "{\\"name\\":\\"Light Strip\\",\\"area\\":\\"Guest Bedroom\\"}"}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_name == "HassTurnOn"
    assert calls[0].tool_args == {
        "name": "Light Strip",
        "area": "Guest Bedroom",
    }
    hass.states.get.assert_not_called()

def test_parse_tool_call_preserves_apostrophes_in_json_strings() -> None:
    """Valid JSON strings containing apostrophes must not be rewritten."""
    helper = ToolHelper(Mock())
    response = """```homeassistant
{"tool": "HassRememberFact", "arguments": {"memory": "The guest's reading lamp is dimmable."}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {"memory": "The guest's reading lamp is dimmable."}


def test_unknown_tool_arguments_are_preserved_exactly_from_live_schema() -> None:
    helper = ToolHelper(Mock())
    arguments = {
        "name": "raw.vendor_target",
        "friendly_name": "schema-owned value",
        "original_name": "schema-owned original",
        "options": {"mode": "turbo", "levels": [1, 2]},
    }
    response = f'''```homeassistant
{{"tool": "VendorExecute", "arguments": {json.dumps(arguments)}}}
```'''

    calls = helper.parse_tool_calls(response, {"VendorExecute": ToolMetadata()})
    execution_call = helper.to_home_assistant_tool_call(calls[0], ToolMetadata())

    assert execution_call.tool_args == arguments


def test_parse_tool_call_uses_device_class_as_missing_domain() -> None:
    """A device class supplies the domain when the model omitted it."""
    hass = Mock()
    hass.states.get.return_value = None
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"name": "Kitchen Window", "device_class": ["window"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {
        "name": "Kitchen Window",
        "device_class": ["window"],
    }

def test_parse_tool_call_prefers_explicit_domain() -> None:
    """An explicit domain takes precedence over the device class fallback."""
    hass = Mock()
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"area": "Living Room", "domain": ["light"], "device_class": ["switch"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert len(calls) == 1
    assert calls[0].tool_args == {
        "area": "Living Room",
        "domain": ["light"],
        "device_class": ["switch"],
    }

def test_parse_tool_call_converts_entity_id_and_resolves_friendly_name(
    monkeypatch,
) -> None:
    """An entity-id target is normalized to its configured friendly name."""
    hass = Mock()
    hass.data = {}
    hass.states.get.return_value = Mock(
        attributes={"friendly_name": "Bedroom Ceiling Light"}
    )
    monkeypatch.setattr(tool_helper_module, "entity_registry", None)
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"entity_id": "light.bedroom_ceiling"}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args == {
        "friendly_name": "Bedroom Ceiling Light",
        "original_name": "light.bedroom_ceiling",
    }
    hass.states.get.assert_called_once_with("light.bedroom_ceiling")

def test_parse_tool_call_finds_domain_from_entity_id(monkeypatch) -> None:
    """A full entity ID supplies the missing domain."""
    hass = Mock()
    hass.data = {}
    hass.states.get.return_value = None
    monkeypatch.setattr(tool_helper_module, "entity_registry", None)
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"entity_id": "switch.bedroom_fan"}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args == {
        "friendly_name": "switch.bedroom_fan",
        "original_name": "switch.bedroom_fan",
    }
    hass.states.get.assert_called_once_with("switch.bedroom_fan")

def test_parse_tool_call_preserves_name_without_domain_aware_metadata() -> None:
    """A name is not resolved when domain-aware metadata is unavailable."""
    hass = Mock()
    hass.states.get.return_value = Mock(
        attributes={"friendly_name": "Bedroom 2 Ceiling Light"}
    )
    helper = ToolHelper(hass)
    response = """```homeassistant
{"tool": "HassTurnOn", "arguments": {"name": "bedroom_2_ceiling_light", "domain": ["light"]}}
```"""

    calls = helper.parse_tool_calls(response)

    assert calls[0].tool_args["name"] == "bedroom_2_ceiling_light"
    assert "original_name" not in calls[0].tool_args
    hass.states.get.assert_not_called()

def test_tool_call_signature_ignores_argument_order() -> None:
    """Equivalent calls have the same signature regardless of key order."""
    helper = ToolHelper(Mock())
    first = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 1"},
    )
    second = Mock(
        tool_name="HassTurnOn",
        tool_args={"area": "Bedroom 1", "domain": ["light"]},
    )

    assert helper.tool_call_signature(first) == helper.tool_call_signature(second)

def test_tool_call_signature_distinguishes_targets() -> None:
    """Calls to different targets remain independently executable."""
    helper = ToolHelper(Mock())
    bedroom_one = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 1"},
    )
    bedroom_two = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom 2"},
    )

    assert helper.tool_call_signature(bedroom_one) != helper.tool_call_signature(
        bedroom_two
    )


def test_identical_failed_retry_ignores_argument_order() -> None:
    helper = ToolHelper(Mock())
    first = Mock(
        tool_name="HassTurnOn",
        tool_args={"domain": ["light"], "area": "Bedroom"},
    )
    retry = Mock(
        tool_name="HassTurnOn",
        tool_args={"area": "Bedroom", "domain": ["light"]},
    )

    failed = {helper.tool_call_signature(first): {"success": False}}

    assert helper.is_identical_failed_retry(retry, failed)


def test_semantic_search_signature_normalizes_query_and_scope() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "  TURN   ON lights ", "scope": "TOOLS"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"scope": "tools", "search_query": "turn on LIGHTS"},
    )

    assert ToolHelper.tool_call_signature(first) == ToolHelper.tool_call_signature(second)


def test_semantic_search_signature_does_not_infer_equivalent_actions() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "switch off kitchen lights", "scope": "tools"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "power off the kitchen light", "scope": "tools"},
    )

    assert ToolHelper.tool_call_signature(first) != ToolHelper.tool_call_signature(second)


def test_semantic_search_signature_distinguishes_different_capabilities() -> None:
    first = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "heater bathroom switch toggle", "scope": "tools"},
    )
    second = Mock(
        tool_name="ha_ragent__HassSemanticSearch",
        tool_args={"search_query": "heater bathroom on/off", "scope": "tools"},
    )

    assert ToolHelper.tool_call_signature(first) != ToolHelper.tool_call_signature(second)


def test_exposed_tool_name_normalizes_only_known_namespace_variants() -> None:
    exposed = {"HassTurnOn", "ha_ragent__HassSemanticSearch"}

    assert ToolHelper.resolve_exposed_tool_name("switch__HassTurnOn", exposed) == "HassTurnOn"
    assert ToolHelper.resolve_exposed_tool_name("HassSemanticSearch", exposed) == (
        "ha_ragent__HassSemanticSearch"
    )
    assert ToolHelper.resolve_exposed_tool_name("switch__HassSwitchToggle", exposed) is None



def test_discovered_tools_are_converted_for_next_iteration() -> None:
    existing_names = {"HassSemanticSearch"}
    discovered = ToolHelper.discovered_tools(
        {
            "candidate_tools": [{
                "name": "HassTurnOn",
                "description": "Turn on a target",
                "parameters": {"properties": {"name": {"type": "string"}}},
                "metadata": {"family": "power", "is_domain_aware": True},
            }],
        },
        existing_names,
    )

    assert [tool.name for tool in discovered] == ["HassTurnOn"]
    assert discovered[0].metadata.family == "power"
    assert discovered[0].metadata.is_domain_aware is True
    assert "HassTurnOn" in existing_names


def test_discovered_unknown_tool_keeps_live_schema_and_generic_metadata() -> None:
    parameters = {"type": "object", "properties": {"mode": {"enum": ["quiet"]}}}
    discovered = ToolHelper.discovered_tools(
        {"candidate_tools": [{
            "name": "VendorExecute",
            "description": "Execute a vendor operation",
            "parameters": parameters,
        }]},
        set(),
    )

    assert discovered[0].parameters is parameters
    assert discovered[0].metadata == ToolMetadata()


def test_device_registry_lookup_uses_device_id(monkeypatch) -> None:
    from custom_components.ha_ragent.src.homeassistant.helpers import tool_helper
    registry = Mock()
    registry.async_get.return_value = Mock(area_id=None)
    monkeypatch.setattr(tool_helper, "device_registry", Mock(async_get=Mock(return_value=registry)))
    ToolHelper(Mock())._parse_area_and_floor(Mock(area_id=None, device_id="device-1"), None, None)
    registry.async_get.assert_called_once_with("device-1")
    registry.async_get_device.assert_not_called()


