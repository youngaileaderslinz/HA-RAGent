from types import SimpleNamespace

import voluptuous as vol

from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor
from custom_components.ha_ragent.src.homeassistant.ragent import RAGent


def test_tool_metadata_uses_explicit_capability_and_schema_domains() -> None:
    extractor = ToolExtractor.__new__(ToolExtractor)
    tool = SimpleNamespace(metadata={
        "action": "stop",
        "expected_states": ["idle", "off"],
    })
    parameters = {
        "properties": {
            "domain": {"enum": ["media_player"]},
            "name": {"type": "string"},
        },
    }

    metadata = extractor._extract_tool_metadata(tool, parameters)

    assert metadata.canonical_action == "stop"
    assert metadata.supported_domains == ("media_player",)
    assert metadata.expected_states == ("idle", "off")


def test_tool_metadata_uses_a_single_schema_action_without_reading_tool_name() -> None:
    extractor = ToolExtractor.__new__(ToolExtractor)
    parameters = {
        "properties": {
            "action": {"const": "pause"},
            "domain": {"enum": ["media_player"]},
            "expected_state": {"const": "paused"},
        },
    }

    first = extractor._extract_tool_metadata(SimpleNamespace(name="StopEverything"), parameters)
    second = extractor._extract_tool_metadata(SimpleNamespace(name="BeliebigerName"), parameters)

    assert first.canonical_action == second.canonical_action == "pause"
    assert first.supported_domains == second.supported_domains == ("media_player",)
    assert first.expected_states == second.expected_states == ("paused",)


def test_runtime_converted_tool_keeps_canonical_metadata(monkeypatch) -> None:
    api_tool = SimpleNamespace(
        name="VendorStop",
        description="Stop playback",
        parameters=object(),
        metadata={
            "action": "stop",
            "supported_domains": ["media_player"],
            "expected_states": ["idle"],
        },
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.ragent.to_openapi",
        lambda *_args, **_kwargs: {"properties": {}},
    )
    agent = RAGent.__new__(RAGent)

    converted = agent._convert_api_tool(api_tool, SimpleNamespace(custom_serializer=None))

    assert converted.canonical_action == "stop"
    assert converted.canonical_supported_domains == ("media_player",)
    assert converted.metadata.expected_states == ("idle",)


def test_native_intent_tool_gets_production_capability_metadata() -> None:
    tool = SimpleNamespace(
        name="intent__HassTurnOn",
        intent_type="HassTurnOn",
        metadata=None,
    )

    metadata = ToolExtractor.extract_tool_metadata(tool, {"properties": {}})

    assert metadata.canonical_action == "turn_on"
    assert metadata.expected_states == ("on",)


def test_extract_tool_metadata_extracts_domain_enums_directly() -> None:
    tool = SimpleNamespace(metadata=None)

    fan = ToolExtractor.extract_tool_metadata(
        tool, {"properties": {"domain": {"enum": ["fan"]}}}
    )
    light = ToolExtractor.extract_tool_metadata(
        tool, {"properties": {"domain": {"enum": ["light"]}}}
    )

    assert fan.supported_domains == ("fan",)
    assert light.supported_domains == ("light",)


def test_extract_tool_metadata_keeps_universal_domain_schema_unrestricted() -> None:
    tool = SimpleNamespace(
        metadata=None,
        parameters=vol.Schema({vol.Required("domain"): str}),
    )

    values, universal, has_field = ToolExtractor._extract_field_constraints(
        tool.parameters.schema, "domain"
    )
    metadata = ToolExtractor.extract_tool_metadata(tool, {"properties": {}})

    assert values == []
    assert universal is True
    assert has_field is True
    assert metadata.supported_domains == ()
