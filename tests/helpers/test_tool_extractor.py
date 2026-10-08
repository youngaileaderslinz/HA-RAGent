import asyncio
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import probatio

from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata


def test_live_tool_selection_uses_current_schemas_and_requested_order() -> None:
    first = SimpleNamespace(
        name="FirstAction", description="Current first action",
        parameters=probatio.Schema({
            probatio.Required("level", description="Current level description"): int,
        }),
        metadata={"action": "set_level", "supported_domains": ["light"]},
    )
    second = SimpleNamespace(name="SecondAction", description="Current second action", parameters={})
    unrequested = SimpleNamespace(name="OtherAction", parameters={})
    api = SimpleNamespace(tools=[first, unrequested, second], custom_serializer=None)

    tools = ToolExtractor.tools_from_api(api, ["SecondAction", "MissingAction", "FirstAction", "SecondAction"])

    assert [tool.name for tool in tools] == ["SecondAction", "FirstAction"]
    assert tools[1].description == first.description
    assert tools[1].parameters["properties"]["level"] == {
        "type": "integer", "description": "Current level description",
    }
    assert tools[1].parameters["required"] == ["level"]
    assert tools[1].canonical_action == "set_level"
    assert tools[1].canonical_supported_domains == ("light",)


def test_live_tool_selection_skips_invalid_schema_and_keeps_other_tools() -> None:
    broken = SimpleNamespace(
        name="BrokenAction",
        parameters={"type": "function", "function": {"parameters": {"type": "array"}}},
    )
    valid = SimpleNamespace(name="ValidAction", parameters={})

    tools = ToolExtractor.tools_from_api(SimpleNamespace(tools=[broken, valid]), [broken.name, valid.name])

    assert [tool.name for tool in tools] == [valid.name]


def test_live_tool_selection_keeps_schema_when_metadata_is_invalid() -> None:
    raw = SimpleNamespace(
        name="VendorAction", description="Current vendor action",
        parameters={"type": "object", "properties": {}},
        metadata={"supported_domains": 42},
    )

    tool, = ToolExtractor.tools_from_api(SimpleNamespace(tools=[raw]), [raw.name])

    assert tool.description == raw.description
    assert tool.parameters["type"] == "object"
    assert tool.metadata == ToolMetadata()


def test_live_tool_selection_passes_api_custom_serializer(monkeypatch) -> None:
    source, serializer = object(), object()
    calls = []

    def convert(parameters, *, custom_serializer):
        calls.append((parameters, custom_serializer))
        return {"type": "object", "properties": {}}

    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi", convert,
    )
    raw = SimpleNamespace(name="VendorAction", parameters=source)

    tool, = ToolExtractor.tools_from_api(
        SimpleNamespace(tools=[raw], custom_serializer=serializer), [raw.name],
    )

    assert tool.name == raw.name
    assert calls == [(source, serializer)]


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _subentry() -> SimpleNamespace:
    return SimpleNamespace(data={}, title="test")


@_run_async
async def test_startup_extraction_embeds_more_than_zero_tools(monkeypatch) -> None:
    raw_tool = SimpleNamespace(
        name="HassFanSetSpeed",
        description="Set fan speed",
        parameters=object(),
        metadata=None,
    )
    api = SimpleNamespace(tools=[raw_tool], custom_serializer=None)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi",
        lambda *_args, **_kwargs: {"properties": {"domain": {"enum": ["fan"]}}},
    )

    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert len(tools) > 0
    assert tools[0].metadata.supported_domains == ("fan",)
    assert tools[0].parameters == {
        "type": "object",
        "properties": {
            "domain": {"type": ["string", "null"], "enum": ["fan", None]},
        },
        "required": [],
        "additionalProperties": False,
    }


@_run_async
async def test_metadata_extraction_exception_does_not_zero_all_tools(monkeypatch) -> None:
    raw_tools = [
        SimpleNamespace(name="BrokenTool", description="broken", parameters=object(), metadata=None),
        SimpleNamespace(name="HassTurnOn", description="turn on", parameters=object(), metadata=None),
    ]
    api = SimpleNamespace(tools=raw_tools, custom_serializer=None)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi",
        lambda *_args, **_kwargs: {"properties": {}},
    )

    original = ToolExtractor.extract_tool_metadata
    calls = 0

    def fail_once(self, tool, parameters):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("bad metadata")
        return original(tool, parameters)

    monkeypatch.setattr(ToolExtractor, "extract_tool_metadata", fail_once)
    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert [tool.name for tool in tools] == ["BrokenTool", "HassTurnOn"]


@_run_async
async def test_startup_embedding_persists_more_than_zero_tools(monkeypatch) -> None:
    subentry = _subentry()
    tool = LlmTool("HassTurnOn", "Turn on", parameters={"properties": {}})
    entry = SimpleNamespace(
        subentries={"sub": subentry},
        embedder_backend=SimpleNamespace(
            async_embed_object=AsyncMock(
                return_value=[LlmToolEmbedding(tool, [1.0])]
            )
        ),
        vector_db_backend=SimpleNamespace(
            async_reset_collection=AsyncMock(),
            async_save_objects=AsyncMock(),
            invalidate_collection_cache=lambda _name: None,
            cache_collection_objects=lambda _name, _objects: None,
        ),
    )
    extractor = ToolExtractor(SimpleNamespace(), entry)
    monkeypatch.setattr(
        extractor,
        "_async_get_embeddable_tools",
        AsyncMock(return_value=[tool]),
    )

    await extractor.async_embed_exposed_tools("sub")

    entry.embedder_backend.async_embed_object.assert_awaited_once()
    entry.vector_db_backend.async_save_objects.assert_awaited_once()


@_run_async
async def test_extractor_preserves_openai_parameter_schema(monkeypatch) -> None:
    schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["auto", "manual"]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    raw_tools = [
        SimpleNamespace(name="Direct", description="Direct schema", parameters=schema, metadata=None),
        SimpleNamespace(
            name="Wrapped",
            description="Wrapped schema",
            parameters={"type": "function", "function": {"parameters": schema}},
            metadata=None,
        ),
    ]
    api = SimpleNamespace(tools=raw_tools, custom_serializer=None)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("already JSON Schema")),
    )
    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert [tool.parameters for tool in tools] == [schema, schema]


def test_tool_dict_includes_empty_openai_parameters() -> None:
    tool = LlmTool("NoArguments", "Run without arguments")

    assert tool.to_tool_dict() == {
        "type": "function",
        "function": {
            "name": "NoArguments",
            "description": "Run without arguments",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def test_openai_parameters_normalizes_nested_optional_fields_without_mutating_input() -> None:
    schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}},
                },
                "required": ["name"],
            },
        },
        "required": ["device"],
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert schema["properties"]["device"]["required"] == ["name"]
    assert normalized["required"] == ["device"]
    assert normalized["additionalProperties"] is False
    device = normalized["properties"]["device"]
    assert device["required"] == ["name"]
    assert device["additionalProperties"] is False
    assert device["properties"]["tags"]["type"] == ["array", "null"]
    item = device["properties"]["tags"]["items"]
    assert item["required"] == []
    assert item["additionalProperties"] is False
    assert item["properties"]["id"]["type"] == ["integer", "null"]


def test_openai_parameters_removes_required_only_root_anyof() -> None:
    schema = {
        "type": "object",
        "properties": {"hours": {"type": "integer"}, "minutes": {"type": "integer"}},
        "anyOf": [{"required": ["hours"]}, {"required": ["minutes"]}],
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert "anyOf" not in normalized
    assert normalized["required"] == []
    assert normalized["additionalProperties"] is False


def test_openai_parameters_flattens_compatible_composition() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"allOf": [{"type": "string"}]}},
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["value"] == {"type": ["string", "null"]}


def test_openai_parameters_rejects_conflicting_composition() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer", "allOf": [{"type": "string"}]}},
    }

    with pytest.raises(ValueError, match="allOf"):
        ToolExtractor._openai_parameters(schema)


def test_optional_string_uses_nullable_type_union() -> None:
    schema = {
        "type": "object",
        "properties": {"conversation_command": {"type": "string"}},
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["conversation_command"] == {
        "type": ["string", "null"],
    }
    assert normalized["required"] == []


def test_existing_nullable_object_type_is_preserved() -> None:
    schema = {
        "type": "object",
        "properties": {
            "device": {
                "type": ["object", "null"],
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["device"]["type"] == ["object", "null"]


def test_optional_integer_from_probatio_uses_nullable_type_union() -> None:
    schema = {
        "type": "object",
        "properties": {
            "minutes": {"type": "integer", "minimum": 1},
            "hours": {"type": "integer", "nullable": True},
        },
        "required": [],
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["minutes"] == {
        "type": ["integer", "null"], "minimum": 1,
    }
    assert normalized["properties"]["hours"] == {"type": ["integer", "null"]}
    assert normalized["required"] == []


def test_probatio_alternative_types_no_longer_use_anyof() -> None:
    schema = {
        "type": "object",
        "properties": {
            "action": {"anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]},
        },
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["action"] == {
        "type": ["string", "array", "null"],
        "items": {"type": "string"},
    }


def test_native_probatio_intent_schema_converts_optional_integer() -> None:
    tool = SimpleNamespace(parameters=probatio.Schema({
        probatio.Required("name"): str,
        probatio.Optional("minutes"): int,
    }))

    parameters = ToolExtractor._tool_parameters(tool, None)

    assert parameters["properties"]["name"] == {"type": "string"}
    assert parameters["properties"]["minutes"] == {"type": ["integer", "null"]}
    assert parameters["required"] == ["name"]
    assert parameters["additionalProperties"] is False


@_run_async
async def test_invalid_parameter_schema_skips_only_affected_tool(monkeypatch) -> None:
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="Invalid", description="", metadata=None,
                parameters={
                    "type": "object",
                    "properties": {"value": {
                        "type": "integer", "allOf": [{"type": "string"}],
                    }},
                },
            ),
            SimpleNamespace(
                name="Valid", description="", metadata=None,
                parameters={"type": "object", "properties": {"count": {"type": "integer"}}},
            ),
        ],
        custom_serializer=None,
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert [tool.name for tool in tools] == ["Valid"]
    assert tools[0].parameters["properties"]["count"]["type"] == ["integer", "null"]


def test_probatio_literal_alternatives_become_enum() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"anyOf": [{"const": "auto"}, {"const": "manual"}]},
        },
    }

    normalized = ToolExtractor._openai_parameters(schema)

    assert normalized["properties"]["mode"] == {
        "type": ["string", "null"], "enum": ["auto", "manual", None],
    }
