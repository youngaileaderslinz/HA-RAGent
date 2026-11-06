import asyncio
from copy import deepcopy
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

    def convert(parameters, *, custom_serializer, openapi_version):
        calls.append((parameters, custom_serializer, openapi_version))
        return {"type": "object", "properties": {}}

    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi", convert,
    )
    raw = SimpleNamespace(name="VendorAction", parameters=source)

    tool, = ToolExtractor.tools_from_api(
        SimpleNamespace(tools=[raw], custom_serializer=serializer), [raw.name],
    )

    assert tool.name == raw.name
    assert calls == [(source, serializer, "3.1.0")]


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
        "properties": {
            "domain": {"enum": ["fan"]},
        },
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
async def test_extractor_preserves_preconverted_parameter_schema(monkeypatch) -> None:
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


def test_tool_dict_includes_empty_function_parameters() -> None:
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


@pytest.mark.parametrize("schema", [
    {
        "type": "object",
        "properties": {
            "device": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
                "additionalProperties": True,
            },
        },
        "required": ["device"],
    },
    {
        "type": "object",
        "properties": {"hours": {"type": "integer"}, "minutes": {"type": "integer"}},
        "anyOf": [{"required": ["hours"]}, {"required": ["minutes"]}],
    },
    {
        "type": "object",
        "properties": {
            "action": {"anyOf": [
                {"type": "string"}, {"type": "array", "items": {"type": "string"}},
            ]},
            "mode": {"oneOf": [{"const": "auto"}, {"const": "manual"}]},
        },
    },
    {
        "type": "object",
        "properties": {"value": {"allOf": [{"type": "integer"}, {"minimum": 1}]}},
    },
    {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email", "default": "user@example.com"},
            "label": {"type": ["string", "null"]},
            "mode": {"const": "auto"},
        },
        "dependentRequired": {"email": ["label"]},
        "if": {"required": ["email"]},
        "then": {"required": ["label"]},
        "else": {"not": {"required": ["label"]}},
        "additionalProperties": {"type": "string"},
    },
    {
        "type": "object",
        "properties": {"target": {"$ref": "#/$defs/target"}},
        "$defs": {"target": {"type": "string", "pattern": "^light\\."}},
        "patternProperties": {"^vendor_": {"type": "string"}},
        "propertyNames": {"minLength": 1},
    },
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_preconverted_schema_preserves_constraints_without_mutating_input(schema, wrapped) -> None:
    original = deepcopy(schema)
    source = {"type": "function", "function": {"parameters": schema}} if wrapped else schema

    parameters = ToolExtractor._tool_parameters(SimpleNamespace(parameters=source), None)

    assert parameters == original
    assert schema == original


def test_native_probatio_schema_matches_home_assistant_formatters() -> None:
    from homeassistant.components.llama_cpp.entity import _format_tool as llama_format
    from homeassistant.components.ollama.entity import _format_tool as ollama_format

    raw = SimpleNamespace(name="ExampleAction", description="Example action", parameters=probatio.Schema({
        probatio.Required("name"): str,
        probatio.Optional("minutes"): probatio.All(int, probatio.Range(min=1)),
        probatio.Optional("target"): probatio.Any(str, [str]),
        probatio.Optional("mode"): probatio.In(["auto", "manual"]),
    }))

    parameters = ToolExtractor._tool_parameters(raw, None)
    tool = LlmTool(raw.name, raw.description, parameters=parameters)

    assert tool.to_tool_dict() == ollama_format(raw, None) == llama_format(raw, None)
    assert parameters["properties"]["minutes"] == {"type": "integer", "minimum": 1}
    assert parameters["required"] == ["name"]


def test_native_conversion_preserves_explicit_null_as_openapi_31() -> None:
    raw = SimpleNamespace(parameters=probatio.Schema({
        probatio.Optional("label"): probatio.Any(str, None),
    }))

    parameters = ToolExtractor._tool_parameters(raw, None)

    assert parameters["properties"]["label"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
    }


@_run_async
async def test_invalid_parameter_schema_skips_only_affected_tool(monkeypatch) -> None:
    api = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name="Invalid", description="", metadata=None,
                parameters={"type": "function", "function": {"parameters": {"type": "array"}}},
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
    assert tools[0].parameters == api.tools[1].parameters
