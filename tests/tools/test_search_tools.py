import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.helpers import llm
from probatio import to_openapi

from custom_components.ha_ragent.src import const
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import (
    RAGentSemanticSearchTool,
)
from custom_components.ha_ragent.src.homeassistant.tools import search_tools as search_tools_module
from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


def _search_tool() -> RAGentSemanticSearchTool:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool.entry_id = "entry"
    tool.subentry_id = "agent"
    tool.set_search_context()
    return tool


def _candidates(tool, result, kind):
    return [tool._results[r["id"]]["data"] for r in result["results"] if r["type"] == kind]


def test_device_search_provides_ids_and_context_without_detail(compact_search):
    search, _ = compact_search
    call = SimpleNamespace(tool_args={"search_queries": ["office lights"], "scope": "devices"})

    result = asyncio.run(search._async_call(call))
    devices, tools = search.result_context(result)

    assert result["status"] == "ok"
    assert {item["entity_id"] for item in result["results"]} == {
        f"light.office_{index}" for index in range(4)
    }
    assert all("off" in item["summary"] for item in result["results"])
    assert all("attributes" not in item for item in result["results"])
    assert {device["name"] for device in devices} == {item["entity_id"] for item in result["results"]}
    assert tools == []
    assert search._details == {}

    repeated = asyncio.run(search._async_call(call))
    assert repeated["status"] == "same_as"
    assert search.result_context(repeated) == (devices, [])
    search.reset_turn()
    assert search.result_context(result) == ([], [])


def test_detail_remains_available_for_missing_facts_and_unexposed_tools(compact_search):
    search, actions = compact_search
    devices = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "scope": "devices",
    })))
    device = devices["results"][0]
    search.hass.states.get(device["entity_id"]).attributes["brightness"] = 128
    detail_call = SimpleNamespace(tool_args={
        "operation": "detail", "result_id": device["id"], "scope": "devices",
    })

    detail = asyncio.run(search._async_call(detail_call))
    assert detail["detail"]["attributes"]["brightness_percent"] == 50
    assert asyncio.run(search._async_call(detail_call))["status"] == "same_as"

    tools = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "scope": "tools",
    })))
    assert search.result_context(tools) == ([], [])
    assert "parameters" not in tools["results"][0]
    tool_detail = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "operation": "detail", "result_id": tools["results"][0]["id"], "scope": "tools",
    })))
    assert search.result_context(tool_detail) == ([], [actions[0].name])
    assert tool_detail["detail"]["parameters"] == actions[0].parameters


def test_semantic_search_schema_explains_parallel_structured_capabilities() -> None:
    schema = to_openapi(RAGentSemanticSearchTool.parameters, openapi_version="3.1.0")

    assert "search_queries" not in schema.get("required", [])
    assert "capabilities" not in schema.get("required", [])
    assert "position" in schema["properties"]["search_queries"]["description"]
    capability_description = schema["properties"]["capabilities"]["description"]
    assert "turn_on" in capability_description
    action_schema = schema["properties"]["capabilities"]["items"]["properties"]["action"]
    assert "canonical action ID" in action_schema["description"]
    assert "action" not in schema["properties"]["capabilities"]["items"].get("required", [])


def test_search_schema_exposes_typed_nested_filters():
    schema = to_openapi(RAGentSemanticSearchTool.parameters, openapi_version="3.1.0")
    filters = schema["properties"]["filters"]

    assert filters["properties"]["area"] == {"type": "string"}
    assert filters["properties"]["domain"]["anyOf"] == [
        {"type": "string"}, {"type": "array", "items": {"type": "string"}},
    ]
    assert filters["additionalProperties"] is False
    assert "area" not in schema["properties"]
    assert "domain" not in schema["properties"]
    assert "scope" not in schema.get("required", [])
    assert schema["properties"]["scope"]["default"] == "devices"


@pytest.mark.parametrize("arguments", [
    {"scope": "devices", "filters": {"domain": [None]}},
    {"scope": "devices", "filters": {"unknown": "Bedroom 1"}},
    {"scope": "devices", "area": "Office", "filters": {"area": "Bedroom 1"}},
    {"scope": "tools", "filters": {"scope": "devices"}},
    {"scope": "devices", "action": "turn_on"},
])
def test_malformed_search_calls_return_correction_without_retrieval(compact_search, arguments):
    search, _ = compact_search
    call = llm.ToolInput(tool_name=search.name, tool_args=arguments)

    result = asyncio.run(search.async_call(search.hass, call, None))
    data = getattr(result, "data", result)

    assert data["status"] == "validation_error"
    assert "inside filters" in data["error"]
    assert "scope is devices" in data["error"]
    assert "Correct the arguments and retry" in data["error"]
    assert "results" not in data
    assert getattr(result, "error", True) is True
    entry = search.hass.data[const.DOMAIN][search.entry_id]
    entry.embedder_backend.async_embed_text.assert_not_awaited()
    entry.vector_db_backend.async_retrieve_scored_objects.assert_not_awaited()


@pytest.mark.parametrize("arguments", [
    {"area": "Office", "domain": ["light"], "scope": "devices"},
    {"action": "search", "domain": ["light", "fan", "sensor"], "area": "Office", "scope": "devices"},
    {"operation": "search", "area": "Office", "domain": ["light"]},
    {"filters": {"area": "Office", "scope": "devices", "domain": "light"}},
])
def test_search_repairs_logged_argument_shapes_without_dropping_filters(compact_search, arguments):
    search, _ = compact_search
    search.set_search_context(latest_request="office devices")
    original = deepcopy(arguments)
    call = llm.ToolInput(tool_name=search.name, tool_args=arguments)

    result = asyncio.run(search.async_call(search.hass, call, None))
    data = getattr(result, "data", result)

    assert data["status"] == "ok"
    assert len(data["results"]) == 4
    assert arguments == original
    repaired = search._repair_arguments(arguments)
    assert repaired["filters"]["area"] == "Office"
    assert repaired["filters"]["domain"] == arguments.get("domain", arguments.get("filters", {}).get("domain"))
    assert search._details == {}


@pytest.mark.parametrize("domains,count", [(["fan", "light"], 4), (["fan", "sensor"], 0)])
def test_domain_alternatives_preserve_filtering(compact_search, domains, count):
    search, _ = compact_search
    result = asyncio.run(search.async_call(search.hass, llm.ToolInput(tool_name=search.name, tool_args={
        "search_queries": ["office devices"], "filters": {"area": "Office", "domain": domains},
    }), None))
    data = getattr(result, "data", result)

    assert data["status"] == ("ok" if count else "no_matches")
    assert len(data["results"]) == count


@pytest.mark.parametrize("field,value", [
    ("area", "  OFFICE  "), ("area", "office_area"), ("area", "BÜRO"),
    ("floor", "  GROUND  "), ("floor", "ground_floor"), ("floor", "Erdgeschoss"),
])
def test_location_filters_resolve_registry_names_ids_and_aliases(compact_search, field, value):
    search, _ = compact_search
    result = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": {field: value},
    })))

    assert result["status"] == "ok"
    assert len(result["results"]) == 4
    assert "ignored_filters" not in result


@pytest.mark.parametrize("scope", ["devices", "tools"])
def test_unknown_location_filters_are_reported_with_cached_results(compact_search, scope):
    search, _ = compact_search
    arguments = {
        "scope": scope, "search_queries": ["office lights"],
        "filters": {"area": "Unknown", "floor": "Missing floor", "domain": "light", "state": "off"},
    }

    result = asyncio.run(search._async_call(SimpleNamespace(tool_args=arguments)))

    assert result["status"] == "ok"
    assert len(result["results"]) == (4 if scope == "devices" else 1)
    assert result["ignored_filters"] == {"area": "Unknown", "floor": "Missing floor"}
    assert MessageHelper.compact_tool_result_value(search.name, result)["ignored_filters"] == result["ignored_filters"]
    assert arguments["filters"]["area"] == "Unknown"

    repeated = asyncio.run(search._async_call(SimpleNamespace(tool_args=arguments)))
    assert repeated["status"] == "same_as"
    assert repeated["ignored_filters"] == result["ignored_filters"]

    arguments["filters"]["area"] = "Another missing room"
    another = asyncio.run(search._async_call(SimpleNamespace(tool_args=arguments)))
    assert another["status"] == "same_as"
    assert another["ignored_filters"]["area"] == "Another missing room"
    assert result["ignored_filters"]["area"] == "Unknown"

    del arguments["filters"]["area"]
    del arguments["filters"]["floor"]
    unfiltered = asyncio.run(search._async_call(SimpleNamespace(tool_args=arguments)))
    assert unfiltered["status"] == "same_as"
    assert "ignored_filters" not in unfiltered


@pytest.mark.parametrize("filters,ignored,effective", [
    ({"area": "Kitchen", "floor": "Missing"}, {"floor": "Missing"}, {"area": "kitchen"}),
    ({"area": "Missing", "floor": "Upper"}, {"area": "Missing"}, {"floor": "upper"}),
    ({"area": "Missing", "domain": "fan"}, {"area": "Missing"}, {"domain": "fan"}),
    ({"area": "Missing", "state": "on"}, {"area": "Missing"}, {"state": {"operator": "eq", "value": "on"}}),
])
def test_ignoring_unknown_location_preserves_other_constraints(compact_search, filters, ignored, effective):
    search, _ = compact_search
    result = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": filters,
    })))

    assert result["status"] == "no_matches"
    assert result["results"] == []
    assert result["ignored_filters"] == ignored
    assert result["filters"] == effective


@pytest.mark.parametrize("field", ["area", "floor"])
def test_location_resolution_preserves_input_length_validation(compact_search, field):
    search, _ = compact_search
    result = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": {field: "x" * 257},
    })))

    assert result["status"] == "validation_error"
    assert "256" in result["error"]


def test_ambiguous_alias_is_ignored_but_real_unknown_room_is_preserved(compact_search):
    search, _ = compact_search
    areas = search_tools_module.area_registry.async_get(search.hass).async_list_areas()
    for area in areas:
        area.aliases.add("Shared alias")
    areas.append(SimpleNamespace(id="unknown_area", name="Unknown", aliases=set()))

    ambiguous = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": {"area": "Shared alias"},
    })))
    assert ambiguous["status"] == "ok"
    assert ambiguous["ignored_filters"] == {"area": "Shared alias"}

    known = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": {"area": "Unknown"},
    })))
    assert known["status"] == "no_matches"
    assert "ignored_filters" not in known


def test_unassigned_devices_remain_eligible_when_location_is_ignored(compact_search):
    search, _ = compact_search
    entry = search.hass.data[const.DOMAIN][search.entry_id]
    devices = asyncio.run(entry.vector_db_backend.async_get_lexical_objects(DeviceEmbedding, {}, "devices_agent"))
    devices[0].area_name = ""
    devices[0].floor_name = ""

    result = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_queries": ["office lights"], "filters": {"area": ""},
    })))

    assert result["status"] == "ok"
    assert result["ignored_filters"] == {"area": ""}
    assert devices[0].id in {item["entity_id"] for item in result["results"]}


@pytest.mark.parametrize("state", ["off", {"operator": "eq", "value": "off"}])
def test_native_search_accepts_location_domain_and_state_filters(compact_search, state):
    search, _ = compact_search
    call = llm.ToolInput(tool_name=search.name, tool_args={
        "scope": "devices", "search_queries": ["office lights"],
        "filters": {"area": "Office", "domain": "light", "state": state},
    })

    result = asyncio.run(search.async_call(search.hass, call, None))
    data = getattr(result, "data", result)

    assert data["status"] == "ok"
    assert len(data["results"]) == 4


def test_model_search_query_can_search_outside_user_request() -> None:
    tool = _search_tool()
    tool.set_search_context(
        latest_request="turn off the light strip",
        area="Guest Bedroom",
        floor="2nd Floor",
        candidates=[{"name": "light.strip", "friendly_name": "Light Strip"}],
    )

    queries = asyncio.run(tool._validate_queries(SimpleNamespace(
        tool_args={"search_query": "find a ventilation control tool"},
    )))

    assert queries == ["find a ventilation control tool"]


@pytest.mark.parametrize("queries,capabilities", [
    (["office", "kitchen"], [{"action": "turn_on"}]),
    (["office"], [{"action": "turn_on"}, {"action": "turn_off"}]),
    (None, [{"action": "turn_on"}, {"action": "turn_off"}]),
])
def test_capability_count_error_explains_alignment_before_retrieval(queries, capabilities):
    tool = _search_tool()
    tool.set_search_context(latest_request="turn on the lights")
    tool._iter_searchable_entries = Mock(side_effect=AssertionError("Retrieval started"))
    arguments = {"scope": "tools", "capabilities": capabilities}
    if queries is not None:
        arguments["search_queries"] = queries

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args=arguments)))

    assert result["status"] == "validation_error"
    assert f"{len(capabilities)} capabilities for {len(queries) if queries is not None else 1} search queries" in result["error"]
    assert "{}" in result["error"]
    assert "omit capabilities" in result["error"]
    tool._iter_searchable_entries.assert_not_called()


@pytest.mark.parametrize("capabilities", [None, {"action": "turn_on"}, "turn_on"])
def test_capabilities_require_a_list_before_retrieval(capabilities):
    tool = _search_tool()
    tool._iter_searchable_entries = Mock(side_effect=AssertionError("Retrieval started"))

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "scope": "tools", "search_queries": ["office"], "capabilities": capabilities,
    })))

    assert result["status"] == "validation_error"
    assert "capabilities must be a list" in result["error"]
    tool._iter_searchable_entries.assert_not_called()


def test_empty_capability_placeholder_preserves_query_alignment():
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter(())
    capabilities = [{}, {"action": "turn_off", "domain": "light"}]
    arguments = tool.parameters({
        "scope": "tools", "search_queries": ["office", "kitchen"],
        "capabilities": capabilities,
    })

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args=arguments)))

    assert result["status"] == "no_matches"
    assert tool.requested_capabilities == [{}, {"action": "turn_off", "domains": ("light",)}]


def test_user_context_is_fallback_without_model_search_query() -> None:
    tool = _search_tool()
    tool._contextual_query = "turn off a light"

    queries = asyncio.run(tool._validate_queries(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert queries == ["turn off a light"]


def test_multi_query_search_normalizes_distinct_focused_intents() -> None:
    tool = _search_tool()

    queries = tool._model_search_queries(SimpleNamespace(tool_args={
        "search_queries": [" turn on devices ", "turn off devices", "TURN ON DEVICES"],
    }))

    assert queries == ["turn on devices", "turn off devices"]


def test_multi_query_results_are_merged_fairly_without_duplicates() -> None:
    merged = RAGentSemanticSearchTool._merge_query_candidates(
        [
            [{"name": "HassTurnOn"}, {"name": "HassLightSet"}],
            [{"name": "HassTurnOff"}, {"name": "HassLightSet"}],
        ],
        3,
    )

    assert [candidate["name"] for candidate in merged] == [
        "HassTurnOn",
        "HassTurnOff",
        "HassLightSet",
    ]


def test_search_context_signature_accepts_empty_keyword_context() -> None:
    tool = _search_tool()

    tool.set_search_context()

    assert tool._contextual_query == ""
    assert tool._candidate_context == []


def test_corrective_search_uses_subentry_retrieval_limits() -> None:
    entry = SimpleNamespace(options={
        const.CONF_MIN_DEVICES_TO_EXTRACT: 7,
        const.CONF_MAX_DEVICES_TO_EXTRACT: 9,
        const.CONF_MIN_TOOLS_TO_EXTRACT: 6,
        const.CONF_MAX_TOOLS_TO_EXTRACT: 8,
    })
    subentry = SimpleNamespace(data={
        const.CONF_MIN_DEVICES_TO_EXTRACT: 2,
        const.CONF_MAX_DEVICES_TO_EXTRACT: 4,
        const.CONF_MIN_TOOLS_TO_EXTRACT: 3,
        const.CONF_MAX_TOOLS_TO_EXTRACT: 5,
    })

    assert RAGentSemanticSearchTool._get_effective_ranges(entry, subentry) == (2, 4, 3, 5)


def test_effective_ranges_clamp_minimums_to_exposure_limits() -> None:
    entry = SimpleNamespace(options={})
    subentry = SimpleNamespace(data={
        const.CONF_MIN_DEVICES_TO_EXTRACT: 8,
        const.CONF_MAX_DEVICES_TO_EXTRACT: 3,
        const.CONF_MIN_TOOLS_TO_EXTRACT: 7,
        const.CONF_MAX_TOOLS_TO_EXTRACT: 2,
    })

    assert RAGentSemanticSearchTool._get_effective_ranges(entry, subentry) == (
        3, 3, 2, 2,
    )


def test_contextual_fallback_includes_trusted_location() -> None:
    tool = _search_tool()
    tool.set_search_context(
        latest_request="turn off the lights",
        area="Kitchen",
        floor="Ground floor",
        candidates=[],
    )

    queries = asyncio.run(tool._validate_queries(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert queries == ["turn off the lights\nKitchen\nGround floor"]


def test_device_search_uses_explicit_model_query() -> None:
    tool = _search_tool()
    tool.set_search_context(latest_request="turn on the bathroom lights")

    query = tool._device_search_query(
        "lights bathroom area switch",
        "fallback query",
    )

    assert query == "lights bathroom area switch"


def test_tool_search_uses_explicit_model_query_with_capability() -> None:
    tool = _search_tool()
    tool.set_search_context(latest_request="turn off the bedroom lights")

    query = tool._tool_search_query(
        "set kitchen fan speed", [],
        {"action": "fan_set_speed", "domain": "fan"},
        fallback_query="turn off the bedroom lights",
    )

    assert query.startswith("set kitchen fan speed\n")
    assert "fan_set_speed" in query
    assert "bedroom" not in query


class _FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def async_embed_text(self, _config: dict, _query: str) -> list[float]:
        self.queries.append(_query)
        return [1.0, 0.0]


@pytest.mark.parametrize("scope", ["devices", "tools"])
@pytest.mark.parametrize("mode,calls", [("automatic", 1), ("vector", 1), ("lexical", 0)])
def test_scoped_search_retrieves_only_requested_collection(mode, calls, scope):
    from custom_components.ha_ragent.src.homeassistant.helpers.message_helper import MessageHelper

    device = Device("fan.a", "Ventilator", "Kitchen", "Ground",
                    domain=["fan"], area_aliases=["Küche"], floor_aliases=["Erdgeschoss"])
    capability = LlmTool("VendorLueften", "Lüften", parameters={
        "properties": {"domain": {"enum": ["fan"]}},
    })
    vectors = []

    async def local(_type, _options, collection):
        return [device] if collection.startswith("devices_") else [capability]

    async def vector(_type, _options, collection, embedding, _limit):
        vectors.append(embedding)
        return [ScoredResult(item, 0.9, 1) for item in await local(_type, _options, collection)]

    embedder = _FakeEmbedder()
    entry = SimpleNamespace(options={const.CONF_RETRIEVAL_METHOD: mode},
                            embedder_backend=embedder,
                            vector_db_backend=SimpleNamespace(
                                async_get_lexical_objects=local, async_retrieve_scored_objects=vector,
                            ))
    subentry = SimpleNamespace(data={}, title="Test")
    search = _search_tool()
    search.hass = Mock()
    search.hass.states.get.return_value = SimpleNamespace(
        state="on", attributes={"speed": "high"},
    )
    search._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 2, 2, 2, 2)])
    search.set_search_context(latest_request="in der Küche lüften")
    result = asyncio.run(search._async_call(SimpleNamespace(tool_args={
        "search_query": "Ventilator starten", "scope": scope,
    })))

    assert result["status"] == "ok"
    assert len(embedder.queries) == calls
    if calls:
        assert len(vectors) == 1
        assert embedder.queries == ["Ventilator starten"]
    if scope == "devices":
        assert _candidates(search, result, "device")[0]["name"] == device.id
        assert _candidates(search, result, "device")[0]["attributes"] == {"speed": "high"}
        compact = MessageHelper._compact_candidate_devices(_candidates(search, result, "device"))
        assert "area_aliases" not in compact[0]
        assert "floor_aliases" not in compact[0]
    else:
        assert _candidates(search, result, "tool")[0]["name"] == capability.name
    assert len(result["results"]) == 1


class _FakeToolVectorDatabase:
    def __init__(self, tools: list[LlmTool]) -> None:
        self.tools = tools

    async def async_retrieve_scored_objects(self, *_args) -> list[ScoredResult[LlmTool]]:
        return [
            ScoredResult(tool, 1.0 - (index * 0.1), index + 1)
            for index, tool in enumerate(self.tools)
        ]

    async def async_get_lexical_objects(self, *_args) -> list[LlmTool]:
        return self.tools


@pytest.mark.parametrize("scope", ["devices", "tools"])
@pytest.mark.parametrize("mode", [const.RETRIEVAL_METHOD_LEXICAL, const.RETRIEVAL_METHOD_VECTOR])
def test_search_mode_skips_unused_backends_and_preserves_ranking(scope, mode):
    if scope == "tools":
        semantic = LlmTool("VendorSemantic", "Control equipment")
        lexical = LlmTool("VendorTurbo", "Turbo ventilation")
    else:
        semantic = Device("fan.semantic", "Equipment", "", "", domain=["fan"])
        lexical = Device("fan.turbo", "Turbo ventilation", "", "", domain=["fan"])
    database = SimpleNamespace(
        async_retrieve_scored_objects=AsyncMock(return_value=[ScoredResult(semantic, 0.99, 1)]),
        async_get_lexical_objects=AsyncMock(return_value=[lexical]),
    )
    embedder = _FakeEmbedder()
    # Entry-level retrieval options must apply to corrective search, too.
    entry = SimpleNamespace(options={const.CONF_RETRIEVAL_METHOD: mode},
                            embedder_backend=embedder, vector_db_backend=database)
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": scope,
    })))

    is_vector = mode == const.RETRIEVAL_METHOD_VECTOR
    candidate_name = _candidates(tool, result, "device" if scope == "devices" else "tool")[0]["name"]
    assert candidate_name == (
        semantic.name if is_vector and scope == "tools"
        else semantic.id if is_vector
        else lexical.name if scope == "tools"
        else lexical.id
    )
    assert len(embedder.queries) == int(is_vector)
    assert database.async_retrieve_scored_objects.await_count == int(is_vector)
    assert database.async_get_lexical_objects.await_count == int(not is_vector)


@pytest.mark.parametrize(
    ("entry_mode", "subentry_mode", "expected_mode"),
    [
        (
            const.RETRIEVAL_METHOD_VECTOR,
            const.RETRIEVAL_METHOD_LEXICAL,
            const.RETRIEVAL_METHOD_LEXICAL,
        ),
        (
            const.RETRIEVAL_METHOD_LEXICAL,
            const.RETRIEVAL_METHOD_VECTOR,
            const.RETRIEVAL_METHOD_VECTOR,
        ),
        (const.RETRIEVAL_METHOD_VECTOR, None, const.RETRIEVAL_METHOD_VECTOR),
        (const.RETRIEVAL_METHOD_LEXICAL, None, const.RETRIEVAL_METHOD_LEXICAL),
        (const.RETRIEVAL_METHOD_VECTOR, const.RETRIEVAL_METHOD_AUTOMATIC, const.RETRIEVAL_METHOD_AUTOMATIC),
    ],
)
def test_search_mode_uses_configuration(
    entry_mode, subentry_mode, expected_mode,
):
    lexical_device = Device("fan.lexical", "Lexical fan", "", "")
    vector_device = Device("fan.vector", "Vector fan", "", "")
    database = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(return_value=[lexical_device]),
        async_retrieve_scored_objects=AsyncMock(
            return_value=[ScoredResult(vector_device, 0.99, 1)]
        ),
    )
    embedder = _FakeEmbedder()
    entry = SimpleNamespace(
        options={const.CONF_RETRIEVAL_METHOD: entry_mode},
        embedder_backend=embedder,
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(
        data={const.CONF_RETRIEVAL_METHOD: subentry_mode} if subentry_mode is not None else {},
        title="Test",
    )
    tool = _search_tool()
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter(
        [(entry, "subentry", subentry, 1, 1, 1, 1)]
    )
    tool.set_search_context(latest_request="fan")

    arguments = {"search_query": "fan", "scope": "devices"}
    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args=arguments)))

    assert result["mode"] == expected_mode
    is_vector = expected_mode == const.RETRIEVAL_METHOD_VECTOR
    if expected_mode != const.RETRIEVAL_METHOD_AUTOMATIC:
        assert _candidates(tool, result, "device")[0]["name"] == (
            vector_device.id if is_vector else lexical_device.id
        )
    assert database.async_get_lexical_objects.await_count == int(not is_vector)
    uses_vectors = expected_mode != const.RETRIEVAL_METHOD_LEXICAL
    assert database.async_retrieve_scored_objects.await_count == int(uses_vectors)
    assert len(embedder.queries) == int(uses_vectors)


@pytest.mark.parametrize("mode", ["automatic", "lexical", "vector", "guess", None])
def test_search_rejects_model_supplied_mode_before_retrieval(mode):
    tool = _search_tool()
    tool._iter_searchable_entries = Mock(side_effect=AssertionError("Retrieval started"))
    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "scope": "devices", "search_queries": ["fan"], "mode": mode,
    })))
    assert result["status"] == "validation_error"
    tool._iter_searchable_entries.assert_not_called()


def test_automatic_device_search_combines_exact_lexical_and_vector_results():
    local_device = Device("fan.kitchen", "Kitchen fan", "Kitchen", "")
    database = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(return_value=[local_device]),
        async_retrieve_scored_objects=AsyncMock(
            return_value=[ScoredResult(local_device, 0.99, 1)]
        ),
    )
    embedder = _FakeEmbedder()
    entry = SimpleNamespace(
        options={const.CONF_RETRIEVAL_METHOD: const.RETRIEVAL_METHOD_AUTOMATIC},
        embedder_backend=embedder,
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter(
        [(entry, "subentry", subentry, 1, 1, 1, 1)]
    )
    tool.set_search_context(latest_request="fan.kitchen")

    result = asyncio.run(
        tool._async_call(
            SimpleNamespace(
                tool_args={"search_query": "fan.kitchen", "scope": "devices"}
            )
        )
    )

    assert _candidates(tool, result, "device")[0]["name"] == local_device.id
    assert embedder.queries == ["fan.kitchen"]
    database.async_get_lexical_objects.assert_awaited_once()
    database.async_retrieve_scored_objects.assert_awaited_once()


@pytest.mark.parametrize("scope", ["devices", "tools"])
def test_zero_maximum_skips_the_requested_search_scope(scope):
    database = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(),
        async_retrieve_scored_objects=AsyncMock(),
    )
    entry = SimpleNamespace(
        options={
            const.CONF_MIN_DEVICES_TO_EXTRACT: 1,
            const.CONF_MAX_DEVICES_TO_EXTRACT: 0 if scope == "devices" else 1,
            const.CONF_MIN_TOOLS_TO_EXTRACT: 1,
            const.CONF_MAX_TOOLS_TO_EXTRACT: 0 if scope == "tools" else 1,
        },
        embedder_backend=SimpleNamespace(async_embed_text=AsyncMock()),
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter(
        [(
            entry,
            "subentry",
            subentry,
            1,
            0 if scope == "devices" else 1,
            1,
            0 if scope == "tools" else 1,
        )]
    )
    tool.set_search_context(latest_request="find it")

    result = asyncio.run(
        tool._async_call(
            SimpleNamespace(tool_args={"search_query": "find it", "scope": scope})
        )
    )

    assert _candidates(tool, result, "device" if scope == "devices" else "tool") == []
    database.async_get_lexical_objects.assert_not_awaited()
    database.async_retrieve_scored_objects.assert_not_awaited()


def test_search_embedding_failure_still_discovers_custom_tool():
    custom = LlmTool("VendorTurbo", "Turbo ventilation")
    database = _FakeToolVectorDatabase([custom])
    database.async_retrieve_scored_objects = AsyncMock(side_effect=AssertionError("No embedding"))
    entry = SimpleNamespace(
        options={},
        embedder_backend=SimpleNamespace(async_embed_text=AsyncMock(side_effect=RuntimeError("offline"))),
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")
    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": "tools",
    })))
    assert _candidates(tool, result, "tool")[0]["name"] == custom.name
    database.async_retrieve_scored_objects.assert_not_awaited()


def test_semantic_tool_search_prioritizes_capability_without_filtering_custom_tools() -> None:
    timer = LlmTool(name="HassTimerCancel", description="Cancel a timer",
                    metadata=ToolMetadata(canonical_action="cancel", supported_domains=("timer",)))
    set_light = LlmTool(name="HassLightSet", description="Set light brightness",
                        metadata=ToolMetadata(canonical_action="set", supported_domains=("light",)))
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off",
                       metadata=ToolMetadata(canonical_action="off", supported_domains=("light",)))
    vector_database = _FakeToolVectorDatabase([timer, set_light, turn_off])
    entry = SimpleNamespace(
        options={},
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
    tool.set_search_context(
        latest_request="turn off the kitchen light",
        candidates=[{"name": "light.kitchen", "domain": ["light"]}],
    )

    result = asyncio.run(
        tool._async_call(
            SimpleNamespace(
                tool_args={
                    "search_query": "switch off lights",
                    "capabilities": [{"action": "off", "domain": "light"}],
                    "scope": "tools",
                },
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in _candidates(tool, result, "tool")]
    assert candidate_names[0] == "HassTurnOff"
    assert set(candidate_names) == {"HassTimerCancel", "HassLightSet", "HassTurnOff"}


def test_structured_search_allows_query_only_discovery() -> None:
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter(())
    tool.set_search_context(latest_request="request")

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "search_queries": ["target operation"],
        "scope": "tools",
    })))

    assert result["status"] == "no_matches"
    assert "requested_capabilities" not in result


def test_capability_ranking_prioritizes_requested_action_without_excluding_alternatives() -> None:
    forget = LlmTool(
        name="HassForgetFact",
        description="Forget stored information",
        metadata=ToolMetadata(canonical_action="forget_fact"),
    )
    turn_on = LlmTool(
        name="HassTurnOn",
        description="Turn on a device",
        metadata=ToolMetadata(canonical_action="turn_on", supported_domains=("switch",)),
    )
    vector_database = _FakeToolVectorDatabase([forget, turn_on])
    entry = SimpleNamespace(
        options={},
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([
        (entry, "subentry", subentry, 2, 6, 1, 6),
    ])
    tool.set_search_context(candidates=[{
        "name": "switch.heater", "domain": ["switch"],
    }])

    result = asyncio.run(tool._async_call(SimpleNamespace(tool_args={
        "search_queries": ["operation target"],
        "capabilities": [{"action": "turn_on", "domain": "switch"}],
        "scope": "tools",
    })))

    assert [candidate["name"] for candidate in _candidates(tool, result, "tool")] == [
        "HassTurnOn",
        "HassForgetFact",
    ]


def test_weak_tool_result_retains_bounded_compact_candidates() -> None:
    vector_database = _FakeToolVectorDatabase([
        LlmTool(name="HassBroadcast", description="Broadcast a message"),
    ])
    entry = SimpleNamespace(
        options={},
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool._async_call(
            SimpleNamespace(tool_args={"search_query": "heater toggle", "scope": "tools"})
        )
    )

    assert [candidate["name"] for candidate in _candidates(tool, result, "tool")] == [
        "HassBroadcast"
    ]
    assert _candidates(tool, result, "device") == []
    assert result["status"] == "ok"
    assert "tool_search_confidence" not in result


def test_empty_tool_index_returns_no_tools_fallback() -> None:
    entry = SimpleNamespace(
        options={},
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=_FakeToolVectorDatabase([]),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool._async_call(
            SimpleNamespace(tool_args={"search_query": "heater", "scope": "tools"})
        )
    )

    assert _candidates(tool, result, "tool") == []
    assert result["status"] == "no_matches"


def test_no_match_does_not_reintroduce_unrelated_context() -> None:
    entry = SimpleNamespace(
        options={},
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=_FakeToolVectorDatabase([]),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = _search_tool()
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(candidates=[
        {"name": "light.first", "domain": ["light"]},
        {"name": "light.second", "domain": ["light"]},
    ])

    result = asyncio.run(tool._async_call(SimpleNamespace(
        tool_args={"search_query": "target", "scope": "devices"},
    )))

    assert result["status"] == "no_matches"
    assert result["results"] == []
