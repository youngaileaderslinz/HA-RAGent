import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src import const
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

from custom_components.ha_ragent.src.homeassistant.tools.search_tools import (
    RAGentSemanticSearchTool,
)


def test_model_search_query_can_search_outside_user_request() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(
        latest_request="turn off the light strip",
        area="Guest Bedroom",
        floor="2nd Floor",
        candidates=[{"name": "light.strip", "friendly_name": "Light Strip"}],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": "find a ventilation control tool"},
    )))

    assert query.startswith("Search intent: find a ventilation control tool")
    assert "Default area when the request has no explicit location: Guest Bedroom" in query
    assert "Default floor when the request has no explicit location: 2nd Floor" in query
    assert "Current candidate: light.strip | Light Strip" in query


def test_user_context_is_fallback_without_model_search_query() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._contextual_query = "Current request: turn off a light"

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert query == "Current request: turn off a light"


def test_multi_query_search_normalizes_distinct_focused_intents() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)

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
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)

    tool.set_search_context()

    assert tool._contextual_query == ""
    assert tool._candidate_context == []


def test_contextual_fallback_includes_trusted_location() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(
        latest_request="turn off the lights",
        area="Kitchen",
        floor="Ground floor",
        candidates=[],
    )

    query = asyncio.run(tool._validate_query(SimpleNamespace(
        tool_args={"search_query": ""},
    )))

    assert "Default area when the request has no explicit location: Kitchen" in query
    assert "Default floor when the request has no explicit location: Ground floor" in query


def test_device_search_ignores_model_guessed_concepts() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(latest_request="turn on the bathroom lights")

    query = tool._device_search_query(
        "lights bathroom area switch",
        "fallback query",
    )

    assert query == "turn on the bathroom lights"


def test_refresh_does_not_restore_pruned_candidates() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    candidates = [
        {"name": "light.kitchen"},
        {"name": "light.dining"},
    ]
    tool.set_search_context(candidates=candidates)

    tool.prune_candidates({"LIGHT.KITCHEN"})
    tool.refresh_candidates(candidates)

    assert tool._candidate_context == [{"name": "light.dining"}]


class _FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def async_embed_text(self, _config: dict, _query: str) -> list[float]:
        self.queries.append(_query)
        return [1.0, 0.0]


@pytest.mark.parametrize("mode,calls", [("automatic", 1), ("vector", 1), ("lexical", 0)])
def test_combined_corrective_search_shares_embedding_and_keeps_location_aliases(mode, calls):
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
    search = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    search.hass = Mock()
    search.hass.states.get.return_value = None
    search._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 2, 2)])
    search.set_search_context(latest_request="in der Küche lüften")
    result = asyncio.run(search.async_call(SimpleNamespace(tool_args={
        "search_query": "Ventilator starten", "scope": "devices_and_tools",
    })))

    assert not result["error"]
    assert len(embedder.queries) == calls
    if calls:
        assert len(vectors) == 2
        assert vectors[0] is vectors[1]
    assert result["candidate_devices"][0]["name"] == device.id
    assert result["candidate_tools"][0]["name"] == capability.name
    compact = MessageHelper._compact_candidate_devices(result["candidate_devices"])
    assert compact[0]["area_aliases"] == ["Küche"]
    assert compact[0]["floor_aliases"] == ["Erdgeschoss"]


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
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")

    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": scope,
    })))

    is_vector = mode == const.RETRIEVAL_METHOD_VECTOR
    expected = semantic if is_vector else lexical
    assert result[f"candidate_{scope}"][0]["name"] == (expected.name if scope == "tools" else expected.id)
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
    ],
)
def test_subentry_retrieval_method_overrides_entry(
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
        data={const.CONF_RETRIEVAL_METHOD: subentry_mode},
        title="Test",
    )
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter(
        [(entry, "subentry", subentry, 1, 1)]
    )
    tool.set_search_context(latest_request="fan")

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "fan", "scope": "devices"})
        )
    )

    is_vector = expected_mode == const.RETRIEVAL_METHOD_VECTOR
    expected_device = vector_device if is_vector else lexical_device
    assert result["candidate_devices"][0]["name"] == expected_device.id
    assert database.async_get_lexical_objects.await_count == int(not is_vector)
    assert database.async_retrieve_scored_objects.await_count == int(is_vector)
    assert len(embedder.queries) == int(is_vector)


def test_automatic_device_search_skips_embedding_for_unique_exact_identity():
    local_device = Device("fan.kitchen", "Kitchen fan", "Kitchen", "")
    database = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(return_value=[local_device]),
        async_retrieve_scored_objects=AsyncMock(),
    )
    embedder = _FakeEmbedder()
    entry = SimpleNamespace(
        options={const.CONF_RETRIEVAL_METHOD: const.RETRIEVAL_METHOD_AUTOMATIC},
        embedder_backend=embedder,
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter(
        [(entry, "subentry", subentry, 1, 1)]
    )
    tool.set_search_context(latest_request="fan.kitchen")

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={"search_query": "fan.kitchen", "scope": "devices"}
            )
        )
    )

    assert result["candidate_devices"][0]["name"] == local_device.id
    assert embedder.queries == []
    database.async_retrieve_scored_objects.assert_not_awaited()


def test_search_embedding_failure_still_discovers_custom_tool():
    custom = LlmTool("VendorTurbo", "Turbo ventilation")
    database = _FakeToolVectorDatabase([custom])
    database.async_retrieve_scored_objects = AsyncMock(side_effect=AssertionError("No embedding"))
    entry = SimpleNamespace(
        embedder_backend=SimpleNamespace(async_embed_text=AsyncMock(side_effect=RuntimeError("offline"))),
        vector_db_backend=database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")
    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": "tools",
    })))
    assert result["candidate_tools"][0]["name"] == custom.name
    database.async_retrieve_scored_objects.assert_not_awaited()


def test_semantic_tool_search_filters_capability_before_vector_rank() -> None:
    timer = LlmTool(name="HassTimerCancel", description="Cancel a timer")
    set_light = LlmTool(name="HassLightSet", description="Set light brightness")
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off")
    vector_database = _FakeToolVectorDatabase([timer, set_light, turn_off])
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn off the kitchen light",
        candidates=[{"name": "light.kitchen", "domain": ["light"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={"search_query": "switch off lights", "scope": "tools"},
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in result["candidate_tools"]]
    assert candidate_names[0] == "HassTurnOff"
    assert set(candidate_names) == {"HassTurnOff", "HassLightSet", "HassTimerCancel"}


def test_tool_search_uses_trusted_action_and_resolved_switch_domain() -> None:
    light_set = LlmTool(name="HassLightSet", description="Set light brightness")
    broadcast = LlmTool(name="HassBroadcast", description="Broadcast a message")
    turn_on = LlmTool(name="HassTurnOn", description="Turn a device on")
    embedder = _FakeEmbedder()
    vector_database = _FakeToolVectorDatabase([light_set, broadcast, turn_on])
    entry = SimpleNamespace(embedder_backend=embedder, vector_db_backend=vector_database)
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{
            "name": "switch.bathroom_heater",
            "friendly_name": "Bathroom heater",
            "domain": ["switch"],
        }],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={
                    "search_query": "lights bathroom area switch toggle",
                    "scope": "tools",
                },
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in result["candidate_tools"]]
    assert candidate_names[0] == "HassTurnOn"
    assert set(candidate_names) == {"HassTurnOn", "HassLightSet", "HassBroadcast"}
    assert result["candidate_tools"][0]["canonical_action"] == ""
    assert result["candidate_tools"][0]["ranking_signals"]["lexical_exact"] > 0
    assert result["tool_search_confidence"] != "high"
    assert "canonical action:" not in result["tool_search_query"]
    assert "supported domains:" not in result["tool_search_query"]
    assert "Search intent: lights bathroom area switch toggle" in result["tool_search_query"]
    assert embedder.queries == [result["tool_search_query"]]


def test_weak_tool_result_returns_explicit_fallback_signal() -> None:
    vector_database = _FakeToolVectorDatabase([
        LlmTool(name="HassBroadcast", description="Broadcast a message"),
    ])
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "heater toggle", "scope": "tools"})
        )
    )

    assert [candidate["name"] for candidate in result["candidate_tools"]] == [
        "HassBroadcast"
    ]
    assert result["candidate_devices"] == []
    assert result["tool_search_status"] == "weak_candidates"
    assert result["tool_search_confidence"] == "low"
    assert result["fallback_required"] is True
    assert "do not invent a tool" in result["tool_search_message"]


def test_empty_tool_index_returns_no_tools_fallback() -> None:
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=_FakeToolVectorDatabase([]),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(
        latest_request="turn on the bathroom heater",
        candidates=[{"name": "switch.bathroom_heater", "domain": ["switch"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "heater", "scope": "tools"})
        )
    )

    assert result["candidate_tools"] == []
    assert result["tool_search_status"] == "no_tools_found"
    assert result["tool_search_confidence"] == "none"
    assert result["fallback_required"] is True


@pytest.mark.parametrize("search_query", ["", "set light color brightness sleep mode"])
def test_compound_search_exposes_light_settings_and_custom_mode(search_query: str) -> None:
    request = (
        "turn on light strip and set the color to red and brightness to 40% "
        "and also enable sleep mode"
    )
    powers = [LlmTool(f"Power{index}TurnOn", "Turn on a light") for index in range(5)]
    light_set = LlmTool("HassLightSet", "Set light brightness and color")
    custom = LlmTool("VendorExecute", "Run a user program", parameters={
        "properties": {"mode": {"oneOf": [{"const": "sleep"}, {"const": "turbo"}]}}
    })
    vector_database = _FakeToolVectorDatabase([*powers, light_set, custom])
    entry = SimpleNamespace(embedder_backend=_FakeEmbedder(), vector_db_backend=vector_database)
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3)])
    tool.set_search_context(latest_request=request, candidates=[{"domain": ["light"]}])

    result = asyncio.run(tool.async_call(SimpleNamespace(
        tool_args={"search_query": search_query, "scope": "tools"},
    )))

    candidates = result["candidate_tools"]
    assert len(candidates) == 3
    assert {light_set.name, custom.name}.issubset({candidate["name"] for candidate in candidates})
    assert next(candidate for candidate in candidates if candidate["name"] == custom.name)[
        "parameters"
    ] is custom.parameters
    assert entry.embedder_backend.queries[0].startswith(request)
    assert result["error"] == []


def test_corrective_search_can_retrieve_a_custom_capability_after_power_action() -> None:
    custom = LlmTool("VendorExecute", "Apply a user profile", parameters={
        "properties": {"profile": {"enum": ["moonlight", "daylight"]}}
    })
    powers = [LlmTool(f"Power{index}TurnOn", "Turn on a light") for index in range(5)]
    entry = SimpleNamespace(embedder_backend=_FakeEmbedder(),
                            vector_db_backend=_FakeToolVectorDatabase([*powers, custom]))
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 2)])
    tool.set_search_context(latest_request="turn on the light", candidates=[{"domain": ["light"]}])

    result = asyncio.run(tool.async_call(SimpleNamespace(
        tool_args={"search_query": "moonlight profile", "scope": "tools"},
    )))

    assert custom.name in {candidate["name"] for candidate in result["candidate_tools"]}
    assert "moonlight profile" in entry.embedder_backend.queries[0]
