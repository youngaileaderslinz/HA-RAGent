import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from probatio import to_openapi

from custom_components.ha_ragent.src import const
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import (
    RAGentSemanticSearchTool,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


def test_semantic_search_schema_explains_parallel_structured_capabilities() -> None:
    schema = to_openapi(RAGentSemanticSearchTool.parameters)

    assert set(schema["required"]) >= {"search_queries", "capabilities"}
    assert "position" in schema["properties"]["search_queries"]["description"]
    capability_description = schema["properties"]["capabilities"]["description"]
    assert "turn_on" in capability_description
    action_schema = schema["properties"]["capabilities"]["items"]["properties"]["action"]
    assert "canonical action ID" in action_schema["description"]


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


def test_effective_ranges_do_not_rewrite_user_boundaries() -> None:
    entry = SimpleNamespace(options={})
    subentry = SimpleNamespace(data={
        const.CONF_MIN_DEVICES_TO_EXTRACT: 8,
        const.CONF_MAX_DEVICES_TO_EXTRACT: 3,
        const.CONF_MIN_TOOLS_TO_EXTRACT: 7,
        const.CONF_MAX_TOOLS_TO_EXTRACT: 2,
    })

    assert RAGentSemanticSearchTool._get_effective_ranges(entry, subentry) == (
        8, 3, 7, 2,
    )


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


@pytest.mark.parametrize("mode,calls", [("automatic", 2), ("vector", 2), ("lexical", 0)])
def test_combined_corrective_search_keeps_retrieval_independent_and_location_aliases(mode, calls):
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
    search._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 2, 2, 2, 2)])
    search.set_search_context(latest_request="in der Küche lüften")
    result = asyncio.run(search.async_call(SimpleNamespace(tool_args={
        "search_query": "Ventilator starten", "scope": "devices_and_tools",
    })))

    assert not result["error"]
    assert len(embedder.queries) == calls
    if calls:
        assert len(vectors) == 2
        assert vectors[0] is not vectors[1]
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
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")

    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": scope,
    })))

    is_vector = mode == const.RETRIEVAL_METHOD_VECTOR
    candidate_name = result[f"candidate_{scope}"][0]["name"]
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
        [(entry, "subentry", subentry, 1, 1, 1, 1)]
    )
    tool.set_search_context(latest_request="fan")

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "fan", "scope": "devices"})
        )
    )

    is_vector = expected_mode == const.RETRIEVAL_METHOD_VECTOR
    assert result["candidate_devices"][0]["name"] == (
        vector_device.id if is_vector else lexical_device.id
    )
    assert database.async_get_lexical_objects.await_count == int(not is_vector)
    assert database.async_retrieve_scored_objects.await_count == int(is_vector)
    assert len(embedder.queries) == int(is_vector)


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
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter(
        [(entry, "subentry", subentry, 1, 1, 1, 1)]
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
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
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
        tool.async_call(
            SimpleNamespace(tool_args={"search_query": "find it", "scope": scope})
        )
    )

    assert result[f"candidate_{scope}"] == []
    database.async_get_lexical_objects.assert_not_awaited()
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
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(latest_request="Turbo ventilation")
    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_query": "Turbo ventilation", "scope": "tools",
    })))
    assert result["candidate_tools"][0]["name"] == custom.name
    database.async_retrieve_scored_objects.assert_not_awaited()


def test_semantic_tool_search_filters_capability_before_vector_rank() -> None:
    timer = LlmTool(name="HassTimerCancel", description="Cancel a timer",
                    metadata=ToolMetadata(canonical_action="cancel", supported_domains=("timer",)))
    set_light = LlmTool(name="HassLightSet", description="Set light brightness",
                        metadata=ToolMetadata(canonical_action="set", supported_domains=("light",)))
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off",
                       metadata=ToolMetadata(canonical_action="off", supported_domains=("light",)))
    vector_database = _FakeToolVectorDatabase([timer, set_light, turn_off])
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
    tool.set_search_context(
        latest_request="turn off the kitchen light",
        candidates=[{"name": "light.kitchen", "domain": ["light"]}],
    )

    result = asyncio.run(
        tool.async_call(
            SimpleNamespace(
                tool_args={
                    "search_query": "switch off lights",
                    "capabilities": [{"action": "off", "domain": "light"}],
                    "scope": "tools",
                },
            )
        )
    )

    candidate_names = [candidate["name"] for candidate in result["candidate_tools"]]
    assert candidate_names[0] == "HassTurnOff"
    assert candidate_names == ["HassTurnOff"]


def test_structured_search_requires_capability_for_each_production_query() -> None:
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.set_search_context(latest_request="request")

    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_queries": ["target operation"],
        "scope": "tools",
    })))

    assert result["requested_capabilities"] == []
    assert "structured capability" in result["error"]


def test_capability_filter_removes_dangerous_memory_distractor() -> None:
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
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=vector_database,
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([
        (entry, "subentry", subentry, 2, 6, 1, 6),
    ])
    tool.set_search_context(candidates=[{
        "name": "switch.heater", "domain": ["switch"],
    }])

    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_queries": ["operation target"],
        "capabilities": [{"action": "turn_on", "domain": "switch"}],
        "scope": "tools",
    })))

    assert [candidate["name"] for candidate in result["candidate_tools"]] == [
        "HassTurnOn",
    ]


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
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
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
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 3, 3, 3, 3)])
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


def test_candidate_context_fallback_obeys_device_limit() -> None:
    entry = SimpleNamespace(
        embedder_backend=_FakeEmbedder(),
        vector_db_backend=_FakeToolVectorDatabase([]),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool._iter_searchable_entries = lambda: iter([(entry, "subentry", subentry, 1, 1, 1, 1)])
    tool.set_search_context(candidates=[
        {"name": "light.first", "domain": ["light"]},
        {"name": "light.second", "domain": ["light"]},
    ])

    result = asyncio.run(tool.async_call(SimpleNamespace(
        tool_args={"search_query": "target", "scope": "devices"},
    )))

    assert result["candidate_devices"] == [
        {"name": "light.first", "domain": ["light"]}
    ]
