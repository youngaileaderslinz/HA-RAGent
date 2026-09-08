import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ({}, RETRIEVAL_METHOD_AUTOMATIC),
        ({CONF_RETRIEVAL_METHOD: "  AuToMaTiC  "}, RETRIEVAL_METHOD_AUTOMATIC),
        ({CONF_RETRIEVAL_METHOD: "  LEXICAL "}, RETRIEVAL_METHOD_LEXICAL),
        ({CONF_RETRIEVAL_METHOD: "Vector"}, RETRIEVAL_METHOD_VECTOR),
        ({CONF_RETRIEVAL_METHOD: "unsupported"}, RETRIEVAL_METHOD_AUTOMATIC),
        ({CONF_RETRIEVAL_METHOD: None}, RETRIEVAL_METHOD_AUTOMATIC),
    ],
)
def test_retrieval_method_normalizes_and_defaults(configured, expected):
    assert RetrievalHelper.retrieval_method(configured) == expected


@pytest.mark.parametrize("query,name", [
    ("KÜCHENLAMPE", "Küchenlampe"),
    ("lampe de cuisine", "Lampe de cuisine"),
    ("厨房吊灯", "厨房吊灯"),
    ("مصباح المطبخ", "مصباح المطبخ"),
    ("Ku\u0308chenlampe", "Küchenlampe"),
])
def test_registered_aliases_resolve_without_language_dictionary(query, name):
    device = Device("light.a", "A", "", "", domain=["light"], aliases=[name])
    unrelated = Device("switch.b", "B", "", "", domain=["switch"])
    assert RetrievalHelper.device_resolution(query, [device, unrelated]) == ("high", (device.id,))


@pytest.mark.parametrize("query", [
    "all lights except Küchenlampe", "nicht Küchenlampe", "Küchenlampe ausschalten",
    "厨房吊灯以外的灯", "there", "dort auch", "on", "lights",
])
def test_raw_commands_and_group_words_never_resolve_a_target_by_themselves(query):
    device = Device("light.a", "Küchenlampe", "Kitchen", "", aliases=["厨房吊灯"])
    assert RetrievalHelper.device_resolution(query, [device])[0] == "weak"
    assert not RetrievalHelper.local_candidates_confident(query, [device])


def test_unknown_tool_compatibility_is_neutral_even_with_an_english_name():
    tool = LlmTool("HassLock", "Lock a door", parameters={})
    signals = RetrievalHelper.tool_ranking_signals(tool, "Licht einschalten", [{"domain": ["light"]}])
    assert signals["domain"] == 0
    assert tool.canonical_action == ""
    assert tool.family == ""


def test_declared_capabilities_round_trip_without_name_inference():
    metadata = ToolMetadata(canonical_action="vendor_mode", supported_domains=("fan",))
    tool = LlmTool("BeliebigerName", "Ventilation", metadata=metadata)
    restored = LlmTool.from_dict(tool.to_dict())
    assert restored.canonical_action == "vendor_mode"
    assert restored.canonical_supported_domains == ("fan",)
    assert RetrievalHelper.tool_ranking_signals(restored, "", [{"domain": ["fan"]}])["domain"] > 0
    assert RetrievalHelper.tool_ranking_signals(restored, "", [{"domain": ["light"]}])["domain"] < 0


def test_area_and_floor_aliases_round_trip_and_are_searchable():
    device = Device("light.a", "Lamp", "Kitchen", "Ground", domain=["light"],
                    area_aliases=["Küche"], floor_aliases=["Erdgeschoss"])
    restored = Device.from_dict(device.to_dict())
    assert "Küche" in RetrievalHelper._candidate_location_values(restored)
    assert "Erdgeschoss" in restored.to_embedding_text()
    legacy = Device.from_dict({"device_id": "light.old"})
    assert legacy.area_aliases is None


@pytest.mark.parametrize("query,expected", [
    ("Küchenlampe", 0), ("Küchenlmpe", 0), ("厨房吊灯", 1), ("مصباح", 2),
])
def test_local_lexical_index_recovers_names_and_typos(query, expected):
    index = lexical_index((("Küchenlampe",), ("厨房吊灯",), ("مصباح المطبخ",)))
    scores = index.scores(query)
    assert scores.index(max(scores)) == expected
    assert max(scores) > 0


def test_lexical_statistics_and_cache_follow_the_metadata_snapshot():
    documents = (("device kitchen lamp",), ("device bedroom lamp",), ("device hallway fan",))
    index = lexical_index(documents)
    assert index.idf["w:device"] < index.idf["w:kitchen"]
    assert lexical_index(documents) is index
    assert lexical_index((("renamed",),)) is not index
    assert index.scores("unrelatedxyz") == [0, 0, 0]


def test_vector_order_does_not_rebuild_local_index():
    tools = [LlmTool("Alpha", "First device"), LlmTool("Beta", "Second device")]
    lexical_index.cache_clear()
    for order in (tools, tools[::-1]):
        RetrievalHelper.rank_scored_candidates(
            [ScoredResult(tool, 0.9, i + 1) for i, tool in enumerate(order)],
            tools, "device", lambda tool: tool.name, lambda tool: (tool.description,), 2,
        )
    assert lexical_index.cache_info().misses == 1


def test_cache_keys_preserve_negation_order_and_distinct_numeric_values():
    signature = RetrievalHelper.canonical_search_signature
    assert signature("  KÜCHE  AUS ") == signature("Küche aus")
    assert signature("Ku\u0308che aus") == signature("Küche aus")
    assert signature("kitchen on bedroom off") != signature("kitchen off bedroom on")
    assert signature("Küche aus") != signature("Küche nicht aus")
    assert signature("set 0.5") != signature("set 05")


def test_recent_unfinished_conversation_is_retained_without_embeddings():
    pending = TurnContext(key="pending", text="Licht einschalten", created_at=880.0)
    selected = RetrievalHelper.select_history_contexts([pending], {}, [], now=1000.0)
    continuity = RetrievalHelper.build_continuity_context(selected)
    assert continuity.selected_turn_keys == {"pending"}
    assert continuity.target_groups == []
    assert RetrievalHelper.select_history_contexts([pending], {}, [], now=1500.0) == []
    assert RetrievalHelper.select_history_contexts([], {}, [], now=1000.0) == []


@pytest.mark.parametrize("mode,query,embeddings,vectors,lexical", [
    ("automatic", "Küchenlampe", 0, 0, 1),
    ("automatic", "schalte sie ein", 1, 1, 1),
    ("lexical", "schalte sie ein", 0, 0, 1),
    ("vector", "Küchenlampe", 1, 1, 0),
])
def test_modes_obey_embedding_budget(mode, query, embeddings, vectors, lexical):
    device = Device("light.a", "Küchenlampe", "", "")
    embed = AsyncMock(return_value=[1.0, 0.0])
    backend = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(return_value=[device]),
        async_retrieve_scored_objects=AsyncMock(return_value=[ScoredResult(device, 1, 1)]),
    )
    asyncio.run(RetrievalHelper.async_retrieve_sources(
        backend, Device, {CONF_RETRIEVAL_METHOD: mode}, "devices", QueryEmbedding(embed), 4, query,
    ))
    assert embed.await_count == embeddings
    assert backend.async_retrieve_scored_objects.await_count == vectors
    assert backend.async_get_lexical_objects.await_count == lexical


@pytest.mark.parametrize(
    ("mode", "lexical_error", "vector_error", "expected_lexical"),
    [
        (RETRIEVAL_METHOD_LEXICAL, RuntimeError("lexical offline"), None, []),
        (RETRIEVAL_METHOD_VECTOR, None, RuntimeError("vector offline"), []),
        (RETRIEVAL_METHOD_AUTOMATIC, None, RuntimeError("vector offline"), ["local"]),
    ],
)
def test_retrieval_modes_isolate_backends_and_fall_back(
    mode, lexical_error, vector_error, expected_lexical,
):
    lexical_result = AsyncMock(
        side_effect=lexical_error,
        return_value=["local"],
    )
    vector_result = AsyncMock(
        side_effect=vector_error,
        return_value=[ScoredResult("semantic", 1.0, 1)],
    )
    backend = SimpleNamespace(
        async_get_lexical_objects=lexical_result,
        async_retrieve_scored_objects=vector_result,
    )
    embedding = AsyncMock(return_value=[1.0, 0.0])

    vector, lexical = asyncio.run(
        RetrievalHelper.async_retrieve_sources(
            backend,
            Device,
            {CONF_RETRIEVAL_METHOD: mode},
            "devices",
            QueryEmbedding(embedding),
            2,
            "ambiguous request",
        )
    )

    assert lexical == expected_lexical
    if vector_error or mode == RETRIEVAL_METHOD_LEXICAL:
        assert vector == []
    assert lexical_result.await_count == int(mode != RETRIEVAL_METHOD_VECTOR)
    assert vector_result.await_count == int(mode != RETRIEVAL_METHOD_LEXICAL)
    assert embedding.await_count == int(mode != RETRIEVAL_METHOD_LEXICAL)


def test_automatic_mode_uses_vector_when_local_identity_is_ambiguous():
    duplicate_devices = [
        Device("light.a", "Lamp", "", ""),
        Device("light.b", "Lamp", "", ""),
    ]
    embedding = AsyncMock(return_value=[1.0, 0.0])
    vector_result = AsyncMock(
        return_value=[ScoredResult(duplicate_devices[1], 0.9, 1)]
    )
    backend = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(return_value=duplicate_devices),
        async_retrieve_scored_objects=vector_result,
    )

    vector, lexical = asyncio.run(
        RetrievalHelper.async_retrieve_sources(
            backend,
            Device,
            {CONF_RETRIEVAL_METHOD: RETRIEVAL_METHOD_AUTOMATIC},
            "devices",
            QueryEmbedding(embedding),
            2,
            "Lamp",
        )
    )

    assert vector[0].item is duplicate_devices[1]
    assert lexical == duplicate_devices
    embedding.assert_awaited_once()
    vector_result.assert_awaited_once()


def test_devices_tools_and_memory_share_one_concurrent_embedding():
    async def run():
        embed = AsyncMock(return_value=[1.0, 0.0])
        query = QueryEmbedding(embed)
        backend = SimpleNamespace(async_get_lexical_objects=AsyncMock(return_value=[]),
                                 async_retrieve_scored_objects=AsyncMock(return_value=[]))
        await asyncio.gather(
            RetrievalHelper.async_retrieve_sources(backend, Device, {}, "devices", query, 4, "dort"),
            RetrievalHelper.async_retrieve_sources(backend, LlmTool, {}, "tools", query, 4, "dort"),
            query.get(),  # The memory consumer uses the same vector.
        )
        assert embed.await_count == 1
        assert backend.async_retrieve_scored_objects.await_count == 2
    asyncio.run(run())


def test_failed_shared_embedding_falls_back_locally_without_retrying():
    async def run():
        embed = AsyncMock(side_effect=RuntimeError("offline"))
        query = QueryEmbedding(embed)
        items = [LlmTool("VendorExecute", "Ventilation")]
        backend = SimpleNamespace(async_get_lexical_objects=AsyncMock(return_value=items),
                                 async_retrieve_scored_objects=AsyncMock())
        for collection in ("devices", "tools"):
            assert await RetrievalHelper.async_retrieve_sources(
                backend, LlmTool, {}, collection, query, 4, "lüften",
            ) == ([], items)
        assert embed.await_count == 1
        backend.async_retrieve_scored_objects.assert_not_awaited()
    asyncio.run(run())


def test_duplicate_aliases_and_zero_limit_do_not_trigger_false_confidence():
    items = [Device("light.a", "Lampe", "", ""), Device("light.b", "Lampe", "", "")]
    assert not RetrievalHelper.local_candidates_confident("Lampe", items)
    embed = AsyncMock()
    backend = SimpleNamespace(async_get_lexical_objects=AsyncMock(), async_retrieve_scored_objects=AsyncMock())
    assert asyncio.run(RetrievalHelper.async_retrieve_sources(
        backend, Device, {}, "devices", QueryEmbedding(embed), 0, "Lampe",
    )) == ([], [])
    embed.assert_not_awaited()
    backend.async_get_lexical_objects.assert_not_awaited()
