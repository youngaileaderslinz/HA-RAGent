"""Regression checks for compound requests, run inside Home Assistant."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RAGENT_MAX_SEARCH_QUERIES,
)
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.translation import RAGentTranslations

# Home Assistant installs its schema compatibility layer during import.
import voluptuous as vol


@pytest.mark.parametrize("query,names,actions", [
    (
        "turn on the ceiling light and turn off the heater in bathroom",
        ("intent__HassTurnOn", "intent__HassTurnOff"),
        ("", ""),
    ),
    (
        "Deckenlicht einschalten und Heizung ausschalten",
        ("vendor__Alpha", "vendor__Beta"),
        ("einschalten", "ausschalten"),
    ),
    (
        "allumer le plafonnier et éteindre le chauffage",
        ("vendor__Alpha", "vendor__Beta"),
        ("allumer", "éteindre"),
    ),
])
def test_opposing_capabilities_survive_a_two_tool_budget(query, names, actions):
    # One capability is absent from vector results; the other description even
    # mentions its opposite. Neither may suppress the distinct live capability.
    first = LlmTool(names[0], "Turn on, not turn off, the equipment",
                    metadata=ToolMetadata(canonical_action=actions[0]))
    second = LlmTool(names[1], "Control equipment",
                     metadata=ToolMetadata(canonical_action=actions[1]))
    noise = [
        LlmTool("light__HassLightSet", "Set light brightness"),
        LlmTool("intent__HassUnpauseTimer", "Resume a timer"),
        LlmTool("todo__HassListRemoveItem", "Remove an item"),
    ]
    vectors = [ScoredResult(tool, 0.99 - i * 0.01, i + 1)
               for i, tool in enumerate([*noise, first])]

    ranked = RetrievalHelper.rank_tool_candidates(
        vectors, [*noise, first, second], query, [], 2,
    )

    assert {tool.name for tool in ranked} == set(names)


def test_action_phrase_keeps_word_order_and_namespace_out_of_matching():
    tool = LlmTool("intent__HassTurnOff", "Control equipment")
    assert RetrievalHelper.tool_ranking_signals(tool, "turn off heater")["lexical_action"] == 1
    assert RetrievalHelper.tool_ranking_signals(tool, "off heater turn")["lexical_action"] == 0
    assert tool.canonical_action == ""  # Retrieval does not infer authorization.


@pytest.mark.parametrize("mode", ["automatic", "vector", "lexical"])
@pytest.mark.parametrize("scope", ["devices", "tools", "devices_and_tools"])
def test_focused_queries_retrieve_each_task_without_compound_context(mode, scope):
    queries = ["turn on bathroom ceiling light", "turn off bathroom heater"]
    devices = [
        Device("light.ceiling", "Bathroom ceiling light", "Bathroom", "", domain=["light"]),
        Device("switch.heater", "Bathroom heater", "Bathroom", "", domain=["switch"]),
    ]
    capabilities = [
        LlmTool("intent__HassTurnOn", "Turn on equipment"),
        LlmTool("intent__HassTurnOff", "Turn off equipment"),
    ]
    embedded = []

    async def embed(_config, query):
        embedded.append(query)
        return [float(queries.index(query))]

    async def local(_type, _options, collection):
        return devices if collection.startswith("devices_") else capabilities

    async def vector(_type, _options, collection, embedding, _limit):
        items = await local(_type, _options, collection)
        primary = int(embedding[0])
        return [ScoredResult(items[primary], 0.99, 1),
                ScoredResult(items[1 - primary], 0.8, 2)]

    entry = SimpleNamespace(
        options={CONF_RETRIEVAL_METHOD: mode},
        embedder_backend=SimpleNamespace(async_embed_text=embed),
        vector_db_backend=SimpleNamespace(
            async_get_lexical_objects=AsyncMock(side_effect=local),
            async_retrieve_scored_objects=AsyncMock(side_effect=vector),
        ),
        llm_backend=Mock(),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.hass = Mock()
    tool.hass.states.get.return_value = None
    tool._iter_searchable_entries = lambda: iter([(entry, "test", subentry, 2, 2)])
    tool.set_search_context(latest_request="turn on the ceiling light and turn off the heater in bathroom")

    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={
        "search_queries": queries, "scope": scope,
    })))

    assert result["error"] == []
    assert embedded == ([] if mode == "lexical" else queries)
    assert result["search_queries"] == queries
    assert {item["name"] for item in result["candidate_devices"]} == (
        set() if scope == "tools" else {device.id for device in devices}
    )
    assert {item["name"] for item in result["candidate_tools"]} == (
        set() if scope == "devices" else {tool.name for tool in capabilities}
    )
    assert not entry.llm_backend.mock_calls


def test_merge_advances_past_duplicates_to_preserve_each_query():
    batches = [
        [{"name": "shared"}, {"name": "first-only"}],
        [{"name": "shared"}, {"name": "second-only"}],
    ]
    assert RAGentSemanticSearchTool._merge_query_candidates(batches, 2) == [
        {"name": "shared"}, {"name": "second-only"},
    ]


def test_query_limit_is_enforced_before_any_retrieval():
    tool = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
    tool.translations = RAGentTranslations.__new__(RAGentTranslations)
    tool.translations._data = RAGentTranslations._load("en")
    tool._iter_searchable_entries = Mock(side_effect=AssertionError("Must not retrieve"))
    queries = ["repeated"] * (RAGENT_MAX_SEARCH_QUERIES + 1)
    result = asyncio.run(tool.async_call(SimpleNamespace(tool_args={"search_queries": queries})))
    assert str(RAGENT_MAX_SEARCH_QUERIES) in result["error"]
    tool._iter_searchable_entries.assert_not_called()
    with pytest.raises(vol.Invalid):
        tool.parameters({"search_queries": queries})
    assert tool.parameters({"search_queries": queries[:RAGENT_MAX_SEARCH_QUERIES]})


def test_partial_query_coverage_requires_follow_up():
    status, fallback, _ = RAGentSemanticSearchTool._tool_search_feedback(
        True, "none", [{"name": "intent__HassTurnOn"}],
    )
    assert status == "weak_candidates"
    assert fallback
