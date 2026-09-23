import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant.helpers import device_ranker, tool_ranker, source_ranker
from custom_components.ha_ragent.src.homeassistant.helpers.conversation_retriever import ConversationRetriever
from custom_components.ha_ragent.src.homeassistant.ragent import RAGent
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext


@pytest.mark.parametrize("workflow", ["initial", "corrective"])
@pytest.mark.parametrize("scope", ["devices", "tools"])
@pytest.mark.parametrize("mode", [
    RETRIEVAL_METHOD_AUTOMATIC, RETRIEVAL_METHOD_LEXICAL, RETRIEVAL_METHOD_VECTOR,
])
@pytest.mark.parametrize("source_state", ["results", "empty", "failure"])
def test_retrieval_mode_controls_sources_and_final_order(workflow, scope, mode, source_state, monkeypatch):
    query = "Turbo ventilation"
    if scope == "devices":
        vector_winner = Device("fan.vector", "Opaque appliance", "", "")
        lexical_winner = Device("fan.lexical", query, "", "")
    else:
        vector_winner = LlmTool("VendorVector", "Opaque appliance")
        lexical_winner = LlmTool("VendorLexical", query)
    vector_results = [ScoredResult(vector_winner, 0.99, 1), ScoredResult(lexical_winner, 0.4, 2)]
    database = SimpleNamespace(
        async_get_lexical_objects=AsyncMock(
            return_value=[] if source_state == "empty" else [lexical_winner],
        ),
        async_retrieve_scored_objects=AsyncMock(
            return_value=[] if source_state == "empty" else vector_results,
            side_effect=RuntimeError("vector backend unavailable") if source_state == "failure" else None,
        ),
    )
    embed = AsyncMock(return_value=[1.0, 0.0])
    entry = SimpleNamespace(
        options={CONF_RETRIEVAL_METHOD: mode},
        vector_db_backend=database,
        embedder_backend=SimpleNamespace(async_embed_text=embed),
    )
    subentry = SimpleNamespace(data={}, title="Test")
    if mode == RETRIEVAL_METHOD_VECTOR:
        def forbidden_lexical_scoring(*_args, **_kwargs):
            pytest.fail("Vector mode evaluated lexical evidence")
        monkeypatch.setattr(device_ranker, "lexical_index", forbidden_lexical_scoring)
        monkeypatch.setattr(tool_ranker, "lexical_index", forbidden_lexical_scoring)
        monkeypatch.setattr(source_ranker, "lexical_index", forbidden_lexical_scoring)

    if workflow == "initial":
        retriever = ConversationRetriever(None, entry, "entry", "agent", subentry)
        embedding = QueryEmbedding(lambda: embed({}, query))
        if scope == "devices":
            result = asyncio.run(retriever.async_retrieve_devices(
                embedding, query, minimum=1, maximum=1,
                continuity=ContinuityContext(), retrieval_method=mode,
            ))
            names = [device.id for device in result]
        else:
            result = asyncio.run(retriever.async_retrieve_tools(
                embedding,
                query,
                minimum=1,
                maximum=1,
                retrieval_method=mode,
            ))
            names = [tool.name for tool in result]
    else:
        search = RAGentSemanticSearchTool.__new__(RAGentSemanticSearchTool)
        search.entry_id = "entry"
        search.hass = Mock()
        search.hass.states.get.return_value = None
        search._iter_searchable_entries = lambda: iter([(entry, "agent", subentry, 1, 1, 1, 1)])
        search.set_search_context(latest_request=query)
        result = asyncio.run(search.async_call(SimpleNamespace(tool_args={
            "search_queries": [query], "scope": scope,
        })))
        assert not result["error"]
        names = [candidate["name"] for candidate in result[f"candidate_{scope}"]]

    if source_state == "empty" or (source_state == "failure" and mode == RETRIEVAL_METHOD_VECTOR):
        assert names == []
    else:
        vector_wins = mode == RETRIEVAL_METHOD_VECTOR or (
            mode == RETRIEVAL_METHOD_AUTOMATIC and scope == "tools" and source_state == "results"
        )
        winner = vector_winner if vector_wins else lexical_winner
        assert names == [winner.id if scope == "devices" else winner.name]
    if mode == RETRIEVAL_METHOD_VECTOR:
        database.async_get_lexical_objects.assert_not_awaited()
    else:
        database.async_get_lexical_objects.assert_awaited_once()
    if mode == RETRIEVAL_METHOD_LEXICAL:
        database.async_retrieve_scored_objects.assert_not_awaited()
        embed.assert_not_awaited()
    else:
        database.async_retrieve_scored_objects.assert_awaited_once()
        embed.assert_awaited_once()


def test_lexical_continuity_keeps_recent_context_without_embeddings():
    context = TurnContext(key="recent", text="previous request", entities=("fan.previous",))
    manager = Mock()
    manager.structured_turn_contexts.return_value = [context]
    embed = AsyncMock(side_effect=AssertionError("Lexical history used an embedding"))
    agent = SimpleNamespace(
        runtime_options={CONF_RETRIEVAL_METHOD: RETRIEVAL_METHOD_LEXICAL},
        _async_embed_retrieval_text=embed,
    )
    continuity = asyncio.run(RAGent._async_build_continuity_context(
        agent, manager, Mock(), QueryEmbedding(lambda: embed("current request")),
    ))
    assert context.key in continuity.selected_turn_keys
    assert "fan.previous" in continuity.entities
    embed.assert_not_awaited()
