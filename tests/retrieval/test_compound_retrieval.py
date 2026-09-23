"""Regression checks for compound requests, run inside Home Assistant."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import voluptuous as vol

from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.const import RAGENT_MAX_SEARCH_QUERIES
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.translation import RAGentTranslations


def test_action_compatibility_uses_metadata_not_word_order():
    tool = LlmTool(
        "intent__OpaqueName",
        "Control equipment",
        metadata=ToolMetadata(canonical_action="stop", supported_domains=("media_player",)),
    )
    capability = {"action": "stop", "domain": "media_player"}
    first = ToolRanker.tool_ranking_signals(
        tool, "stop the player", requested_capability=capability,
    )
    reordered = ToolRanker.tool_ranking_signals(
        tool, "player the stop", requested_capability=capability,
    )
    assert first["capability"] == reordered["capability"] == 1.5


def test_semantic_rank_is_not_added_to_tool_score():
    signals = {
        "semantic_rank": 1.0,
        "semantic_similarity": 0.8,
    }

    assert ToolRanker.tool_signal_score(signals) == pytest.approx(0.8)


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
    valid_queries = queries[:RAGENT_MAX_SEARCH_QUERIES]
    assert tool.parameters({
        "search_queries": valid_queries,
        "capabilities": [
            {"action": f"operation_{index}"}
            for index, _query in enumerate(valid_queries)
        ],
    })


def test_partial_query_coverage_requires_follow_up():
    status, fallback, _ = RAGentSemanticSearchTool._tool_search_feedback(
        True, "none", [{"name": "intent__HassTurnOn"}],
    )
    assert status == "weak_candidates"
    assert fallback
