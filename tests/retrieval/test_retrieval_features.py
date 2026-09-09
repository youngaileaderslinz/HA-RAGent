import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.homeassistant.ragent import RAGent
from custom_components.ha_ragent.src.const import CONF_RETRIEVAL_METHOD


@pytest.mark.parametrize("query,name", [("ab", "xabz"), ("x", "xyz"), ("厨房", "厨房吊灯"), ("مص", "مصباح")])
def test_short_substrings_are_retrievable_without_identity_confidence(query, name):
    tools = [LlmTool("unrelated", "zzz"), LlmTool("target", name)]
    index = lexical_index((("zzz",), (name,)))
    assert index.scores(query)[1] > index.scores(query)[0]
    assert RetrievalHelper.rank_tool_candidates([], tools, query, [], 1) == [tools[1]]
    assert not RetrievalHelper.local_candidates_confident(query, tools)


@pytest.mark.parametrize("query", ["", "ab", "a", "Alpha Beta", "alpha btea", "VendorMode", "厨房", "zzz"])
def test_indexed_field_scores_preserve_exhaustive_matches(query):
    documents = (("Alpha Beta", "area"), ("Beta Gamma", "area"), ("VendorMode", "xabz"), ("厨房吊灯",))
    indexed = lexical_index(documents).match_scores(query)
    for position, parts in enumerate(documents):
        assert indexed.get(position, (0.0, 0.0)) == RetrievalHelper._match_scores(query, parts)


def test_nested_schema_edits_refresh_cached_features():
    tool = LlmTool("Vendor", "", parameters={"properties": {"domain": {"enum": ["alpha"]}}})
    assert tool.canonical_supported_domains == ("alpha",)
    first = tool.canonical_schema_parts
    assert tool.canonical_schema_parts is first
    tool.parameters["properties"]["domain"]["enum"][:] = ["beta"]
    assert tool.canonical_supported_domains == ("beta",)
    assert tool.canonical_schema_parts != first
    assert "choices beta" in tool.canonical_schema_parts


def test_tool_schema_is_prepared_once_for_repeated_ranking(monkeypatch):
    tool = LlmTool("Vendor", "Execute", parameters={"properties": {"mode": {"type": "string"}}})
    RetrievalHelper.rank_tool_candidates([], [tool], "execute", [], 1)
    monkeypatch.setattr(LlmTool, "_schema_search_parts", lambda *_args: pytest.fail("Unchanged schema was rebuilt"))
    assert RetrievalHelper.rank_tool_candidates([], [tool], "mode", [], 1) == [tool]


def test_request_ranking_runs_outside_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    tool = LlmTool("Vendor", "Capability")
    def rank(*_args, **_kwargs):
        assert threading.get_ident() != event_loop_thread
        return [tool]
    monkeypatch.setattr(RetrievalHelper, "rank_tool_candidates", rank)
    backend = SimpleNamespace(async_get_lexical_objects=AsyncMock(return_value=[tool]))
    agent = SimpleNamespace(
        entry=SimpleNamespace(options={}, vector_db_backend=backend),
        subentry=SimpleNamespace(data={CONF_RETRIEVAL_METHOD: "lexical"}),
        subentry_id="agent",
    )
    assert asyncio.run(RAGent._async_retrieve_tools(agent, [], "capability", 1, ContinuityContext())) == [tool]
