import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_confidence import RetrievalConfidence
from custom_components.ha_ragent.src.const import RETRIEVAL_METHOD_AUTOMATIC
from custom_components.ha_ragent.src.homeassistant.helpers.conversation_retriever import ConversationRetriever
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index
from custom_components.ha_ragent.src.models.retrieval.confidence_assessment import ConfidenceAssessment
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool


@pytest.mark.parametrize("query,name", [("ab", "xabz"), ("x", "xyz"), ("厨房", "厨房吊灯"), ("مص", "مصباح")])
def test_short_substrings_are_retrievable_without_identity_confidence(query, name):
    tools = [LlmTool("unrelated", "zzz"), LlmTool("target", name)]
    index = lexical_index((("zzz",), (name,)))
    assert index.scores(query)[1] > index.scores(query)[0]
    assert ToolRanker.rank_tool_candidates([], tools, query, [], 1) == [tools[1]]
    assert not SourceRanker.local_candidates_confident(query, tools)


@pytest.mark.parametrize("query", ["", "ab", "a", "Alpha Beta", "alpha btea", "VendorMode", "厨房", "zzz"])
def test_indexed_field_scores_preserve_exhaustive_matches(query):
    documents = (("Alpha Beta", "area"), ("Beta Gamma", "area"), ("VendorMode", "xabz"), ("厨房吊灯",))
    indexed = lexical_index(documents).match_scores(query)
    for position, parts in enumerate(documents):
        assert indexed.get(position, (0.0, 0.0)) == SourceRanker.match_scores(query, parts)


def test_nested_schema_edits_refresh_cached_features():
    tool = LlmTool("Vendor", "", parameters={"properties": {"domain": {"enum": ["alpha"]}}})
    assert tool.canonical_supported_domains == ("alpha",)
    first = tool.canonical_schema_parts
    assert tool.canonical_schema_parts is first
    tool.parameters["properties"]["domain"]["enum"][:] = ["beta"]
    assert tool.canonical_supported_domains == ("beta",)
    assert tool.canonical_schema_parts != first
    assert "beta" in tool.canonical_schema_parts


def test_local_schema_references_and_deep_custom_fields_are_indexed_fairly():
    tool = LlmTool("Vendor", "", parameters={
        "$defs": {"profile": {"type": "string", "enum": ["hydroponics"]}},
        "properties": {
            "mode": {"$ref": "#/$defs/profile"},
            "other": {"properties": {"a": {"properties": {"b": {"properties": {"c": {"enum": ["deep-value"]}}}}}}},
        },
    })
    parts = " ".join(tool.canonical_schema_parts)
    assert "hydroponics" in parts
    assert "deep-value" in parts


def test_tool_pruning_uses_calibrated_confidence_not_raw_rrf_scale():
    confidence = ConfidenceAssessment(
        level="high",
        candidate_scores=(("first", 1.0), ("second", 0.98)),
        final_candidate_scores=(("first", 0.0492), ("second", 0.0484)),
    )
    assert RetrievalConfidence.prune_confidence_band(
        confidence, 2, absolute_floor=0.20, relative_floor=0.60, gap_threshold=0.11,
    ) == ["first", "second"]


def test_inferred_domain_does_not_exclude_custom_tool():
    custom = LlmTool("VendorDomainSetting", "Calibrate irrigation", parameters={
        "properties": {"domain": {"enum": ["configuration"]}},
    })
    assert ToolRanker.tool_capability_compatibility(
        custom, {"action": "calibrate", "domain": "light"},
    ) == 0.0


def test_fallback_search_query_contains_only_user_text_and_literal_metadata():
    assert RAGentSemanticSearchTool._build_search_query(
        latest_request="Allume la lampe",
        area="Cuisine",
        floor="Rez-de-chaussée",
    ) == "Allume la lampe\nCuisine\nRez-de-chaussée"


def test_tool_schema_is_prepared_once_for_repeated_ranking(monkeypatch):
    tool = LlmTool("Vendor", "Execute", parameters={"properties": {"mode": {"type": "string"}}})
    ToolRanker.rank_tool_candidates([], [tool], "execute", [], 1)
    monkeypatch.setattr(LlmTool, "_schema_search_parts", lambda *_args: pytest.fail("Unchanged schema was rebuilt"))
    assert ToolRanker.rank_tool_candidates([], [tool], "mode", [], 1) == [tool]


def test_request_ranking_runs_outside_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    tool = LlmTool("Vendor", "Capability")
    def rank(*_args, **_kwargs):
        assert threading.get_ident() != event_loop_thread
        return [tool]
    monkeypatch.setattr(ToolRanker, "rank_tool_candidates", rank)
    entry = SimpleNamespace(
        options={},
        vector_db_backend=SimpleNamespace(
            async_get_lexical_objects=AsyncMock(return_value=[tool]),
        ),
    )
    retriever = ConversationRetriever(
        None, entry, "entry", "subentry", SimpleNamespace(data={}),
    )
    assert asyncio.run(retriever.async_retrieve_tools(
        [],
        "capability",
        minimum=1,
        maximum=1,
        retrieval_method=RETRIEVAL_METHOD_AUTOMATIC,
    )) == [tool]
