import pytest

from dataclasses import dataclass

from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.target_group import TargetGroup
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext


def test_unrelated_history_is_not_embedded() -> None:
    text = RetrievalHelper.build_retrieval_text("turn on the kitchen lights")

    assert text == "turn on the kitchen lights"


@dataclass
class Candidate:
    name: str


def test_exact_match_can_recover_candidate_outside_vector_results() -> None:
    candidates = [
        Candidate("Bedroom lamp"),
        Candidate("Kitchen ceiling light"),
        Candidate("Patio light"),
    ]

    result = RetrievalHelper.rank_scored_candidates(
        [
            ScoredResult(candidates[0], 0.9, 1),
            ScoredResult(candidates[2], 0.8, 2),
        ],
        candidates,
        "turn on the kitchen ceiling light",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        2,
    )

    assert result[0] == candidates[1]
    assert len(result) == 2


def test_fuzzy_match_is_fused_with_vector_rank() -> None:
    candidates = [Candidate("Bedroom lamp"), Candidate("Kitchen ceiling light")]

    result = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[0], 0.05, 1)],
        candidates,
        "kithen ceiling light",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        1,
    )

    assert result == [candidates[1]]


def test_turn_on_query_selects_turn_on_tool() -> None:
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off")
    turn_on = LlmTool(name="HassTurnOn", description="Turn a device on")

    result = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(turn_off, 0.9, 1), ScoredResult(turn_on, 0.8, 2)],
        [turn_off, turn_on],
        "turn on the kitchen light",
        lambda tool: tool.name,
        lambda tool: tool.canonical_search_parts,
        1,
    )

    assert result == [turn_on]


def test_camel_case_tool_name_provides_action_keywords() -> None:
    tool = LlmTool(name="HassTurnOff", description="")

    assert tool.canonical_name_parts == ("hass", "turn", "off")
    assert tool.canonical_action_keywords == ("turn", "off")
    assert tool.canonical_action == ""
    assert "switch off" not in tool.canonical_search_parts
    assert "turn off" in tool.canonical_search_parts
    assert "action keywords: turn off" in tool.to_embedding_text()
    assert "action aliases:" not in tool.to_embedding_text()


def test_action_ranking_prefers_match_without_erasing_alternatives() -> None:
    turn_off = LlmTool(name="HassTurnOff", description="Turn a device off")
    turn_on = LlmTool(name="HassTurnOn", description="Turn a device on")
    set_light = LlmTool(name="HassLightSet", description="Set brightness and color")
    timer = LlmTool(name="HassTimerCancel", description="Cancel a timer")

    result = RetrievalHelper.rank_tools_for_query(
        [timer, set_light, turn_on, turn_off],
        "switch off the kitchen lights",
    )

    assert result[0] == turn_off
    assert set(tool.name for tool in result) == {
        "HassTimerCancel",
        "HassLightSet",
        "HassTurnOn",
        "HassTurnOff",
    }


def test_misspelled_action_uses_fuzzy_recall_without_hard_filtering() -> None:
    broadcast = LlmTool(name="HassBroadcast", description="Broadcast a message")
    turn_on = LlmTool(name="HassTurnOn", description="Turn on a device")
    switch = Device(
        id="switch.bathroom_heater",
        friendly_name="Bathroom heater",
        area_name="Bathroom",
        floor_name="Ground floor",
        domain=["switch"],
    )

    result = RetrievalHelper.rank_tool_candidates(
        [ScoredResult(broadcast, 0.9, 1)],
        [broadcast, turn_on],
        "trun on the bathrom heater",
        [switch],
        2,
    )

    assert result == [turn_on, broadcast]


def test_domain_signal_ranks_matching_tool_first() -> None:
    light_off = LlmTool(
        name="LightTurnOff",
        description="Turn off a light",
        parameters={"properties": {"domain": {"enum": ["light"]}}},
    )
    switch_off = LlmTool(
        name="SwitchTurnOff",
        description="Turn off a switch",
        parameters={"properties": {"domain": {"enum": ["switch"]}}},
    )

    result = RetrievalHelper.rank_tools_for_query(
        [switch_off, light_off],
        "power off the lights",
    )

    assert result == [light_off, switch_off]


def test_synonyms_are_not_assumed_equivalent_for_cache_keys() -> None:
    first = RetrievalHelper.canonical_search_signature("switch off kitchen lights")
    second = RetrievalHelper.canonical_search_signature("power off the kitchen light")

    assert first != second


def test_different_capabilities_have_distinct_cache_keys() -> None:
    first = RetrievalHelper.canonical_search_signature("heater bathroom switch toggle")
    second = RetrievalHelper.canonical_search_signature("heater bathroom on/off")

    assert first != second


def test_tool_search_query_uses_action_and_resolved_device_domain() -> None:
    heater = Device(
        id="switch.bathroom_heater",
        friendly_name="Bathroom heater",
        area_name="Bathroom",
        floor_name="Ground floor",
        domain=["switch"],
    )

    query = RetrievalHelper.build_tool_search_query(
        "turn on the bathroom heater",
        "lights bathroom area switch",
        [heater],
    )

    assert "canonical action:" not in query
    assert "switch on" not in query
    assert "supported domains:" not in query
    assert query.startswith("turn on the bathroom heater\n")
    assert "Search intent: lights bathroom area switch" in query


def test_action_aliases_do_not_add_false_switch_domain() -> None:
    light = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="Ground floor",
        domain=["light"],
    )
    light_tool = LlmTool(
        name="LightTurnOn",
        description="Turn on a light",
        parameters={"properties": {"domain": {"enum": ["light"]}}},
    )
    switch_tool = LlmTool(
        name="SwitchTurnOn",
        description="Turn on a switch",
        parameters={"properties": {"domain": {"enum": ["switch"]}}},
    )
    query = RetrievalHelper.build_tool_search_query("turn on lights", "", [light])

    assert RetrievalHelper.rank_tools_for_query(
        [switch_tool, light_tool],
        query,
        [light],
    ) == [light_tool, switch_tool]


@pytest.mark.parametrize("query", [
    "do it again", "turn it on there", "same room", "in the bathroom",
    "dort auch", "schalte sie aus", "all except the kitchen lamp",
    "turn on bedroom light", "打开它",
])
def test_history_is_separate_data_and_never_rewrites_current_request(query):
    group = TargetGroup(entities=("light.kitchen",), areas=("Kitchen",),
                        action="HassTurnOff", tool="HassTurnOff")
    continuity = ContinuityContext(target_groups=[(group, 0.9)])
    groups = RetrievalHelper.continuity_groups(continuity)

    assert groups[0]["entities"] == ["light.kitchen"]
    assert groups[0]["action"] == "HassTurnOff"
    assert RetrievalHelper.build_retrieval_text(query) == query
    assert RetrievalHelper.build_tool_search_query(query, "", []) == query
    assert not RetrievalHelper.target_is_confident(
        query, [Device("light.kitchen", "Kitchen lamp", "Kitchen", "")], continuity,
    )


def test_continuity_groups_are_bounded_and_empty_without_successful_targets():
    assert RetrievalHelper.continuity_groups(ContinuityContext()) == []
    groups = [(TargetGroup(entities=tuple(f"light.{i}" for i in range(30))), 0.5)] * 10
    continuity_groups = RetrievalHelper.continuity_groups(
        ContinuityContext(target_groups=groups),
    )
    assert len(continuity_groups) == 2
    assert len(continuity_groups[0]["entities"]) == 12
    assert "device_classes" in continuity_groups[0]


def test_literal_name_resolves_one_identity_candidate() -> None:
    devices = [
        Device(
            id="light.bathroom_ceiling",
            friendly_name="Bathroom ceiling light",
            area_name="Bathroom",
            floor_name="Ground floor",
            domain=["light"],
        ),
        Device(
            id="light.kitchen_ceiling",
            friendly_name="Kitchen ceiling light",
            area_name="Kitchen",
            floor_name="Ground floor",
            domain=["light"],
        ),
    ]

    status, names = RetrievalHelper.device_resolution(
        "Bathroom ceiling light",
        devices,
    )

    assert status == "high"
    assert names == ("light.bathroom_ceiling",)
    assert RetrievalHelper.reduce_confident_devices(
        "Bathroom ceiling light",
        devices,
    ) == [devices[0]]


def test_ambiguous_singular_request_does_not_authorize_top_rank() -> None:
    devices = [
        Device(
            id="light.bathroom_ceiling",
            friendly_name="Bathroom ceiling light",
            area_name="Bathroom",
            floor_name="Ground floor",
            domain=["light"],
        ),
        Device(
            id="light.bathroom_mirror",
            friendly_name="Bathroom mirror light",
            area_name="Bathroom",
            floor_name="Ground floor",
            domain=["light"],
        ),
    ]

    status, names = RetrievalHelper.device_resolution("turn on the bathroom light", devices)

    assert status == "ambiguous"
    assert set(names) == {"light.bathroom_ceiling", "light.bathroom_mirror"}
    assert RetrievalHelper.reduce_confident_devices("turn on the bathroom light", devices) == devices
    assert len(RetrievalHelper.select_device_candidates("turn on the bathroom light", devices, 1)) == 2


def test_explicit_location_mismatch_is_not_authorized_by_exact_entity_name() -> None:
    device = Device(
        id="light.kitchen_ceiling",
        friendly_name="Kitchen ceiling light",
        area_name="Kitchen",
        floor_name="Ground floor",
        domain=["light"],
    )

    status, _ = RetrievalHelper.device_resolution(
        "turn on light.kitchen_ceiling in the bathroom",
        [device],
    )

    assert status == "weak"


def test_normalization_supports_unicode_without_language_patterns() -> None:
    assert RetrievalHelper._normalize("KÜCHE") == "küche"


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    scores = RetrievalHelper.reciprocal_rank_fusion(
        (["a", "b"], ["b", "c"], ["b", "a"]),
    )

    assert scores["b"] > scores["a"] > scores["c"]


def test_metadata_rank_breaks_equal_text_match() -> None:
    candidates = [Candidate("sensor"), Candidate("sensor")]

    result = RetrievalHelper.rank_scored_candidates(
        [
            ScoredResult(candidates[0], 0.8, 1),
            ScoredResult(candidates[1], 0.8, 2),
        ],
        candidates,
        "kitchen sensor",
        lambda candidate: str(id(candidate)),
        lambda candidate: (candidate.name,),
        1,
        metadata_score=lambda candidate: 1.0 if candidate is candidates[1] else 0.0,
    )

    assert result == [candidates[1]]


def test_native_vector_magnitude_does_not_change_rank_fusion() -> None:
    candidates = [Candidate("first"), Candidate("second")]

    low_scores = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[0], 0.01, 1), ScoredResult(candidates[1], 0.0, 2)],
        candidates,
        "unrelated request",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        2,
    )
    high_scores = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[0], 1.0, 1), ScoredResult(candidates[1], 0.99, 2)],
        candidates,
        "unrelated request",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        2,
    )

    assert low_scores == high_scores == candidates


def test_strong_current_match_suppresses_continuity() -> None:
    candidates = [Candidate("Bedroom lamp"), Candidate("Kitchen ceiling light")]
    continuity_calls: list[Candidate] = []

    result = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[0], 0.9, 1), ScoredResult(candidates[1], 0.8, 2)],
        candidates,
        "turn on the kitchen ceiling light",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        1,
        continuity_score=lambda candidate: continuity_calls.append(candidate) or 10.0,
    )

    assert result == [candidates[1]]
    assert continuity_calls == []


def test_weak_current_match_allows_continuity() -> None:
    candidates = [Candidate("Bedroom lamp"), Candidate("Kitchen ceiling light")]
    continuity_calls: list[Candidate] = []

    RetrievalHelper.rank_scored_candidates(
        [ScoredResult(candidates[1], 0.9, 1)],
        candidates,
        "adjust it",
        lambda candidate: candidate.name,
        lambda candidate: (candidate.name,),
        1,
        continuity_score=lambda candidate: continuity_calls.append(candidate) or (
            1.0 if candidate is candidates[0] else 0.0
        ),
    )

    assert sorted(candidate.name for candidate in continuity_calls) == sorted(
        candidate.name for candidate in candidates
    )


def test_device_compatibility_promotes_matching_tool_schema() -> None:
    device = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="Ground floor",
        domain=["light"],
    )
    incompatible = LlmTool(
        name="CoverControl",
        description="Control cover",
        parameters={"properties": {"domain": {"enum": ["cover"]}}},
        metadata=ToolMetadata(is_domain_aware=True),
    )
    compatible = LlmTool(
        name="LightControl",
        description="Control light",
        parameters={"properties": {"domain": {"enum": ["light"]}}},
        metadata=ToolMetadata(is_domain_aware=True),
    )

    result = RetrievalHelper.rerank_tools_for_devices(
        [incompatible, compatible],
        [device],
        1,
    )

    assert result == [compatible]


def test_device_compatibility_demotes_explicit_mismatch() -> None:
    device = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="",
        domain=["light"],
    )
    incompatible = LlmTool(
        name="CoverControl",
        description="Control cover",
        parameters={"properties": {"domain": {"enum": ["cover"]}}},
    )
    neutral = LlmTool(name="Search", description="Search", parameters={})

    assert RetrievalHelper.rerank_tools_for_devices(
        [incompatible, neutral], [device], 2
    ) == [neutral, incompatible]


def test_device_domain_mismatch_is_a_negative_soft_signal() -> None:
    light = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="",
        domain=["light"],
    )
    lock_tool = LlmTool(name="HassLock", description="Lock a lock",
                        parameters={"properties": {"domain": {"enum": ["lock"]}}})

    signals = RetrievalHelper.tool_ranking_signals(
        lock_tool,
        "turn on the light",
        [light],
    )

    assert signals["domain"] < 0


def test_tool_confidence_distinguishes_search_from_matching_action() -> None:
    light = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="",
        domain=["light"],
    )
    search = LlmTool(name="HassSemanticSearch", description="Search", parameters={})
    turn_on = LlmTool(name="HassTurnOn", description="Turn on", parameters={})

    assert RetrievalHelper.tool_search_confidence(
        [search],
        "turn on the light",
        [light],
    ) == "low"
    assert RetrievalHelper.tool_search_confidence(
        [turn_on, search],
        "turn on the light",
        [light],
    ) == "medium"


def test_trusted_location_ranks_current_area_before_retrieval() -> None:
    kitchen = Device(
        id="light.kitchen",
        friendly_name="Light",
        area_name="Kitchen",
        floor_name="Ground floor",
    )
    bedroom = Device(
        id="light.bedroom",
        friendly_name="Light",
        area_name="Bedroom",
        floor_name="First floor",
    )

    result = RetrievalHelper.rank_scored_candidates(
        [
            ScoredResult(bedroom, 0.9, 1),
            ScoredResult(kitchen, 0.8, 2),
        ],
        [bedroom, kitchen],
        "turn on the light",
        lambda device: device.id,
        lambda device: (device.friendly_name,),
        1,
        metadata_score=lambda device: RetrievalHelper.trusted_location_score(
            device,
            "Kitchen",
            "Ground floor",
        ),
    )

    assert result == [kitchen]


def test_adaptive_candidate_limit_is_bounded() -> None:
    assert RetrievalHelper.adaptive_candidate_limit(4) == 24
    assert RetrievalHelper.adaptive_candidate_limit(100) == 64
    assert RetrievalHelper.adaptive_candidate_limit(0) == 0
    assert RetrievalHelper.expanded_tool_limit(4) == 12
    assert RetrievalHelper.expanded_tool_limit(8) == 20


def test_semantic_history_uses_similarity_recency_and_expiry() -> None:
    relevant = TurnContext(
        key="relevant",
        text="kitchen light",
        entities=("light.kitchen",),
        created_at=990.0,
    )
    unrelated = TurnContext(
        key="unrelated",
        text="front door",
        entities=("lock.front_door",),
        created_at=995.0,
    )
    expired = TurnContext(
        key="expired",
        text="old kitchen light",
        entities=("light.old",),
        created_at=500.0,
    )

    selected = RetrievalHelper.select_history_contexts(
        [relevant, unrelated, expired],
        {
            "relevant": [1.0, 0.0],
            "unrelated": [0.0, 1.0],
            "expired": [1.0, 0.0],
        },
        [1.0, 0.0],
        max_age_seconds=300.0,
        now=1000.0,
    )

    assert [context.key for context, _ in selected] == ["relevant", "unrelated"]
    assert selected[0][1] > selected[1][1]


def test_structured_continuity_boosts_recent_canonical_entity() -> None:
    context = TurnContext(
        key="turn",
        text="",
        entities=("light.kitchen",),
        areas=("Kitchen",),
        domains=("light",),
        tools=("HassTurnOn",),
        actions=("HassTurnOn",),
    )
    continuity = RetrievalHelper.build_continuity_context([(context, 0.8)])
    device = Device(
        id="light.kitchen",
        friendly_name="Kitchen light",
        area_name="Kitchen",
        floor_name="Ground floor",
        domain=["light"],
    )
    tool = LlmTool(name="HassTurnOn", description="", parameters={})

    assert continuity.device_score(device) > 1.0
    assert continuity.tool_score(tool) > 1.0


def test_successful_target_group_is_preserved_for_weak_followup() -> None:
    previous = Device(
        id="light.bedroom",
        friendly_name="Bedroom lamp",
        area_name="Bedroom",
        floor_name="",
    )
    vector_match = Device(
        id="switch.kitchen",
        friendly_name="Kitchen switch",
        area_name="Kitchen",
        floor_name="",
    )
    context = TurnContext(
        key="turn",
        text="turn on the bedroom lamp",
        target_groups=(TargetGroup(entities=(previous.id,), tool="HassTurnOn"),),
    )
    continuity = RetrievalHelper.build_continuity_context([(context, 0.8)])

    result = RetrievalHelper.rank_scored_candidates(
        [ScoredResult(vector_match, 0.9, 1)],
        [previous, vector_match],
        "adjust it",
        lambda device: device.id,
        lambda device: (device.id, device.friendly_name),
        1,
        preserve_score=continuity.successful_target_score,
    )

    assert result == [previous]
    assert not RetrievalHelper.target_is_confident("adjust it", result, continuity)


def test_canonical_tool_name_parts_and_family_are_embedded() -> None:
    tool = LlmTool(name="HassTurnOn", description="Control a target",
                   metadata=ToolMetadata(family="power"))

    assert tool.canonical_name_parts == ("hass", "turn", "on")
    assert tool.family == "power"
    assert "canonical parts: hass turn on" in tool.to_embedding_text()
    assert "family: power" in tool.to_embedding_text()


def test_supported_tool_domains_are_indexed() -> None:
    tool = LlmTool(
        name="HassTurnOn",
        description="Turn on a device",
        parameters={"properties": {"domain": {"enum": ["switch", "light"]}}},
    )

    assert tool.canonical_supported_domains == ("light", "switch")
    assert "supported domains: light, switch" in tool.to_embedding_text()


def test_compound_and_informational_requests_are_preserved():
    for query in (
        "turn on the kitchen light and turn off the bathroom fan",
        "is the kitchen light on",
        "kitchen light on",
        "Küchenlicht einschalten und Ventilator ausschalten",
    ):
        assert RetrievalHelper.build_tool_search_query(query, "", []) == query


def test_unknown_tool_schema_is_searchable_and_remains_live() -> None:
    parameters = {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {
                "type": "string",
                "description": "Cleaning program to start",
                "enum": ["quiet", "turbo"],
            },
        },
    }
    tool = LlmTool(name="VendorExecute", description="Run a vendor capability", parameters=parameters)

    assert "parameter mode" in tool.canonical_search_parts
    assert "Cleaning program to start" in tool.canonical_search_parts
    assert "required mode" in tool.to_embedding_text()
    assert "choices quiet turbo" in tool.to_embedding_text()
    assert tool.to_tool_dict()["function"]["parameters"] is parameters


def test_compound_request_preserves_custom_capabilities_in_embedding_query() -> None:
    request = (
        "turn on light strip and set the color to red and brightness to 40% "
        "and also enable sleep mode"
    )
    query = RetrievalHelper.build_tool_search_query(request, "", [{"domain": ["light"]}])

    assert query.startswith(request)
    assert query == request


def test_compound_retrieval_covers_schema_capabilities_among_power_tools() -> None:
    powers = [
        LlmTool(f"Vendor{index}TurnOn", "Turn on a light",
                parameters={"properties": {"domain": {"enum": ["light"]}}})
        for index in range(5)
    ]
    light_set = LlmTool("HassLightSet", "Set light brightness and color",
                        parameters={"properties": {
                            "brightness": {"type": "integer"},
                            "color": {"type": "string"},
                        }})
    custom = LlmTool("VendorExecute", "Run a user-defined program",
                     parameters={"properties": {"mode": {"enum": ["sleep", "turbo"]}}})
    query = RetrievalHelper.build_tool_search_query(
        "turn on light strip and set the color to red and brightness to 40% "
        "and also enable sleep mode", "", [{"domain": ["light"]}],
    )
    result = RetrievalHelper.rank_tool_candidates(
        [ScoredResult(tool, 0.99, index + 1) for index, tool in enumerate(powers)],
        [*powers, light_set, custom], query, [{"domain": ["light"]}], 3,
    )

    assert len(result) == 3
    assert any(tool in powers for tool in result)
    assert light_set in result
    assert custom in result


def test_compound_action_candidates_do_not_claim_complete_intent_resolution() -> None:
    turn_on = LlmTool("HassTurnOn", "Turn on a device")
    turn_off = LlmTool("HassTurnOff", "Turn off a device")
    query = RetrievalHelper.build_tool_search_query(
        "turn on the light and turn off the fan", "", [],
    )

    assert RetrievalHelper.tool_ranking_signals(turn_off, query)["lexical_exact"] > 0
    assert RetrievalHelper.tool_search_confidence([turn_on], query, []) != "high"
    assert RetrievalHelper.tool_search_confidence([turn_on, turn_off], query, []) != "high"
    assert set(t.name for t in RetrievalHelper.rank_tools_for_query([turn_on, turn_off], query)) == {"HassTurnOn", "HassTurnOff"}


def test_high_scoring_power_tool_does_not_hide_other_candidates() -> None:
    turn_on = LlmTool("HassTurnOn", "Turn on a light")
    custom = LlmTool("BedtimeRoutine", "Start a user-defined routine")
    result = RetrievalHelper.rank_tool_candidates(
        [ScoredResult(turn_on, 1.0, 1)], [turn_on, custom],
        "turn on the light and start bedtime routine", [], 4,
    )

    assert result == [turn_on, custom]
