import unittest

from custom_components.ha_ragent.src.homeassistant.helpers.source_ranker import SourceRanker
from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.retrieval.lexical_index import lexical_index


class ExtensibleRankingTests(unittest.TestCase):
    def test_identifier_tokenization(self):
        index = lexical_index((("LivingRoomLamp",), ("LivingRoomSpeaker",)))
        self.assertGreater(index.scores("living room lamp")[0], 0.99)
        self.assertGreater(index.scores("living room lamp")[0], index.scores("living room lamp")[1])

    def test_custom_tool_description_recovers_without_vectors(self):
        for description in ("Bewässerung kalibrieren", "灌溉校准", "معايرة الري"):
            target = LlmTool("VendorX42", description)
            distractor = LlmTool("VendorX41", "ZZZZ")
            ranked = ToolRanker.rank_tool_candidates([], [distractor, target], description, [], 1)
            self.assertEqual(ranked, [target])

    def test_custom_schema_parameter_recovers_without_action_metadata(self):
        target = LlmTool("VendorX42", "", parameters={
            "properties": {"calibration_profile": {"enum": ["hydroponics"]}},
        })
        ranked = ToolRanker.rank_tool_candidates(
            [], [LlmTool("VendorX41", ""), target], "hydroponics", [], 1,
        )
        self.assertEqual(ranked, [target])

    def test_custom_capability_and_domain_survive(self):
        self.assertEqual(ToolRanker.normalize_requested_capability({
            "action": "Vendor_Calibrate", "domain": "irrigation_controller",
        }), {"action": "vendor_calibrate", "domains": ("irrigation_controller",)})

    def test_missing_tool_metadata_remains_neutral(self):
        self.assertEqual(ToolRanker.tool_capability_compatibility(
            LlmTool("Unknown", ""), {"action": "vendor_calibrate", "domain": "vendor"},
        ), 0.0)

    def test_query_preserves_custom_description(self):
        query = ToolRanker.build_tool_search_query(
            "", "Bewässerung kalibrieren", [],
            {"action": "vendor_calibrate", "domain": "irrigation_controller"},
        )
        self.assertEqual(query, "Bewässerung kalibrieren\nvendor_calibrate\nirrigation_controller")

    def test_device_typo_recovers_without_vectors(self):
        target = Device("light.chandelier", "Dining Room Chandelier", "Dining Room", "")
        other = Device("light.bedroom", "Bedroom Ceiling Light", "Bedroom", "")
        ranked = SourceRanker.rank_scored_candidates(
            [], [other, target], "dining room candelier", lambda d: d.id,
            lambda d: (d.friendly_name, d.area_name), 1,
        )
        self.assertEqual(ranked, [target])
