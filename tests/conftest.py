from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from homeassistant.helpers import area_registry, floor_registry

from custom_components.ha_ragent.src.const import (
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MAX_TOOLS_TO_EXTRACT,
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_RETRIEVAL_METHOD,
    DOMAIN,
    RAGENT_PREFIXED_TOOL_NAMES_BY_NAME,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.translation import RAGentTranslations


@pytest.fixture
def compact_search(monkeypatch):
    """Provide four office lights and one action through an in-memory backend."""
    devices = [
        Device(f"light.office_{index}", f"Office light {index}", "Office", "Ground", domain=["light"])
        for index in range(4)
    ]
    actions = [LlmTool(
        "VendorOfficeLights", "Control office lights",
        metadata=ToolMetadata(canonical_action="turn_on", supported_domains=("light",)),
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    )]
    catalog = {DeviceEmbedding: devices, LlmToolEmbedding: actions}

    async def list_objects(object_type, *_args):
        return list(catalog[object_type])

    async def retrieve_objects(object_type, *_args):
        return [ScoredResult(item, 1.0, index + 1) for index, item in enumerate(catalog[object_type])]

    subentry = SimpleNamespace(data={
        CONF_RETRIEVAL_METHOD: RETRIEVAL_METHOD_VECTOR,
        CONF_MIN_DEVICES_TO_EXTRACT: 4,
        CONF_MAX_DEVICES_TO_EXTRACT: 4,
        CONF_MAX_TOOLS_TO_EXTRACT: 1,
    })
    entry = SimpleNamespace(
        options={}, subentries={"agent": subentry},
        embedder_backend=SimpleNamespace(async_embed_text=AsyncMock(return_value=[1.0, 0.0])),
        vector_db_backend=SimpleNamespace(
            async_get_lexical_objects=AsyncMock(side_effect=list_objects),
            async_retrieve_scored_objects=AsyncMock(side_effect=retrieve_objects),
        ),
    )
    states = {device.id: SimpleNamespace(state="off", attributes={}) for device in devices}
    hass = SimpleNamespace(data={DOMAIN: {"entry": entry}}, states=SimpleNamespace(get=states.get))
    areas = [
        SimpleNamespace(id="office_area", name="Office", aliases={"Workroom", "Büro"}),
        SimpleNamespace(id="kitchen_area", name="Kitchen", aliases={"Küche"}),
    ]
    floors = [
        SimpleNamespace(floor_id="ground_floor", name="Ground", aliases={"Downstairs", "Erdgeschoss"}),
        SimpleNamespace(floor_id="upper_floor", name="Upper", aliases={"Upstairs"}),
    ]
    area_reg = SimpleNamespace(async_list_areas=lambda: areas)
    floor_reg = SimpleNamespace(async_list_floors=lambda: floors)
    monkeypatch.setattr(area_registry, "async_get", lambda _hass: area_reg)
    monkeypatch.setattr(floor_registry, "async_get", lambda _hass: floor_reg)
    RAGentTranslations.default("en")
    search = RAGentSemanticSearchTool(hass, "entry", "agent", "en")
    search.name = RAGENT_PREFIXED_TOOL_NAMES_BY_NAME[search.name]
    return search, actions
