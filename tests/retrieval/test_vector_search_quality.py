import asyncio

import faiss
import pytest
from homeassistant.core import HomeAssistant
from custom_components.ha_ragent.src.backends.database.faiss_backend import (
    FaissDbBackend,
)
from custom_components.ha_ragent.src.const import (
    CONF_VECTOR_DB_NAME,
    TRANSLATION_PROMPT_INSTRUCTIONS,
)
from custom_components.ha_ragent.src.translation import RAGentTranslations
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.device import Device


@pytest.fixture
def hass(tmp_path):
    """Provide a native Home Assistant instance with isolated storage."""
    loop = asyncio.new_event_loop()

    async def create() -> HomeAssistant:
        instance = HomeAssistant(str(tmp_path))
        await instance.async_start()
        return instance

    instance = loop.run_until_complete(create())
    try:
        yield instance
    finally:
        loop.run_until_complete(instance.async_stop(force=True))
        loop.close()


def _backend(hass: HomeAssistant) -> FaissDbBackend:
    return FaissDbBackend(hass, {CONF_VECTOR_DB_NAME: "test"})


def test_faiss_uses_cosine_similarity(hass: HomeAssistant) -> None:
    backend = _backend(hass)
    collection = "tools"
    aligned = LlmTool(name="aligned", description="", parameters={})
    nearby = LlmTool(name="nearby", description="", parameters={})

    backend._save_device_embeddings(collection, [
        LlmToolEmbedding(aligned, [10.0, 0.0]),
        LlmToolEmbedding(nearby, [1.0, 1.0]),
    ])

    assert backend._indices[collection].metric_type == faiss.METRIC_INNER_PRODUCT
    result = backend._query_scored_devices(collection, [1.0, 0.0], 1)
    assert result[0][0]["name"] == "aligned"


def test_faiss_returns_normalized_vector_scores(hass: HomeAssistant) -> None:
    backend = _backend(hass)
    tool = LlmTool(name="aligned", description="", parameters={})
    backend._save_device_embeddings("tools", [LlmToolEmbedding(tool, [1.0, 0.0])])

    result = hass.loop.run_until_complete(
        backend.async_retrieve_scored_objects(
            LlmToolEmbedding,
            {},
            "tools",
            [1.0, 0.0],
            1,
        )
    )

    assert result[0].item == tool
    assert result[0].score == 1.0
    assert result[0].rank == 1


def test_tool_embedding_includes_parameter_schema() -> None:
    tool = LlmTool(
        name="HassTurnOff",
        description="Turn off a Home Assistant device.",
        parameters={"properties": {"area": {"type": "string"}}},
    )

    embedding_text = tool.to_embedding_text()

    assert "canonical parts: hass turn off" in embedding_text
    assert "parameter area" in embedding_text
    assert "type string" in embedding_text


def test_device_embedding_includes_device_class() -> None:
    device = Device(
        id="binary_sensor.patio_door",
        friendly_name="Patio door",
        area_name="Living room",
        floor_name="Ground floor",
        domain=["binary_sensor"],
        device_class="door",
    )

    assert "Device Class: door" in device.to_embedding_text()


def test_recovery_instruction_searches_for_missing_action_tool() -> None:
    prompt = RAGentTranslations._load("en")["Prompts"][TRANSLATION_PROMPT_INSTRUCTIONS]
    assert "never additional tasks or authorization to act" in prompt
    assert "scope `tools`" in prompt
