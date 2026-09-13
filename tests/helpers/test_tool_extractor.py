import asyncio
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding


def _run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _subentry() -> SimpleNamespace:
    return SimpleNamespace(data={}, title="test")


@_run_async
async def test_startup_extraction_embeds_more_than_zero_tools(monkeypatch) -> None:
    raw_tool = SimpleNamespace(
        name="HassFanSetSpeed",
        description="Set fan speed",
        parameters=object(),
        metadata=None,
    )
    api = SimpleNamespace(tools=[raw_tool], custom_serializer=None)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi",
        lambda *_args, **_kwargs: {"properties": {"domain": {"enum": ["fan"]}}},
    )

    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert len(tools) > 0
    assert tools[0].metadata.supported_domains == ("fan",)


@_run_async
async def test_metadata_extraction_exception_does_not_zero_all_tools(monkeypatch) -> None:
    raw_tools = [
        SimpleNamespace(name="BrokenTool", description="broken", parameters=object(), metadata=None),
        SimpleNamespace(name="HassTurnOn", description="turn on", parameters=object(), metadata=None),
    ]
    api = SimpleNamespace(tools=raw_tools, custom_serializer=None)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.llm.async_get_api",
        AsyncMock(return_value=api),
    )
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor.to_openapi",
        lambda *_args, **_kwargs: {"properties": {}},
    )

    original = ToolExtractor._extract_tool_metadata
    calls = 0

    def fail_once(self, tool, parameters):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("bad metadata")
        return original(tool, parameters)

    monkeypatch.setattr(ToolExtractor, "_extract_tool_metadata", fail_once)
    extractor = ToolExtractor(SimpleNamespace(), SimpleNamespace())
    extractor._register_fake_timer_device = lambda: None
    extractor._remove_fake_timer_device = lambda: None

    tools = await extractor._async_get_embeddable_tools(_subentry())

    assert [tool.name for tool in tools] == ["BrokenTool", "HassTurnOn"]


@_run_async
async def test_startup_embedding_persists_more_than_zero_tools(monkeypatch) -> None:
    subentry = _subentry()
    tool = LlmTool("HassTurnOn", "Turn on", parameters={"properties": {}})
    entry = SimpleNamespace(
        subentries={"sub": subentry},
        embedder_backend=SimpleNamespace(
            async_embed_object=AsyncMock(
                return_value=[LlmToolEmbedding(tool, [1.0])]
            )
        ),
        vector_db_backend=SimpleNamespace(
            async_reset_collection=AsyncMock(),
            async_save_objects=AsyncMock(),
            invalidate_collection_cache=lambda _name: None,
            cache_collection_objects=lambda _name, _objects: None,
        ),
    )
    extractor = ToolExtractor(SimpleNamespace(), entry)
    monkeypatch.setattr(
        extractor,
        "_async_get_embeddable_tools",
        AsyncMock(return_value=[tool]),
    )

    await extractor.async_embed_exposed_tools("sub")

    entry.embedder_backend.async_embed_object.assert_awaited_once()
    entry.vector_db_backend.async_save_objects.assert_awaited_once()
