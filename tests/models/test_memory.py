from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.const import CONF_MAX_MEMORY_ENTRIES, CONF_VECTOR_DB_NAME, DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.tools.forget_fact import RAGentForgetTool
from custom_components.ha_ragent.src.homeassistant.tools.remember_fact import RAGentRememberTool
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.translation import RAGentTranslations


def test_missing_translation_language_falls_back_to_english() -> None:
    RAGentTranslations._cache.clear()

    missing = RAGentTranslations._load("missing")
    english = RAGentTranslations._load("en")

    assert missing == english


def test_missing_translation_values_use_safe_defaults() -> None:
    translations = RAGentTranslations.__new__(RAGentTranslations)
    translations._data = {}

    assert translations.get("Tools", "missing_tool", "fallback") == "fallback"
    assert not translations.has_tool("missing_tool")


class FakeEmbedder:
    async def async_embed_text(
        self, config: dict[str, Any], text: str, input_type: str = "query",
    ) -> list[float]:
        return [1.0, float(len(text)), 0.5]


class FakeVectorDb:
    def __init__(self) -> None:
        self.objects: dict[str, list[MemoryEmbedding]] = {}

    async def async_collection_has_objects(self, config: dict[str, Any], collection_name: str) -> bool:
        return bool(self.objects.get(collection_name))

    async def async_ensure_collection_exists(self, config: dict[str, Any], collection_name: str, embedding_length: int) -> None:
        self.objects.setdefault(collection_name, [])

    async def async_delete_objects(self, config: dict[str, Any], collection_name: str, id_field: str, object_ids: list[str]) -> int:
        current = self.objects.get(collection_name, [])
        retained = [item for item in current if item.to_dict().get(id_field) not in object_ids]
        deleted = len(current) - len(retained)
        self.objects[collection_name] = retained
        return deleted

    async def async_save_objects(self, config: dict[str, Any], collection_name: str, embeddings: list[MemoryEmbedding]) -> None:
        self.objects.setdefault(collection_name, []).extend(embeddings)

    async def async_upsert_objects(self, config: dict[str, Any], collection_name: str, id_field: str, embeddings: list[MemoryEmbedding]) -> None:
        incoming_ids = {embedding.to_dict()[id_field] for embedding in embeddings}
        retained = [
            item for item in self.objects.get(collection_name, [])
            if item.to_dict().get(id_field) not in incoming_ids
        ]
        self.objects[collection_name] = [*retained, *embeddings]

    async def async_list_objects(self, object_type, config_subentry: dict[str, Any], collection_name: str):
        return [
            object_type.parse_object(item.to_dict())
            for item in self.objects.get(collection_name, [])
        ]

    async def async_retrieve_scored_objects(self, object_type, config_subentry: dict[str, Any], collection_name: str, query_embedding: list[float], top_k: int):
        return [
            ScoredResult(object_type.parse_object(item.to_dict()), 1.0, rank)
            for rank, item in enumerate(self.objects.get(collection_name, [])[:top_k], start=1)
        ]


def create_memory_hass(*, max_entries: int | None = None) -> tuple[SimpleNamespace, FakeVectorDb]:
    RAGentTranslations._load("en")
    vector_db = FakeVectorDb()
    subentry_data = {"model": "embed"}
    if max_entries is not None:
        subentry_data[CONF_MAX_MEMORY_ENTRIES] = max_entries
    entry = SimpleNamespace(
        subentries={"agent": SimpleNamespace(data=subentry_data)},
        embedder_backend=FakeEmbedder(),
        vector_db_backend=vector_db,
        translations=RAGentTranslations("en"),
    )
    hass = SimpleNamespace(data={DOMAIN: {"entry": entry}})
    return hass, vector_db


@pytest.fixture
def memory_now(monkeypatch) -> datetime:
    now = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "custom_components.ha_ragent.src.homeassistant.helpers.memory_manager.dt_util.utcnow",
        lambda: now,
    )
    return now


def test_memory_model_round_trip() -> None:
    memory = Memory(id="0123456789abcdef", content="The reading lamp is beside the sofa.", created_at="2026-09-01T12:00:00+00:00")
    embedding = MemoryEmbedding(memory, [1.0, 2.0])

    assert MemoryEmbedding.parse_object(embedding.to_dict()) == memory
    assert "reading lamp" in memory.to_embedding_text()
    assert memory.to_dict()["id"] == memory.id
    assert embedding.to_dict()["id"] == memory.id
    assert "memory_id" not in embedding.to_dict()


def test_memory_confidence_controls_exposed_count() -> None:
    memories = [
        Memory(str(index), f"Memory {index}", "2026-09-01T12:00:00+00:00")
        for index in range(4)
    ]

    weak = [
        ScoredResult(memory, score, rank)
        for rank, (memory, score) in enumerate(
            zip(memories, (0.59, 0.58, 0.57, 0.56)), start=1
        )
    ]
    assert MemoryManager.select_confident_memories(weak, 0, 4) == []
    assert MemoryManager.select_confident_memories(weak, 1, 4) == [memories[0]]

    mixed = [
        ScoredResult(memory, score, rank)
        for rank, (memory, score) in enumerate(
            zip(memories, (0.95, 0.82, 0.70, 0.55)), start=1
        )
    ]
    assert MemoryManager.select_confident_memories(mixed, 0, 4) == memories[:2]

    strong = [
        ScoredResult(memory, score, rank)
        for rank, (memory, score) in enumerate(
            zip(memories, (0.95, 0.90, 0.86, 0.82)), start=1
        )
    ]
    assert MemoryManager.select_confident_memories(strong, 0, 4) == memories


def test_memory_manager_remember_recall_replace_and_forget() -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass()
        manager = MemoryManager(hass, "entry", "agent")

        first = await manager.async_remember("  The reading   lamp is beside the sofa.  ")
        replacement = await manager.async_remember("The reading lamp is beside the sofa.")

        assert first.id == replacement.id
        assert replacement.content == "The reading lamp is beside the sofa."
        assert len(vector_db.objects[manager.collection_name]) == 1
        assert await manager.async_recall([1.0, 1.0, 1.0], 0, 4) == [replacement]
        assert await manager.async_forget(replacement.id) is True
        assert await manager.async_forget(replacement.id) is False
        assert await manager.async_recall([1.0, 1.0, 1.0], 0, 4) == []

    asyncio.run(run())


@pytest.mark.parametrize("max_entries", [1, 3])
def test_memory_limit_reached_keeps_all_entries(memory_now, max_entries) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=max_entries)
        manager = MemoryManager(hass, "entry", "agent")

        saved = [
            await manager.async_remember(f"Fact {index}")
            for index in range(max_entries)
        ]
        assert all(memory is not None for memory in saved)
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert stored == saved
        assert len(stored) == max_entries

    asyncio.run(run())


@pytest.mark.parametrize("max_entries", [1, 3])
def test_memory_limit_exceeded_evicts_oldest_on_equal_retrieval_counts(memory_now, max_entries) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=max_entries)
        manager = MemoryManager(hass, "entry", "agent")
        existing = [
            Memory(
                f"{index + 1:016x}", f"Existing fact {index}",
                (memory_now - timedelta(minutes=max_entries - index)).isoformat(),
            )
            for index in range(max_entries)
        ]
        vector_db.objects[manager.collection_name] = [
            MemoryEmbedding(memory, [1.0, 1.0, 1.0]) for memory in reversed(existing)
        ]

        added = await manager.async_remember("New fact")

        assert added is not None
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert len(stored) == max_entries
        assert {memory.id for memory in stored} == {
            added.id, *(memory.id for memory in existing[1:]),
        }

    asyncio.run(run())


def test_memory_limit_eviction_prioritizes_retrieval_count_over_age(memory_now) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=3)
        manager = MemoryManager(hass, "entry", "agent")
        oldest = Memory(
            "1111111111111111", "Frequently recalled old fact",
            (memory_now - timedelta(minutes=3)).isoformat(), retrieval_count=5,
        )
        middle = Memory(
            "2222222222222222", "Occasionally recalled fact",
            (memory_now - timedelta(minutes=2)).isoformat(), retrieval_count=1,
        )
        newest = Memory(
            "3333333333333333", "Unrecalled recent fact",
            (memory_now - timedelta(minutes=1)).isoformat(), retrieval_count=0,
        )
        vector_db.objects[manager.collection_name] = [
            MemoryEmbedding(memory, [1.0, 1.0, 1.0]) for memory in (oldest, middle, newest)
        ]

        added = await manager.async_remember("New fact")

        assert added is not None
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert {memory.id for memory in stored} == {oldest.id, middle.id, added.id}
        assert len(stored) == 3

    asyncio.run(run())


@pytest.mark.parametrize("max_entries", [1, 3])
def test_replacing_memory_at_limit_keeps_other_entries(memory_now, max_entries) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=max_entries)
        manager = MemoryManager(hass, "entry", "agent")
        original = await manager.async_remember("The reading lamp is beside the sofa.")
        assert original is not None
        others = [
            await manager.async_remember(f"Other fact {index}")
            for index in range(max_entries - 1)
        ]
        assert all(memory is not None for memory in others)

        replacement = await manager.async_remember("  The reading   lamp is beside the sofa.  ")

        assert replacement is not None
        assert replacement.id == original.id
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert len(stored) == max_entries
        assert {memory.id for memory in stored} == {original.id, *(memory.id for memory in others)}
        assert next(memory for memory in stored if memory.id == original.id) == replacement

    asyncio.run(run())


def test_remember_tool_reports_success_for_immediately_evicted_memory(memory_now) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=2)
        manager = MemoryManager(hass, "entry", "agent")
        existing = [
            Memory(
                f"{index + 1:016x}", f"Recalled fact {index}",
                (memory_now - timedelta(minutes=index + 1)).isoformat(), retrieval_count=index + 1,
            )
            for index in range(2)
        ]
        vector_db.objects[manager.collection_name] = [
            MemoryEmbedding(memory, [1.0, 1.0, 1.0]) for memory in existing
        ]
        remember = RAGentRememberTool(hass, "entry", "agent")

        result = await remember._async_call(SimpleNamespace(tool_args={"memory": "New fact"}))

        assert result["success"] is True
        assert result["memory"] == "New fact"
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert stored == existing
        assert result["memory_id"] not in {memory.id for memory in stored}

    asyncio.run(run())


def test_lowered_memory_limit_removes_all_excess_entries(memory_now) -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass(max_entries=5)
        manager = MemoryManager(hass, "entry", "agent")
        existing = [
            Memory(
                f"{index + 1:016x}", f"Existing fact {index}",
                (memory_now - timedelta(minutes=5 - index)).isoformat(),
            )
            for index in range(5)
        ]
        vector_db.objects[manager.collection_name] = [
            MemoryEmbedding(memory, [1.0, 1.0, 1.0]) for memory in reversed(existing)
        ]
        hass.data[DOMAIN]["entry"].subentries["agent"].data[CONF_MAX_MEMORY_ENTRIES] = 2

        added = await manager.async_remember("New fact")

        assert added is not None
        stored = await vector_db.async_list_objects(MemoryEmbedding, {}, manager.collection_name)
        assert len(stored) == 2
        assert {memory.id for memory in stored} == {existing[-1].id, added.id}

    asyncio.run(run())


def test_memory_tools() -> None:
    async def run() -> None:
        hass, _ = create_memory_hass()
        remember = RAGentRememberTool(hass, "entry", "agent")
        remember_result = await remember._async_call(
            SimpleNamespace(tool_args={"memory": "The thermostat target is 21 C."})
        )

        assert remember_result["success"] is True
        memory_id = remember_result["memory_id"]

        forget = RAGentForgetTool(hass, "entry", "agent")
        forget_result = await forget._async_call(SimpleNamespace(tool_args={"memory_id": memory_id}))
        assert forget_result == {
            "success": True,
            "memory_id": memory_id,
            "forgotten": True,
        }

        missing_result = await forget._async_call(SimpleNamespace(tool_args={"memory_id": memory_id}))
        assert missing_result["success"] is False
        assert missing_result["error"] == "memory not found"

    asyncio.run(run())


def test_faiss_memory_persistence_and_delete(tmp_path: Path) -> None:
    class Hass:
        def __init__(self) -> None:
            self.config = SimpleNamespace(path=lambda name: str(tmp_path / name))

        async def async_add_executor_job(self, target, *args):
            return target(*args)

    async def run() -> None:
        hass = Hass()
        config = {CONF_VECTOR_DB_NAME: "memory_test"}
        collection = "memories_agent"
        first = Memory("1111111111111111", "First memory", "2026-09-01T12:00:00+00:00")
        second = Memory("2222222222222222", "Second memory", "2026-09-01T12:01:00+00:00")

        backend = FaissDbBackend(hass, config)
        await backend.async_ensure_collection_exists(config, collection, 3)
        await backend.async_save_objects(
            config,
            collection,
            [MemoryEmbedding(first, [1.0, 0.0, 0.0]), MemoryEmbedding(second, [0.0, 1.0, 0.0])],
        )
        assert await backend.async_delete_objects(config, collection, "id", [first.id]) == 1

        reloaded_backend = FaissDbBackend(hass, config)
        recalled = await reloaded_backend.async_retrieve_objects(
            MemoryEmbedding,
            config,
            collection,
            [0.0, 1.0, 0.0],
            top_k=4,
        )
        assert recalled == [second]

    asyncio.run(run())
