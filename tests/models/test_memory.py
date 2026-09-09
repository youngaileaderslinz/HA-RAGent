from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.const import CONF_VECTOR_DB_NAME, DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.homeassistant.tools.forget_fact import RAGentForgetTool
from custom_components.ha_ragent.src.homeassistant.tools.remember_fact import RAGentRememberTool
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.translation import RAGentTranslations


class FakeEmbedder:
    async def async_embed_text(self, config: dict[str, Any], text: str) -> list[float]:
        return [1.0, float(len(text)), 0.5]


class FakeVectorDb:
    def __init__(self) -> None:
        self.objects: dict[str, list[MemoryEmbedding]] = {}

    def async_collection_has_objects(self):
        async def check(config: dict[str, Any], collection_name: str) -> bool:
            return bool(self.objects.get(collection_name))
        return check

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

    async def async_retrieve_objects(self, object_type, config_subentry: dict[str, Any], collection_name: str, query_embedding: list[float], top_k: int):
        return [object_type.parse_object(item.to_dict()) for item in self.objects.get(collection_name, [])[:top_k]]


def create_memory_hass() -> tuple[SimpleNamespace, FakeVectorDb]:
    RAGentTranslations._load("en")
    vector_db = FakeVectorDb()
    entry = SimpleNamespace(
        subentries={"agent": SimpleNamespace(data={"model": "embed"})},
        embedder_backend=FakeEmbedder(),
        vector_db_backend=vector_db,
    )
    hass = SimpleNamespace(data={DOMAIN: {"entry": entry}})
    return hass, vector_db


def test_memory_model_round_trip() -> None:
    memory = Memory(id="0123456789abcdef", content="The reading lamp is beside the sofa.", created_at="2026-09-01T12:00:00+00:00")
    embedding = MemoryEmbedding(memory, [1.0, 2.0])

    assert MemoryEmbedding.parse_object(embedding.to_dict()) == memory
    assert "reading lamp" in memory.to_embedding_text()
    assert memory.to_dict()["id"] == memory.id
    assert embedding.to_dict()["id"] == memory.id
    assert "memory_id" not in embedding.to_dict()


def test_memory_manager_remember_recall_replace_and_forget() -> None:
    async def run() -> None:
        hass, vector_db = create_memory_hass()
        manager = MemoryManager(hass, "entry", "agent")

        first = await manager.async_remember("  The reading   lamp is beside the sofa.  ")
        replacement = await manager.async_remember("The reading lamp is beside the sofa.")

        assert first.id == replacement.id
        assert replacement.content == "The reading lamp is beside the sofa."
        assert len(vector_db.objects[manager.collection_name]) == 1
        assert await manager.async_recall([1.0, 1.0, 1.0], 4) == [replacement]
        assert await manager.async_forget(replacement.id) is True
        assert await manager.async_forget(replacement.id) is False
        assert await manager.async_recall([1.0, 1.0, 1.0], 4) == []

    asyncio.run(run())


def test_memory_tools() -> None:
    async def run() -> None:
        hass, _ = create_memory_hass()
        remember = RAGentRememberTool(hass, "entry", "agent")
        remember_result = await remember.async_call(
            SimpleNamespace(tool_args={"memory": "The thermostat target is 21 C."})
        )

        assert remember_result["success"] is True
        memory_id = remember_result["memory_id"]

        forget = RAGentForgetTool(hass, "entry", "agent")
        forget_result = await forget.async_call(SimpleNamespace(tool_args={"memory_id": memory_id}))
        assert forget_result == {
            "success": True,
            "memory_id": memory_id,
            "forgotten": True,
        }

        missing_result = await forget.async_call(SimpleNamespace(tool_args={"memory_id": memory_id}))
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
