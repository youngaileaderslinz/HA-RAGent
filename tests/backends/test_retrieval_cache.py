import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.backends.database.chromadb_backend import ChromaDbBackend
from custom_components.ha_ragent.src.backends.database.mongodb_backend import MongoDbBackend
from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.database.base_backend import invalidates_cache
from custom_components.ha_ragent.src.const import CONF_VECTOR_DB_NAME, DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding


def make_backend(tmp_path):
    hass = SimpleNamespace(config=SimpleNamespace(path=lambda name: str(tmp_path / name)),
                           async_add_executor_job=asyncio.to_thread, data={})
    return hass, FaissDbBackend(hass, {CONF_VECTOR_DB_NAME: "review"})


def record(name="one"):
    return MemoryEmbedding(Memory(name, name, "2026-01-01"), [1.0, 0.0])


def test_cache_tracks_save_upsert_delete_reset_and_cleanup(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        async def listed():
            return [m.id for m in await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")]
        assert await listed() == []
        assert not await db.async_collection_has_objects({}, "memories")
        await db.async_save_objects({}, "memories", [record()])
        assert await listed() == ["one"]
        assert await db.async_collection_has_objects({}, "memories")
        await db.async_upsert_objects({}, "memories", "id", [record("two")])
        assert await listed() == ["one", "two"]
        await db.async_delete_objects({}, "memories", "id", ["one"])
        assert await listed() == ["two"]
        await db.async_reset_collection({}, "memories", 2)
        assert await listed() == []
        await db.async_save_objects({}, "memories", [record()])
        assert await listed() == ["one"]
        await db.async_cleanup_collection({}, "memories")
        assert await listed() == []
        await db.async_save_objects({}, "memories", [record()])
        assert await listed() == ["one"]
        await db.async_cleanup_database()
        assert await listed() == []
    asyncio.run(run())


def test_empty_snapshots_are_cached_but_failures_are_not(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        db.async_list_objects = AsyncMock(side_effect=[RuntimeError("offline"), [], [record().embedded_object]])
        with pytest.raises(RuntimeError):
            await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")
        assert await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories") == []
        assert await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories") == []
        assert db.async_list_objects.await_count == 2
        db.invalidate_collection_cache("memories")
        assert await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")
    asyncio.run(run())


@pytest.mark.parametrize("backend_class", [ChromaDbBackend, MongoDbBackend])
def test_remote_delete_invalidates_lexical_and_presence_caches(backend_class):
    async def run():
        db = backend_class.__new__(backend_class)
        ABaseDbBackend.__init__(db, SimpleNamespace(async_add_executor_job=asyncio.to_thread), {})
        objects = [record().embedded_object]
        def delete(*_args, **_kwargs):
            objects.clear()
            return 1
        if backend_class is ChromaDbBackend:
            db._delete_objects = delete
        else:
            db._get_connection = lambda: SimpleNamespace(close=AsyncMock())
            db._async_collection_exists = AsyncMock(return_value=True)
            async def delete_many(*_args, **_kwargs):
                return SimpleNamespace(deleted_count=delete())
            db._get_collection = lambda *_args: SimpleNamespace(delete_many=delete_many)
        db.async_list_objects = AsyncMock(side_effect=lambda *_args: list(objects))
        db._async_collection_has_objects = AsyncMock(side_effect=lambda *_args: bool(objects))
        db.cache_collection_objects("memories", objects)
        assert await db.async_collection_has_objects({}, "memories")
        assert await db.async_delete_objects(config_subentry={}, collection_name="memories", id_field="id", object_ids=["one"]) == 1
        assert await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories") == []
        assert not await db.async_collection_has_objects({}, "memories")
    asyncio.run(run())


def test_failed_mutation_does_not_leave_old_cache(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        await db.async_save_objects({}, "memories", [record()])
        assert await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")
        original = db._save_device_embeddings
        def partial_write(*args):
            original(*args)
            raise RuntimeError("failed after write")
        db._save_device_embeddings = partial_write
        with pytest.raises(RuntimeError):
            await db.async_save_objects({}, "memories", [record("two")])
        assert [m.id for m in await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")] == ["one", "two"]
    asyncio.run(run())


def test_failed_presence_probe_is_retried(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        db._async_collection_has_objects = AsyncMock(side_effect=[RuntimeError("offline"), True])
        with pytest.raises(RuntimeError):
            await db.async_collection_has_objects({}, "memories")
        assert await db.async_collection_has_objects({}, "memories")
        assert await db.async_collection_has_objects({}, "memories")
        assert db._async_collection_has_objects.await_count == 2
    asyncio.run(run())


def test_cancelled_write_invalidates_after_storage_finishes(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        started, release = asyncio.Event(), asyncio.Event()
        objects = [record("old").embedded_object]
        @invalidates_cache
        async def write(backend, config, collection_name):
            started.set()
            await release.wait()
            objects[:] = [record("new").embedded_object]
        db.async_list_objects = AsyncMock(side_effect=lambda *_args: list(objects))
        task = asyncio.create_task(write(db, {}, "memories"))
        await started.wait()
        task.cancel()
        # A read overlapping the write may observe old data, but must not
        # leave that snapshot cached after the write completes.
        await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [m.id for m in await db.async_get_lexical_objects(MemoryEmbedding, {}, "memories")] == ["new"]
    asyncio.run(run())


def test_read_started_before_invalidation_cannot_publish_stale_snapshot(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        started, release = asyncio.Event(), asyncio.Event()
        async def read(*_args):
            started.set()
            await release.wait()
            return [record("old").embedded_object]
        db.async_list_objects = AsyncMock(side_effect=read)
        task = asyncio.create_task(db.async_get_lexical_objects(MemoryEmbedding, {}, "memories"))
        await started.wait()
        db.invalidate_collection_cache("memories")
        db.cache_collection_objects("memories", [record("new").embedded_object])
        release.set()
        assert [m.id for m in await task] == ["new"]
    asyncio.run(run())


def test_empty_memory_recall_skips_embedding_and_observes_later_writes(tmp_path):
    async def run():
        hass, db = make_backend(tmp_path)
        hass.data = {DOMAIN: {"entry": SimpleNamespace(subentries={"agent": SimpleNamespace(data={})}, vector_db_backend=db)}}
        manager = MemoryManager(hass, "entry", "agent")
        embed = AsyncMock(return_value=[1.0, 0.0])
        assert await manager.async_recall(QueryEmbedding(embed), 3) == []
        embed.assert_not_awaited()
        await db.async_save_objects({}, manager.collection_name, [record()])
        assert await manager.async_recall(QueryEmbedding(embed), 3)
        embed.assert_awaited_once()
        await db.async_delete_objects({}, manager.collection_name, "id", ["one"])
        assert await manager.async_recall(QueryEmbedding(embed), 3) == []
        embed.assert_awaited_once()
        await db.async_flush()
    asyncio.run(run())


def test_counter_updates_batch_metadata_only_and_survive_flush(tmp_path):
    async def run():
        hass, db = make_backend(tmp_path)
        await db.async_save_objects({}, "memories", [record()])
        with patch("custom_components.ha_ragent.src.backends.database.faiss_backend.faiss.write_index") as write_index, patch.object(db, "_save_metadata_to_disk", wraps=db._save_metadata_to_disk) as write_metadata:
            await db.async_increment_memory_retrieval_counts({}, "memories", ["one"])
            await db.async_increment_memory_retrieval_counts({}, "memories", ["one"])
            write_index.assert_not_called()
            write_metadata.assert_not_called()
            assert (await db.async_list_objects(MemoryEmbedding, {}, "memories"))[0].retrieval_count == 2
            await db.async_flush()
            write_index.assert_not_called()
            write_metadata.assert_called_once()
        restored = FaissDbBackend(hass, {CONF_VECTOR_DB_NAME: "review"})
        assert (await restored.async_list_objects(MemoryEmbedding, {}, "memories"))[0].retrieval_count == 2
    asyncio.run(run())


def test_pending_counter_flush_does_not_recreate_deleted_collection(tmp_path):
    async def run():
        _, db = make_backend(tmp_path)
        await db.async_save_objects({}, "memories", [record()])
        await db.async_increment_memory_retrieval_counts({}, "memories", ["one"])
        await db.async_cleanup_collection({}, "memories")
        await db.async_flush()
        assert not any(Path(p).exists() for p in db._get_paths("memories"))
    asyncio.run(run())
