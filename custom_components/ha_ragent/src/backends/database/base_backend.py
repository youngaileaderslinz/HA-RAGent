import asyncio
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from functools import wraps

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


def invalidates_cache(method):
    """Invalidate before and after writes, including partially failed writes."""
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        collection = kwargs.get("collection_name", args[1] if len(args) > 1 else None)
        self.invalidate_collection_cache(collection)
        operation = asyncio.create_task(method(self, *args, **kwargs))
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Executor writes continue after caller cancellation. Wait before
            # the final invalidation so their completion cannot stale the cache.
            try:
                await operation
            except Exception:
                pass
            raise
        finally:
            self.invalidate_collection_cache(collection)
    return wrapped


class ABaseDbBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options
        self._lexical_object_cache: dict[str, tuple[Device | LlmTool | Memory, ...]] = {}
        self._lexical_cache_locks: dict[str, asyncio.Lock] = {}
        self._cache_revision = 0
        self._collection_revisions: dict[str, int] = {}
        self._collection_presence: dict[str, bool] = {}

    def cache_collection_objects(self, collection_name: str, objects: List[Device | LlmTool | Memory]) -> None:
        """Replace the in-memory metadata snapshot for a collection."""
        self._lexical_object_cache[collection_name] = tuple(objects)
        self._collection_presence[collection_name] = bool(objects)
        self._collection_revisions[collection_name] = self._collection_revisions.get(collection_name, 0) + 1

    def _collection_revision(self, collection_name: str) -> tuple[int, int]:
        return self._cache_revision, self._collection_revisions.get(collection_name, 0)

    def invalidate_collection_cache(self, collection_name: str | None = None, *, contents_changed: bool = True) -> None:
        """Invalidate one metadata snapshot or all collection snapshots."""
        if collection_name is None:
            self._cache_revision += 1
            self._collection_revisions.clear()
            self._lexical_object_cache.clear()
            if contents_changed:
                self._collection_presence.clear()
        else:
            self._collection_revisions[collection_name] = self._collection_revisions.get(collection_name, 0) + 1
            self._lexical_object_cache.pop(collection_name, None)
            if contents_changed:
                self._collection_presence.pop(collection_name, None)

    async def async_collection_has_objects(self, config_subentry: dict, collection_name: str) -> bool:
        """Cache an authoritative existence check until the collection changes."""
        lock = self._lexical_cache_locks.setdefault(collection_name, asyncio.Lock())
        async with lock:
            while collection_name not in self._collection_presence:
                revision = self._collection_revision(collection_name)
                present = await self._async_collection_has_objects(config_subentry, collection_name)
                if revision == self._collection_revision(collection_name):
                    self._collection_presence[collection_name] = present
            return self._collection_presence[collection_name]

    async def _async_collection_has_objects(self, config_subentry: dict, collection_name: str) -> bool:
        return bool(await self.async_list_objects(MemoryEmbedding, config_subentry, collection_name))

    async def async_flush(self) -> None:
        """Persist deferred backend bookkeeping before unloading."""

    async def async_close(self) -> None:
        """Finish deferred work and release backend lifecycle subscriptions."""
        await self.async_flush()

    async def async_get_lexical_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str) -> List[Device | LlmTool | Memory]:
        """Return cached lexical metadata, loading it once when necessary."""
        cached = self._lexical_object_cache.get(collection_name)
        if cached is None:
            lock = self._lexical_cache_locks.setdefault(collection_name, asyncio.Lock())
            async with lock:
                cached = self._lexical_object_cache.get(collection_name)
                while cached is None:
                    revision = self._collection_revision(collection_name)
                    objects = await self.async_list_objects(
                        object_type,
                        config_subentry,
                        collection_name,
                    )
                    if revision == self._collection_revision(collection_name):
                        cached = tuple(objects)
                        self._lexical_object_cache[collection_name] = cached
                        self._collection_presence[collection_name] = bool(cached)
                    else:
                        cached = self._lexical_object_cache.get(collection_name)
        return list(cached)

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the name of the database backend."""
        return "DB"
    
    @staticmethod
    @abstractmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        """Validate the connection to the database backend."""
        raise NotImplementedError()

    @abstractmethod
    async def async_ensure_collection_exists(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        """Create a collection without removing existing objects."""
        raise NotImplementedError()

    @abstractmethod
    async def async_cleanup_database(self) -> None:
        """Cleanup the database, removing all collections and data."""
        raise NotImplementedError()

    @abstractmethod
    async def async_reset_collection(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        """Delete and recreate a collection."""
        raise NotImplementedError()

    @abstractmethod
    async def async_cleanup_collection(self, config_subentry: dict, collection_name: str) -> None:
        """Delete all objects in a collection."""
        raise NotImplementedError()

    @abstractmethod
    async def async_upsert_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_embeddings: List[MemoryEmbedding]) -> None:
        """Insert objects or replace records with the same id."""
        raise NotImplementedError()

    @abstractmethod
    async def async_save_objects(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding]) -> None:
        """Insert objects without replacing existing records."""
        raise NotImplementedError()

    async def async_retrieve_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device | LlmTool | Memory]:
        """Retrieve objects, discarding scores from the canonical result."""
        results = await self.async_retrieve_scored_objects(object_type, config_subentry, collection_name, query_embedding, top_k)
        return [result.item for result in results]

    @abstractmethod
    async def async_retrieve_scored_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[ScoredResult[Device | LlmTool | Memory]]:
        """Retrieve ranked objects with normalized confidence."""
        raise NotImplementedError()

    @abstractmethod
    async def async_list_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str) -> List[Device | LlmTool | Memory]:
        """List all objects in a collection."""
        raise NotImplementedError()

    async def async_increment_memory_retrieval_counts(self, config_subentry: dict, collection_name: str, memory_ids: List[str]) -> None:
        """Increment retrieval counts for memory records."""
        raise NotImplementedError()

    @abstractmethod
    async def async_delete_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_ids: List[str]) -> int:
        """Delete objects by id."""
        raise NotImplementedError()
