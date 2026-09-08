import asyncio
from typing import Any, Dict, List
from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

class ABaseDbBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options
        self._lexical_object_cache: dict[str, tuple[Device | LlmTool | Memory, ...]] = {}
        self._lexical_cache_locks: dict[str, asyncio.Lock] = {}

    def cache_collection_objects(self, collection_name: str, objects: List[Device | LlmTool | Memory]) -> None:
        """Replace the in-memory metadata snapshot for a collection."""
        self._lexical_object_cache[collection_name] = tuple(objects)

    def invalidate_collection_cache(self, collection_name: str | None = None) -> None:
        """Invalidate one metadata snapshot or all collection snapshots."""
        if collection_name is None:
            self._lexical_object_cache.clear()
        else:
            self._lexical_object_cache.pop(collection_name, None)

    async def async_get_lexical_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str) -> List[Device | LlmTool | Memory]:
        """Return cached lexical metadata, loading it once when necessary."""
        cached = self._lexical_object_cache.get(collection_name)
        if cached is None:
            lock = self._lexical_cache_locks.setdefault(collection_name, asyncio.Lock())
            async with lock:
                cached = self._lexical_object_cache.get(collection_name)
                if cached is None:
                    objects = await self.async_list_objects(
                        object_type,
                        config_subentry,
                        collection_name,
                    )
                    if objects:
                        cached = tuple(objects)
                        self._lexical_object_cache[collection_name] = cached
                    else:
                        return []
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
