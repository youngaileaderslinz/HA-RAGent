from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    CONF_MAX_MEMORY_ENTRIES,
    DOMAIN,
    RAGENT_MEMORY_LOCKS,
)
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.utils import get_setting_value

_logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.subentry_id = subentry_id

    @property
    def collection_name(self) -> str:
        return f"memories_{self.subentry_id}"

    def _get_entry_and_config(self) -> tuple[Any | None, dict[str, Any]]:
        entry = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
        if entry is None:
            _logger.error("The HA-RAGent integration entry is not available.")
            return None, {}

        subentry = entry.subentries.get(self.subentry_id)
        if subentry is None:
            _logger.error("The HA-RAGent agent entry is not available.")
            return None, {}
        
        return entry, dict(subentry.data)

    def _get_lock(self) -> asyncio.Lock:
        domain_data = self.hass.data.setdefault(DOMAIN, {})
        locks = domain_data.setdefault(RAGENT_MEMORY_LOCKS, {})
        return locks.setdefault(self.subentry_id, asyncio.Lock())

    @staticmethod
    def normalize_content(content: str) -> str:
        return " ".join(content.split()).strip()

    @staticmethod
    def memory_id_for_content(content: str) -> str:
        normalized = MemoryManager.normalize_content(content).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    async def async_remember(self, content: str) -> Memory | None:
        normalized_content = self.normalize_content(content)
        if not normalized_content:
            _logger.error("Memory content must not be empty.")
            return None

        entry, config = self._get_entry_and_config()
        if entry is None:
            return None
        memory = Memory(
            id=self.memory_id_for_content(normalized_content),
            content=normalized_content,
            created_at=dt_util.utcnow().isoformat(),
        )
        vector = await entry.embedder_backend.async_embed_text(config, memory.to_embedding_text())
        if not vector:
            _logger.error("The embedding backend returned an empty memory embedding.")
            return None

        async with self._get_lock():
            await entry.vector_db_backend.async_ensure_collection_exists(
                config,
                self.collection_name,
                len(vector),
            )
            await entry.vector_db_backend.async_upsert_objects(
                config,
                self.collection_name,
                "id",
                [MemoryEmbedding(memory, vector)],
            )

            list_objects = getattr(entry.vector_db_backend, "async_list_objects", None)
            if list_objects:
                memories = await list_objects(
                    object_type=MemoryEmbedding,
                    config_subentry=config,
                    collection_name=self.collection_name,
                )
                max_entries = int(get_setting_value(CONF_MAX_MEMORY_ENTRIES, config))
                if max_entries > 0 and len(memories) > max_entries:
                    memories_to_delete = sorted(
                        (item for item in memories if isinstance(item, Memory)),
                        key=lambda item: (item.retrieval_count, item.created_at),
                    )[: len(memories) - max_entries]
                    await entry.vector_db_backend.async_delete_objects(
                        config,
                        self.collection_name,
                        "id",
                        [memory.id for memory in memories_to_delete],
                    )

        return memory

    async def async_forget(self, memory_id: str) -> bool:
        entry, config = self._get_entry_and_config()
        if entry is None:
            return False
        async with self._get_lock():
            deleted = await entry.vector_db_backend.async_delete_objects(
                config,
                self.collection_name,
                "id",
                [memory_id],
            )
        return deleted > 0

    async def async_recall(self, query_embedding: list[float], limit: int) -> list[Memory]:
        if limit <= 0:
            return []
        entry, config = self._get_entry_and_config()
        if entry is None:
            return []
        async with self._get_lock():
            memories = await entry.vector_db_backend.async_retrieve_objects(
                object_type=MemoryEmbedding,
                config_subentry=config,
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                top_k=limit,
            )
            memory_ids = [memory.id for memory in memories if isinstance(memory, Memory) and memory.id]
            increment_counts = getattr(entry.vector_db_backend, "async_increment_memory_retrieval_counts", None)
            if memory_ids and increment_counts:
                try:
                    await increment_counts(config, self.collection_name, memory_ids)
                except Exception as err:
                    _logger.warning("Failed to update memory retrieval counts: %s", err)
        return [memory for memory in memories if isinstance(memory, Memory) and memory.id and memory.content]

    async def async_list(self) -> list[Memory]:
        entry, config = self._get_entry_and_config()
        if entry is None:
            return []
        async with self._get_lock():
            memories = await entry.vector_db_backend.async_list_objects(
                object_type=MemoryEmbedding,
                config_subentry=config,
                collection_name=self.collection_name,
            )
        return [memory for memory in memories if isinstance(memory, Memory) and memory.id and memory.content]
