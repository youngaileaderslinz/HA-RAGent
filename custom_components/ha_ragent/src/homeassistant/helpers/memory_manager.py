from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.ha_ragent.src.const import (
    CONF_MAX_MEMORY_ENTRIES,
    DOMAIN,
    MEMORY_ABSOLUTE_CONFIDENCE_FLOOR,
    MEMORY_RELATIVE_CONFIDENCE_FLOOR,
    RAGENT_MEMORY_LOCKS,
)
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.utils import get_setting_value

_logger = BaseLogger(__name__)

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
            _logger.log_string(logging.ERROR, "The HA-RAGent integration entry is not available.")
            return None, {}

        subentry = entry.subentries.get(self.subentry_id)
        if subentry is None:
            _logger.log_string(logging.ERROR, "The HA-RAGent agent entry is not available.")
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

    async def async_export_memories(self) -> str:
        """Export this agent's memories as portable JSON without embeddings."""
        entry, config = self._get_entry_and_config()
        if entry is None:
            raise ValueError("The HA-RAGent agent subentry is not available.")
        async with self._get_lock():
            memories = await entry.vector_db_backend.async_list_objects(
                MemoryEmbedding, config, self.collection_name,
            )
            return json.dumps(
                {"memories": [memory.to_dict() for memory in sorted(memories, key=lambda item: item.id)]},
                ensure_ascii=False,
            )

    async def async_import_memories(self, payload: str | dict[str, Any]) -> int:
        """Validate an export, rebuild embeddings, and merge memories by ID."""
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except (TypeError, ValueError) as err:
            raise ValueError("Memories must be valid JSON.") from err
        if not isinstance(data, dict) or not isinstance(data.get("memories"), list):
            raise ValueError('Expected a JSON object containing a "memories" list.')

        memories: list[Memory] = []
        seen_ids: set[str] = set()
        for record in data["memories"]:
            if not isinstance(record, dict):
                raise ValueError("Each memory must be a JSON object.")
            memory_id = record.get("id")
            content = record.get("content")
            created_at = record.get("created_at")
            retrieval_count = record.get("retrieval_count", 0)
            if not isinstance(memory_id, str) or not re.fullmatch(r"[0-9a-f]{16}", memory_id):
                raise ValueError("Each memory ID must be 16 lowercase hexadecimal characters.")
            if memory_id in seen_ids:
                raise ValueError(f"Duplicate memory ID: {memory_id}.")
            if not isinstance(content, str) or not content.strip() or len(content) > 1000:
                raise ValueError("Memory content must contain 1 to 1000 characters.")
            if not isinstance(created_at, str) or dt_util.parse_datetime(created_at) is None:
                raise ValueError("Each memory must have a valid created_at timestamp.")
            if type(retrieval_count) is not int or retrieval_count < 0:
                raise ValueError("Memory retrieval_count must be a non-negative integer.")
            seen_ids.add(memory_id)
            memories.append(Memory(memory_id, content, created_at, retrieval_count))

        entry, config = self._get_entry_and_config()
        if entry is None:
            raise ValueError("The HA-RAGent agent subentry is not available.")
        embeddings: list[MemoryEmbedding] = []
        for memory in memories:
            vector = await entry.embedder_backend.async_embed_text(
                config, memory.to_embedding_text(entry.translations), input_type="document",
            )
            if not vector:
                raise ValueError("The embedding backend returned an empty memory embedding.")
            embeddings.append(MemoryEmbedding(memory, vector))

        if embeddings:
            async with self._get_lock():
                await entry.vector_db_backend.async_ensure_collection_exists(
                    config, self.collection_name, len(embeddings[0].vector_embedding),
                )
                await entry.vector_db_backend.async_upsert_objects(
                    config, self.collection_name, "id", embeddings,
                )
                await self._async_prune_memories(entry, config)
        return len(memories)

    async def _async_prune_memories(self, entry: Any, config: dict[str, Any]) -> None:
        """Apply the storage limit while the caller holds the memory lock."""
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
                    config, self.collection_name, "id",
                    [memory.id for memory in memories_to_delete],
                )

    async def async_remember(self, content: str) -> Memory | None:
        normalized_content = self.normalize_content(content)
        if not normalized_content:
            _logger.log_string(logging.ERROR, "Memory content must not be empty.")
            return None

        entry, config = self._get_entry_and_config()
        if entry is None:
            return None
        memory = Memory(
            id=self.memory_id_for_content(normalized_content),
            content=normalized_content,
            created_at=dt_util.utcnow().isoformat(),
        )
        vector = await entry.embedder_backend.async_embed_text(
            config, memory.to_embedding_text(entry.translations), input_type="document",
        )
        if not vector:
            _logger.log_string(logging.ERROR, "The embedding backend returned an empty memory embedding.")
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

            await self._async_prune_memories(entry, config)

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

    @staticmethod
    def select_confident_memories(
        results: list[ScoredResult[Memory]],
        minimum: int,
        maximum: int,
    ) -> list[Memory]:
        """Select valid memories whose vector confidence is close to the best match."""
        maximum = max(0, int(maximum))
        minimum = min(maximum, max(0, int(minimum)))
        if maximum <= 0:
            return []

        ranked = [
            result
            for result in results
            if isinstance(result.item, Memory) and result.item.id and result.item.content
        ][:maximum]
        if not ranked:
            return []

        top_score = max(0.0, float(ranked[0].score))
        confidence_floor = max(
            MEMORY_ABSOLUTE_CONFIDENCE_FLOOR,
            top_score * MEMORY_RELATIVE_CONFIDENCE_FLOOR,
        )
        selected = [
            result.item
            for result in ranked
            if float(result.score) >= confidence_floor
        ]
        if len(selected) < minimum:
            selected_ids = {memory.id for memory in selected}
            for result in ranked:
                if result.item.id not in selected_ids:
                    selected.append(result.item)
                    selected_ids.add(result.item.id)
                if len(selected) >= minimum:
                    break
        return selected[:maximum]

    async def async_recall(
        self,
        query_embedding: list[float] | QueryEmbedding,
        minimum: int,
        maximum: int,
    ) -> list[Memory]:
        try:
            maximum = max(0, int(maximum))
            minimum = min(maximum, max(0, int(minimum)))
        except (TypeError, ValueError, OverflowError):
            return []
        if maximum <= 0:
            return []
        entry, config = self._get_entry_and_config()
        if entry is None:
            return []
        async with self._get_lock():
            if not await entry.vector_db_backend.async_collection_has_objects(
                config, self.collection_name
            ):
                return []
            if isinstance(query_embedding, QueryEmbedding):
                query_embedding = await query_embedding.get()
            if not query_embedding:
                return []
            results = await entry.vector_db_backend.async_retrieve_scored_objects(
                object_type=MemoryEmbedding,
                config_subentry=config,
                collection_name=self.collection_name,
                query_embedding=query_embedding,
                top_k=maximum,
            )
            memories = self.select_confident_memories(results, minimum, maximum)
            memory_ids = [memory.id for memory in memories]
            increment_counts = getattr(entry.vector_db_backend, "async_increment_memory_retrieval_counts", None)
            if memory_ids and increment_counts:
                try:
                    await increment_counts(config, self.collection_name, memory_ids)
                except Exception as err:
                    _logger.log_string(logging.WARNING, f"Failed to update memory retrieval counts: {err}")
        _logger.log_payload(
            "retrieval.memory_exposure",
            configured_minimum=minimum,
            configured_maximum=maximum,
            vector_scores=[
                (result.item.id, round(float(result.score), 6))
                for result in results
                if isinstance(result.item, Memory)
            ],
            selected_memory_ids=memory_ids,
        )
        return memories
