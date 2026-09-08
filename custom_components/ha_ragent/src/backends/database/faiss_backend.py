import logging
import os
from typing import Any, Dict, List

from homeassistant.core import HomeAssistant

import faiss
import numpy as np
import pickle

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

from custom_components.ha_ragent.src.const import (
    CONF_VECTOR_DB_NAME
)

_logger = logging.getLogger(__name__)

class FaissDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._storage_path = hass.config.path("ha_ragent_storage")
        self.db_name = self.client_options.get(CONF_VECTOR_DB_NAME)

        os.makedirs(os.path.join(self._storage_path, self.db_name), exist_ok=True)
        
        self._indices: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def get_name() -> str:
        return f"{ABaseDbBackend.get_name()}: Local FAISS"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        return None
    
    def _get_paths(self, collection_name: str):
        index_path = os.path.join(self._storage_path, self.db_name, f"{collection_name}.index")
        meta_path = os.path.join(self._storage_path, self.db_name, f"{collection_name}.pkl")
        return index_path, meta_path

    def _load_collection(self, collection_name: str, embedding_length: int = 1536):
        """Lazy load or initialize a cosine-similarity index."""
        idx_path, meta_path = self._get_paths(collection_name)
        
        if collection_name not in self._indices:
            if os.path.exists(idx_path) and os.path.exists(meta_path):
                try:
                    self._indices[collection_name] = faiss.read_index(idx_path)
                    with open(meta_path, "rb") as f:
                        self._metadata[collection_name] = pickle.load(f)

                except Exception as e:
                    _logger.error(f"Failed to load collection {collection_name}: {e}")
                    self._create_empty(collection_name, embedding_length)
            else:
                self._create_empty(collection_name, embedding_length)

    def _create_empty(self, collection_name: str, embedding_length: int):
        # Cosine similarity is inner product over unit-normalized vectors.
        self._indices[collection_name] = faiss.IndexFlatIP(embedding_length)
        self._metadata[collection_name] = []

    def _save_to_disk(self, collection_name: str):
        idx_path, meta_path = self._get_paths(collection_name)
        faiss.write_index(self._indices[collection_name], idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata[collection_name], f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_device_embeddings(self, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding]):
        if not device_embeddings:
            return

        dim = len(device_embeddings[0].vector_embedding)
        self._load_collection(collection_name, dim)

        vectors = np.asarray([emb.vector_embedding for emb in device_embeddings], dtype=np.float32)
        faiss.normalize_L2(vectors)
        metadatas = [emb.to_dict() for emb in device_embeddings]

        self._indices[collection_name].add(vectors)
        self._metadata[collection_name].extend(metadatas)
        
        self._save_to_disk(collection_name)
        _logger.info(f"Saved {len(device_embeddings)} embeddings to local FAISS index: {collection_name}")

    def _query_scored_devices(self, collection_name: str, query_embedding: List[float], top_k: int):
        self._load_collection(collection_name, len(query_embedding))
        
        query_vector = np.asarray([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        scores, indices = self._indices[collection_name].search(query_vector, top_k)

        metadata = self._metadata[collection_name]
        return [
            (metadata[idx], min(1.0, max(0.0, (float(score) + 1.0) / 2.0)))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and idx < len(metadata)
        ]

    def _cleanup_database(self):
        db_path = os.path.join(self._storage_path, self.db_name)
        for filename in os.listdir(db_path):
            if filename.endswith((".index", ".pkl")):
                os.remove(os.path.join(db_path, filename))

        os.rmdir(db_path)
        if not os.listdir(self._storage_path):
            os.rmdir(self._storage_path)
        self._indices.clear()
        self._metadata.clear()
        
    def _reset_collection(self, collection_name: str, embedding_length: int):
        self._cleanup_collection(collection_name)

        self._create_empty(collection_name, embedding_length)
        self._save_to_disk(collection_name)

    def _ensure_collection(self, collection_name: str, embedding_length: int):
        self._load_collection(collection_name, embedding_length)
        index_path, metadata_path = self._get_paths(collection_name)
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            self._save_to_disk(collection_name)

    def _delete_objects(self, collection_name: str, id_field: str, object_ids: List[str]) -> int:
        index_path, metadata_path = self._get_paths(collection_name)
        if collection_name not in self._indices and not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            return 0

        self._load_collection(collection_name)
        ids = set(object_ids)
        current_metadata = self._metadata[collection_name]
        remaining_metadata = [item for item in current_metadata if item.get(id_field) not in ids]
        deleted = len(current_metadata) - len(remaining_metadata)
        if not deleted:
            return 0

        dimension = self._indices[collection_name].d
        self._create_empty(collection_name, dimension)
        if remaining_metadata:
            vectors = np.asarray([item["vector_embedding"] for item in remaining_metadata], dtype=np.float32)
            faiss.normalize_L2(vectors)
            self._indices[collection_name].add(vectors)
            self._metadata[collection_name] = remaining_metadata

        self._save_to_disk(collection_name)
        return deleted

    def _upsert_object_embeddings(self, collection_name: str, id_field: str, object_embeddings: List[MemoryEmbedding]):
        if not object_embeddings:
            return

        dimension = len(object_embeddings[0].vector_embedding)
        self._load_collection(collection_name, dimension)
        incoming_metadata = [embedding.to_dict() for embedding in object_embeddings]
        incoming_ids = {item[id_field] for item in incoming_metadata}
        retained_metadata = [
            item for item in self._metadata[collection_name]
            if item.get(id_field) not in incoming_ids
        ]
        combined_metadata = [*retained_metadata, *incoming_metadata]
        vectors = np.asarray([item["vector_embedding"] for item in combined_metadata], dtype=np.float32)
        faiss.normalize_L2(vectors)

        self._create_empty(collection_name, dimension)
        self._indices[collection_name].add(vectors)
        self._metadata[collection_name] = combined_metadata
        self._save_to_disk(collection_name)

    def _cleanup_collection(self, collection_name: str):
        self._indices.pop(collection_name, None)
        self._metadata.pop(collection_name, None)

        idx_path, meta_path = self._get_paths(collection_name)
        for path in (idx_path, meta_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as err:
                    _logger.warning(f"Failed to remove stale FAISS file {path}: {err}")

    def _list_objects(self, object_type, collection_name: str):
        self._load_collection(collection_name)
        return [object_type.parse_object(metadata) for metadata in self._metadata[collection_name]]

    def _increment_memory_retrieval_counts(self, collection_name: str, memory_ids: List[str]):
        self._load_collection(collection_name)
        ids = set(memory_ids)
        changed = False
        for metadata in self._metadata[collection_name]:
            if metadata.get("id") in ids:
                metadata["retrieval_count"] = int(metadata.get("retrieval_count", 0) or 0) + 1
                changed = True

        if changed:
            self._save_to_disk(collection_name)

    async def async_cleanup_database(self) -> None:
        try:
            await self.hass.async_add_executor_job(self._cleanup_database)
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_reset_collection(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._reset_collection, collection_name, embedding_length)
        except Exception as e:
            _logger.error(f"Error resetting collection: {e}", exc_info=True)

    async def async_ensure_collection_exists(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._ensure_collection, collection_name, embedding_length)
        except Exception as e:
            _logger.error(f"Error ensuring collection: {e}", exc_info=True)
            raise
    
    async def async_cleanup_collection(self, config_subentry: dict, collection_name: str) -> None:
        try:
            await self.hass.async_add_executor_job(self._cleanup_collection, collection_name)
        except Exception as e:
            _logger.error(f"Error cleaning up collection: {e}", exc_info=True)

    async def async_save_objects(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding]) -> None:
        try:
            await self.hass.async_add_executor_job(self._save_device_embeddings, collection_name, device_embeddings)
        except Exception as e:
            _logger.error(f"Error saving device embeddings: {e}", exc_info=True)
            raise

    async def async_upsert_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_embeddings: List[MemoryEmbedding]) -> None:
        try:
            await self.hass.async_add_executor_job(
                self._upsert_object_embeddings,
                collection_name,
                id_field,
                object_embeddings,
            )
        except Exception as e:
            _logger.error(f"Error upserting objects: {e}", exc_info=True)
            raise

    async def async_retrieve_scored_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[ScoredResult[Device | LlmTool | Memory]]:
        try:
            results = await self.hass.async_add_executor_job(self._query_scored_devices, collection_name, query_embedding, top_k)
            return [
                ScoredResult(object_type.parse_object(metadata), score, rank)
                for rank, (metadata, score) in enumerate(results, start=1)
            ]
        except Exception as err:
            _logger.error(f"Error retrieving scored objects: {err}", exc_info=True)
            return []

    async def async_list_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str) -> List[Device | LlmTool | Memory]:
        try:
            return await self.hass.async_add_executor_job(self._list_objects, object_type, collection_name)
        except Exception as e:
            _logger.error(f"Error listing objects: {e}", exc_info=True)
            return []

    async def async_increment_memory_retrieval_counts(self, config_subentry: dict, collection_name: str, memory_ids: List[str]) -> None:
        await self.hass.async_add_executor_job(self._increment_memory_retrieval_counts, collection_name, memory_ids)

    async def async_delete_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_ids: List[str]) -> int:
        try:
            return await self.hass.async_add_executor_job(self._delete_objects, collection_name, id_field, object_ids)
        except Exception as e:
            _logger.error(f"Error deleting objects: {e}", exc_info=True)
            raise

