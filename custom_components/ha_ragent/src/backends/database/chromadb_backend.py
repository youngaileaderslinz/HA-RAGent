import logging
import asyncio
from typing import Any, Dict, List, Optional
from uuid import uuid4

from homeassistant.core import HomeAssistant

from chromadb import Client
from chromadb.config import Settings

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult

from custom_components.ha_ragent.src.const import (
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
)

_logger = logging.getLogger(__name__)

class ChromaDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._settings = Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=self.client_options.get(CONF_VECTOR_DB_HOST),
                chroma_server_http_port=self.client_options.get(CONF_VECTOR_DB_PORT),
                chroma_server_ssl_enabled=self.client_options.get(CONF_VECTOR_DB_SSL),
            )
            
        self._client = None

    @staticmethod
    def get_name() -> str:
        return f"{ABaseDbBackend.get_name()}: ChromaDB"

    @staticmethod
    def _validate_connection(client_options: dict[str, Any]) -> Optional[str]:
        host = client_options.get(CONF_VECTOR_DB_HOST)
        port = client_options.get(CONF_VECTOR_DB_PORT)
        ssl = client_options.get(CONF_VECTOR_DB_SSL)
        settings = Settings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=host,
            chroma_server_http_port=port,
            chroma_server_ssl_enabled=ssl,
        )
        client = Client(settings=settings)
        client.list_collections()
        return None

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            return await hass.async_add_executor_job(ChromaDbBackend._validate_connection, user_input)
        except Exception as e:
            _logger.error(f"Error validating ChromaDB connection: {e}", exc_info=True)
            return str(e)

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(settings=self._settings)

        return self._client
    
    def _collection_exists(self, client: Client, collection_name: str) -> bool:
        collections = [col.name for col in client.list_collections()]
        return collection_name in collections

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove values that are invalid or redundant in Chroma metadata."""
        return {
            key: value
            for key, value in metadata.items()
            if key != "vector_embedding"
            and value is not None
            and not (isinstance(value, list) and not value)
        }
    
    def _save_device_embeddings(self, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding]):
        collection = self._get_client().get_or_create_collection(name=collection_name)

        metadatas = [
            self._sanitize_metadata(embedding.to_dict())
            for embedding in device_embeddings
        ]

        ids = [str(uuid4()) for emb in device_embeddings]
        embeddings = [emb.vector_embedding for emb in device_embeddings]
        collection.add(ids=ids, metadatas=metadatas, embeddings=embeddings)
        _logger.info(f"Saved {len(device_embeddings)} device embeddings to collection {collection_name}")

    def _query_scored_devices(self, collection_name: str, query_embedding: List[float], top_k: int):
        client = self._get_client()
        if not self._collection_exists(client, collection_name):
            return {"metadatas": []}
        
        collection = client.get_collection(name=collection_name)
        object_count = collection.count()
        if object_count == 0:
            return {"metadatas": []}
        
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, object_count),
            include=["metadatas", "distances"],
        )

    def _reset_collection(self, collection_name: str):
        try:
            client = self._get_client()
            collection = client.get_or_create_collection(name=collection_name)

            existing_ids = collection.get(include=[]).get("ids", [])
            if existing_ids:
                collection.delete(ids=existing_ids)

            _logger.info(f"Collection {collection_name} reset successfully")
        except Exception as e:
            _logger.error(f"Error resetting Chroma collection: {e}", exc_info=True)

    def _ensure_collection(self, collection_name: str):
        self._get_client().get_or_create_collection(name=collection_name)

    def _delete_objects(self, collection_name: str, id_field: str, object_ids: List[str]) -> int:
        client = self._get_client()
        if not object_ids or not self._collection_exists(client, collection_name):
            return 0

        collection = client.get_collection(name=collection_name)
        where = {id_field: object_ids[0]} if len(object_ids) == 1 else {id_field: {"$in": object_ids}}
        matching_ids = collection.get(where=where, include=[]).get("ids", [])
        if matching_ids:
            collection.delete(ids=matching_ids)

        return len(matching_ids)

    def _upsert_object_embeddings(self, collection_name: str, id_field: str, object_embeddings: List[MemoryEmbedding]):
        if not object_embeddings:
            return

        collection = self._get_client().get_or_create_collection(name=collection_name)
        metadatas = [
            self._sanitize_metadata(embedding.to_dict())
            for embedding in object_embeddings
        ]
        collection.upsert(
            ids=[str(metadata[id_field]) for metadata in metadatas],
            metadatas=metadatas,
            embeddings=[embedding.vector_embedding for embedding in object_embeddings],
        )
    
    def _cleanup_collection(self, collection_name: str):
        try:
            client = self._get_client()
            if self._collection_exists(client, collection_name):
                client.delete_collection(name=collection_name)
                _logger.info(f"Collection {collection_name} deleted successfully")

        except Exception as e:
            _logger.error(f"Error deleting Chroma collection: {e}", exc_info=True)

    def _list_objects(self, object_type, collection_name: str):
        client = self._get_client()
        if not self._collection_exists(client, collection_name):
            return []
        metadata = client.get_collection(name=collection_name).get().get("metadatas") or []
        return [object_type.parse_object(item) for item in metadata if item]

    def _increment_memory_retrieval_counts(self, collection_name: str, memory_ids: List[str]):
        client = self._get_client()
        if not memory_ids or not self._collection_exists(client, collection_name):
            return
        
        collection = client.get_collection(name=collection_name)
        result = collection.get(ids=memory_ids, include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        updated = []
        for metadata in metadatas:
            metadata = dict(metadata or {})
            metadata["retrieval_count"] = int(metadata.get("retrieval_count", 0) or 0) + 1
            updated.append(self._sanitize_metadata(metadata))

        if ids:
            collection.update(ids=ids, metadatas=updated)
    
    def _cleanup_database(self):
        try:
            for col in self._get_client().list_collections():
                self._get_client().delete_collection(col.name)

            _logger.info(f"Database cleanup for {self._get_client()} successful.")
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_cleanup_database(self) -> None:
        try:
            await self.hass.async_add_executor_job(self._cleanup_database)
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_reset_collection(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._reset_collection, collection_name)
        except Exception as e:
            _logger.error(f"Error resetting collection: {e}", exc_info=True)

    async def async_ensure_collection_exists(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._ensure_collection, collection_name)
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
            result = await self.hass.async_add_executor_job(self._query_scored_devices, collection_name, query_embedding, top_k)
            metadata_rows = result.get("metadatas") or []
            distance_rows = result.get("distances") or []
            metadatas = metadata_rows[0] if metadata_rows else []
            distances = distance_rows[0] if distance_rows else []
            return [
                ScoredResult(
                    object_type.parse_object(metadata),
                    1.0 / (1.0 + max(0.0, float(distance))),
                    rank,
                )
                for rank, (metadata, distance) in enumerate(zip(metadatas, distances), start=1)
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

