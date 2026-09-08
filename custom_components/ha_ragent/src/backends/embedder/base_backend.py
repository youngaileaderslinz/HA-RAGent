from abc import ABC, abstractmethod
import aiohttp
from typing import Any, Dict, List

from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    STREAM_READ_TIMEOUT,
    HTTP_REQUEST_TIMEOUT,
    STREAM_CONNECT_TIMEOUT,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.embedding.memory import Memory
from custom_components.ha_ragent.src.models.embedding.memory_embedding import MemoryEmbedding


class ABaseEmbedder(ABC):
    _default_timeout = aiohttp.ClientTimeout(total=HTTP_REQUEST_TIMEOUT)
    _chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=STREAM_CONNECT_TIMEOUT, sock_read=STREAM_READ_TIMEOUT)

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self._hass = hass
        self._client_options = client_options
        self._api_key = ABaseEmbedder.normalize_api_key(client_options.get(CONF_EMBEDDING_API_KEY))
        self._url_base = {
            "hostname": client_options.get(CONF_EMBEDDING_HOST),
            "port": client_options.get(CONF_EMBEDDING_PORT),
            "ssl": client_options.get(CONF_EMBEDDING_SSL),
        }

    @staticmethod
    def normalize_api_key(api_key: str) -> str:
        """Normalize the API key by stripping whitespaces and returning an empty string if None."""
        return str(api_key or "").strip()
        
    @staticmethod
    def format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
        """Format the URL for the embedding backend."""
        return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

    @staticmethod
    def get_name() -> str:
        """Return the name of the embedding backend."""
        return "Embedder"

    @staticmethod
    def build_embedding_records(objects: List[EmbeddableModel], vectors: List[List[float]]) -> List[EmbeddingRecord]:
        """Pair objects with validated vectors using their persisted record type."""
        if len(objects) != len(vectors):
            raise ValueError(f"Embedding backend returned {len(vectors)} vectors for {len(objects)} objects")

        records: List[EmbeddingRecord] = []
        dimension = len(vectors[0]) if vectors else 0
        if not dimension and objects:
            raise ValueError("Embedding backend returned an empty vector")

        for obj, vector in zip(objects, vectors):
            if len(vector) != dimension:
                raise ValueError("Embedding backend returned vectors with inconsistent dimensions")
            if isinstance(obj, Device):
                records.append(DeviceEmbedding(obj, vector))
            elif isinstance(obj, LlmTool):
                records.append(LlmToolEmbedding(obj, vector))
            elif isinstance(obj, Memory):
                records.append(MemoryEmbedding(obj, vector))
            else:
                raise TypeError(f"Unsupported embeddable object: {type(obj).__name__}")

        return records
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        """Validate the connection to the embedding backend."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        """Return information about the specified embedding model."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_preload_model(self, config_subentry: dict) -> None:
        """Preload the model and any resources associated with it."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_unload_model(self, config_subentry: dict) -> None:
        """Unload the model and free any resources associated with it."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_get_available_models(self) -> List[str]:
        """Return a list of available embedding models."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> List[float]:
        """Embed a single text string and return the vector embedding."""
        raise NotImplementedError()
    
    @abstractmethod
    async def async_embed_object(self, config_subentry: dict, objects: List[EmbeddableModel]) -> List[EmbeddingRecord]:
        """Embed a list of objects and return the vector embeddings."""
        raise NotImplementedError()
