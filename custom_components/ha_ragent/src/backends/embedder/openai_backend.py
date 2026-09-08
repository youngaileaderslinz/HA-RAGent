from functools import partial
import logging
from typing import Any, Dict, List
from openai import AsyncOpenAI, InternalServerError



from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_EMBEDDING_MODEL,
    RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS,
    RAGENT_EMBEDDING_TRUNCATE_RETRIES,
    RAGENT_EMBEDDING_BATCH_SIZE,
    CONNECTION_RETRIES,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder

_logger = logging.getLogger(__name__)
    
class OpenAiEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._openai_url = ABaseEmbedder.format_url(**self._url_base, path="/v1")
        self._client: AsyncOpenAI | None = None

    async def _async_get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = await self._hass.async_add_executor_job(
                partial(
                    AsyncOpenAI,
                    base_url=self._openai_url,
                    api_key=self._api_key or "not-needed",
                    max_retries=CONNECTION_RETRIES,
                )
            )
        return self._client
    
    @staticmethod
    def get_name() -> str:
        return f"{ABaseEmbedder.get_name()}: OpenAI API"

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        return isinstance(error, InternalServerError) and error.status_code == 500 and "increase the physical batch size" in str(error).lower()

    @staticmethod
    def _truncate_inputs(inputs: List[str], max_chars: int = RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS) -> List[str]:
        """Keep embedding requests within a conservative context-size limit."""
        return [text[:max_chars] if len(text) > max_chars else text for text in inputs]

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        client = None
        try:
            base_url = ABaseEmbedder.format_url(
                hostname=user_input.get(CONF_EMBEDDING_HOST),
                port=user_input.get(CONF_EMBEDDING_PORT),
                ssl=user_input.get(CONF_EMBEDDING_SSL),
                path="/v1",
            )
            api_key = ABaseEmbedder.normalize_api_key(user_input.get(CONF_EMBEDDING_API_KEY))
            client = await hass.async_add_executor_job(
                partial(
                    AsyncOpenAI,
                    base_url=base_url,
                    api_key=api_key or "not-needed",
                    max_retries=CONNECTION_RETRIES,
                )
            )
            await client.models.list()
            return None
        except Exception as ex:
            return str(ex)
        finally:
            if client:
                await client.close()

    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        try:
            client = await self._async_get_client()
            models = await client.models.list()
            model = next((m for m in models.data if m.id == model_name), None)

            if not model:
                raise ValueError(f"Model not found: {model_name}")

            data = model.model_dump() 
            meta = data.get("meta") or {}
            context_size = int(meta.get("n_ctx") or 0) 

            return ModelInfo( 
                name=model.id, 
                context_size=context_size, 
                is_embedding_model=None, 
                is_tool_model=None
            )
        except Exception as ex:
            _logger.error(f"Error retrieving model info for {model_name}: {ex}", exc_info=True)
            raise

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.info("Preloading not supported for OpenAI Compatible Embedder backend.")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.info("Unloading not supported for OpenAI Compatible Embedder backend.")

    async def async_get_available_models(self) -> List[str]:
        client = await self._async_get_client()
        result = await client.models.list()
        return [model.id for model in result.data if model.id]

    async def _async_embed_batch(self, config_subentry: dict, inputs: List[str]) -> List[List[float]]:
        if not inputs:
            return []
        
        max_chars = RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS

        client = await self._async_get_client()
        for attempt in range(RAGENT_EMBEDDING_TRUNCATE_RETRIES + 1):
            request_inputs = self._truncate_inputs(inputs, max_chars)
            try:
                response = await client.embeddings.create(
                    model=config_subentry[CONF_EMBEDDING_MODEL],
                    input=request_inputs,
                    encoding_format="float",
                )
                break
            except Exception as err:
                if not self._is_context_length_error(err) or attempt == RAGENT_EMBEDDING_TRUNCATE_RETRIES:
                    raise

                max_chars //= 2
                _logger.warning(f"Embedding input is too large. Retrying with inputs limited to {max_chars} characters.")

        # OpenAI-compatible embedding responses contain
        # an index for every input. Sort explicitly instead
        # of relying on response ordering.
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> List[float]:
        embeddings = await self._async_embed_batch(config_subentry, [text])
        return embeddings[0] if embeddings else []

    async def async_embed_object(self, config_subentry: dict, objects: List[EmbeddableModel]) -> List[EmbeddingRecord]:
        if not objects:
            return []

        object_embeddings: List[EmbeddingRecord] = []
        for i in range(0, len(objects), RAGENT_EMBEDDING_BATCH_SIZE):
            chunk = objects[i:i + RAGENT_EMBEDDING_BATCH_SIZE]
            texts = [obj.to_embedding_text() for obj in chunk]
            vectors = await self._async_embed_batch(config_subentry, texts)
            object_embeddings.extend(self.build_embedding_records(chunk, vectors))
        return object_embeddings
