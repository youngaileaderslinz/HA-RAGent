from __future__ import annotations

from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from typing import Any

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.homeassistant.helpers.lexical_retriever import LexicalRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.vector_retriever import VectorRetriever
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.utils import get_setting_value


_logger = BaseLogger(__name__)


class SourceRetriever:
    @staticmethod
    def retrieval_method(options: dict) -> str:
        """Return the configured retrieval mode, defaulting safely to hybrid."""
        method = str(get_setting_value(CONF_RETRIEVAL_METHOD, options)).strip().lower()
        return method if method in {
            RETRIEVAL_METHOD_AUTOMATIC, RETRIEVAL_METHOD_LEXICAL, RETRIEVAL_METHOD_VECTOR,
        } else RETRIEVAL_METHOD_AUTOMATIC

    @classmethod
    async def async_retrieve_sources(
        cls,
        backend: Any,
        object_type: type,
        options: dict,
        collection: str,
        embedding: list[float] | QueryEmbedding,
        limit: int,
        query: str = "",
    ) -> tuple[list, list]:
        """Retrieve sources selected by the configured mode and combine them."""
        if limit <= 0:
            return [], []
        method = cls.retrieval_method(options)
        _logger.log_payload(
            "retrieval.sources.request", 
            collection=collection,
            object_type=getattr(object_type, "__name__", str(object_type)),
            method=method, 
            query=query, 
            limit=limit,
            embedding_deferred=isinstance(embedding, QueryEmbedding)
        )

        lexical = []
        if method != RETRIEVAL_METHOD_VECTOR:
            lexical = await LexicalRetriever.async_retrieve(
                backend, object_type, options, collection,
            )
        if method == RETRIEVAL_METHOD_LEXICAL:
            _logger.log_payload(
                "retrieval.sources.result", 
                collection=collection,
                method=method, 
                vector=[], 
                lexical=lexical,
            )
            return [], lexical

        vector, resolved_embedding, failure = await VectorRetriever.async_retrieve(
            backend, object_type, options, collection, embedding, limit,
        )
        payload: dict[str, object] = {
            "collection": collection,
            "method": method,
            "vector": vector,
            "lexical": lexical,
        }
        if resolved_embedding is not None:
            payload["embedding"] = resolved_embedding
        if failure is not None:
            payload["failure"] = failure
        _logger.log_payload(
            "retrieval.sources.result", 
            **payload
        )
        return vector, lexical
