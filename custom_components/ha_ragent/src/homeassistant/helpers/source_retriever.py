from __future__ import annotations

import logging
from typing import Any

from custom_components.ha_ragent.src.const import (
    CONF_RETRIEVAL_METHOD,
    RETRIEVAL_METHOD_AUTOMATIC,
    RETRIEVAL_METHOD_LEXICAL,
    RETRIEVAL_METHOD_VECTOR,
)
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.utils import get_setting_value


_logger = BaseLogger(__name__)


class SourceRetriever:
    """Fetch lexical and vector candidates, isolating source failures."""

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
            lexical = await cls._async_retrieve_lexical(
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

        vector, resolved_embedding, failure = await cls._async_retrieve_vector(
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

    @staticmethod
    async def _async_retrieve_lexical(
        backend: Any,
        object_type: type,
        options: dict,
        collection: str,
    ) -> list:
        """Return lexical candidates, treating a backend failure as no result."""
        try:
            return await backend.async_get_lexical_objects(
                object_type, options, collection,
            )
        except Exception as err:
            _logger.log_string(logging.WARNING, f"Lexical retrieval failed for {collection}: {err}")
            return []

    @staticmethod
    async def _async_retrieve_vector(
        backend: Any,
        object_type: type,
        options: dict,
        collection: str,
        embedding: list[float] | QueryEmbedding,
        limit: int,
    ) -> tuple[list, list[float] | None, dict[str, str] | None]:
        """Return candidates, the resolved embedding, and an optional failure."""
        if isinstance(embedding, QueryEmbedding):
            try:
                embedding = await embedding.get()
            except Exception as err:
                _logger.log_string(logging.WARNING, f"Query embedding failed for {collection}: {err}")
                return [], None, {"stage": "embedding", "error": repr(err)}
        if not embedding:
            return [], None, {"stage": "embedding", "error": "empty embedding"}
        try:
            raw_results = await backend.async_retrieve_scored_objects(
                object_type, options, collection, embedding, limit,
            )
        except Exception as err:
            _logger.log_string(logging.WARNING, f"Vector retrieval failed for {collection}: {err}")
            return [], embedding, {"stage": "vector", "error": repr(err)}
        return [
            ScoredResult(result.item, result.score, rank)
            for rank, result in enumerate(
                sorted(raw_results, key=lambda result: (-result.score, result.rank)),
                start=1,
            )
        ], embedding, None
