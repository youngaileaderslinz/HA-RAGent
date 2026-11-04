from __future__ import annotations

import logging
from typing import Any
from collections.abc import Callable

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
        predicate: Callable[[Any], bool] | None = None,
        failures: list[str] | None = None,
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

        catalog_failures: list[str] = []
        lexical = []
        if method != RETRIEVAL_METHOD_VECTOR:
            lexical = await cls._async_retrieve_lexical(
                backend, object_type, options, collection, catalog_failures,
            )
        if predicate is not None and catalog_failures:
            if failures is not None:
                failures.extend(catalog_failures)
            return [], []
        if method == RETRIEVAL_METHOD_LEXICAL:
            if predicate is not None:
                lexical = [item for item in lexical if predicate(item)]
            _logger.log_payload(
                "retrieval.sources.result", 
                collection=collection,
                method=method, 
                vector=[], 
                lexical=lexical,
            )
            return [], lexical

        vector_limit = limit
        if predicate is not None:
            # Filter before ranking/selection, including matches outside the
            # ordinary vector shortlist. Catalog access is not lexical ranking.
            catalog = lexical if method != RETRIEVAL_METHOD_VECTOR else await cls.async_retrieve_catalog(
                backend, object_type, options, collection, catalog_failures,
            )
            if catalog_failures:
                if failures is not None:
                    failures.extend(catalog_failures)
                return [], []
            vector_limit = max(limit, len(catalog))
        vector, resolved_embedding, failure = await cls._async_retrieve_vector(
            backend, object_type, options, collection, embedding, vector_limit,
        )
        if predicate is not None:
            vector = [result for result in vector if predicate(result.item)]
            lexical = [item for item in lexical if predicate(item)]
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
        failures: list[str] | None = None,
    ) -> list:
        """Return lexical candidates, treating a backend failure as no result."""
        return await SourceRetriever.async_retrieve_catalog(backend, object_type, options, collection, failures)

    @staticmethod
    async def async_retrieve_catalog(
        backend: Any,
        object_type: type,
        options: dict,
        collection: str,
        failures: list[str] | None = None,
    ) -> list:
        """Guard catalog reads and distinguish failures from an empty catalog."""
        try:
            return await backend.async_get_lexical_objects(
                object_type, options, collection,
            )
        except Exception as err:
            message = f"Catalog retrieval failed for {collection}"
            _logger.log_string(logging.WARNING, f"{message}: {err}")
            if failures is not None:
                failures.append(message)
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
