"""Vector source retrieval for Home Assistant entities and tools."""

from __future__ import annotations

from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from typing import Any

from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult


_logger = BaseLogger(__name__)


class VectorRetriever:
    """Resolve an embedding and retrieve normalized scored vector candidates."""

    @staticmethod
    async def async_retrieve(
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
                _logger.warning("Query embedding failed for %s: %s", collection, err)
                return [], None, {"stage": "embedding", "error": repr(err)}
        if not embedding:
            return [], None, {"stage": "embedding", "error": "empty embedding"}
        try:
            raw_results = await backend.async_retrieve_scored_objects(
                object_type, options, collection, embedding, limit,
            )
        except Exception as err:
            _logger.warning("Vector retrieval failed for %s: %s", collection, err)
            return [], embedding, {"stage": "vector", "error": repr(err)}
        return [
            ScoredResult(result.item, result.score, rank)
            for rank, result in enumerate(
                sorted(raw_results, key=lambda result: (-result.score, result.rank)),
                start=1,
            )
        ], embedding, None
