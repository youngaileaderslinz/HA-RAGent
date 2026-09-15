"""Lexical source retrieval for Home Assistant entities and tools."""

from __future__ import annotations

import logging
from typing import Any


_logger = logging.getLogger(__name__)


class LexicalRetriever:
    """Retrieve the locally indexed lexical corpus without vector dependencies."""

    @staticmethod
    async def async_retrieve(
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
            _logger.warning("Lexical retrieval failed for %s: %s", collection, err)
            return []
