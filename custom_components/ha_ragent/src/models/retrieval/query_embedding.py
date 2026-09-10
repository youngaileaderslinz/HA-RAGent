from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from custom_components.ha_ragent.src.debug import log_debug_payload

_logger = logging.getLogger(__name__)


class QueryEmbedding:
    def __init__(self, embed: Callable[[], Awaitable[list[float] | None]]) -> None:
        self._embed = embed
        self._task: asyncio.Task | None = None

    async def get(self) -> list[float]:
        if self._task is None:
            log_debug_payload(_logger, "embedding.cache_miss")
            self._task = asyncio.create_task(self._embed())
        else:
            log_debug_payload(
                _logger, "embedding.cache_hit", done=self._task.done(),
            )
        embedding = await self._task or []
        log_debug_payload(
            _logger, "embedding.value", dimensions=len(embedding), vector=embedding,
        )
        return embedding
