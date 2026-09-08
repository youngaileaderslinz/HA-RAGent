from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class QueryEmbedding:
    def __init__(self, embed: Callable[[], Awaitable[list[float] | None]]) -> None:
        self._embed = embed
        self._task: asyncio.Task | None = None

    async def get(self) -> list[float]:
        if self._task is None:
            self._task = asyncio.create_task(self._embed())
        return await self._task or []
