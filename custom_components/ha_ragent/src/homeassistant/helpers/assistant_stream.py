from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing

from custom_components.ha_ragent.src.const import TOOL_END, TOOL_START


class AssistantStream:
    def __init__(self, stream: AsyncGenerator[str, None]) -> None:
        self._stream = stream
        self._chunks: list[str] = []
        self._content_chunks: list[str] = []
        self._pending = ""
        self._inside_tool = False

    @property
    def response(self) -> str:
        """Return the original response, including calls for validation."""
        return "".join(self._chunks)

    @property
    def content(self) -> str:
        """Return exactly the text delivered to Home Assistant."""
        return "".join(self._content_chunks)

    def _filter_chunk(self, chunk: str) -> str:
        """Remove tool fences even when their markers span several chunks."""
        self._pending += chunk
        visible: list[str] = []
        while self._pending:
            if self._inside_tool:
                end = self._pending.find(TOOL_END)
                if end == -1:
                    self._pending = self._pending[-(len(TOOL_END) - 1):]
                    break
                self._pending = self._pending[end + len(TOOL_END):]
                self._inside_tool = False
                continue

            start = self._pending.find(TOOL_START)
            if start != -1:
                visible.append(self._pending[:start])
                self._pending = self._pending[start + len(TOOL_START):]
                self._inside_tool = True
                continue

            keep = next((
                size for size in range(min(len(self._pending), len(TOOL_START) - 1), 0, -1)
                if TOOL_START.startswith(self._pending[-size:])
            ), 0)
            if keep:
                visible.append(self._pending[:-keep])
                self._pending = self._pending[-keep:]
            else:
                visible.append(self._pending)
                self._pending = ""
            break
        return "".join(visible)

    async def async_deltas(self) -> AsyncGenerator[dict[str, str], None]:
        """Yield text-only HA deltas; tool execution stays with RAGent."""
        started = False
        whitespace = ""
        async with aclosing(self._stream):
            async for chunk in self._stream:
                self._chunks.append(chunk)
                text = whitespace + self._filter_chunk(chunk)
                if not started:
                    text = text.lstrip()
                content = text.rstrip()
                whitespace = text[len(content):]
                if content:
                    self._content_chunks.append(content)
                    delta = {"content": content}
                    if not started:
                        delta["role"] = "assistant"
                        started = True
                    yield delta

        # Preserve ordinary markdown suffixes, but hide incomplete tool blocks.
        partial_tool_marker = self._pending.startswith("```h")
        if self._pending and not self._inside_tool and not partial_tool_marker:
            content = (whitespace + self._pending).rstrip()
            if not started:
                content = content.lstrip()
            if content:
                self._content_chunks.append(content)
                delta = {"content": content}
                if not started:
                    delta["role"] = "assistant"
                yield delta
