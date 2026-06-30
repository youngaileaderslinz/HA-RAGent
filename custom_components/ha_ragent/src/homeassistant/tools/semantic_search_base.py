from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    CONF_NUM_DEVICES_TO_EXTRACT,
    CONF_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DOMAIN,
)


class RAGentSemanticSearchBaseTool(llm.Tool):
    """Semantic search base tool over embedded devices or tools."""

    name = ""
    description = ""
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
        }
    )

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    @staticmethod
    def _get_effective_limits(entry: Any) -> tuple[int, int]:
        """Use the same configured limits as normal retrieval."""
        entry_options = getattr(entry, "options", {}) or {}
        device_limit = int(entry_options.get(CONF_NUM_DEVICES_TO_EXTRACT, DEFAULT_NUM_DEVICES_TO_EXTRACT))
        tool_limit = int(entry_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT))
        return device_limit, tool_limit

    async def _validate_query(self, tool_input: llm.ToolInput) -> str | None:
        query = str(tool_input.tool_args.get("query", "")).strip()
        return query or None

    def _iter_searchable_entries(self):
        """Yield searchable entry and subentry combinations."""
        domain_data = self.hass.data.get(DOMAIN, {})
        for _, entry in domain_data.items():
            if not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
                continue

            device_limit, tool_limit = self._get_effective_limits(entry)

            for subentry_id, subentry in entry.subentries.items():
                if subentry.data.get(CONF_LLM_HASS_API) == "none":
                    continue

                yield entry, subentry_id, subentry, device_limit, tool_limit

    async def _embed_query_for_subentry(self, entry: Any, subentry: Any, query: str) -> list[float]:
        """Embed a search query for a specific subentry."""
        return await entry.embedder_backend.async_embed_text(dict(subentry.data), query)
