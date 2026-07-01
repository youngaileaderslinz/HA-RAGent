from __future__ import annotations

from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.const import RAGENT_SEARCH_DEVICES_TOOL_NAME

from .semantic_search_base import RAGentSemanticSearchBaseTool


class RAGentSemanticSearchDevicesTool(RAGentSemanticSearchBaseTool):
    """Semantic search tool over embedded devices."""

    name = RAGENT_SEARCH_DEVICES_TOOL_NAME
    description = (
        "Search Home Assistant devices using semantic similarity. "
        "Use this when you need to find matching entities, areas, domains, or device names from a natural-language query."
    )

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "query must not be empty"}

        results: list[dict[str, object]] = []
        errors: list[str] = []
        seen_device_ids: set[str] = set()
        device_limit = 0

        for entry, subentry_id, subentry, device_limit, _tool_limit in self._iter_searchable_entries():
            try:
                query_embedding = await self._embed_query_for_subentry(entry, subentry, query)
            except Exception as err:
                errors.append(f"Failed to embed query for subentry {subentry.title}: {err}")
                continue

            try:
                if len(results) >= device_limit:
                    continue

                devices = await entry.vector_db_backend.async_retrieve_objects(
                    object_type=DeviceEmbedding,
                    config_subentry=dict(subentry.data),
                    collection_name=f"devices_{subentry_id}",
                    query_embedding=query_embedding,
                    top_k=device_limit,
                )
                for device in devices:
                    if not isinstance(device, Device) or device.id in seen_device_ids:
                        continue
                    seen_device_ids.add(device.id)
                    state = self.hass.states.get(device.id)
                    results.append(
                        {
                            "entity_id": device.id,
                            "name": device.name,
                            "area": device.area_name,
                            "domain": device.domain,
                            "aliases": device.aliases or [],
                            "state": state.state if state else None,
                        }
                    )
                    if len(results) >= device_limit:
                        break
            except Exception as err:
                errors.append(f"Failed to search subentry {subentry.title}: {err}")

        return {"query": query, "devices": results[:device_limit], "errors": errors}
