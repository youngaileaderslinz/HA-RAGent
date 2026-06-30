from __future__ import annotations

from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.const import RAGENT_SEARCH_TOOLS_TOOL_NAME

from .semantic_search_base import RAGentSemanticSearchBaseTool


class RAGentSemanticSearchToolsTool(RAGentSemanticSearchBaseTool):
    """Semantic search tool over embedded tools."""

    name = RAGENT_SEARCH_TOOLS_TOOL_NAME
    description = (
        "Search available Home Assistant tools using semantic similarity. "
        "Use this when you need to find matching tool names or capabilities from a natural-language query."
    )

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "query must not be empty"}

        results: list[dict[str, object]] = []
        errors: list[str] = []
        seen_tool_names: set[str] = set()
        tool_limit = 0

        for entry, subentry_id, subentry, _device_limit, tool_limit in self._iter_searchable_entries():
            try:
                query_embedding = await self._embed_query_for_subentry(entry, subentry, query)
            except Exception as err:
                errors.append(f"Failed to embed query for subentry {subentry.title}: {err}")
                continue

            try:
                if len(results) >= tool_limit:
                    continue

                tools = await entry.vector_db_backend.async_retrieve_objects(
                    object_type=LlmToolEmbedding,
                    config_subentry=dict(subentry.data),
                    collection_name=f"tools_{subentry_id}",
                    query_embedding=query_embedding,
                    top_k=tool_limit,
                )
                for tool in tools:
                    if not isinstance(tool, LlmTool) or tool.name in seen_tool_names:
                        continue
                    seen_tool_names.add(tool.name)
                    results.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters or {}
                        }
                    )
                    if len(results) >= tool_limit:
                        break
            except Exception as err:
                errors.append(f"Failed to search subentry {subentry.title}: {err}")

        return {"query": query, "tools": results[:tool_limit], "errors": errors}
