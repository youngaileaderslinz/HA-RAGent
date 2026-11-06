from __future__ import annotations
import logging

from custom_components.ha_ragent.src.homeassistant.helpers.tool_ranker import ToolRanker
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
import json
from typing import Any

import probatio

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry, entity_registry, floor_registry, llm
from custom_components.ha_ragent.src.homeassistant.tools.base_tool import RAGentTool

from custom_components.ha_ragent.src.const import (
    CONF_MIN_DEVICES_TO_EXTRACT,
    CONF_MAX_DEVICES_TO_EXTRACT,
    CONF_MIN_TOOLS_TO_EXTRACT,
    CONF_MAX_TOOLS_TO_EXTRACT,
    DOMAIN,
    RAGENT_MAX_SEARCH_QUERY_CHARS,
    RAGENT_MAX_SEARCH_QUERIES,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.homeassistant.helpers.source_retriever import SourceRetriever
from custom_components.ha_ragent.src.homeassistant.helpers.conversation_retriever import ConversationRetriever
from custom_components.ha_ragent.src.models.retrieval.query_embedding import QueryEmbedding
from custom_components.ha_ragent.src.models.retrieval.search_contract import SearchFilters, normalized, opaque_id
from custom_components.ha_ragent.src.translation import RAGentTranslations
from custom_components.ha_ragent.src.utils import get_setting_value

_logger = BaseLogger(__name__)


class RAGentSemanticSearchTool(RAGentTool):
    name = RAGENT_SEMANTIC_SEARCH_TOOL_NAME
    parameters = probatio.Schema(
        {
            probatio.Optional("operation", description="Default: search."): probatio.In(["search", "detail"]),
            probatio.Optional("result_id", description="Current result ID for detail."): str,
            probatio.Optional("filters", description=(
                "AND constraints; include only fields requested by the user."
            )): {
                probatio.Optional("area"): str,
                probatio.Optional("floor"): str,
                probatio.Optional("domain", description="Omit unless the user specifies device types."): probatio.Any(str, [str]),
                probatio.Optional("integration"): str,
                probatio.Optional("entity_type", description="Entity domain."): str,
                probatio.Optional("device_type", description="Device class."): str,
                probatio.Optional("state"): probatio.Any(str, {
                    probatio.Required("operator"): probatio.In(["eq", "ne", "gt", "gte", "lt", "lte"]),
                    probatio.Required("value"): probatio.Any(str, int, float),
                }),
            },
            probatio.Optional("search_queries", description="Self-contained target-group queries; capabilities align by position."
            ): probatio.All([str], probatio.Length(min=1, max=RAGENT_MAX_SEARCH_QUERIES)),
            probatio.Optional("capabilities", description=(
                "One hint/query in order; {} for no hint. Without search_queries, one context hint. "
                "E.g. {action:turn_on,domain:light}."
            )
            ): probatio.All(
                [{
                    probatio.Optional("action", description="Stable canonical action ID, e.g. turn_on."): str,
                    probatio.Optional("domain", description="HA domain(s)."): probatio.Any(str, [str]),
                }],
                probatio.Length(min=1, max=RAGENT_MAX_SEARCH_QUERIES),
            ),
            probatio.Optional("scope", default="devices", description="devices (default) or tools.",
            ): probatio.In(["devices", "tools"]),
        }
    )

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.subentry_id = subentry_id
        self.translations = RAGentTranslations(language or "en")
        self.description = self.translations.tool(RAGENT_SEMANTIC_SEARCH_TOOL_NAME)
        self._latest_request = ""
        self._contextual_query = ""
        self._candidate_context: list[dict[str, object]] = []
        self._requested_capabilities: list[dict[str, object]] = []
        self.reset_turn()

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext,
    ) -> llm.ToolResult | dict[str, object]:
        """Repair search input and distinguish invalid arguments from no matches."""
        try:
            arguments = self._repair_arguments(tool_input.tool_args)
            repaired_input = llm.ToolInput(
                tool_name=tool_input.tool_name, tool_args=arguments,
                id=tool_input.id, external=tool_input.external,
            )
            return await super().async_call(hass, repaired_input, llm_context)
        except (probatio.Invalid, ValueError) as error:
            data = {
                "status": "validation_error",
                "error": (
                    f"Invalid search arguments: {error}. "
                    "scope is devices (default) or tools. Put area/floor strings and domain strings/lists "
                    "inside filters. Use operation, not action; omit operation for search. "
                    "Correct the arguments and retry. This is not a no_matches result."
                ),
            }
            result_type = getattr(llm, "ToolResult", None)
            return result_type(data=data, error=True) if result_type else data

    @staticmethod
    def _repair_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        """Repair misplaced search fields without guessing or discarding constraints."""
        repaired = dict(arguments)
        if repaired.get("action") in ("search", "detail"):
            action = repaired.pop("action")
            if "operation" in repaired and repaired["operation"] != action:
                raise ValueError("Conflicting action and operation")
            repaired["operation"] = action
        filters = repaired.get("filters", {})
        if not isinstance(filters, dict):
            return repaired
        filters = dict(filters)
        if "scope" in filters:
            scope = filters.pop("scope")
            if "scope" in repaired and repaired["scope"] != scope:
                raise ValueError("Conflicting scope values")
            repaired["scope"] = scope
        for field in SearchFilters.FIELDS:
            if field in repaired:
                value = repaired.pop(field)
                if field in filters and filters[field] != value:
                    raise ValueError(f"Conflicting {field} filters")
                filters[field] = value
        if filters or "filters" in repaired:
            repaired["filters"] = filters
        return repaired

    def reset_turn(self) -> None:
        """Invalidate cached searches and detail access at a turn/scope boundary."""
        self._searches: dict[str, dict] = {}
        self._results: dict[str, dict] = {}
        self._details: dict[str, str] = {}
        self._requested_capabilities = []

    def invalidate_results(self) -> None:
        """Allow fresh searches after actions while retaining scoped result IDs."""
        self._searches.clear()
        self._details.clear()

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _candidate_summary(cls, candidate: dict[str, object]) -> str:
        """Return compact searchable text for a current device candidate."""
        domains = candidate.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        values = (
            candidate.get("name"),
            candidate.get("friendly_name"),
            candidate.get("area"),
            candidate.get("floor"),
            *domains,
            candidate.get("device_class"),
            candidate.get("state"),
            candidate.get("unit_of_measurement"),
        )
        return " | ".join(cls._clean(value) for value in values if cls._clean(value))

    @classmethod
    def _build_search_query(
        cls,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> str:
        """Build a bounded fallback query from user requests and trusted location."""
        sections: list[str] = []
        latest = cls._clean(latest_request)
        if latest:
            sections.append(latest)

        current_area = cls._clean(area)
        current_floor = cls._clean(floor)
        if current_area:
            sections.append(current_area)
        if current_floor:
            sections.append(current_floor)

        for candidate in candidates or []:
            summary = cls._candidate_summary(candidate)
            if summary:
                sections.append(summary)

        return "\n".join(sections)[:RAGENT_MAX_SEARCH_QUERY_CHARS].strip()

    def set_search_context(
        self,
        *,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> None:
        """Bind trusted context for the current conversation turn."""
        self.reset_turn()
        self._latest_request = self._clean(latest_request)
        self._contextual_query = self._build_search_query(
            latest_request=latest_request,
            area=area,
            floor=floor,
            candidates=candidates,
        )
        self._candidate_context = list(candidates or [])
        _logger.log_payload(
            "search.context_set", 
            entry_id=getattr(self, "entry_id", ""),
            subentry_id=getattr(self, "subentry_id", ""), 
            latest_request=self._latest_request,
            contextual_query=self._contextual_query,
            candidate_context=self._candidate_context,
        )

    @property
    def requested_capabilities(self) -> list[dict[str, object]]:
        """Return structured operations supplied during the latest search."""
        return list(self._requested_capabilities)

    def refresh_candidates(self, candidates: list[dict[str, object]]) -> None:
        """Replace candidate context after a corrective search."""
        self._candidate_context = list(candidates)
        _logger.log_payload(
            "search.context_refreshed", 
            candidates=candidates,
            candidate_context=self._candidate_context,
        )
    @staticmethod
    def _get_effective_ranges(entry: Any, subentry: Any) -> tuple[int, int, int, int]:
        """Return normalized minimum and maximum exposure limits."""
        entry_options = getattr(entry, "options", {}) or {}
        subentry_data = getattr(subentry, "data", {}) or {}
        runtime_options = {**entry_options, **subentry_data}
        requested_min_devices = max(0, int(get_setting_value(CONF_MIN_DEVICES_TO_EXTRACT, runtime_options)))
        requested_min_tools = max(0, int(get_setting_value(CONF_MIN_TOOLS_TO_EXTRACT, runtime_options)))
        max_devices = max(0, int(get_setting_value(CONF_MAX_DEVICES_TO_EXTRACT, runtime_options)))
        max_tools = max(0, int(get_setting_value(CONF_MAX_TOOLS_TO_EXTRACT, runtime_options)))
        if max_devices:
            requested_min_devices = min(requested_min_devices, max_devices)
        if max_tools:
            requested_min_tools = min(requested_min_tools, max_tools)
        return (
            requested_min_devices,
            max_devices,
            requested_min_tools,
            max_tools,
        )

    def _model_search_queries(self, tool_input: llm.ToolInput) -> list[str]:
        """Return the distinct focused search intents supplied by the model."""
        raw_queries = tool_input.tool_args.get("search_queries")
        if raw_queries is None:
            raw_queries = [tool_input.tool_args.get("search_query")]
        elif isinstance(raw_queries, str):
            raw_queries = [raw_queries]

        queries: list[str] = []
        seen: set[str] = set()
        for raw_query in raw_queries if isinstance(raw_queries, list) else []:
            query = self._clean(raw_query)
            normalized = query.casefold()
            if query and normalized not in seen:
                queries.append(query)
                seen.add(normalized)
        return queries

    async def _validate_queries(self, tool_input: llm.ToolInput) -> list[str]:
        """Use model search intents; fall back to turn context when absent."""
        queries = self._model_search_queries(tool_input)
        if queries:
            return [query[:RAGENT_MAX_SEARCH_QUERY_CHARS] for query in queries]
        return [self._contextual_query] if self._contextual_query else []

    @staticmethod
    def _merge_query_candidates(
        candidate_batches: list[list[dict[str, object]]],
        limit: int,
    ) -> list[dict[str, object]]:
        """Take each query's next unseen candidate in turn within the limit."""
        merged: list[dict[str, object]] = []
        seen_names: set[str] = set()
        batches = [iter(candidates) for candidates in candidate_batches]
        while len(merged) < limit:
            added = False
            for candidates in batches:
                for candidate in candidates:
                    name = str(candidate["name"])
                    if name in seen_names:
                        continue
                    merged.append(candidate)
                    seen_names.add(name)
                    added = True
                    break
                if len(merged) >= limit:
                    break
            if not added:
                break
        return merged

    def _iter_searchable_entries(self):
        """Yield only the active entry and subentry."""
        domain_data = self.hass.data.get(DOMAIN, {})
        entry = domain_data.get(self.entry_id)
        if not entry or not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
            return
        subentry = entry.subentries.get(self.subentry_id)
        if not subentry or get_setting_value(CONF_LLM_HASS_API, subentry.data) == "none":
            return
        limits = self._get_effective_ranges(entry, subentry)
        yield entry, self.subentry_id, subentry, *limits

    async def _embed_query_for_subentry(self, entry: Any, subentry: Any, query: str) -> list[float]:
        """Embed a search query for a specific subentry."""
        try:
            return await entry.embedder_backend.async_embed_text(dict(subentry.data), query) or []
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Search embedding failed: {err}")
            return []

    def _shared_query_embedding(
        self,
        cache: dict[tuple[object, str, str], QueryEmbedding],
        entry: Any,
        subentry: Any,
        query: str,
    ) -> QueryEmbedding:
        """Reuse an identical configured embedding request within this turn."""
        configuration = json.dumps(getattr(subentry, "data", {}) or {}, sort_keys=True, default=str)
        key = (id(getattr(entry, "embedder_backend", None)), configuration, query)
        if key not in cache:
            cache[key] = QueryEmbedding(
                lambda: self._embed_query_for_subentry(entry, subentry, query)
            )
        return cache[key]

    def _device_search_query(
        self, model_search_query: str, fallback_query: str,
    ) -> str:
        """Search the requested target, using context only as a fallback."""
        return (model_search_query or fallback_query)[:RAGENT_MAX_SEARCH_QUERY_CHARS]

    def _tool_search_query(
        self,
        model_search_query: str,
        devices: list[Device | dict[str, object]],
        requested_capability: object = None,
        fallback_query: str = "",
    ) -> str:
        """Search the requested operation with any declared capability."""
        return ToolRanker.build_tool_search_query(
            model_search_query or fallback_query,
            "",
            devices,
            requested_capability,
        )[:RAGENT_MAX_SEARCH_QUERY_CHARS]

    def _device_candidate(self, device: Device) -> dict[str, object]:
        """Return enough current device data to answer without another search."""
        state = self.hass.states.get(device.id)
        return {
            "name": device.id,
            "friendly_name": device.friendly_name,
            "area": device.area_name,
            "floor": device.floor_name,
            "area_aliases": device.area_aliases or [],
            "floor_aliases": device.floor_aliases or [],
            "domain": device.domain,
            "device_class": device.device_class,
            "aliases": device.aliases or [],
            "device_labels": device.device_labels or [],
            "services": device.services or [],
            "state": state.state if state else None,
            "attributes": Device.clean_attributes(device.domain, state.attributes) if state else {},
            "unit_of_measurement": (
                state.attributes.get("unit_of_measurement")
                if state
                else device.unit_of_measurement
            ),
        }

    def _result_id(self, kind: str, identity: str) -> str:
        return opaque_id(self.entry_id, self.subentry_id, kind, identity)

    def _resolve_location_filters(self, raw: object) -> tuple[SearchFilters, dict[str, str]]:
        """Resolve registry locations and report omitted discovery constraints."""
        if not isinstance(raw, dict):
            raise ValueError("filters must be an object")
        values = dict(raw)
        ignored: dict[str, str] = {}
        for field in ("area", "floor"):
            value = values.get(field)
            if not isinstance(value, str):
                continue
            if len(value) > 256:
                raise ValueError(f"{field} must be at most 256 characters")
            if field == "area":
                entries = area_registry.async_get(self.hass).async_list_areas()
                id_field = "id"
            else:
                entries = floor_registry.async_get(self.hass).async_list_floors()
                id_field = "floor_id"
            key = normalized(value)
            exact = []
            aliases = []
            for entry in entries:
                if key in {normalized(entry.name), normalized(getattr(entry, id_field))}:
                    exact.append(entry)
                elif key in {normalized(alias) for alias in entry.aliases}:
                    aliases.append(entry)
            matches = exact or aliases
            if len(matches) == 1:
                values[field] = matches[0].name
            else:
                ignored[field] = values.pop(field)
        return SearchFilters.parse(values), ignored

    def result_context(self, result: dict) -> tuple[list[dict], list[str]]:
        """Resolve device matches and selected tool details to trusted context."""
        if result.get("status") not in ("ok", "same_as"):
            return [], []
        if "detail" in result:
            record = self._results.get(result["detail"].get("id"))
            if not record:
                return [], []
            return ([record["data"]], []) if record["type"] == "device" else ([], [record["data"]["name"]])
        devices = [self._results[result_id]["data"] for result_id in self.history_candidate_entities(result)]
        return devices, []

    def history_candidate_entities(self, result: dict) -> dict[str, str]:
        """Resolve compact references to canonical device IDs for internal history."""
        if result.get("status") not in ("ok", "same_as"):
            return {}
        candidates = result.get("results", [])
        if not isinstance(candidates, list):
            return {}
        result_ids = [candidate.get("id") for candidate in candidates if isinstance(candidate, dict)]
        references = result.get("result_ids", [])
        if isinstance(references, list):
            result_ids.extend(references)
        return {result_id: record["data"]["name"] for result_id in result_ids
                if isinstance(result_id, str) and (record := self._results.get(result_id))
                and record["type"] == "device"
                and result_id == self._result_id("device", record["data"]["name"])}

    def _device_filter(self, filters: SearchFilters, device: Device) -> bool:
        data = self._device_candidate(device)
        data["entity_type"] = device.id.partition(".")[0]
        data["device_type"] = data["device_class"]
        if "integration" in filters.values:
            entry = entity_registry.async_get(self.hass).async_get(device.id)
            data["integration"] = entry.platform if entry else None
        return filters.matches(data)

    @staticmethod
    def _tool_filter(filters: SearchFilters, tool: LlmTool, devices: list[Device]) -> bool:
        values = dict(filters.values)
        domains = tool.canonical_supported_domains
        # Location, live state and entity integration constrain targets, then
        # tool applicability; a tool itself has no room or live entity state.
        target_fields = {"area", "floor", "state", "integration", "device_type"}
        if target_fields & values.keys():
            if not devices:
                return False
            if domains and not any(set(d.domain or ()) & set(domains) for d in devices):
                return False
            for field in target_fields:
                values.pop(field, None)
        return SearchFilters(values).matches({"domain": domains, "entity_type": domains})

    def _compact_result(self, kind: str, data: dict, query: str, filters: SearchFilters) -> dict:
        identity = data["name"]
        result_id = self._result_id(kind, identity)
        self._results[result_id] = {"type": kind, "data": data}
        if kind == "device":
            name = data.get("friendly_name") or identity
            domains = data.get("domain") or [identity.partition(".")[0]]
            summary = data.get("device_class") or "/".join(domains)
            if data.get("state") is not None:
                summary += "; " + str(data["state"])
                if data.get("unit_of_measurement"):
                    summary += " " + str(data["unit_of_measurement"])
            location = "/".join(str(data[key]) for key in ("floor", "area") if data.get(key))
        else:
            name = identity
            summary = data.get("action") or data.get("description") or identity
            if data.get("domains"):
                summary += "; " + "/".join(data["domains"])
            location = ""
        # Codes describe deterministic evidence; no free-form ranking payload.
        if normalized(query) in {normalized(identity), normalized(str(name))}:
            rationale = "name_match"
        elif "state" in filters.values:
            rationale = "state_match"
        elif "area" in filters.values or "floor" in filters.values:
            rationale = "area_match"
        elif kind == "tool" and data.get("action"):
            rationale = "capability_match"
        else:
            rationale = None
        compact = {"id": result_id, "type": kind, "name": str(name)[:160], "summary": str(summary)[:160]}
        if kind == "device":
            compact["entity_id"] = identity
        if location:
            compact["location"] = location[:160]
        if rationale:
            compact["rationale"] = rationale
        return compact

    def _detail(self, arguments: dict) -> dict:
        result_id = arguments.get("result_id")
        if set(arguments) - {"operation", "result_id", "scope"} or not isinstance(result_id, str) or not result_id:
            return {"status": "detail_error", "error": "detail requires exactly one result_id"}
        record = self._results.get(result_id)
        if (record is None or arguments["scope"] != record["type"] + "s"
                or result_id != self._result_id(record["type"], record["data"]["name"])):
            return {"status": "detail_error", "error": "Unknown, stale or out-of-scope result_id"}
        data = dict(record["data"])
        if record["type"] == "device":
            state = self.hass.states.get(data["name"])
            if state is None:
                return {"status": "detail_error", "error": "Device is no longer available"}
            data["state"] = state.state
            data["attributes"] = Device.clean_attributes(data.get("domain"), state.attributes)
            data["unit_of_measurement"] = state.attributes.get("unit_of_measurement", data.get("unit_of_measurement"))
            data = {key: value for key, value in data.items() if value is not None and value != "" and value != [] and value != {}}
        signature = opaque_id(data)
        if self._details.get(result_id) == signature:
            return {"status": "same_as", "result_ids": [result_id]}
        self._details[result_id] = signature
        record["data"] = data
        return {"status": "ok", "detail": {"id": result_id, "type": record["type"], **data}}

    async def _async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        arguments = tool_input.tool_args
        operation = arguments.get("operation", "search")
        scope = arguments.get("scope", "devices")
        if scope not in ("devices", "tools"):
            return {"status": "detail_error" if operation == "detail" else "validation_error",
                    "error": "scope must be devices or tools"}
        if operation == "detail":
            return self._detail(arguments)
        try:
            if operation != "search" or set(arguments) - {
                "operation", "search_queries", "search_query", "scope", "filters", "capabilities"
            }:
                raise ValueError("Invalid search operation or arguments")
            filters, ignored_filters = self._resolve_location_filters(arguments.get("filters", {}))
            raw_queries = arguments.get("search_queries", [arguments.get("search_query", "")])
            if not isinstance(raw_queries, list) or not 1 <= len(raw_queries) <= RAGENT_MAX_SEARCH_QUERIES:
                raise ValueError(f"Supply 1 to {RAGENT_MAX_SEARCH_QUERIES} search_queries")
            if not all(isinstance(q, str) and len(q) <= RAGENT_MAX_SEARCH_QUERY_CHARS for q in raw_queries):
                raise ValueError("Invalid search query")
            raw_capabilities = arguments.get("capabilities", [])
            if not isinstance(raw_capabilities, list):
                raise ValueError("capabilities must be a list of objects; omit it when no hints are needed")
            if raw_capabilities and len(raw_capabilities) != len(raw_queries):
                raise ValueError(
                    f"Received {len(raw_capabilities)} capabilities for {len(raw_queries)} search queries. "
                    "Supply one capability per query in the same order; use {} for queries without a hint, "
                    "or omit capabilities entirely"
                )
            if any(not isinstance(c, dict) or set(c) - {"action", "domain"}
                   or ("action" in c and not isinstance(c["action"], str))
                   or ("domain" in c and not (isinstance(c["domain"], str)
                       or isinstance(c["domain"], list) and all(isinstance(d, str) for d in c["domain"])))
                   for c in raw_capabilities):
                raise ValueError("Invalid capability")
            # Keep capability/query pairs aligned, including duplicate text.
            pairs = []
            for index, query in enumerate(raw_queries):
                query = self._clean(query)
                capability = ToolRanker.normalize_requested_capability(raw_capabilities[index]) if raw_capabilities else {}
                pair = (query or self._contextual_query, capability)
                if not any(normalized(pair[0]) == normalized(q) and capability == c for q, c in pairs):
                    pairs.append(pair)
            if not pairs or any(not query for query, _ in pairs):
                raise ValueError("Search query must not be empty")
        except (ValueError, TypeError) as error:
            return {"status": "validation_error", "error": str(error)}

        metadata = {"ignored_filters": ignored_filters} if ignored_filters else {}
        searchable_entries = list(self._iter_searchable_entries())
        options = {}
        if searchable_entries:
            entry, _, subentry, *_ = searchable_entries[0]
            options = {
                **getattr(entry, "options", {}), **subentry.data,
            }
        mode = SourceRetriever.retrieval_method(options)

        self._requested_capabilities = [capability for _, capability in pairs]
        key = opaque_id(self.entry_id, self.subentry_id, scope, mode, filters.values,
                        [(normalized(query), capability) for query, capability in pairs])
        if key in self._searches:
            previous = self._searches[key]
            return {"status": "same_as", "search_id": previous["search_id"],
                    "result_ids": [result["id"] for result in previous["results"]], **metadata}
        device_batches, tool_batches = [], []
        device_limit = tool_limit = 0
        embedding_cache = {}
        failures: list[str] = []
        for entry, subentry_id, subentry, min_devices, max_devices, min_tools, max_tools in searchable_entries:
            retriever = ConversationRetriever(self.hass, entry, self.entry_id, subentry_id, subentry)
            device_limit, tool_limit = max_devices, max_tools
            for query, capability in pairs:
                embedding = self._shared_query_embedding(embedding_cache, entry, subentry, query)
                devices = list(self._candidate_context)
                # Tools-only searches with target constraints use the entire
                # filtered target catalog, without wasting model result slots.
                target_filters = bool({"area", "floor", "state", "integration", "device_type"} & filters.values.keys())
                if scope == "tools" and target_filters:
                    from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
                    catalog = await SourceRetriever.async_retrieve_catalog(
                        entry.vector_db_backend, DeviceEmbedding,
                        {**getattr(entry, "options", {}), **subentry.data}, f"devices_{subentry_id}", failures)
                    if failures:
                        return {"status": "retrieval_error", "error": failures[0], **metadata}
                    devices = [d for d in catalog if self._device_filter(filters, d)]
                if scope == "devices" and max_devices > 0:
                    devices = await retriever.async_retrieve_search_devices(
                        embedding, query, minimum=min_devices, maximum=max_devices, retrieval_method=mode,
                        predicate=(lambda d: self._device_filter(filters, d)) if filters.values else None,
                        failures=failures)
                    if failures:
                        return {"status": "retrieval_error", "error": failures[0], **metadata}
                    device_batches.append([self._device_candidate(d) for d in devices])
                if scope == "tools" and max_tools > 0:
                    tool_query = self._tool_search_query(query, devices, capability)
                    tool_embedding = self._shared_query_embedding(embedding_cache, entry, subentry, tool_query)
                    tools, scored, evidence = await retriever.async_retrieve_search_tools(
                        tool_embedding, tool_query, devices=devices, maximum=max_tools,
                        requested_capability=capability, retrieval_method=mode,
                        predicate=(lambda t: self._tool_filter(filters, t, [d for d in devices if isinstance(d, Device)]))
                        if filters.values else None, failures=failures)
                    if failures:
                        return {"status": "retrieval_error", "error": failures[0], **metadata}
                    _logger.log_payload("search.ranking", query=tool_query, vector=scored, evidence=evidence)
                    tool_batches.append([{
                        "name": t.name, "description": t.description, "parameters": t.parameters or {},
                        "metadata": t.metadata.to_dict() if t.metadata else {},
                        "action": t.canonical_action, "domains": list(t.canonical_supported_domains),
                    } for t in tools[:max_tools]])
        devices = self._merge_query_candidates(device_batches, device_limit)
        tools = self._merge_query_candidates(tool_batches, tool_limit)
        results = [self._compact_result(kind, data, pairs[0][0], filters)
                   for kind, batch in (("device", devices), ("tool", tools)) for data in batch]
        response = {"status": "ok" if results else "no_matches", "search_id": key, "mode": mode,
                    "results": results, **metadata}
        if not results and filters.values:
            response["filters"] = filters.values
        self._searches[key] = response
        return response
