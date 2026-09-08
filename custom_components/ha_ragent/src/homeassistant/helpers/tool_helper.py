import logging
import json
from typing import Any, Dict, List, Tuple

from homeassistant.core import HomeAssistant, JsonObjectType
from homeassistant.helpers.llm import ToolInput
from homeassistant.helpers import area_registry, device_registry, entity_registry, floor_registry
from homeassistant.helpers.entity_registry import RegistryEntry as EntityEntry

from custom_components.ha_ragent.src.const import (
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
    TOOL_REGEX_PATTERN,
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper

_logger = logging.getLogger(__name__)

class ToolHelper:
    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    @staticmethod
    def _copy_tool_input(tool_call: ToolInput, tool_name: str, arguments: dict[str, Any]) -> ToolInput:
        """Copy a tool input across Home Assistant and lightweight test models."""
        if hasattr(tool_call, "id"):
            return ToolInput(id=tool_call.id, tool_name=tool_name, tool_args=arguments)
        return ToolInput(tool_name=tool_name, tool_args=arguments)

    def _save_json_load(self, json_string: str) -> dict | None:
        """Safely load a JSON string into a dictionary."""
        if not isinstance(json_string, str):
            return None

        json_string = json_string.strip()

        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            _logger.debug(f"Failed to parse JSON: {e}")
            return None

    def _tool_string_to_dict(self, tool_string: str) -> dict | None:
        """Convert a tool string to a dictionary."""
        if not isinstance(tool_string, str):
            return None

        tool_string = tool_string.strip()

        tool_json = None
        first_brace = tool_string.find('{')
        last_brace = tool_string.rfind('}')

        if first_brace >= 0 and last_brace > first_brace:
            json_str = tool_string[first_brace:last_brace + 1]
            tool_json = self._save_json_load(json_str)
                
        return tool_json

    def _parse_original_name(self, original_name: str | None, parameters: dict[str, Any]) -> str | None:
        """Parse the original name from parameters."""
        original_name = parameters.get("name", None)

        if "entity_id" in parameters:
            original_name = parameters.pop("entity_id")

        return original_name

    def _parse_domain(
        self,
        original_name: str | None,
        is_domain_aware: bool,
        parameters: dict[str, Any],
    ) -> List[str] | None:
        """Parse the domain from the parameters and set it in the parameters dictionary."""
        domain = parameters.get("domain", None)

        if not is_domain_aware:
            return None

        if domain:
            return domain if isinstance(domain, list) else [domain]

        if original_name and "." in original_name:
            domain = original_name.split(".", 1)[0]
            return [domain]

        return None

    def _parse_friendly_name_and_entity_entry(
        self,
        original_name: str | None,
        domain: List[str] | None,
    ) -> Tuple[str | None, EntityEntry | None]:
        """Parse the friendly name from the parameters and set it in the parameters dictionary."""
        friendly_name = original_name

        if isinstance(original_name, str) and "." not in original_name and domain:
            temp_name = f"{domain[0]}.{original_name}"
        else:
            temp_name = original_name

        state = self._hass.states.get(temp_name)
        if state:
            friendly_name = state.attributes.get("friendly_name", original_name)

        entity_reg = entity_registry.async_get(self._hass) if entity_registry else None
        entity_entry = entity_reg.async_get(temp_name) if entity_reg else None

        aliases = []
        if entity_entry:
            aliases = entity_registry.async_get_entity_aliases(self._hass, entity_entry)

        if aliases:
            friendly_name = aliases[0]

        return friendly_name, entity_entry

    def _parse_area_and_floor(
        self,
        entity_entry: EntityEntry | None,
        original_area: str | None,
        original_floor: str | None,
    ) -> Tuple[str | None, str | None]:
        """Parse area and floor from the parameters and set them in the parameters dictionary."""
        area = None
        floor = None
        area_id = entity_entry.area_id if entity_entry else None

        if not area_id and entity_entry and entity_entry.device_id and device_registry:
            device = device_registry.async_get(self._hass).async_get(entity_entry.device_id)
            area_id = device.area_id if device else None

        if area_id and area_registry:
            area = area_registry.async_get(self._hass).async_get_area(area_id)
            if area and area.floor_id and floor_registry:
                floor = floor_registry.async_get(self._hass).async_get_floor(area.floor_id)

        area_name = original_area if isinstance(original_area, str) and original_area else area.name if area and area.name else None
        floor_name = original_floor if isinstance(original_floor, str) and original_floor else floor.name if floor and floor.name else None
        return (area_name, floor_name)

    def _parse_parameters(self, parameters: dict[str, Any], tool_metadata: ToolMetadata | None) -> None:
        """Parse and normalize tool parameters."""
        if tool_metadata and not any((
            tool_metadata.is_domain_aware,
            tool_metadata.is_area_aware,
            tool_metadata.is_device_class_aware,
        )):
            return
        is_domain_aware = tool_metadata.is_domain_aware if tool_metadata else False
        is_area_aware = tool_metadata.is_area_aware if tool_metadata else False

        original_name = self._parse_original_name(parameters.get("name"), parameters)
        domain = self._parse_domain(original_name, is_domain_aware, parameters)
        area = parameters.get("area", None)
        floor = parameters.get("floor", None)

        if not isinstance(original_name, str) or ("." not in original_name and not isinstance(domain, list)):
            return

        friendly_name, entity_entry = self._parse_friendly_name_and_entity_entry(original_name, domain)
        area_name, floor_name = self._parse_area_and_floor(entity_entry, area, floor)

        parameters["original_name"] = original_name
        parameters["friendly_name"] = friendly_name

        if is_domain_aware:
            parameters["domain"] = domain

        if is_area_aware:
            if area_name:
                parameters["area"] = area_name
            if floor_name:
                parameters["floor"] = floor_name

    def parse_tool_calls(
        self,
        llm_response: str,
        tool_metadata_dic: Dict[str, ToolMetadata] | None = None,
    ) -> List[ToolInput]:
        """Parse tool calls from LLM response."""
        parsed_calls = []
        tool_metadata_dic = tool_metadata_dic or {}
        
        for match in TOOL_REGEX_PATTERN.finditer(llm_response):
            tool_json = self._tool_string_to_dict(match.group(1))

            if tool_json is None:
                _logger.debug(f"Failed to parse tool call from LLM response: {match.group(1)}")
                continue

            tool_name = tool_json.get("tool")
            if not tool_name:
                _logger.debug(f"Tool name missing in tool call: {tool_json}")
                continue

            parameters = tool_json.get("arguments")
            if isinstance(parameters, str):
                parameters = self._save_json_load(parameters)

            if not isinstance(parameters, dict):
                _logger.debug(f"Empty tool arguments: {tool_json.get('arguments')}")
                continue

            self._parse_parameters(parameters, tool_metadata_dic.get(tool_name))
            parsed_call = ToolInput(tool_name=tool_name, tool_args=parameters)
            parsed_calls.append(parsed_call)

        return parsed_calls

    @staticmethod
    def tool_call_signature(tool_call: ToolInput) -> str:
        """Return a stable signature for a tool name and its arguments."""
        arguments = tool_call.tool_args
        if ToolHelper.is_semantic_search_tool(tool_call.tool_name):
            arguments = {
                "search_query": ToolHelper._normalize_search_query(arguments.get("search_query", "")),
                "scope": str(arguments.get("scope", "devices_and_tools")).strip().casefold(),
            }
        return json.dumps(
            {
                "tool": tool_call.tool_name,
                "arguments": arguments,
            },
            sort_keys=True
        )

    @staticmethod
    def _normalize_search_query(query: object) -> str:
        """Normalize semantically identical search text for turn-local reuse."""
        return RetrievalHelper.canonical_search_signature(query)

    @staticmethod
    def is_semantic_search_tool(tool_name: str) -> bool:
        """Return whether a name identifies the semantic-search tool."""
        return str(tool_name or "").rsplit("__", 1)[-1] == RAGENT_SEMANTIC_SEARCH_TOOL_NAME


    @staticmethod
    def merge_candidates(existing: list[dict[str, object]], discovered: list[dict[str, object]]) -> list[dict[str, object]]:
        """Retain earlier targets while refreshing newly retrieved candidates."""
        merged = {str(item.get("name")): item for item in existing}
        merged.update({str(item.get("name")): item for item in discovered})
        return list(merged.values())


    @staticmethod
    def resolve_exposed_tool_name(tool_name: str, exposed_names: set[str]) -> str | None:
        """Resolve harmless namespace variants without accepting invented tools."""
        requested = str(tool_name or "").casefold()
        requested_suffix = requested.rsplit("__", 1)[-1]
        matches = [
            exposed_name
            for exposed_name in exposed_names
            if exposed_name.casefold() == requested
            or exposed_name.casefold().rsplit("__", 1)[-1] == requested_suffix
        ]
        return matches[0] if len(matches) == 1 else None

    def normalize_exposed_tool_call(
        self,
        tool_call: ToolInput,
        exposed_names: set[str],
        metadata_by_name: dict[str, ToolMetadata] | None = None,
    ) -> ToolInput | None:
        """Return a call using an exposed name, or reject an invented name."""
        tool_name = ToolHelper.resolve_exposed_tool_name(tool_call.tool_name, exposed_names)
        if tool_name is None:
            return None
        if tool_name == tool_call.tool_name:
            return tool_call
        arguments = dict(tool_call.tool_args)
        self._parse_parameters(arguments, (metadata_by_name or {}).get(tool_name))
        return self._copy_tool_input(tool_call, tool_name, arguments)

    @staticmethod
    def candidate_devices(result: object) -> list[dict[str, object]]:
        """Return candidate devices from a semantic-search result."""
        if not isinstance(result, dict):
            return []
        candidates = result.get("candidate_devices", result.get("devices", []))
        if not isinstance(candidates, list):
            return []
        return [candidate for candidate in candidates if isinstance(candidate, dict)]

    @staticmethod
    def discovered_tools(result: object, existing_names: set[str]) -> list[LlmTool]:
        """Convert newly discovered candidate tools for the next iteration."""
        if not isinstance(result, dict):
            return []
        candidates = result.get("candidate_tools", result.get("tools", []))
        if not isinstance(candidates, list):
            return []

        discovered: list[LlmTool] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name", "") or "")
            if not name or name in existing_names:
                continue
            metadata = candidate.get("metadata")
            discovered.append(
                LlmTool(
                    name=name,
                    description=str(candidate.get("description", "") or ""),
                    parameters=candidate.get("parameters") or {},
                    metadata=ToolMetadata.from_dict(metadata) if isinstance(metadata, dict) else ToolMetadata(),
                )
            )
            existing_names.add(name)
        return discovered

    @staticmethod
    def successful_target_names(tool_call: ToolInput, result: object) -> list[str]:
        """Return target names confirmed by a successful tool result."""
        success = result.get("success") if isinstance(result, dict) else None
        if isinstance(success, str):
            return [success]
        if isinstance(success, (list, tuple, set)):
            return [str(value) for value in success if value]

        names = tool_call.tool_args.get("name") if isinstance(tool_call.tool_args, dict) else None
        if isinstance(names, str):
            return [names]
        if isinstance(names, (list, tuple, set)):
            return [str(value) for value in names if value]
        return []

    @staticmethod
    def is_identical_failed_retry(tool_call: ToolInput, failed_signatures: set[str] | dict[str, Any]) -> bool:
        """Return whether the same canonical call has already failed."""
        return ToolHelper.tool_call_signature(tool_call) in failed_signatures


    @staticmethod
    def parse_tool_results(tool_result: JsonObjectType) -> Dict[str, Any]:
        """Parse tool results from LLM response."""
        if not isinstance(tool_result, dict):
            return {"result": tool_result}

        data = tool_result.get("data", {})
        if not isinstance(data, dict):
            return tool_result
        success = data.get("success", [])
        failed = data.get("failed", [])
        if not isinstance(success, list) or not isinstance(failed, list):
            return tool_result
        parsed_result: Dict[str, Any] = {}
        
        success_ids = [x["id"] for x in success if isinstance(x, dict) and x.get("type") == "entity" and "id" in x]
        if success_ids:
            parsed_result["success"] = success_ids

        failed_ids = [x["id"] for x in failed if isinstance(x, dict) and x.get("type") == "entity" and "id" in x]
        if failed_ids:
            parsed_result["failed"] = failed_ids

        if parsed_result:
            return parsed_result

        return tool_result

    @staticmethod
    def to_home_assistant_tool_call(tool_call: ToolInput, metadata: ToolMetadata | None = None) -> ToolInput:
        """Create the Home Assistant call with parser metadata removed."""
        args = dict(tool_call.tool_args)
        if metadata and not any((
            metadata.is_domain_aware,
            metadata.is_area_aware,
            metadata.is_device_class_aware,
        )):
            return ToolHelper._copy_tool_input(tool_call, tool_call.tool_name, args)
        args.pop("original_name", None)
        friendly_name = args.pop("friendly_name", None)
        if friendly_name is not None:
            args["name"] = friendly_name
        return ToolHelper._copy_tool_input(tool_call, tool_call.tool_name, args)

    @staticmethod
    def to_history_tool_call(tool_call: ToolInput) -> ToolInput:
        """Create the history call with the original name restored."""
        args = dict(tool_call.tool_args)
        args.pop("friendly_name", None)
        original_name = args.pop("original_name", None)
        if original_name is not None:
            args["name"] = original_name
        return ToolHelper._copy_tool_input(tool_call, tool_call.tool_name, args)
