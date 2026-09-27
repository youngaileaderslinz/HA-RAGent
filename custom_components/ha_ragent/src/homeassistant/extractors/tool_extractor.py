from __future__ import annotations

import logging
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from collections.abc import Iterable
from typing import Any, List, Tuple

import probatio
from probatio import to_openapi

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.intent import async_register_timer_handler
from homeassistant.components.intent.timers import TimerEventType, TimerInfo
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.llm import LLMContext

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    CONF_EXCLUDED_TOOLS,
    RAGENT_PREFIXED_REQUIRED_TOOL_NAMES,
    RAGENT_TIMER_DEVICE_ID,
)

from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    ToolMetadata,
    split_canonical_name,
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.schema_constraints import root_property_values
from custom_components.ha_ragent.src.homeassistant.ragent_api import resolve_llm_api_id
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.utils import get_setting_value

_logger = BaseLogger(__name__)

class ToolExtractor:
    _timer_handlers: dict[int, tuple[Any, int]] = {}

    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._fake_timer_remove = None
        self._timer_handler_key = id(hass)

    @staticmethod
    def _normalize_strings(values: Iterable[Any]) -> set[str]:
        return {str(value).lower() for value in values if isinstance(value, str) and value}

    @classmethod
    def _extract_values_from_validator(cls, validator: Any) -> Tuple[set[str], bool]:
        values: set[str] = set()
        universal = False

        if isinstance(validator, probatio.In):
            return cls._normalize_strings(validator.container), False

        if isinstance(validator, probatio.All):
            constrained: list[set[str]] = []
            for nested in validator.validators:
                nested_values, nested_universal = cls._extract_values_from_validator(nested)
                if nested_values:
                    constrained.append(nested_values)
                universal = universal or nested_universal
            # A free-form conjunct is neutral; multiple literal conjuncts
            # narrow their intersection rather than broadening a domain list.
            return (set.intersection(*constrained) if constrained else set()), universal

        if isinstance(validator, probatio.Any):
            for nested in validator.validators:
                nested_values, nested_universal = cls._extract_values_from_validator(nested)
                values.update(nested_values)
                universal = universal or nested_universal
            return values, universal

        if isinstance(validator, (list, tuple, set)):
            for nested in validator:
                nested_values, nested_universal = cls._extract_values_from_validator(nested)
                values.update(nested_values)
                universal = universal or nested_universal
            return values, universal

        if callable(validator) and getattr(validator, "__name__", "") in {"string", "str"}:
            return set(), True

        return set(), False

    @classmethod
    def _extract_field_constraints(cls, schema_dict: dict[Any, Any], field_name: str) -> Tuple[List[str], bool, bool]:
        values = set()
        universal = False
        has_field = False

        for raw_key, validator in schema_dict.items():
            if str(getattr(raw_key, "schema", raw_key)) != field_name:
                continue

            has_field = True
            found_values, found_universal = cls._extract_values_from_validator(validator)
            values.update(found_values)
            universal = universal or found_universal

        return list(values), universal, has_field

    @staticmethod
    def _schema_values(schema: object) -> set[str]:
        """Extract explicit const/enum values from an OpenAPI schema."""
        values: set[str] = set()
        if isinstance(schema, dict):
            constant = schema.get("const")
            if isinstance(constant, (str, int, float)):
                values.add(str(constant).casefold())
            enum = schema.get("enum")
            if isinstance(enum, list):
                values.update(str(value).casefold() for value in enum)
            for keyword in ("anyOf", "oneOf", "allOf"):
                for nested in schema.get(keyword, []):
                    values.update(ToolExtractor._schema_values(nested))
        return values

    @staticmethod
    def _metadata_value(metadata: object, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(metadata, dict) and name in metadata:
                return metadata[name]
            value = getattr(metadata, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def extract_tool_metadata(cls, tool: Any, parameters: Any) -> ToolMetadata:
        """Build capability metadata from explicit tool metadata and schema only."""
        metadata = ToolMetadata()
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        if not isinstance(properties, dict):
            properties = {}

        metadata.is_domain_aware = "domain" in properties
        metadata.is_area_aware = "area" in properties or "floor" in properties
        metadata.is_device_class_aware = "device_class" in properties

        source = getattr(tool, "metadata", None)
        action = cls._metadata_value(source, "canonical_action", "action", "capability_id", default="")
        if not action and (intent_type := getattr(tool, "intent_type", "")):
            parts = split_canonical_name(intent_type)
            action = "_".join(parts[1:] if parts[:1] == ("hass",) else parts)
        if not action:
            schema_actions = cls._schema_values(properties.get("action", {}))
            if len(schema_actions) == 1:
                action = next(iter(schema_actions))
        metadata.canonical_action = str(action or "").casefold()

        domains = cls._metadata_value(source, "supported_domains", "domains", "domain", default=())
        if isinstance(domains, str):
            domains = (domains,)
        # Metadata restrictions and ranking use the same conservative schema
        # interpreter.  Literal values in an unrestricted alternative remain
        # searchable schema content, but are not a supported-domain claim.
        schema_domains = root_property_values(parameters, "domain")
        # Keep the original Probatio schema as a fallback. Some HA/API
        # adapters flatten constrained fields while converting to OpenAPI and
        # silently drop the enum, even though the source schema still has it.
        raw_parameters = getattr(tool, "parameters", None)
        raw_schema = (
            raw_parameters
            if isinstance(raw_parameters, dict)
            else getattr(raw_parameters, "schema", None)
        )
        if isinstance(raw_schema, dict):
            raw_domains, _universal, has_domain = cls._extract_field_constraints(
                raw_schema, "domain",
            )
            if has_domain:
                # Probatio can express a union with a free-form string.  In
                # that case the source schema is likewise neutral.
                if not _universal:
                    schema_domains.update(raw_domains)
        metadata.supported_domains = tuple(sorted({
            *(str(value).casefold() for value in (domains or ())),
            *schema_domains,
        }))

        expected_states = cls._metadata_value(source, "expected_states", "expected_state", default=())
        if isinstance(expected_states, str):
            expected_states = (expected_states,)
        if not expected_states:
            expected_states = {
                *cls._schema_values(properties.get("expected_state", {})),
                *cls._schema_values(properties.get("expected_states", {})),
            }
        if not expected_states:
            expected_states = {
                "turn_on": ("on",),
                "turn_off": ("off",),
            }.get(metadata.canonical_action, ())
        metadata.expected_states = tuple(sorted(str(value).casefold() for value in (expected_states or ())))
        return metadata

    @staticmethod
    def _openai_parameters(schema: dict[str, Any]) -> dict[str, Any]:
        """Translate a Probatio OpenAPI schema into OpenAI tool parameters."""
        if not isinstance(schema, dict):
            raise ValueError("Tool parameters must be an object schema")

        def type_name(value: Any) -> str:
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, int):
                return "integer"
            if isinstance(value, float):
                return "number"
            if isinstance(value, str):
                return "string"
            raise ValueError("Unsupported enum value in tool schema")

        def branch_type(branch: dict[str, Any]) -> str | None:
            if isinstance(branch.get("type"), str):
                return branch["type"]
            values = branch.get("enum", [branch["const"]] if "const" in branch else [])
            return ("null" if values[0] is None else type_name(values[0])) if values else None

        def normalize(node: dict[str, Any], *, optional: bool = False) -> dict[str, Any]:
            node = dict(node)
            for keyword in ("anyOf", "oneOf"):
                if keyword not in node:
                    continue
                branches = node.pop(keyword)
                if not isinstance(branches, list) or not all(isinstance(part, dict) for part in branches):
                    raise ValueError(f"Unsupported {keyword} in tool schema")
                # Intent slot groups use alternatives of required-only objects.
                if "properties" in node and all(set(part) == {"required"} for part in branches):
                    continue
                null_branches = [part for part in branches if part.get("type") == "null"]
                other_branches = [part for part in branches if part.get("type") != "null"]
                if len(null_branches) == 1 and len(other_branches) == 1:
                    node = {**other_branches[0], **node}
                    optional = True
                elif other_branches and not null_branches and all(
                    branch_type(part) is not None for part in other_branches
                ):
                    types = list(dict.fromkeys(branch_type(part) for part in other_branches))
                    node["type"] = types[0] if len(types) == 1 else types
                    if all("enum" in part or "const" in part for part in other_branches):
                        values = []
                        for part in other_branches:
                            values.extend(part.get("enum", [part["const"]] if "const" in part else []))
                        node["enum"] = list(dict.fromkeys(values))
                    array_branches = [part for part in other_branches if branch_type(part) == "array"]
                    if len(array_branches) == 1 and "items" in array_branches[0]:
                        node["items"] = array_branches[0]["items"]
                else:
                    raise ValueError(f"Unsupported {keyword} in tool schema")
            if "allOf" in node:
                branches = node.pop("allOf")
                if not isinstance(branches, list) or not all(isinstance(part, dict) for part in branches):
                    raise ValueError("Unsupported allOf in tool schema")
                for part in branches:
                    for key, value in part.items():
                        if key in node and node[key] != value:
                            raise ValueError(f"Conflicting allOf {key} in tool schema")
                        node[key] = value
            for keyword in ("not", "if", "then", "else"):
                if keyword in node:
                    raise ValueError(f"Unsupported {keyword} in tool schema")
            optional = optional or node.pop("nullable", False) is True
            node.pop("default", None)
            node.pop("format", None)
            for keyword in ("dependentRequired", "dependentSchemas", "propertyNames", "patternProperties"):
                if keyword in node:
                    raise ValueError(f"Unsupported {keyword} in tool schema")
            if "const" in node:
                node["enum"] = [node.pop("const")]
            if "enum" in node and "type" not in node and node["enum"]:
                types = list(dict.fromkeys(
                    "null" if value is None else type_name(value)
                    for value in node["enum"]
                ))
                node["type"] = types[0] if len(types) == 1 else types
            if "properties" in node or node.get("type") == "object":
                properties = node.get("properties", {})
                if not isinstance(properties, dict):
                    raise ValueError("Object properties must be a mapping")
                required = set(node.get("required", []))
                node["type"] = "object"
                node["properties"] = {
                    name: normalize(value, optional=name not in required)
                    for name, value in properties.items()
                }
                node["required"] = list(properties)
                node["additionalProperties"] = False
            if "items" in node:
                node["items"] = normalize(node["items"])
            if "$defs" in node:
                node["$defs"] = {
                    name: normalize(value) for name, value in node["$defs"].items()
                }
            if optional:
                types = node.get("type")
                if isinstance(types, str):
                    node["type"] = [types, "null"] if types != "null" else ["null"]
                elif isinstance(types, list):
                    node["type"] = list(dict.fromkeys([*types, "null"]))
                else:
                    raise ValueError("Optional property has no JSON Schema type")
                if "enum" in node and None not in node["enum"]:
                    node["enum"] = [*node["enum"], None]
            return node

        if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
            schema = schema["function"].get("parameters", {})
        if schema and schema.get("type", "object") != "object":
            raise ValueError("Tool parameters must be an object schema")
        return normalize(schema or {"type": "object", "properties": {}})

    @classmethod
    def _tool_parameters(cls, tool: Any, custom_serializer: Any) -> dict[str, Any]:
        """Convert a Home Assistant tool's native schema once."""
        source = getattr(tool, "parameters", None)
        if isinstance(source, dict) and (
            source.get("type") == "object"
            or source.get("type") == "function"
            or "properties" in source
        ):
            parameters = source
        else:
            parameters = to_openapi(
                source or probatio.Schema({}),
                custom_serializer=custom_serializer,
            )
        return cls._openai_parameters(parameters)

    def _register_fake_timer_device(self) -> None:
        @callback
        def handle_timer_event(event_type: TimerEventType, timer: TimerInfo) -> None:
            pass

        try:
            shared = self._timer_handlers.get(self._timer_handler_key)
            if shared is not None:
                remove, count = shared
                self._timer_handlers[self._timer_handler_key] = (remove, count + 1)
                self._fake_timer_remove = True
                return
            remove = async_register_timer_handler(self._hass, RAGENT_TIMER_DEVICE_ID, handle_timer_event)
            self._timer_handlers[self._timer_handler_key] = (remove, 1)
            self._fake_timer_remove = True
            _logger.log_string(logging.DEBUG, "Registered timer support for HA-RAGent")
        except Exception as err:
            _logger.log_string(logging.WARNING, f"Failed to register timer device: {err}")
    
    def _remove_fake_timer_device(self) -> None:
        if not self._fake_timer_remove:
            return
        
        try:
            shared = self._timer_handlers.get(self._timer_handler_key)
            if shared is None:
                self._fake_timer_remove = None
                return
            remove, count = shared
            if count > 1:
                self._timer_handlers[self._timer_handler_key] = (remove, count - 1)
            else:
                remove()
                self._timer_handlers.pop(self._timer_handler_key, None)
            self._fake_timer_remove = None
            _logger.log_string(logging.DEBUG, "Unregistered timer support for HA-RAGent")
        except Exception as err:
            _logger.log_string(logging.WARNING, f"Failed to unregister timer device: {err}")

    async def _async_get_embeddable_tools(self, subentry: ConfigSubentry) -> List[LlmTool]:
        tool_list: list[LlmTool] = []
        seen_tool_names: set[str] = set()
        selected_api = get_setting_value(CONF_LLM_HASS_API, subentry.data) or "default"
        excluded_tools = {
            name
            for name in (get_setting_value(CONF_EXCLUDED_TOOLS, subentry.data) or [])
            if isinstance(name, str) and name.strip()
        }

        if selected_api == "none":
            return tool_list

        try:
            self._register_fake_timer_device()
            llm_context = LLMContext(
                platform=DOMAIN,
                context=None,
                language=None,
                assistant=CONVERSATION_DOMAIN,
                device_id=RAGENT_TIMER_DEVICE_ID,
            )

            llm_api = await llm.async_get_api(
                self._hass,
                resolve_llm_api_id(selected_api),
                llm_context=llm_context,
            )

            if not llm_api or not hasattr(llm_api, "tools"):
                _logger.log_string(logging.DEBUG, f"LLM API {selected_api} did not expose any tools attribute for subentry {subentry.title}")
                return tool_list

            _logger.log_string(logging.DEBUG, f"LLM API {selected_api} exposed {len(llm_api.tools)} raw tools for subentry {subentry.title}")

            for tool in llm_api.tools:
                tool_name = getattr(tool, "name", "unknown")
                base_tool_name = tool_name.rsplit("__", 1)[-1]
                if (
                    base_tool_name == "GetLiveContext"
                    or tool_name in RAGENT_PREFIXED_REQUIRED_TOOL_NAMES
                    or tool_name in excluded_tools
                    or base_tool_name in excluded_tools
                    or tool_name in seen_tool_names
                ):
                    continue

                try:
                    parameters = self._tool_parameters(tool, llm_api.custom_serializer)
                except Exception as param_err:
                    _logger.log_string(logging.WARNING, f"Could not convert parameters for tool {tool_name}: {param_err}")
                    continue

                try:
                    metadata = self.extract_tool_metadata(tool, parameters)
                except Exception as metadata_err:
                    # One malformed tool must not erase every other tool from
                    # the startup index. Preserve the live schema for ranking
                    # and use neutral metadata for this individual tool.
                    _logger.log_string(logging.WARNING, f"Could not extract metadata for tool {tool_name}: {metadata_err}")
                    metadata = ToolMetadata()

                tool_list.append(
                    LlmTool(
                        name=tool_name,
                        description=getattr(tool, "description", ""),
                        parameters=parameters,
                        metadata=metadata,
                    )
                )
                seen_tool_names.add(tool_name)

        except HomeAssistantError as err:
            _logger.log_string(logging.WARNING, f"Error getting LLM API for tool extraction: {err}")
            return []
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Error extracting tools from LLM API: {err}")
            return []
        finally:
            self._remove_fake_timer_device()

        _logger.log_payload("tools", logging.WARNING, tools=tool_list)
        return tool_list

    async def async_get_embeddable_tool_names(self, subentry: ConfigSubentry) -> list[str]:
        """Return the tool names currently produced by the extractor."""
        tools = await self._async_get_embeddable_tools(subentry)
        return [tool.name for tool in tools or []]

    async def async_embed_exposed_tools(self, subentry_id: str) -> None:
        total_embedded_tools = 0
        try:
            _logger.log_string(logging.DEBUG, "Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.log_string(logging.DEBUG, "No subentries found in config entry. Cannot embed tools.")
                return

            subentry = self._entry.subentries.get(subentry_id)
            if not subentry:
                _logger.log_string(logging.DEBUG, "No matching subentries found for tool embedding.")
                return
            try:
                exposed_tools = await self._async_get_embeddable_tools(subentry)
                _logger.log_string(logging.DEBUG, f"Tool embedding starting: {len(exposed_tools)} exposed to conversation. ({[tool.name for tool in exposed_tools]})")
                if not exposed_tools:
                    collection_name = f"tools_{subentry_id}"
                    await self._entry.vector_db_backend.async_cleanup_collection(dict(subentry.data), collection_name)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, [])
                    _logger.log_string(logging.INFO, f"Cleared tool embeddings for empty subentry {subentry_id}")
                    return

                collection_name = f"tools_{subentry_id}"
                tool_embeddings = await self._entry.embedder_backend.async_embed_object(
                    dict(subentry.data), exposed_tools,
                    getattr(self._entry, "translations", None),
                )

                if tool_embeddings:
                    embedding_len = len(tool_embeddings[0].vector_embedding)
                    self._entry.vector_db_backend.invalidate_collection_cache(collection_name)
                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)
                    _logger.log_string(logging.DEBUG, f"Saving {len(tool_embeddings)} tool embeddings to collection {collection_name}.")
                    await self._entry.vector_db_backend.async_save_objects(dict(subentry.data), collection_name, tool_embeddings)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, exposed_tools)
                    total_embedded_tools += len(tool_embeddings)
                else:
                    _logger.log_string(logging.WARNING, f"No tools to embed for subentry {subentry_id}")
            except Exception as err:
                _logger.log_string(logging.ERROR, f"Error in background embedding job for subentry {subentry_id}: {err}")
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Error in tool embedding job: {err}")
        finally:
            if _logger.is_enabled_for(logging.DEBUG):
                _logger.log_string(logging.DEBUG, f"Tool embedding function finished with {total_embedded_tools} embedded tools.")
            else:
                _logger.log_string(logging.INFO, f"Finished embedding {total_embedded_tools} tools.")
