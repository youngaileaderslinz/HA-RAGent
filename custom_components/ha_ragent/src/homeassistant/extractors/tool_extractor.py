from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, List, Tuple

import voluptuous as vol
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

from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.homeassistant.ragent_api import resolve_llm_api_id
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

_logger = logging.getLogger(__name__)

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

    def _extract_values_from_validator(self, validator: Any) -> Tuple[set[str], bool]:
        values: set[str] = set()
        universal = False

        if isinstance(validator, vol.In):
            return self._normalize_strings(validator.container), False

        if isinstance(validator, vol.All) or isinstance(validator, vol.Any):
            for nested in validator.validators:
                nested_values, nested_universal = self._extract_values_from_validator(nested)
                values.update(nested_values)
                universal = universal or nested_universal
            return values, universal

        if isinstance(validator, (list, tuple, set)):
            for nested in validator:
                nested_values, nested_universal = self._extract_values_from_validator(nested)
                values.update(nested_values)
                universal = universal or nested_universal
            return values, universal

        if callable(validator) and getattr(validator, "__name__", "") == "string":
            return set(), True

        return set(), False

    def _extract_field_constraints(self, schema_dict: dict[Any, Any], field_name: str) -> Tuple[List[str], bool, bool]:
        values = set()
        universal = False
        has_field = False

        for raw_key, validator in schema_dict.items():
            if str(getattr(raw_key, "schema", raw_key)) != field_name:
                continue

            has_field = True
            found_values, found_universal = self._extract_values_from_validator(validator)
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
        if not action:
            schema_actions = cls._schema_values(properties.get("action", {}))
            if len(schema_actions) == 1:
                action = next(iter(schema_actions))
        metadata.canonical_action = str(action or "").casefold()

        domains = cls._metadata_value(source, "supported_domains", "domains", "domain", default=())
        if isinstance(domains, str):
            domains = (domains,)
        metadata.supported_domains = tuple(sorted({
            *(str(value).casefold() for value in (domains or ())),
            *cls._schema_values(properties.get("domain", {})),
        }))

        expected_states = cls._metadata_value(source, "expected_states", "expected_state", default=())
        if isinstance(expected_states, str):
            expected_states = (expected_states,)
        if not expected_states:
            expected_states = {
                *cls._schema_values(properties.get("expected_state", {})),
                *cls._schema_values(properties.get("expected_states", {})),
            }
        metadata.expected_states = tuple(sorted(str(value).casefold() for value in (expected_states or ())))
        metadata.family = cls._metadata_value(source, "family", default=None)

        return metadata

    def _extract_tool_metadata(self, tool: Any, parameters: Any) -> ToolMetadata:
        """Compatibility wrapper for existing callers."""
        return self.extract_tool_metadata(tool, parameters)

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
            _logger.debug("Registered timer support for HA-RAGent")
        except Exception as err:
            _logger.warning(f"Failed to register timer device: {err}")
    
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
            _logger.debug("Unregistered timer support for HA-RAGent")
        except Exception as err:
            _logger.warning(f"Failed to unregister timer device: {err}")

    async def _async_get_embeddable_tools(self, subentry: ConfigSubentry) -> List[LlmTool]:
        tool_list: list[LlmTool] = []
        seen_tool_names: set[str] = set()
        selected_api = subentry.data.get(CONF_LLM_HASS_API, "default")
        excluded_tools = {
            name
            for name in subentry.data.get(CONF_EXCLUDED_TOOLS, [])
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
                _logger.debug(f"LLM API {selected_api} did not expose any tools attribute for subentry {subentry.title}")
                return tool_list

            _logger.debug(f"LLM API {selected_api} exposed {len(llm_api.tools)} raw tools for subentry {subentry.title}")

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

                if hasattr(tool, "parameters") and tool.parameters:
                    try:
                        parameters = to_openapi(tool.parameters, custom_serializer=llm_api.custom_serializer)
                        if not isinstance(parameters, dict):
                            _logger.warning(f"Could not convert parameters for tool {tool_name}: converter returned {type(parameters).__name__}")
                            parameters = {}
                    except Exception as param_err:
                        _logger.warning(f"Could not convert parameters for tool {tool_name}: {param_err}")
                        parameters = {}
                else:
                    parameters = {}

                tool_list.append(
                    LlmTool(
                        name=tool_name,
                        description=getattr(tool, "description", ""),
                        parameters=parameters,
                        metadata=self._extract_tool_metadata(tool, parameters),
                    )
                )
                seen_tool_names.add(tool_name)

        except HomeAssistantError as err:
            _logger.warning(f"Error getting LLM API for tool extraction: {err}")
            return []
        except Exception as err:
            _logger.error(f"Error extracting tools from LLM API: {err}", exc_info=True)
            return []
        finally:
            self._remove_fake_timer_device()

        return tool_list

    async def async_get_embeddable_tool_names(self, subentry: ConfigSubentry) -> list[str]:
        """Return the tool names currently produced by the extractor."""
        tools = await self._async_get_embeddable_tools(subentry)
        return [tool.name for tool in tools or []]

    async def async_embed_exposed_tools(self, subentry_id: str) -> None:
        total_embedded_tools = 0
        try:
            _logger.debug("Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.debug("No subentries found in config entry. Cannot embed tools.")
                return

            subentry = self._entry.subentries.get(subentry_id)
            if not subentry:
                _logger.debug("No matching subentries found for tool embedding.")
                return
            try:
                exposed_tools = await self._async_get_embeddable_tools(subentry)
                _logger.debug(f"Tool embedding starting: {len(exposed_tools)} exposed to conversation. ({[tool.name for tool in exposed_tools]})")
                if not exposed_tools:
                    collection_name = f"tools_{subentry_id}"
                    await self._entry.vector_db_backend.async_cleanup_collection(dict(subentry.data), collection_name)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, [])
                    _logger.info("Cleared tool embeddings for empty subentry %s", subentry_id)
                    return

                collection_name = f"tools_{subentry_id}"
                tool_embeddings = await self._entry.embedder_backend.async_embed_object(dict(subentry.data), exposed_tools)

                if tool_embeddings:
                    embedding_len = len(tool_embeddings[0].vector_embedding)
                    self._entry.vector_db_backend.invalidate_collection_cache(collection_name)
                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)
                    _logger.debug(f"Saving {len(tool_embeddings)} tool embeddings to collection {collection_name}.")
                    await self._entry.vector_db_backend.async_save_objects(dict(subentry.data), collection_name, tool_embeddings)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, exposed_tools)
                    total_embedded_tools += len(tool_embeddings)
                else:
                    _logger.warning(f"No tools to embed for subentry {subentry_id}")
            except Exception as err:
                _logger.error(f"Error in background embedding job for subentry {subentry_id}: {err}", exc_info=True)
        except Exception as err:
            _logger.error(f"Error in tool embedding job: {err}", exc_info=True)
        finally:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(f"Tool embedding function finished with {total_embedded_tools} embedded tools.")
            else:
                _logger.info(f"Finished embedding {total_embedded_tools} tools.")
