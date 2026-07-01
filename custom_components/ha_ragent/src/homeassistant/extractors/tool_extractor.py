from __future__ import annotations

import logging
from typing import Any, Iterable, List, Tuple

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.intent import async_register_timer_handler
from homeassistant.components.intent.timers import TIMER_DATA, TimerEventType, TimerInfo

import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.llm import LLMContext

from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding

from ..ragent_config_entry import RAGentConfigEntry
from ...models.tool import LlmTool
from ...const import DOMAIN, RAGENT_TIMER_DEVICE_ID

_logger = logging.getLogger(__name__)

class ToolExtractor:
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
        self._hass = hass
        self._entry = entry
        self._fake_timer_remove = None

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

    def _extract_tool_metadata(self, tool: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "domains": {},
            "device_classes": {},
            "is_domain_universal": False,
            "is_domain_aware": False,
        }

        parameters = getattr(tool, "parameters", None)
        schema_dict = getattr(parameters, "schema", None) if parameters else None

        if not isinstance(schema_dict, dict):
            return metadata

        domains, domain_universal, has_domain = self._extract_field_constraints(schema_dict, "domain")
        device_classes, device_class_universal, has_device_class = self._extract_field_constraints(schema_dict, "device_class")

        metadata["domains"] = list(domains)
        metadata["is_domain_universal"] = domain_universal
        metadata["is_domain_aware"] = has_domain or has_device_class
        metadata["device_classes"] = list(device_classes)
        metadata["is_device_class_universal"] = device_class_universal
        metadata["is_device_class_aware"] = has_device_class
        return metadata

    def _register_fake_timer_device(self) -> None:
        @callback
        def handle_timer_event(event_type: TimerEventType, timer: TimerInfo) -> None:
            pass

        try:
            self._fake_timer_remove = async_register_timer_handler(self._hass, RAGENT_TIMER_DEVICE_ID, handle_timer_event)
            _logger.debug("Registered timer support for HA-RAGent")
        except Exception as err:
            _logger.warning("Failed to register timer device: %s", err)
    
    def _remove_fake_timer_device(self) -> None:
        if not self._fake_timer_remove:
            return
        
        try:
            self._fake_timer_remove()
            _logger.debug("Unregistered timer support for HA-RAGent")
        except Exception as err:
            _logger.warning("Failed to unregister timer device: %s", err)

    async def _async_get_embeddable_tools(self, subentry: ConfigSubentry) -> List[LlmTool]:
        tool_list: list[LlmTool] = []
        seen_tool_names: set[str] = set()
        selected_api = subentry.data.get(CONF_LLM_HASS_API, "default")

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
                selected_api,
                llm_context=llm_context,
            )

            if not llm_api or not hasattr(llm_api, "tools"):
                _logger.debug(
                    f"LLM API {selected_api} did not expose any tools attribute for subentry {subentry.title}",
                )
                return tool_list

            _logger.debug(
                f"LLM API {selected_api} exposed {len(llm_api.tools)} raw tools for subentry {subentry.title}",
            )

            for tool in llm_api.tools:
                tool_name = getattr(tool, "name", "unknown")
                if tool_name == "GetLiveContext" or tool_name in seen_tool_names:
                    continue

                if hasattr(tool, "parameters") and tool.parameters:
                    try:
                        parameters = convert(
                            tool.parameters,
                            custom_serializer=llm_api.custom_serializer,
                        )
                    except Exception as param_err:
                        _logger.warning(
                            "Could not convert parameters for tool %s: %s",
                            tool_name,
                            param_err,
                        )
                        parameters = {}
                else:
                    parameters = {}

                tool_list.append(
                    LlmTool(
                        name=tool_name,
                        description=getattr(tool, "description", ""),
                        parameters=parameters,
                        metadata=self._extract_tool_metadata(tool),
                    )
                )
                seen_tool_names.add(tool_name)

        except HomeAssistantError as err:
            _logger.warning(f"Error getting LLM API for tool extraction: {err}")
        except Exception as err:
            _logger.error(f"Error extracting tools from LLM API: {err}", exc_info=True)
        finally:
            self._remove_fake_timer_device()

        return tool_list

    async def async_embed_all_exposed_tools(self) -> None:
        total_embedded_tools = 0
        try:
            _logger.debug("Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.debug("No subentries found in config entry. Cannot embed tools.")
                return

            _logger.debug(f"Found {len(self._entry.subentries)} subentries to process.")

            for subentry_id, subentry in self._entry.subentries.items():
                try:
                    exposed_tools = await self._async_get_embeddable_tools(subentry)
                    _logger.debug(f"Tool embedding starting: {len(exposed_tools)} exposed to conversation. ({[tool.name for tool in exposed_tools]})")

                    if not exposed_tools:
                        _logger.debug(f"No tools to embed for subentry {subentry_id}")
                        continue

                    collection_name = f"tools_{subentry_id}"
                    embedding_len = len(await self._entry.embedder_backend.async_embed_text(dict(subentry.data), "Test"))

                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)    
                    tool_embeddings = await self._entry.embedder_backend.async_embed_object(LlmToolEmbedding, dict(subentry.data), exposed_tools)

                    if tool_embeddings:
                        _logger.debug(f"Saving {len(tool_embeddings)} tool embeddings to collection {collection_name}.")
                        await self._entry.vector_db_backend.async_save_object_embeddings(dict(subentry.data), collection_name, tool_embeddings)
                        total_embedded_tools += len(tool_embeddings)
                    else:
                        _logger.warning(f"No tools to embed for subentry {subentry_id}")
                except Exception as err:
                    _logger.error(f"Error in background embedding job for subentry {subentry_id}: {err}", exc_info=True)
                    continue
        except Exception as err:
            _logger.error(f"Error in tool embedding job: {err}", exc_info=True)
        finally:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug("Tool embedding function finished with %s embedded tools.", total_embedded_tools)
            else:
                _logger.info("Finished embedding %s tools.", total_embedded_tools)
