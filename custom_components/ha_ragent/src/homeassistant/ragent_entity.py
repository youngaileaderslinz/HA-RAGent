from __future__ import annotations
import logging
from typing import List, Any, Optional, Dict, Literal

from homeassistant.components.conversation.const import DOMAIN as CONVERSATION_DOMAIN
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import  device_registry, entity

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    CONF_SELECTED_LANGUAGE,
    CONF_LLM_MODEL
)
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

_logger = logging.getLogger(__name__)

class RAGentEntity(entity.Entity):
    hass: HomeAssistant
    entry_id: str
    in_context_examples: Optional[List[Dict[str, str]]]

    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry, subentry: ConfigSubentry) -> None:
        self._attr_name = subentry.title
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = device_registry.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            model=subentry.data.get(CONF_LLM_MODEL),
            entry_type=device_registry.DeviceEntryType.SERVICE,
        )

        self.hass = hass
        self.entry_id = entry.entry_id
        self.subentry_id = subentry.subentry_id

        # create update handler
        self.async_on_remove(entry.add_update_listener(self._async_update_options))

    async def _async_update_options(self, hass: HomeAssistant, config_entry: RAGentConfigEntry) -> None:
        for subentry in config_entry.subentries.values():
            if subentry.subentry_id == self.subentry_id:
                hass.config_entries.async_update_entry(config_entry, options=self.runtime_options)

    @property
    def entry(self) -> RAGentConfigEntry:
        try:
            return self.hass.data[DOMAIN][self.entry_id]
        except KeyError as ex:
            raise Exception("Attempted to use self.entry during startup.") from ex

    @property
    def subentry(self) -> ConfigSubentry:
        try:
            return self.entry.subentries[self.subentry_id]
        except KeyError as ex:
            raise Exception("Attempted to use self.subentry during startup.") from ex
        
    @property
    def runtime_options(self) -> dict[str, Any]:
        """Return the runtime options for this entity."""
        return {**self.entry.options, **self.subentry.data}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return self.entry.options.get(CONF_SELECTED_LANGUAGE, MATCH_ALL)
