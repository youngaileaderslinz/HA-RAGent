from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.homeassistant.ragent import RAGent

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up HA Ragent Conversation from a config entry."""
    for subentry in entry.subentries.values():
        # create one agent entity per conversation subentry
        agent_entity = RAGent(hass, entry, subentry)

        # register the agent entity
        async_add_entities([agent_entity], config_subentry_id=subentry.subentry_id)

    return True
