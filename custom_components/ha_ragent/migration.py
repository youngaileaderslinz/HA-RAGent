from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.const import (
    CONF_RULE_PROMPT,
    CONF_SELECTED_LANGUAGE,
    CONFIG_FLOW_VERSION,
)
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

LEGACY_PROMPT_KEY = "rag_prompt"

async def async_migrate_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> bool:
    """Move legacy language and agent prompts to their current settings."""
    if entry.version > CONFIG_FLOW_VERSION:
        return False
    if entry.version == CONFIG_FLOW_VERSION:
        return True

    data = dict(entry.data)
    options = dict(entry.options)
    legacy_language = options.pop(CONF_SELECTED_LANGUAGE, None)
    if CONF_SELECTED_LANGUAGE not in data and legacy_language is not None:
        data[CONF_SELECTED_LANGUAGE] = legacy_language

    for subentry in entry.subentries.values():
        subentry_data = dict(subentry.data)
        if LEGACY_PROMPT_KEY not in subentry_data:
            continue
        legacy_prompt = subentry_data.pop(LEGACY_PROMPT_KEY, None)
        if CONF_RULE_PROMPT not in subentry_data and legacy_prompt is not None:
            subentry_data[CONF_RULE_PROMPT] = legacy_prompt
        hass.config_entries.async_update_subentry(entry, subentry, data=subentry_data)

    hass.config_entries.async_update_entry(entry, data=data, options=options, version=CONFIG_FLOW_VERSION)
    return True
