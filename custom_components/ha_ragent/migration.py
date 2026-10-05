from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.const import (
    CONF_PROMPT_LAYOUT,
    CONF_RULE_PROMPT,
    CONF_SELECTED_LANGUAGE,
    CONF_STATE_PROMPT,
    CONFIG_FLOW_VERSION,
    PROMPT_LAYOUT_COMBINED,
)
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

LEGACY_PROMPT_KEY = "rag_prompt"


def _migrate_to_version_2(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    """Move version 1 language and prompts to their version 2 settings."""
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
        subentry_data[CONF_STATE_PROMPT] = ""
        subentry_data[CONF_PROMPT_LAYOUT] = PROMPT_LAYOUT_COMBINED
        hass.config_entries.async_update_subentry(entry, subentry, data=subentry_data)

    hass.config_entries.async_update_entry(entry, data=data, options=options, version=2)


async def async_migrate_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> bool:
    """Apply each config entry migration in version order."""
    version = entry.version
    while version < CONFIG_FLOW_VERSION:
        if version == 1:
            _migrate_to_version_2(hass, entry)
            version = 2
        else:
            return False

    return version == CONFIG_FLOW_VERSION
