from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import DOMAIN, RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME, RAGENT_SCHEDULED_ACTIONS
from custom_components.ha_ragent.src.translation import RAGentTranslations


class RAGentListPlannedActionsTool(llm.Tool):
    name = RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME
    parameters = vol.Schema({})

    def __init__(self, hass: HomeAssistant, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.subentry_id = subentry_id
        self.translations = RAGentTranslations(language or "en")
        self.description = self.translations.tool(RAGENT_LIST_PLANNED_ACTIONS_TOOL_NAME)

    async def async_call(self, _tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        actions = self.hass.data.get(DOMAIN, {}).get(self.subentry_id, {}).get(RAGENT_SCHEDULED_ACTIONS, {})
        return {"success": True, "actions": list(actions.values()), "count": len(actions)}
