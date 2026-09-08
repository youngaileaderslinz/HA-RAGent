from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_SCHEDULED_ACTION_CANCELLERS,
    RAGENT_SCHEDULED_ACTIONS,
)
from custom_components.ha_ragent.src.translation import RAGentTranslations


class RAGentCancelAllPlannedActionsTool(llm.Tool):
    name = RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME
    parameters = vol.Schema({})

    def __init__(self, hass: HomeAssistant, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.subentry_id = subentry_id
        self.translations = RAGentTranslations(language or "en")
        self.description = self.translations.tool(RAGENT_CANCEL_ALL_PLANNED_ACTIONS_TOOL_NAME)

    async def async_call(self, _tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        domain_data = self.hass.data.setdefault(DOMAIN, {})
        subentry_data = domain_data.setdefault(self.subentry_id, {})
        cancellers = subentry_data.get(RAGENT_SCHEDULED_ACTION_CANCELLERS, set())
        subentry_data[RAGENT_SCHEDULED_ACTION_CANCELLERS] = set()
        subentry_data[RAGENT_SCHEDULED_ACTIONS] = {}
        for cancel in cancellers:
            cancel()
        return {"success": True, "cancelled": len(cancellers)}
