from __future__ import annotations

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from custom_components.ha_ragent.src.const import RAGENT_FORGET_TOOL_NAME, TRANSLATION_ERROR_MEMORY_ID_INVALID, TRANSLATION_ERROR_MEMORY_NOT_FOUND
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.translation import RAGentTranslations


class RAGentForgetTool(llm.Tool):
    name = RAGENT_FORGET_TOOL_NAME
    parameters = vol.Schema({vol.Required("memory_id"): vol.All(str, vol.Match(r"^[0-9a-f]{16}$"))})

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass, self.entry_id, self.subentry_id = hass, entry_id, subentry_id
        self.translations = RAGentTranslations(language or "en")
        self.description = self.translations.tool(RAGENT_FORGET_TOOL_NAME)

    async def async_call(self, tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        memory_id = str(tool_input.tool_args.get("memory_id", "")).strip().lower()
        if len(memory_id) != 16 or any(char not in "0123456789abcdef" for char in memory_id):
            return {"success": False, "error": self.translations.error(TRANSLATION_ERROR_MEMORY_ID_INVALID)}

        forgotten = await MemoryManager(self.hass, self.entry_id, self.subentry_id).async_forget(memory_id)
        return {"success": forgotten, "memory_id": memory_id, "forgotten": forgotten, **({} if forgotten else {"error": self.translations.error(TRANSLATION_ERROR_MEMORY_NOT_FOUND)})}
