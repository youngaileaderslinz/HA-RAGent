import asyncio
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from functools import partial

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation, entity_registry, target

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.extractors.device_extractor import DeviceExtractor
from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

_logger = BaseLogger(__name__)

async def _handle_embed_subentry(hass: HomeAssistant, call: ServiceCall) -> None:
    entity_reg = entity_registry.async_get(hass)
    target_selector = target.TargetSelection(call.data)
    referenced = target.async_extract_referenced_entity_ids(hass, target_selector)

    processed_subentries: set[tuple[str, str]] = set()

    for entity_id in referenced.referenced | referenced.indirectly_referenced:
        entry = entity_reg.async_get(entity_id)
        if not entry or entry.platform != DOMAIN or not entry.config_subentry_id:
            continue

        parent: RAGentConfigEntry = hass.config_entries.async_get_entry(entry.config_entry_id)
        if not parent:
            continue

        subentry = parent.subentries.get(entry.config_subentry_id)
        if not subentry:
            continue

        subentry_key = (parent.entry_id, entry.config_subentry_id)
        if subentry_key in processed_subentries:
            continue

        processed_subentries.add(subentry_key)
        _logger.debug("Embedding devices and tools for subentry: %s", subentry.title)

        tool_extractor = ToolExtractor(hass, parent)
        device_extractor = DeviceExtractor(hass, parent)
        await asyncio.gather(
            tool_extractor.async_embed_exposed_tools(entry.config_subentry_id),
            device_extractor.async_embed_exposed_devices(entry.config_subentry_id),
        )

def register_embed_subentry_service(hass: HomeAssistant) -> None:
    hass.services.async_register(
        DOMAIN,
        "embed_subentry",
        partial(_handle_embed_subentry, hass),
        schema=vol.Schema({}).extend(config_validation.TARGET_SERVICE_FIELDS),
    )
