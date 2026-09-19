from custom_components.ha_ragent.src.logging.base_logger import BaseLogger
from functools import partial

import voluptuous as vol
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation, entity_registry, target

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry

_logger = BaseLogger(__name__)

async def _handle_unload_models(hass: HomeAssistant, call: ServiceCall) -> None:
    entity_reg = entity_registry.async_get(hass)
    target_selector = target.TargetSelection(call.data)
    referenced = target.async_extract_referenced_entity_ids(hass, target_selector)

    for entity_id in referenced.referenced | referenced.indirectly_referenced:
        entry = entity_reg.async_get(entity_id)
        if not entry or entry.platform != DOMAIN or not entry.config_subentry_id:
            continue

        parent: RAGentConfigEntry = hass.config_entries.async_get_entry(entry.config_entry_id)
        if not parent:
            continue

        sub = parent.subentries.get(entry.config_subentry_id)
        if not sub:
            continue

        _logger.debug("Unloading model for: %s", sub.title)
        await parent.embedder_backend.async_unload_model(dict(sub.data))
        await parent.llm_backend.async_unload_model(dict(sub.data))


def register_unload_models_service(hass: HomeAssistant) -> None:
    hass.services.async_register(
        DOMAIN,
        "unload_models",
        partial(_handle_unload_models, hass),
        schema=vol.Schema({}).extend(config_validation.TARGET_SERVICE_FIELDS),
    )
