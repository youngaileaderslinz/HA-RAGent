import asyncio
import logging
from typing import Any

from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STARTED
from homeassistant.config_entries import ConfigEntryState, OperationNotAllowed
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import entity_registry, target

from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.homeassistant.device_extractor import DeviceExtractor
from custom_components.ha_ragent.src.homeassistant.tool_extractor import ToolExtractor

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_LLM_BACKEND_TYPE,
    
    DEFAULT_VECTOR_DB_BACKEND_TYPE,
    DEFAULT_EMBEDDING_BACKEND_TYPE,
    DEFAULT_LLM_BACKEND_TYPE,    
    STARTUP_EMBEDDING_RUNNING_FLAG
)

import voluptuous as vol
from homeassistant.helpers import config_validation

from custom_components.ha_ragent.src.utils import vector_db_to_class, embedding_backend_to_class, llm_backend_to_class

_logger = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)

def _create_vector_db_client(hass: HomeAssistant, vector_db_backend_type: str, entry: RAGentConfigEntry) -> ABaseDbBackend:
    _logger.debug("Creating Vector DB client of type %s", vector_db_backend_type)
    return vector_db_to_class(vector_db_backend_type)(hass, dict(entry.options))

def _create_embedding_client(hass: HomeAssistant, embedding_backend_type: str, entry: RAGentConfigEntry) -> ABaseEmbedder:
    _logger.debug("Creating Embedding client of type %s", embedding_backend_type)
    return embedding_backend_to_class(embedding_backend_type)(hass, dict(entry.options))

def _create_llm_client(hass: HomeAssistant, llm_backend_type: str, entry: RAGentConfigEntry) -> ALlmBaseBackend:
    _logger.debug("Creating LLM client of type %s", llm_backend_type)
    return llm_backend_to_class(llm_backend_type)(hass, dict(entry.options))

async def _async_cleanup_subentry_collections(entry: RAGentConfigEntry, subentry_id: str, subentry_data: dict[str, Any]) -> None:
    collection_names = [f"devices_{subentry_id}", f"tools_{subentry_id}"]

    for collection_name in collection_names:
        _logger.debug("Cleaning up collection %s for deleted subentry %s", collection_name, subentry_id)
        await entry.vector_db_backend.async_cleanup_collection(subentry_data, collection_name)

async def _async_update_listener(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    subentry_ids_by_entry = hass.data[DOMAIN].setdefault("subentry_ids", {})
    subentry_data_by_entry = hass.data[DOMAIN].setdefault("subentry_data", {})

    previous_subentry_ids = set(subentry_ids_by_entry.get(entry.entry_id, set()))
    current_subentry_ids = set(entry.subentries)
    removed_subentry_ids = previous_subentry_ids - current_subentry_ids
    removed_data = subentry_data_by_entry.get(entry.entry_id, {})

    if removed_subentry_ids:
        for subentry_id in removed_subentry_ids:
            subentry_data = removed_data.get(subentry_id, {})
            await _async_cleanup_subentry_collections(entry, subentry_id, subentry_data)

    subentry_ids_by_entry[entry.entry_id] = current_subentry_ids
    subentry_data_by_entry[entry.entry_id] = {
        subentry_id: dict(subentry.data)
        for subentry_id, subentry in entry.subentries.items()
    }

    if entry.state != ConfigEntryState.LOADED:
        _logger.debug(
            "Skipped config entry reload after subentry cleanup because entry is not loaded (%s) for %s",
            entry.state,
            entry.entry_id,
        )
        return

    try:
        await hass.config_entries.async_reload(entry.entry_id)
    except OperationNotAllowed:
        _logger.warning(
            "Config entry %s is unloading, skipping reload after subentry change",
            entry.entry_id,
        )

async def _register_services(hass: HomeAssistant):
    async def _handle_preload_models(call: ServiceCall) -> None:
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
            if sub:
                _logger.debug("Preloading model for: %s", sub.title)
                await parent.embedder_backend.async_preload_model(dict(sub.data))
                await parent.llm_backend.async_preload_model(dict(sub.data))

    hass.services.async_register(
        DOMAIN,
        "preload_models",
        _handle_preload_models,
        schema=vol.Schema({}).extend(config_validation.TARGET_SERVICE_FIELDS)
    )
    
    async def _handle_unload_models(call: ServiceCall) -> None:
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
            if sub:
                _logger.debug("Unloading model for: %s", sub.title)
                await parent.embedder_backend.async_unload_model(dict(sub.data))
                await parent.llm_backend.async_unload_model(dict(sub.data))

    hass.services.async_register(
        DOMAIN,
        "unload_models",
        _handle_unload_models,
        schema=vol.Schema({}).extend(config_validation.TARGET_SERVICE_FIELDS)
    )


async def _async_run_startup_embeddings(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    """Run embedding of exposed tools and devices at startup and prevent concurrent runs."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if domain_data.get(STARTUP_EMBEDDING_RUNNING_FLAG):
        _logger.info(
            "Skipping startup embeddings for %s because a run is already in progress",
            entry.entry_id,
        )
        return

    domain_data[STARTUP_EMBEDDING_RUNNING_FLAG] = True
    try:
        tool_extractor = ToolExtractor(hass, entry)
        device_extractor = DeviceExtractor(hass, entry)
        await asyncio.gather(
            tool_extractor.async_embed_all_exposed_tools(),
            device_extractor.async_embed_all_exposed_devices(),
        )
    finally:
        domain_data[STARTUP_EMBEDDING_RUNNING_FLAG] = False
    

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry):
    """Set up HA Ragent from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry
    hass.data[DOMAIN].setdefault("subentry_ids", {})[entry.entry_id] = set(entry.subentries)
    hass.data[DOMAIN].setdefault("subentry_data", {})[entry.entry_id] = {
        subentry_id: dict(subentry.data)
        for subentry_id, subentry in entry.subentries.items()
    }
    
    vector_db_backend_type = entry.data.get(CONF_VECTOR_DB_BACKEND_TYPE, DEFAULT_VECTOR_DB_BACKEND_TYPE)
    embedding_backend_type = entry.data.get(CONF_EMBEDDING_BACKEND_TYPE, DEFAULT_EMBEDDING_BACKEND_TYPE)
    llm_backend_type = entry.data.get(CONF_LLM_BACKEND_TYPE, DEFAULT_LLM_BACKEND_TYPE)

    entry.vector_db_backend = _create_vector_db_client(hass, vector_db_backend_type, entry)
    entry.embedder_backend = _create_embedding_client(hass, embedding_backend_type, entry)    
    entry.llm_backend = _create_llm_client(hass, llm_backend_type, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    if hass.is_running:
        hass.async_create_task(_async_run_startup_embeddings(hass, entry))
    else:
        hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STARTED,
            lambda _event: hass.async_create_task(_async_run_startup_embeddings(hass, entry))
        )

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    await _register_services(hass)
    return True
    
async def async_unload_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    hass.data[DOMAIN].pop(entry.entry_id)
    hass.data[DOMAIN].get("subentry_ids", {}).pop(entry.entry_id, None)
    return True

async def async_remove_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    await entry.vector_db_backend.async_cleanup_database()