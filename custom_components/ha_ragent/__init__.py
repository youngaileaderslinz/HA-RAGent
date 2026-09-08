import asyncio
import logging
from typing import Any

from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.config_entries import ConfigEntryState, OperationNotAllowed
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.homeassistant.services.embed_subentry_service import register_embed_subentry_service
from custom_components.ha_ragent.src.homeassistant.services.preload_service import register_preload_models_service
from custom_components.ha_ragent.src.homeassistant.services.unload_service import register_unload_models_service
from custom_components.ha_ragent.src.homeassistant.extractors.device_extractor import DeviceExtractor
from custom_components.ha_ragent.src.homeassistant.ragent_api import RAGentLLMAPI
from custom_components.ha_ragent.src.homeassistant.extractors.tool_extractor import ToolExtractor

from custom_components.ha_ragent.src.const import (
    CONF_ALLOW_AUTO_EMBEDDING,
    DOMAIN,
    PLATFORMS,
    
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_LLM_BACKEND_TYPE,
    
    RAGENT_LLM_API_ID,
    STARTUP_EMBEDDING_RUNNING_FLAG,
    RAGENT_SCHEDULED_ACTION_CANCELLERS,
    RAGENT_MEMORY_LOCKS,
)

from custom_components.ha_ragent.src.utils import (
    vector_db_to_class,
    embedding_backend_to_class,
    llm_backend_to_class,
    get_setting_value,
)
from custom_components.ha_ragent.src.translation import RAGentTranslations

_logger = logging.getLogger(__name__)

def _ensure_llm_api_registered(hass: HomeAssistant) -> None:
    if any(api.id == RAGENT_LLM_API_ID for api in llm.async_get_apis(hass)):
        return

    llm.async_register_api(hass, RAGentLLMAPI(hass))
    _logger.debug("Registered HA RAGent LLM API: %s", RAGENT_LLM_API_ID)

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
    collection_names = [f"devices_{subentry_id}", f"tools_{subentry_id}", f"memories_{subentry_id}"]

    for collection_name in collection_names:
        _logger.debug("Cleaning up collection %s for deleted subentry %s", collection_name, subentry_id)
        await entry.vector_db_backend.async_cleanup_collection(subentry_data, collection_name)


def _cancel_scheduled_actions(hass: HomeAssistant, subentry_ids: Any) -> None:
    """Cancel and remove pending scheduled actions for subentries."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    for subentry_id in subentry_ids:
        subentry_data = domain_data.get(subentry_id)
        if not subentry_data:
            continue
        cancellers = subentry_data.pop(RAGENT_SCHEDULED_ACTION_CANCELLERS, set())
        for cancel in cancellers:
            cancel()
        if not subentry_data:
            domain_data.pop(subentry_id, None)

async def _async_update_listener(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    subentry_ids_by_entry = hass.data[DOMAIN].setdefault("subentry_ids", {})
    subentry_data_by_entry = hass.data[DOMAIN].setdefault("subentry_data", {})

    previous_subentry_ids = set(subentry_ids_by_entry.get(entry.entry_id, set()))
    current_subentry_ids = set(entry.subentries)
    removed_subentry_ids = previous_subentry_ids - current_subentry_ids
    removed_data = subentry_data_by_entry.get(entry.entry_id, {})

    if removed_subentry_ids:
        _cancel_scheduled_actions(hass, removed_subentry_ids)
        memory_locks = hass.data[DOMAIN].get(RAGENT_MEMORY_LOCKS, {})
        for subentry_id in removed_subentry_ids:
            memory_locks.pop(subentry_id, None)
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
    if not hass.services.has_service(DOMAIN, "embed_subentry"):
        register_embed_subentry_service(hass)

    if not hass.services.has_service(DOMAIN, "preload_models"):
        register_preload_models_service(hass)

    if not hass.services.has_service(DOMAIN, "unload_models"):
        register_unload_models_service(hass)


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Set up the HA RAGent component."""
    hass.data.setdefault(DOMAIN, {})
    _ensure_llm_api_registered(hass)
    await _register_services(hass)
    return True


async def _async_run_startup_embeddings(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    """Run embedding of exposed tools and devices at startup and prevent concurrent runs."""
    auto_embedding_subentry_ids = [
        subentry_id
        for subentry_id, subentry in entry.subentries.items()
        if get_setting_value(CONF_ALLOW_AUTO_EMBEDDING, subentry.data)
    ]

    if not auto_embedding_subentry_ids:
        _logger.debug("Skipping startup embeddings for %s because auto embedding is disabled for all subentries", entry.entry_id)
        return

    domain_data = hass.data.setdefault(DOMAIN, {})
    running_entries = domain_data.get(STARTUP_EMBEDDING_RUNNING_FLAG)
    if not isinstance(running_entries, set):
        running_entries = set()
        domain_data[STARTUP_EMBEDDING_RUNNING_FLAG] = running_entries

    if entry.entry_id in running_entries:
        _logger.info(
            "Skipping startup embeddings for %s because a run is already in progress",
            entry.entry_id,
        )
        return

    running_entries.add(entry.entry_id)
    try:
        tool_extractor = ToolExtractor(hass, entry)
        device_extractor = DeviceExtractor(hass, entry)
        await asyncio.gather(
            *(
                extractor_method(subentry_id)
                for extractor_method in (
                    tool_extractor.async_embed_exposed_tools,
                    device_extractor.async_embed_exposed_devices,
                )
                for subentry_id in auto_embedding_subentry_ids
            )
        )
    finally:
        running_entries.discard(entry.entry_id)
    

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry):
    """Set up HA Ragent from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    selected_language = entry.data.get("rag_selected_language", "en")
    entry.translations = await RAGentTranslations.async_create(hass, selected_language)

    _ensure_llm_api_registered(hass)
    _cancel_scheduled_actions(hass, entry.subentries)

    hass.data[DOMAIN][entry.entry_id] = entry
    hass.data[DOMAIN].setdefault("subentry_ids", {})[entry.entry_id] = set(entry.subentries)
    hass.data[DOMAIN].setdefault("subentry_data", {})[entry.entry_id] = {
        subentry_id: dict(subentry.data)
        for subentry_id, subentry in entry.subentries.items()
    }
    
    vector_db_backend_type = get_setting_value(CONF_VECTOR_DB_BACKEND_TYPE, entry.data)
    embedding_backend_type = get_setting_value(CONF_EMBEDDING_BACKEND_TYPE, entry.data)
    llm_backend_type = get_setting_value(CONF_LLM_BACKEND_TYPE, entry.data)

    entry.vector_db_backend = _create_vector_db_client(hass, vector_db_backend_type, entry)
    entry.embedder_backend = _create_embedding_client(hass, embedding_backend_type, entry)    
    entry.llm_backend = _create_llm_client(hass, llm_backend_type, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    if hass.is_running:
        hass.async_create_task(_async_run_startup_embeddings(hass, entry))
    else:
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, lambda _event: hass.add_job(_async_run_startup_embeddings(hass, entry)))

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
