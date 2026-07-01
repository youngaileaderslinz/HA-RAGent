import logging

from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding

from ...models.device import Device

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry, device_registry, entity_registry, label_registry, llm
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

from ..ragent_config_entry import RAGentConfigEntry
from ...const import (
    DOMAIN
)

_logger = logging.getLogger(__name__)

class DeviceExtractor:
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry):
        self._hass = hass
        self._entry = entry


    async def _async_get_services_for_domain(self, target_domain: str):
        services = self._hass.services.async_services()

        if target_domain not in services:
            return []

        return [service_name for service_name in services[target_domain]]

    async def _async_get_embeddable_devices(self, exposed_entities: list[str]) -> list[Device]:
        area_reg = area_registry.async_get(self._hass)
        device_reg = device_registry.async_get(self._hass)
        entity_reg = entity_registry.async_get(self._hass)
        label_reg = label_registry.async_get(self._hass)
        
        devices = []
        
        for entity_id in exposed_entities:
            state = self._hass.states.get(entity_id)
            if not state:
                continue

            friendly_name = state.attributes.get("friendly_name", entity_id)
            domain = entity_id.split(".")[0] if "." in entity_id else "unknown"

            area_name = ""
            entity_entry = entity_reg.async_get(entity_id)
            if entity_entry:
                if entity_entry.area_id:
                    area = area_reg.async_get_area(entity_entry.area_id)
                    area_name = area.name if area else ""
                elif entity_entry.device_id:
                    device = device_reg.async_get(entity_entry.device_id)
                    if device and device.area_id:
                        area = area_reg.async_get_area(device.area_id)
                        area_name = area.name if area else ""
                
            device_labels = []
            if entity_entry and entity_entry.labels:
                for label_id in entity_entry.labels:
                    label = label_reg.async_get_label(label_id)
                    if label:
                        device_labels.append(label.name)

            aliases = []
            if entity_entry and entity_entry.aliases:
                aliases = [alias for alias in entity_entry.aliases if isinstance(alias, str)]

            services = await self._async_get_services_for_domain(domain)

            devices.append(Device(
                id=entity_id,
                name=friendly_name,
                domain=[domain],
                area_name=area_name,
                device_labels=device_labels,
                aliases=aliases,
                services=services
            ))
        
        return devices
    
    async def async_embed_all_exposed_devices(self) -> None:
        total_embedded_devices = 0
        try:
            _logger.debug("Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.debug("No subentries found in config entry! Cannot embed devices.")
                return

            _logger.debug(f"Found {len(self._entry.subentries)} subentries to process.")

            all_entities = list(self._hass.states.async_entity_ids())
            exposed_entities = [entity_id for entity_id in all_entities if async_should_expose(self._hass, "conversation", entity_id)]
            _logger.debug(f"Device embedding starting: {len(all_entities)} total entities, {len(exposed_entities)} exposed to conversation.")

            if not exposed_entities:
                _logger.warning("No entities are exposed to Conversation. Skipping embedding and preserving existing vectors.")
                return

            for subentry_id, subentry in self._entry.subentries.items():
                try:
                    collection_name = f"devices_{subentry_id}"
                    embedding_len = len(await self._entry.embedder_backend.async_embed_text(dict(subentry.data), "Test"))
                    
                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)                    
                    device_list = await self._async_get_embeddable_devices(exposed_entities)
                    device_embeddings = await self._entry.embedder_backend.async_embed_object(DeviceEmbedding, dict(subentry.data), device_list)

                    if device_embeddings:
                        _logger.debug(f"Saving {len(device_embeddings)} device embeddings to collection {collection_name}.")
                        await self._entry.vector_db_backend.async_save_object_embeddings(dict(subentry.data), collection_name, device_embeddings)
                        total_embedded_devices += len(device_embeddings)
                    else:
                        _logger.warning("No devices to embed for subentry %s", subentry_id)
                except Exception as err:
                    _logger.error(f"Error in background embedding job for subentry {subentry_id}: {err}", exc_info=True)
                    continue
        except Exception as err:
            _logger.error(f"Error in tool embedding job: {err}", exc_info=True)
        finally:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug("Device embedding function finished with %s embedded devices.", total_embedded_devices)
            else:
                _logger.info("Finished embedding %s devices.", total_embedded_devices)

