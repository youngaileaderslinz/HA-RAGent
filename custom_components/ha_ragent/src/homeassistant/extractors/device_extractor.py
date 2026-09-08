import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry, device_registry, entity_registry, floor_registry, label_registry, llm
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.const import HOME_ASSISTANT_SCRIPT_DOMAIN as SCRIPT_DOMAIN

_logger = logging.getLogger(__name__)

class DeviceExtractor:
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry):
        self._hass = hass
        self._entry = entry


    async def _async_get_embeddable_devices(self, exposed_entities: list[str]) -> list[Device]:
        area_reg = area_registry.async_get(self._hass)
        device_reg = device_registry.async_get(self._hass)
        entity_reg = entity_registry.async_get(self._hass)
        floor_reg = floor_registry.async_get(self._hass)
        label_reg = label_registry.async_get(self._hass)
        services_by_domain = self._hass.services.async_services()
        
        devices = []
        
        for entity_id in exposed_entities:
            state = self._hass.states.get(entity_id)
            if not state:
                continue

            friendly_name = state.attributes.get("friendly_name", entity_id)
            domain = entity_id.split(".")[0] if "." in entity_id else "unknown"

            area_name = ""
            floor_name = ""
            area_aliases = []
            floor_aliases = []
            entity_entry = entity_reg.async_get(entity_id)
            device_entry = (
                device_reg.async_get(entity_entry.device_id)
                if entity_entry and entity_entry.device_id
                else None
            )
            if entity_entry:
                area = None
                if entity_entry.area_id:
                    area = area_reg.async_get_area(entity_entry.area_id)
                elif device_entry and device_entry.area_id:
                    area = area_reg.async_get_area(device_entry.area_id)

                if area:
                    area_name = area.name
                    area_aliases = sorted(getattr(area, "aliases", ()) or ())
                    if area.floor_id:
                        floor = floor_reg.async_get_floor(area.floor_id)
                        floor_name = floor.name if floor else ""
                        floor_aliases = sorted(getattr(floor, "aliases", ()) or ())
                
            device_labels = []
            if entity_entry and entity_entry.labels:
                for label_id in entity_entry.labels:
                    label = label_reg.async_get_label(label_id)
                    if label:
                        device_labels.append(label.name)

            aliases = []
            if entity_entry:
                aliases = entity_registry.async_get_entity_aliases(self._hass, entity_entry)

            if aliases:
                friendly_name = aliases[0]
            
            services = list(services_by_domain.get(domain, {}))

            devices.append(Device(
                id=entity_id,
                friendly_name=friendly_name,
                domain=[domain],
                floor_name=floor_name,
                area_name=area_name,
                area_aliases=area_aliases,
                floor_aliases=floor_aliases,
                device_labels=device_labels,
                aliases=aliases,
                services=services,
                unit_of_measurement=state.attributes.get("unit_of_measurement"),
                device_class=state.attributes.get("device_class"),
            ))
        
        return devices
    
    async def async_embed_exposed_devices(self, subentry_id: str) -> None:
        total_embedded_devices = 0
        try:
            _logger.debug("Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.debug("No subentries found in config entry! Cannot embed devices.")
                return

            subentry = self._entry.subentries.get(subentry_id)
            if not subentry:
                _logger.debug("No matching subentries found for device embedding.")
                return

            all_entities = list(self._hass.states.async_entity_ids())
            exposed_entities = [entity_id for entity_id in all_entities if async_should_expose(self._hass, "conversation", entity_id)]
            entities_to_embed = [entity_id for entity_id in exposed_entities if entity_id.partition(".")[0] != SCRIPT_DOMAIN]
            _logger.debug(f"Device embedding starting: {len(all_entities)} total entities, "f"{len(exposed_entities)} exposed to conversation, "f"{len(entities_to_embed)} without script entities.")

            if not exposed_entities:
                _logger.warning("No entities are exposed to Conversation. Skipping embedding and preserving existing vectors.")
                return

            try:
                collection_name = f"devices_{subentry_id}"
                device_list = await self._async_get_embeddable_devices(entities_to_embed)
                if not device_list:
                    await self._entry.vector_db_backend.async_cleanup_collection(dict(subentry.data), collection_name)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, [])
                    _logger.info("Cleared device embeddings for empty subentry %s", subentry_id)
                    return
                device_embeddings = await self._entry.embedder_backend.async_embed_object(dict(subentry.data), device_list)

                if device_embeddings:
                    embedding_len = len(device_embeddings[0].vector_embedding)
                    self._entry.vector_db_backend.invalidate_collection_cache(collection_name)
                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)
                    _logger.debug(f"Saving {len(device_embeddings)} device embeddings to collection {collection_name}.")
                    await self._entry.vector_db_backend.async_save_objects(dict(subentry.data), collection_name, device_embeddings)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, device_list)
                    total_embedded_devices += len(device_embeddings)
                else:
                    _logger.warning("No devices to embed for subentry %s", subentry_id)
            except Exception as err:
                _logger.error(f"Error in background embedding job for subentry {subentry_id}: {err}", exc_info=True)
        except Exception as err:
            _logger.error(f"Error in tool embedding job: {err}", exc_info=True)
        finally:
            if _logger.isEnabledFor(logging.DEBUG):
                _logger.debug(f"Device embedding function finished with {total_embedded_devices} embedded devices.")
            else:
                _logger.info(f"Finished embedding {total_embedded_devices} devices.")

