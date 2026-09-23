import logging
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry, device_registry, entity_registry, floor_registry, label_registry
from homeassistant.helpers import service as service_helper
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.const import HOME_ASSISTANT_SCRIPT_DOMAIN as SCRIPT_DOMAIN

_logger = BaseLogger(__name__)

class DeviceExtractor:
    def __init__(self, hass: HomeAssistant, entry: RAGentConfigEntry):
        self._hass = hass
        self._entry = entry

    @staticmethod
    def _required_features(description: object) -> int:
        """Return the supported-feature mask declared by a service description."""
        required = 0
        if isinstance(description, dict):
            for name, value in description.items():
                if name == "supported_features" and isinstance(value, list):
                    for feature in value:
                        try:
                            required |= int(service_helper.validate_supported_feature(feature))
                        except Exception:
                            # Dynamic services may not use HA's feature notation.
                            continue
                else:
                    required |= DeviceExtractor._required_features(value)
        elif isinstance(description, list):
            for value in description:
                required |= DeviceExtractor._required_features(value)
        return required

    @classmethod
    def _supported_services(
        cls, descriptions: dict[str, object], supported_features: int,
    ) -> list[str]:
        """Return services whose declared feature requirements the entity meets."""
        return sorted(
            service_name
            for service_name, description in descriptions.items()
            if not (required := cls._required_features(description))
            or supported_features & required == required
        )

    async def _async_get_embeddable_devices(self, exposed_entities: list[str]) -> list[Device]:
        area_reg = area_registry.async_get(self._hass)
        device_reg = device_registry.async_get(self._hass)
        entity_reg = entity_registry.async_get(self._hass)
        floor_reg = floor_registry.async_get(self._hass)
        label_reg = label_registry.async_get(self._hass)
        service_descriptions = await service_helper.async_get_all_descriptions(self._hass)
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

            services = self._supported_services(
                service_descriptions.get(domain, {}),
                int(state.attributes.get("supported_features", 0) or 0),
            )

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
                # HA exposes this on state attributes for some integrations,
                # but the entity registry is the authoritative fallback.
                device_class=(
                    state.attributes.get("device_class")
                    or getattr(entity_entry, "device_class", None)
                ),
            ))
        
        return devices
    
    async def async_embed_exposed_devices(self, subentry_id: str) -> None:
        total_embedded_devices = 0
        try:
            _logger.log_string(logging.DEBUG, "Device embedding function starting, checking for subentries")
            if not hasattr(self._entry, "subentries") or not self._entry.subentries:
                _logger.log_string(logging.DEBUG, "No subentries found in config entry! Cannot embed devices.")
                return

            subentry = self._entry.subentries.get(subentry_id)
            if not subentry:
                _logger.log_string(logging.DEBUG, "No matching subentries found for device embedding.")
                return

            all_entities = list(self._hass.states.async_entity_ids())
            exposed_entities = [entity_id for entity_id in all_entities if async_should_expose(self._hass, "conversation", entity_id)]
            entities_to_embed = [entity_id for entity_id in exposed_entities if entity_id.partition(".")[0] != SCRIPT_DOMAIN]
            _logger.log_string(logging.DEBUG, f"Device embedding starting: {len(all_entities)} total entities, "f"{len(exposed_entities)} exposed to conversation, "f"{len(entities_to_embed)} without script entities.")

            if not exposed_entities:
                _logger.log_string(logging.WARNING, "No entities are exposed to Conversation. Skipping embedding and preserving existing vectors.")
                return

            try:
                collection_name = f"devices_{subentry_id}"
                device_list = await self._async_get_embeddable_devices(entities_to_embed)
                if not device_list:
                    await self._entry.vector_db_backend.async_cleanup_collection(dict(subentry.data), collection_name)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, [])
                    _logger.log_string(logging.INFO, f"Cleared device embeddings for empty subentry {subentry_id}")
                    return
                device_embeddings = await self._entry.embedder_backend.async_embed_object(
                    dict(subentry.data), device_list,
                    getattr(self._entry, "translations", None),
                )

                if device_embeddings:
                    embedding_len = len(device_embeddings[0].vector_embedding)
                    self._entry.vector_db_backend.invalidate_collection_cache(collection_name)
                    await self._entry.vector_db_backend.async_reset_collection(dict(subentry.data), collection_name, embedding_len)
                    _logger.log_string(logging.DEBUG, f"Saving {len(device_embeddings)} device embeddings to collection {collection_name}.")
                    await self._entry.vector_db_backend.async_save_objects(dict(subentry.data), collection_name, device_embeddings)
                    self._entry.vector_db_backend.cache_collection_objects(collection_name, device_list)
                    total_embedded_devices += len(device_embeddings)
                else:
                    _logger.log_string(logging.WARNING, f"No devices to embed for subentry {subentry_id}")
            except Exception as err:
                _logger.log_string(logging.ERROR, f"Error in background embedding job for subentry {subentry_id}: {err}")
        except Exception as err:
            _logger.log_string(logging.ERROR, f"Error in tool embedding job: {err}")
        finally:
            if _logger.is_enabled_for(logging.DEBUG):
                _logger.log_string(logging.DEBUG, f"Device embedding function finished with {total_embedded_devices} embedded devices.")
            else:
                _logger.log_string(logging.INFO, f"Finished embedding {total_embedded_devices} devices.")

