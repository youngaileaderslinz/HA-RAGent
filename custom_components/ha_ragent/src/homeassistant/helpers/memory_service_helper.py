from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import entity_registry, target

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager


def get_memory_manager(hass: HomeAssistant, call: ServiceCall) -> MemoryManager:
    """Resolve the selected conversation entities to exactly one agent subentry."""
    registry = entity_registry.async_get(hass)
    referenced = target.async_extract_referenced_entity_ids(hass, target.TargetSelection(call.data))
    subentries: set[tuple[str, str]] = set()
    for entity_id in referenced.referenced | referenced.indirectly_referenced:
        entity = registry.async_get(entity_id)
        if (
            entity is None
            or entity.platform != DOMAIN
            or not entity_id.startswith("conversation.")
            or not entity.config_subentry_id
        ):
            if entity_id in referenced.referenced:
                raise ServiceValidationError("Select an available HA-RAGent conversation entity.")
            continue
        entry = hass.data.get(DOMAIN, {}).get(entity.config_entry_id)
        if entry is None or entity.config_subentry_id not in entry.subentries:
            raise ServiceValidationError("The selected HA-RAGent agent subentry is not available.")
        subentries.add((entity.config_entry_id, entity.config_subentry_id))
    if len(subentries) != 1:
        raise ServiceValidationError("Select exactly one HA-RAGent agent subentry.")
    entry_id, subentry_id = next(iter(subentries))
    return MemoryManager(hass, entry_id, subentry_id)
