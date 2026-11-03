from functools import partial

import probatio
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_service_helper import get_memory_manager


async def _handle_import_memories(hass: HomeAssistant, call: ServiceCall) -> None:
    manager = get_memory_manager(hass, call)
    try:
        await manager.async_import_memories(call.data["json"])
    except ValueError as err:
        raise ServiceValidationError(str(err)) from err


def register_import_memories_service(hass: HomeAssistant) -> None:
    hass.services.async_register(
        DOMAIN,
        "import_memories",
        partial(_handle_import_memories, hass),
        schema=probatio.Schema({
            probatio.Required("json"): probatio.Any(str, dict),
        }).extend(config_validation.TARGET_SERVICE_FIELDS),
    )
