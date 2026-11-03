from functools import partial
from typing import Any

import probatio
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation

from custom_components.ha_ragent.src.const import DOMAIN
from custom_components.ha_ragent.src.homeassistant.helpers.memory_service_helper import get_memory_manager


async def _handle_export_memories(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    manager = get_memory_manager(hass, call)
    try:
        return {"json": await manager.async_export_memories()}
    except ValueError as err:
        raise ServiceValidationError(str(err)) from err


def register_export_memories_service(hass: HomeAssistant) -> None:
    hass.services.async_register(
        DOMAIN,
        "export_memories",
        partial(_handle_export_memories, hass),
        schema=probatio.Schema({}).extend(config_validation.TARGET_SERVICE_FIELDS),
        supports_response=SupportsResponse.ONLY,
    )
