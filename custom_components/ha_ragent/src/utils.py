import socket
from typing import Any

from custom_components.ha_ragent.src.const import CONF_SELECTED_LANGUAGE, DEFAULT_SETTINGS


def get_value(value: object, default: object) -> object:
    """Return the value when not null, otherwise the default parameter."""
    return value if value else default


def try_parse_int(value: str, default: int = 0) -> int:
    """Parse a string as an integer, returning the default if parsing fails."""
    try:
        return int(value)
    except ValueError:
        return default


def get_setting_value(setting_key: str, settings: dict) -> Any:
    """Return a configured setting or its default value."""
    return settings[setting_key] if setting_key in settings else DEFAULT_SETTINGS.get(setting_key)


def get_entry_language(entry: Any) -> str:
    """Return the entry language, preferring current data over legacy options."""
    settings = {**(getattr(entry, "options", {}) or {}), **(getattr(entry, "data", {}) or {})}
    return get_setting_value(CONF_SELECTED_LANGUAGE, settings)


def is_valid_host(host: str) -> bool:
    """Check whether a hostname or IP address resolves."""
    try:
        socket.gethostbyname(host)
        return True
    except socket.gaierror:
        return False
