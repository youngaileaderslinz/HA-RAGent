"""Structured logging for retrieval and conversation diagnostics."""

from dataclasses import asdict, is_dataclass
import json
import logging
from typing import Any


TRACE = logging.DEBUG - 5
"""A logging level more verbose than :data:`logging.DEBUG`."""

logging.addLevelName(TRACE, "TRACE")


def _log_level(level: int | str) -> int:
    """Normalize a standard level name or ``trace`` to its numeric value."""
    if isinstance(level, int):
        return level

    if level.casefold() == "trace":
        return TRACE

    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Unknown log level: {level!r}")
    return resolved


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: item
            for key, item in attributes.items()
            if not key.startswith("_")
        }
    return str(value)


def log_debug_payload(
    logger: logging.Logger,
    event: str,
    *,
    level: int | str = "trace",
    **payload: Any,
) -> None:
    """Log a complete, readable payload at ``level`` (``trace`` by default)."""
    log_level = _log_level(level)
    if not logger.isEnabledFor(log_level):
        return
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    except Exception:  # Logging must never interrupt a conversation turn.
        logger.log(
            log_level,
            "HA-RAGent debug %s (payload serialization failed): %r",
            event,
            payload,
            exc_info=True,
        )
        return
    logger.log(log_level, "HA-RAGent debug %s:\n%s", event, serialized)
