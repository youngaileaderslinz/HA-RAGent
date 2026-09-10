"""Structured debug logging for retrieval and conversation diagnostics."""

from dataclasses import asdict, is_dataclass
import json
import logging
from typing import Any


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


def log_debug_payload(logger: logging.Logger, event: str, **payload: Any) -> None:
    """Log a complete, readable payload only when debug logging is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    except Exception:  # Logging must never interrupt a conversation turn.
        logger.debug(
            "HA-RAGent debug %s (payload serialization failed): %r",
            event,
            payload,
            exc_info=True,
        )
        return
    logger.debug(
        "HA-RAGent debug %s:\n%s",
        event,
        serialized,
    )
