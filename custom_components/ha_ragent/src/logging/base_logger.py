"""Component logging with structured payload support."""

from dataclasses import asdict, is_dataclass
import json
import logging
from typing import Any

from custom_components.ha_ragent.src.const import TRACE

logging.addLevelName(TRACE, "TRACE")

LogLevel = int | str


class BaseLogger:
    """Component logger that supports standard and structured messages."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    @staticmethod
    def _parse_level(level: LogLevel) -> int:
        if isinstance(level, int):
            return level
        if level.casefold() == "trace":
            return TRACE
        resolved = logging.getLevelNamesMapping().get(level.upper())
        if resolved is None:
            raise ValueError(f"Unknown log level: {level!r}")
        return resolved

    @staticmethod
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
                key: item for key, item in attributes.items()
                if not key.startswith("_")
            }
        return str(value)

    def log(self, level: LogLevel, message: str, *args: Any, **kwargs: Any) -> None:
        """Log using a numeric standard-library logging level."""
        self._logger.log(self._parse_level(level), message, *args, **kwargs)

    def log_string(self, level: LogLevel, message: str) -> None:
        """Log a string message at the specified level."""
        if self._logger.isEnabledFor(self._parse_level(level)):
            self.log(level, message)

    def log_payload(self, event: str, level: LogLevel = TRACE, **payload: Any) -> None:
        """Log a complete, readable payload at the specified level."""
        log_level = self._parse_level(level)
        if not self._logger.isEnabledFor(log_level):
            return
        try:
            serialized = json.dumps(
                payload, ensure_ascii=False, indent=2, default=self._json_default,
            )
        except Exception:
            self._logger.log(
                log_level,
                f"HA-RAGent debug {event} (payload serialization failed): {payload}",
                exc_info=True,
            )
            return
        self._logger.log(log_level, f"HA-RAGent debug {event}:\n{serialized}")