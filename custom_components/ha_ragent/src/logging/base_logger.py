import json
import logging
from typing import Any

TRACE = 5  # Custom log level for trace messages
logging.addLevelName(TRACE, "TRACE")
LogLevel = int | str

class BaseLogger:
    """Base logger class used to simplify logging across the custom component."""
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def __getattr__(self, name: str):
        """Delegate standard logging methods to the wrapped logger."""
        return getattr(self._logger, name)

    @staticmethod
    def _parse_level(level: LogLevel) -> int:
        if isinstance(level, int):
            return level
        if level.casefold() == "trace":
            return TRACE
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown log level: {level!r}")
        return resolved

    def log(self, level: LogLevel, message: str, *args: Any, **kwargs: Any) -> None:
        """Log using a numeric standard-library logging level."""
        self._logger.log(self._parse_level(level), message, *args, **kwargs)

    def log_string(self, level: LogLevel, message: str):
        """Log a string message at the specified level."""
        if not self._logger.isEnabledFor(self._parse_level(level)):
            return
        self.log(level, message, exc_info=True)

    def log_payload(self, event: str, level: LogLevel = TRACE, **payload):
        """Log a complete, readable payload at the specified level."""
        if not self._logger.isEnabledFor(self._parse_level(level)):
            return
        try:
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            self.log(level, f"{event}: {serialized}", exc_info=True)
        except Exception as e:
            self.log("ERROR", f"Failed to serialize payload for event '{event}': {e}")