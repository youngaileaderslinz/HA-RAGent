import json
import logging
from typing import Literal

TRACE = 5  # Custom log level for trace messages
logging.addLevelName(TRACE, "TRACE")
LogLevel = Literal[5, 10, 20, 30, 40, 50]

class BaseLogger:
    """Base logger class used to simplify logging across the custom component."""
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def log_string(self, level: LogLevel, message: str):
        """Log a string message at the specified level."""
        if not self._logger.isEnabledFor(getattr(logging, level.upper(), logging.DEBUG)):
            return
        self.log(level, message)

    def log_payload(self, event: str, level: LogLevel = logging.TRACE, **payload):
        """Log a complete, readable payload at the specified level."""
        if not self._logger.isEnabledFor(getattr(logging, level.upper(), logging.DEBUG)):
            return
        try:
            serialized = json.dumps(payload, indent=2)
            self.log(level, f"{event}: {serialized}")
        except Exception as e:
            self.log("ERROR", f"Failed to serialize payload for event '{event}': {e}")