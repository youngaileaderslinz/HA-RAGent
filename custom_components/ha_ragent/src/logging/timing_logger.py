
import time
from custom_components.ha_ragent.src.logging.base_logger import BaseLogger, LogLevel, TRACE

class TimingLogger(BaseLogger):
    def __init__(self, name: str):
        super().__init__(name)
        self.reset_timing()

    def reset_timing(self):
        """Reset the timing mark to the current time."""
        self._start_time = time.perf_counter()
        self._timing_mark = self._start_time

    def log_timed_string(self, level: LogLevel = TRACE, message: str = ""):
        """Log a string message at the specified level, including the time taken since the last log or reset."""
        now = time.perf_counter()
        self.log_string(level, f"[Time taken: {now - self._timing_mark:.3f}s] {message}")
        self._timing_mark = now

    def log_timed_payload(self, event: str, level: LogLevel = TRACE, **payload):
        """Log a complete, readable payload at the specified level, including the time taken since the last log or reset."""
        now = time.perf_counter()
        self.log_payload(f"[Time taken: {now - self._timing_mark:.3f}s] {event}", level, **payload)
        self._timing_mark = now