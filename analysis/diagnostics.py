"""Small thread-safe diagnostic stream for the local analysis GUI."""

from collections import deque
from datetime import datetime, timezone
from threading import RLock


VALID_LEVELS = {"info", "success", "warning", "error"}


class DiagnosticBuffer:
    """Keep recent diagnostic events in memory for browser polling."""

    def __init__(self, maximum_events=500):
        self._events = deque(maxlen=maximum_events)
        self._next_id = 1
        self._lock = RLock()

    def publish(self, message, *, level="info", source="APP"):
        normalized_level = str(level).lower()
        if normalized_level not in VALID_LEVELS:
            normalized_level = "info"

        with self._lock:
            event = {
                "id": self._next_id,
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z"),
                "level": normalized_level,
                "source": str(source).upper(),
                "message": str(message),
            }
            self._next_id += 1
            self._events.append(event)
            return event.copy()

    def read_after(self, event_id):
        """Return events newer than ``event_id`` and the latest cursor."""
        with self._lock:
            events = [event.copy() for event in self._events if event["id"] > event_id]
            return events, self._next_id - 1


DIAGNOSTICS = DiagnosticBuffer()


def PRINT_T(*values, sep=" ", level="info", source="APP"):
    """Print values to stdout and publish them to the GUI diagnostics terminal.

    ``PRINT_T`` intentionally mirrors the familiar shape of ``print`` while
    adding an optional severity and source label.
    """
    if not isinstance(sep, str):
        raise TypeError("sep must be a string")

    message = sep.join(str(value) for value in values)
    event = DIAGNOSTICS.publish(message, level=level, source=source)
    print(
        f"[GUI TERMINAL] {event['level'].upper():7} "
        f"{event['source']}: {event['message']}",
        flush=True,
    )
    return event
