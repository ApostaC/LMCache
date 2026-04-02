# SPDX-License-Identifier: Apache-2.0

"""CSV file subscriber — appends events to a CSV file periodically."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import csv
import json
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

_CSV_FIELDS = ["timestamp", "event_name", "session_id", "attributes"]

_FLUSH_INTERVAL_SECONDS = 5.0


class CSVWriterSubscriber(EventSubscriber):
    """Collects MP server events and flushes them to a CSV file periodically.

    Thread-safe: callbacks append to an internal buffer under a lock;
    a background timer swaps the buffer and writes to disk without
    holding the lock during I/O.
    """

    def __init__(self, output_path: str) -> None:
        self._output_path = output_path
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._stopped = False

        # Write CSV header if the file doesn't exist yet
        if not os.path.exists(output_path):
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()

        self._schedule_flush()

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_event,
            EventType.MP_STORE_END: self._on_event,
            EventType.MP_RETRIEVE_START: self._on_event,
            EventType.MP_RETRIEVE_END: self._on_event,
            EventType.MP_LOOKUP_PREFETCH_START: self._on_event,
            EventType.MP_LOOKUP_PREFETCH_END: self._on_event,
        }

    def _on_event(self, event: Event) -> None:
        row = {
            "timestamp": event.timestamp,
            "event_name": event.event_type.value,
            "session_id": event.session_id,
            "attributes": json.dumps(event.metadata),
        }
        with self._lock:
            self._buffer.append(row)

    def _schedule_flush(self) -> None:
        if self._stopped:
            return
        self._timer = threading.Timer(_FLUSH_INTERVAL_SECONDS, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        # Swap buffer under lock, write outside lock
        with self._lock:
            rows = self._buffer
            self._buffer = []

        if rows:
            try:
                with open(self._output_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                    writer.writerows(rows)
            except OSError:
                logger.exception(
                    "Failed to write %d events to %s",
                    len(rows),
                    self._output_path,
                )

        self._schedule_flush()

    def shutdown(self) -> None:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
        # Final flush
        self._flush()
