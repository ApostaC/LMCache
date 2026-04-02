# SPDX-License-Identifier: Apache-2.0

"""Tests for CSVWriterSubscriber."""

# Standard
import csv
import json
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.subscribers.csv_writer import (
    CSVWriterSubscriber,
)


@pytest.fixture
def bus():
    return EventBus(EventBusConfig(enabled=True, max_queue_size=100))


@pytest.fixture
def csv_path(tmp_path):
    return str(tmp_path / "events.csv")


@pytest.fixture
def subscriber(bus, csv_path):
    sub = CSVWriterSubscriber(csv_path)
    bus.register_subscriber(sub)
    return sub


class TestCSVWriterSubscriber:
    def test_subscriptions_cover_all_mp_server_events(self, subscriber):
        subs = subscriber.get_subscriptions()
        assert EventType.MP_STORE_START in subs
        assert EventType.MP_STORE_END in subs
        assert EventType.MP_RETRIEVE_START in subs
        assert EventType.MP_RETRIEVE_END in subs
        assert EventType.MP_LOOKUP_PREFETCH_START in subs
        assert EventType.MP_LOOKUP_PREFETCH_END in subs

    def test_events_written_to_csv(self, bus, subscriber, csv_path):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_START,
                session_id="req-1",
                metadata={"device": "cuda:0", "num_tokens": 512},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_STORE_END,
                session_id="req-1",
                metadata={"device": "cuda:0", "stored_count": 2, "num_tokens": 512},
            )
        )
        # Wait for drain thread to dispatch, then stop (which calls shutdown)
        time.sleep(0.15)
        bus.stop()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["event_name"] == "mp.store.start"
        assert rows[0]["session_id"] == "req-1"
        attrs = json.loads(rows[0]["attributes"])
        assert attrs["device"] == "cuda:0"
        assert attrs["num_tokens"] == 512

        assert rows[1]["event_name"] == "mp.store.end"
        end_attrs = json.loads(rows[1]["attributes"])
        assert end_attrs["stored_count"] == 2

    def test_lookup_events_with_hit_info(self, bus, subscriber, csv_path):
        bus.start()
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_START,
                session_id="req-2",
                metadata={"num_tokens": 1024},
            )
        )
        bus.publish(
            Event(
                event_type=EventType.MP_LOOKUP_PREFETCH_END,
                session_id="req-2",
                metadata={
                    "found_count": 3,
                    "num_tokens": 1024,
                    "l1_hit_chunks": 2,
                    "l2_hit_chunks": 1,
                },
            )
        )
        time.sleep(0.15)
        bus.stop()

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        end_attrs = json.loads(rows[1]["attributes"])
        assert end_attrs["l1_hit_chunks"] == 2
        assert end_attrs["l2_hit_chunks"] == 1

    def test_csv_header_present(self, csv_path):
        CSVWriterSubscriber(csv_path).shutdown()
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["timestamp", "event_name", "session_id", "attributes"]

    def test_empty_flush_no_error(self, subscriber):
        # Flushing with no events should not raise
        subscriber.shutdown()
