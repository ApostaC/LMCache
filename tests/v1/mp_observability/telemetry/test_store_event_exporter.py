# SPDX-License-Identifier: Apache-2.0

"""Tests for StoreEventExporter and StoreEventExporterConfig."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)
from lmcache.v1.mp_observability.telemetry.processors.store_event_exporter import (
    StoreEventExporter,
    StoreEventExporterConfig,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestStoreEventExporterConfig:
    def test_from_dict_valid(self):
        config = StoreEventExporterConfig.from_dict(
            {
                "endpoint": "http://localhost:5768/api/v1/telemetry",
            }
        )
        assert config.endpoint == "http://localhost:5768/api/v1/telemetry"
        assert config.timeout == 5.0
        assert config.export_workers == 2

    def test_from_dict_custom_values(self):
        config = StoreEventExporterConfig.from_dict(
            {
                "endpoint": "http://example.com/telemetry",
                "timeout": 10.0,
                "export_workers": 4,
            }
        )
        assert config.endpoint == "http://example.com/telemetry"
        assert config.timeout == 10.0
        assert config.export_workers == 4

    def test_from_dict_missing_endpoint_raises(self):
        with pytest.raises(ValueError, match="endpoint"):
            StoreEventExporterConfig.from_dict({})

    def test_help_returns_string(self):
        h = StoreEventExporterConfig.help()
        assert isinstance(h, str)
        assert "endpoint" in h

    def test_registered_as_store_exporter(self):
        # First Party
        from lmcache.v1.mp_observability.telemetry.processors.base import (
            _PROCESSOR_CONFIG_REGISTRY,
        )

        assert "store_exporter" in _PROCESSOR_CONFIG_REGISTRY
        assert _PROCESSOR_CONFIG_REGISTRY["store_exporter"] is StoreEventExporterConfig


# ---------------------------------------------------------------------------
# Processor tests
# ---------------------------------------------------------------------------


def _make_store_end_event(
    session_id: str = "req-1",
    model_name: str = "test-model",
    world_size: int = 1,
    kv_rank: int = 0,
    **extra_metadata,
) -> TelemetryEvent:
    metadata = {
        "model_name": model_name,
        "world_size": world_size,
        "kv_rank": kv_rank,
        **extra_metadata,
    }
    return TelemetryEvent(
        name="store",
        event_type=EventType.END,
        session_id=session_id,
        metadata=metadata,
    )


class TestStoreEventExporter:
    @pytest.fixture()
    def exporter(self):
        """Create a StoreEventExporter with a mocked telemetry exporter."""
        config = StoreEventExporterConfig(
            endpoint="http://localhost:5768/api/v1/telemetry",
        )
        with patch(
            "lmcache.v1.mp_observability.telemetry.processors"
            ".store_event_exporter.RequestTelemetryFactory"
        ) as mock_factory:
            mock_telemetry = MagicMock()
            mock_factory.create.return_value = mock_telemetry
            proc = StoreEventExporter(config)

        proc._mock_telemetry = mock_telemetry
        return proc

    def test_store_end_event_triggers_export(self, exporter):
        event = _make_store_end_event(session_id="req-42")
        exporter.on_new_event(event)

        exporter._mock_telemetry.on_request_store_finished.assert_called_once_with(
            request_ids_set={"req-42"},
            model_name="test-model",
            world_size=1,
            kv_rank=0,
        )

    def test_store_start_event_ignored(self, exporter):
        event = TelemetryEvent(
            name="store",
            event_type=EventType.START,
            session_id="req-1",
            metadata={"model_name": "m", "world_size": 1, "kv_rank": 0},
        )
        exporter.on_new_event(event)
        exporter._mock_telemetry.on_request_store_finished.assert_not_called()

    def test_non_store_end_event_ignored(self, exporter):
        event = TelemetryEvent(
            name="retrieve",
            event_type=EventType.END,
            session_id="req-1",
            metadata={"model_name": "m", "world_size": 1, "kv_rank": 0},
        )
        exporter.on_new_event(event)
        exporter._mock_telemetry.on_request_store_finished.assert_not_called()

    def test_missing_model_name_skips(self, exporter):
        event = TelemetryEvent(
            name="store",
            event_type=EventType.END,
            session_id="req-1",
            metadata={"world_size": 1, "kv_rank": 0},
        )
        exporter.on_new_event(event)
        exporter._mock_telemetry.on_request_store_finished.assert_not_called()

    def test_missing_world_size_skips(self, exporter):
        event = TelemetryEvent(
            name="store",
            event_type=EventType.END,
            session_id="req-1",
            metadata={"model_name": "m", "kv_rank": 0},
        )
        exporter.on_new_event(event)
        exporter._mock_telemetry.on_request_store_finished.assert_not_called()

    def test_missing_kv_rank_skips(self, exporter):
        event = TelemetryEvent(
            name="store",
            event_type=EventType.END,
            session_id="req-1",
            metadata={"model_name": "m", "world_size": 1},
        )
        exporter.on_new_event(event)
        exporter._mock_telemetry.on_request_store_finished.assert_not_called()

    def test_shutdown_closes_exporter(self, exporter):
        exporter.shutdown()
        exporter._mock_telemetry.close.assert_called_once()

    def test_multiple_events_exported_independently(self, exporter):
        for i in range(3):
            event = _make_store_end_event(session_id=f"req-{i}")
            exporter.on_new_event(event)

        assert exporter._mock_telemetry.on_request_store_finished.call_count == 3

    def test_extra_metadata_does_not_interfere(self, exporter):
        event = _make_store_end_event(
            session_id="req-99",
            stored_count=5,
            device="cuda:0",
        )
        exporter.on_new_event(event)

        exporter._mock_telemetry.on_request_store_finished.assert_called_once_with(
            request_ids_set={"req-99"},
            model_name="test-model",
            world_size=1,
            kv_rank=0,
        )


# ---------------------------------------------------------------------------
# Integration with create_processors
# ---------------------------------------------------------------------------


class TestCreateProcessorsIntegration:
    def test_create_processors_handles_store_exporter_config(self):
        # First Party
        from lmcache.v1.mp_observability.telemetry.config import TelemetryConfig
        from lmcache.v1.mp_observability.telemetry.controller import (
            create_processors,
        )

        config = TelemetryConfig(
            enabled=True,
            processor_configs=[
                StoreEventExporterConfig(
                    endpoint="http://localhost:5768/api/v1/telemetry",
                ),
            ],
        )

        with patch(
            "lmcache.v1.mp_observability.telemetry.processors"
            ".store_event_exporter.RequestTelemetryFactory"
        ):
            processors = create_processors(config)

        assert len(processors) == 1
        assert isinstance(processors[0], StoreEventExporter)
