# SPDX-License-Identifier: Apache-2.0
"""Export store finished event for PD orchestration."""

# Future
from __future__ import annotations

# First Party
from lmcache.integration.request_telemetry.factory import RequestTelemetryFactory
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.telemetry.event import EventType, TelemetryEvent
from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
    TelemetryProcessorConfig,
    register_telemetry_processor_type,
)

logger = init_logger(__name__)


class StoreEventExporterConfig(TelemetryProcessorConfig):
    """Config for the store event exporter processor."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 5.0,
        export_workers: int = 2,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self.export_workers = export_workers

    @classmethod
    def from_dict(cls, d: dict) -> StoreEventExporterConfig:
        endpoint = d.get("endpoint", None)
        if endpoint is None:
            raise ValueError("StoreEventExporterConfig requires setting `endpoint`.")
        timeout = d.get("timeout", 5.0)
        export_workers = d.get("export_workers", 2)
        return cls(endpoint=endpoint, timeout=timeout, export_workers=export_workers)

    @classmethod
    def help(cls) -> str:
        return (
            "StoreEventExporterConfig for exporting store events to a FastAPI "
            "endpoint.\n"
            "Config dict keys:\n"
            " - endpoint: The FastAPI endpoint URL to send telemetry events"
            " to. (required)\n"
            " - timeout: Timeout in seconds for HTTP requests. Defaults to 5.0.\n"
            " - export_workers: Maximum number of threads for async HTTP"
            " requests. Defaults to 2."
        )


register_telemetry_processor_type("store_exporter", StoreEventExporterConfig)


class StoreEventExporter(TelemetryProcessor):
    """Telemetry processor that exports store-finished events for PD orchestration.

    Watches for ``store`` END events (emitted by MPCacheEngine and
    BlendEngineV2) and forwards them to a FastAPI endpoint via
    ``RequestTelemetry.on_request_store_finished``.

    Assumes ``event.session_id`` is the request_id from the serving engine.

    Required event metadata keys:
        model_name: The name of the model being served.
        world_size: The world size of the serving engine.
        kv_rank: The rank of the engine process.
    """

    def __init__(self, config: StoreEventExporterConfig) -> None:
        self.telemetry_exporter = RequestTelemetryFactory.create(
            telemetry_type="fastapi",
            config={
                "endpoint": config.endpoint,
                "timeout": config.timeout,
                "max_workers": config.export_workers,
            },
            use_singleton=False,
        )

    def _check_and_get_event_metadata(
        self,
        event: TelemetryEvent,
    ) -> tuple[str, int, int] | None:
        model_name = event.metadata.get("model_name", None)
        world_size = event.metadata.get("world_size", None)
        kv_rank = event.metadata.get("kv_rank", None)
        if model_name is None or world_size is None or kv_rank is None:
            logger.warning(
                "Missing required metadata in store event: %s. "
                "Required keys: model_name, world_size, kv_rank.",
                event.metadata,
            )
            return None
        return model_name, world_size, kv_rank

    def on_new_event(self, event: TelemetryEvent) -> None:
        """Forward store END events to the telemetry exporter."""
        if event.event_type == EventType.END and event.name == "store":
            necessary_metadata = self._check_and_get_event_metadata(event)
            if necessary_metadata is None:
                return

            model_name, world_size, kv_rank = necessary_metadata
            self.telemetry_exporter.on_request_store_finished(
                request_ids_set={event.session_id},
                model_name=model_name,
                world_size=world_size,
                kv_rank=kv_rank,
            )

    def shutdown(self) -> None:
        """Shut down the underlying telemetry exporter."""
        self.telemetry_exporter.close()
