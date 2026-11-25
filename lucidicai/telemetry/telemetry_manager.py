"""Telemetry Manager - Singleton for shared OpenTelemetry infrastructure.

Manages the global TracerProvider and client registry for multi-client support.
Telemetry is shared across all LucidicAI client instances, with spans routed
to the correct client based on client_id captured in span attributes.
"""

import asyncio
import logging
import threading
from typing import Dict, List, Optional, TYPE_CHECKING, Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .lucidic_exporter import LucidicSpanExporter
from .context_capture_processor import ContextCaptureProcessor
from .telemetry_init import instrument_providers

if TYPE_CHECKING:
    from ..client import LucidicAI

logger = logging.getLogger("Lucidic")


class TelemetryManager:
    """Singleton manager for shared OpenTelemetry infrastructure.

    This class manages a single TracerProvider that is shared across all
    LucidicAI client instances. Spans are routed to the correct client
    based on the client_id captured in span attributes at creation time.
    """

    _instance: Optional["TelemetryManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TelemetryManager":
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Only initialize once
        if self._initialized:
            return

        self._tracer_provider: Optional[TracerProvider] = None
        self._exporter: Optional[LucidicSpanExporter] = None
        self._context_processor: Optional[ContextCaptureProcessor] = None
        self._instrumentors: Dict[str, Any] = {}
        self._client_registry: Dict[str, "LucidicAI"] = {}
        self._registry_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._initialized = True

    @property
    def tracer_provider(self) -> Optional[TracerProvider]:
        """Get the shared TracerProvider."""
        return self._tracer_provider

    @property
    def is_telemetry_initialized(self) -> bool:
        """Check if telemetry has been initialized."""
        return self._tracer_provider is not None

    def ensure_initialized(self, providers: List[str]) -> None:
        """Initialize telemetry infrastructure if not already done.

        This method is idempotent - calling it multiple times with different
        providers will add new providers to the instrumentation.

        Args:
            providers: List of provider names to instrument (e.g., ["openai", "anthropic"])
        """
        with self._init_lock:
            if self._tracer_provider is None:
                # First initialization - create TracerProvider
                logger.info("[Telemetry] Initializing shared TracerProvider")

                # the sdk shuts the provider down itself; otel's atexit would wait ~30 s on a stuck export
                self._tracer_provider = TracerProvider(shutdown_on_exit=False)

                # Add context capture processor (captures session_id, parent_event_id, client_id)
                self._context_processor = ContextCaptureProcessor()
                self._tracer_provider.add_span_processor(self._context_processor)

                # Add our exporter via BatchSpanProcessor
                self._exporter = LucidicSpanExporter()
                export_processor = BatchSpanProcessor(self._exporter)
                self._tracer_provider.add_span_processor(export_processor)

                logger.info("[Telemetry] TracerProvider initialized with Lucidic exporter")

            # Instrument providers (idempotent - only instruments new ones)
            if providers:
                new_instrumentors = instrument_providers(
                    providers,
                    self._tracer_provider,
                    self._instrumentors
                )
                self._instrumentors.update(new_instrumentors)

    def register_client(self, client: "LucidicAI") -> None:
        """Register a client with the telemetry system.

        This allows the exporter to route spans to the correct client
        based on the client_id captured in span attributes.

        Args:
            client: The LucidicAI client to register
        """
        with self._registry_lock:
            self._client_registry[client._client_id] = client
            if self._exporter:
                self._exporter.register_client(client)
            logger.debug(f"[Telemetry] Registered client {client._client_id[:8]}...")

    def unregister_client(self, client_id: str) -> None:
        """Unregister a client from the telemetry system.

        Args:
            client_id: The client ID to unregister
        """
        with self._registry_lock:
            self._client_registry.pop(client_id, None)
            if self._exporter:
                self._exporter.unregister_client(client_id)
            logger.debug(f"[Telemetry] Unregistered client {client_id[:8]}...")

    def get_client(self, client_id: str) -> Optional["LucidicAI"]:
        """Get a registered client by ID.

        Args:
            client_id: The client ID to look up

        Returns:
            The client if found, None otherwise
        """
        with self._registry_lock:
            return self._client_registry.get(client_id)

    def force_flush(self, timeout_millis: int = 5000) -> bool:
        """Force flush all pending spans.

        Args:
            timeout_millis: Maximum time to wait for flush

        Returns:
            True if successful, False otherwise
        """
        if self._tracer_provider:
            return self._tracer_provider.force_flush(timeout_millis=timeout_millis)
        return True

    def shutdown(self) -> None:
        """Shutdown the telemetry system."""
        if self._tracer_provider:
            logger.info("[Telemetry] Shutting down telemetry system")
            self._tracer_provider.shutdown()
            self._tracer_provider = None
            self._exporter = None
            self._context_processor = None

        with self._registry_lock:
            self._client_registry.clear()


# Module-level singleton accessor
_telemetry_manager: Optional[TelemetryManager] = None


def get_telemetry_manager() -> TelemetryManager:
    """Get the singleton TelemetryManager instance.

    Returns:
        The TelemetryManager singleton
    """
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


_stuck_flush_lock = threading.Lock()
_stuck_flush: Optional[threading.Thread] = None


def flush_spans(timeout_millis: int = 5000) -> bool:
    """Export pending spans, waiting at most timeout_millis; never raises.

    Returns False when the flush is incomplete or skipped because an earlier one is stuck.
    """
    global _stuck_flush
    manager = get_telemetry_manager()
    if not manager.is_telemetry_initialized:
        return True
    with _stuck_flush_lock:
        # the exporter is stuck, so waiting on it again only burns the caller's budget
        if _stuck_flush is not None and _stuck_flush.is_alive():
            logger.debug("[Telemetry] Skipping span flush: an earlier flush is still stuck exporting")
            return False
    flushed = []

    def flush():
        try:
            flushed.append(manager.force_flush(timeout_millis=timeout_millis))
        except Exception as e:
            logger.debug(f"[Telemetry] Error flushing telemetry: {e}")

    # off the caller's thread: otel ignores the timeout, and a signal handler that
    # interrupts an export on the main thread would wait on its export lock
    try:
        worker = threading.Thread(target=flush, name="LucidicSpanFlush", daemon=True)
        worker.start()
        worker.join(timeout_millis / 1000)
        if worker.is_alive():
            with _stuck_flush_lock:
                _stuck_flush = worker
    except Exception as e:
        logger.debug(f"[Telemetry] Error flushing telemetry: {e}")
    if flushed != [True]:
        logger.warning("[Telemetry] Telemetry flush may be incomplete")
        return False
    return True


async def aflush_spans(timeout_millis: int = 5000) -> bool:
    """flush_spans() awaited off the event loop; never raises."""
    if not get_telemetry_manager().is_telemetry_initialized:
        return True
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def settle(result):
        if not done.done():
            done.set_result(result)

    def run():
        result = flush_spans(timeout_millis)
        try:
            loop.call_soon_threadsafe(settle, result)
        except RuntimeError:
            pass  # the loop closed while we flushed

    # its own thread, not the loop's executor, so a busy executor can't delay the end
    threading.Thread(target=run, name="LucidicSpanFlushWait", daemon=True).start()
    return await done
