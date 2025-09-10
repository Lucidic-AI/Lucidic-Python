"""Parallel event queue for efficient event processing.

This module provides a high-performance event queue that processes events
in parallel while respecting parent-child dependencies.
"""
import gzip
import io
import json
import logging
import queue
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.config import get_config

logger = logging.getLogger("Lucidic")


class EventQueue:
    """High-performance parallel event queue."""
    
    def __init__(self, client):
        """Initialize the event queue."""
        self.config = get_config()
        self._client = client
        
        # Queue configuration
        self.max_queue_size = self.config.event_queue.max_queue_size
        self.flush_interval_ms = self.config.event_queue.flush_interval_ms
        self.flush_at_count = self.config.event_queue.flush_at_count
        self.blob_threshold = self.config.event_queue.blob_threshold
        self.max_workers = self.config.event_queue.max_parallel_workers
        self.retry_failed = self.config.event_queue.retry_failed
        
        # Runtime state
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._stopped = threading.Event()
        self._flush_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._sent_ids: Set[str] = set()
        self._deferred_queue: List[Dict[str, Any]] = []
        self._deferred_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="LucidicSender"
        )
        
        # Thread safety
        self._flush_lock = threading.Lock()
        self._processing_count = 0
        self._processing_lock = threading.Lock()
        
        # Start background worker
        self._start_worker()
        
        if self.config.debug:
            logger.debug(f"[EventQueue] Initialized with {self.max_workers} parallel workers")

    def queue_event(self, event_request: Dict[str, Any]) -> None:
        """Enqueue an event for background processing."""
        if "defer_count" not in event_request:
            event_request["defer_count"] = 0
        
        try:
            self._queue.put(event_request, block=True, timeout=0.001)
            
            if self.config.debug:
                event_id = event_request.get('client_event_id', 'unknown')
                logger.debug(f"[EventQueue] Queued event {event_id}, queue size: {self._queue.qsize()}")
            
            # Wake worker if batch large enough
            if self._queue.qsize() >= self.flush_at_count:
                self._flush_event.set()
                
        except queue.Full:
            if self.config.debug:
                logger.debug(f"[EventQueue] Queue at max size {self.max_queue_size}, dropping event")

    def force_flush(self, timeout_seconds: float = 5.0) -> None:
        """Flush current queue synchronously (best-effort)."""
        with self._flush_lock:
            if self.config.debug:
                logger.debug(f"[EventQueue] Force flush requested, queue size: {self._queue.qsize()}")
            
            # Signal the worker to flush immediately
            self._flush_event.set()
            
            # Wait for the queue to be processed
            end_time = time.time() + timeout_seconds
            last_size = -1
            stable_count = 0
            
            while time.time() < end_time:
                current_size = self._queue.qsize()
                
                with self._processing_lock:
                    processing = self._processing_count
                
                # Check if we're done
                if current_size == 0 and processing == 0:
                    if stable_count >= 2:
                        if self.config.debug:
                            logger.debug("[EventQueue] Force flush complete")
                        return
                    stable_count += 1
                else:
                    stable_count = 0
                
                # Check for progress
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 10:  # 0.5 seconds of no progress
                        break
                else:
                    stable_count = 0
                    last_size = current_size
                
                self._flush_event.set()
                time.sleep(0.05)

    def is_empty(self) -> bool:
        """Check if queue is completely empty."""
        with self._processing_lock:
            queue_empty = self._queue.empty()
            not_processing = self._processing_count == 0
        with self._deferred_lock:
            deferred_empty = len(self._deferred_queue) == 0
        return queue_empty and not_processing and deferred_empty

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the event queue."""
        if self.config.debug:
            logger.debug(f"[EventQueue] Shutdown requested")
        
        # Flush remaining events
        self.force_flush(timeout_seconds=timeout)
        
        # Shutdown executor (timeout param added in Python 3.9+)
        try:
            self._executor.shutdown(wait=True, timeout=timeout)
        except TypeError:
            # Fallback for older Python versions
            self._executor.shutdown(wait=True)
        
        # Signal stop
        self._stopped.set()
        self._flush_event.set()
        
        # Wait for worker
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)

    # --- Internal Implementation ---
    
    def _start_worker(self) -> None:
        """Start the background worker thread."""
        if self._worker and self._worker.is_alive():
            return
        
        self._worker = threading.Thread(
            target=self._run_loop,
            name="LucidicEventQueue",
            daemon=self.config.event_queue.daemon_mode
        )
        self._worker.start()

    def _run_loop(self) -> None:
        """Main worker loop."""
        while not self._stopped.is_set():
            batch = self._collect_batch()
            
            if batch:
                with self._processing_lock:
                    self._processing_count = len(batch)
                
                try:
                    self._process_batch(batch)
                except Exception as e:
                    if self.config.debug:
                        logger.debug(f"[EventQueue] Batch processing error: {e}")
                finally:
                    with self._processing_lock:
                        self._processing_count = 0

    def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect a batch of events from the queue."""
        batch: List[Dict[str, Any]] = []
        deadline = time.time() + (self.flush_interval_ms / 1000.0)
        
        while True:
            # Check for force flush
            if self._flush_event.is_set():
                self._flush_event.clear()
                # Drain entire queue
                while not self._queue.empty():
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if batch:
                    break
            
            # Check batch size
            if len(batch) >= self.flush_at_count:
                break
            
            # Check deadline
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            
            # Try to get an item
            try:
                timeout = min(remaining_time, 0.05)
                item = self._queue.get(block=True, timeout=timeout)
                batch.append(item)
            except queue.Empty:
                if self._stopped.is_set():
                    # Drain remaining on shutdown
                    while not self._queue.empty():
                        try:
                            batch.append(self._queue.get_nowait())
                        except queue.Empty:
                            break
                    break
                if batch and time.time() >= deadline:
                    break
        
        return batch

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process batch with parallel sending."""
        if self.config.debug:
            logger.debug(f"[EventQueue] Processing batch of {len(batch)} events")
        
        # Add deferred events
        with self._deferred_lock:
            if self._deferred_queue:
                batch.extend(self._deferred_queue)
                self._deferred_queue.clear()
        
        # Group by dependencies
        dependency_groups = self._group_by_dependencies(batch)
        
        # Process each group in parallel
        for group_index, group in enumerate(dependency_groups):
            if self.config.debug:
                logger.debug(f"[EventQueue] Processing group {group_index + 1}/{len(dependency_groups)} with {len(group)} events")
            
            # Submit all events in group for parallel processing
            futures_to_event = {}
            for event in group:
                future = self._executor.submit(self._send_event_safe, event)
                futures_to_event[future] = event
            
            # Wait for completion
            for future in as_completed(futures_to_event):
                event = futures_to_event[future]
                try:
                    success = future.result(timeout=30)
                    if success:
                        if event_id := event.get("client_event_id"):
                            self._sent_ids.add(event_id)
                except Exception as e:
                    if self.config.debug:
                        logger.debug(f"[EventQueue] Failed to send event: {e}")
                    if self.retry_failed:
                        self._retry_event(event)

    def _group_by_dependencies(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group events by dependency levels for parallel processing."""
        groups = []
        remaining = list(events)
        processed_ids = set(self._sent_ids)
        
        while remaining:
            current_group = []
            next_remaining = []
            
            for event in remaining:
                parent_id = event.get("client_parent_event_id")
                if not parent_id or parent_id in processed_ids:
                    current_group.append(event)
                    if event_id := event.get("client_event_id"):
                        processed_ids.add(event_id)
                else:
                    next_remaining.append(event)
            
            if current_group:
                groups.append(current_group)
                remaining = next_remaining
            elif remaining:
                # Circular dependency or orphaned events
                if self.config.debug:
                    logger.debug(f"[EventQueue] Found {len(remaining)} events with unresolved dependencies")
                groups.append(remaining)
                break
        
        return groups

    def _send_event_safe(self, event_request: Dict[str, Any]) -> bool:
        """Send event with error suppression if configured."""
        if self.config.error_handling.suppress_errors:
            try:
                return self._send_event(event_request)
            except Exception as e:
                if self.config.debug:
                    logger.debug(f"[EventQueue] Suppressed send error: {e}")
                return False
        else:
            return self._send_event(event_request)

    def _send_event(self, event_request: Dict[str, Any]) -> bool:
        """Send a single event to the backend."""
        # Check parent dependency
        parent_id = event_request.get("client_parent_event_id")
        if parent_id and parent_id not in self._sent_ids:
            # Defer if parent not sent yet
            if event_request.get("defer_count", 0) < 5:
                event_request["defer_count"] = event_request.get("defer_count", 0) + 1
                with self._deferred_lock:
                    self._deferred_queue.append(event_request)
                return True
        
        # Check for blob offloading
        payload = event_request.get("payload", {})
        raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        should_offload = len(raw_bytes) > self.blob_threshold
        
        send_body: Dict[str, Any] = dict(event_request)
        if should_offload:
            send_body["needs_blob"] = True
            send_body["payload"] = self._create_preview(send_body.get("type"), payload)
        else:
            send_body["needs_blob"] = False
        
        # Send event
        try:
            response = self._client.make_request("events", "POST", send_body)
            
            # Handle blob upload if needed
            if should_offload:
                blob_url = response.get("blob_url")
                if blob_url:
                    compressed = self._compress_json(payload)
                    self._upload_blob(blob_url, compressed)
                else:
                    logger.error("[EventQueue] No blob_url received for large payload")
                    return False
            
            return True
            
        except Exception as e:
            if self.config.debug:
                logger.debug(f"[EventQueue] Failed to send event: {e}")
            return False

    def _retry_event(self, event: Dict[str, Any]) -> None:
        """Retry a failed event."""
        event["retry_count"] = event.get("retry_count", 0) + 1
        if event["retry_count"] <= 3:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass

    @staticmethod
    def _compress_json(payload: Dict[str, Any]) -> bytes:
        """Compress JSON payload using gzip."""
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(raw)
        return buf.getvalue()

    def _upload_blob(self, blob_url: str, data: bytes) -> None:
        """Upload compressed blob to presigned URL."""
        headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
        resp = requests.put(blob_url, data=data, headers=headers)
        resp.raise_for_status()

    @staticmethod
    def _create_preview(event_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create preview of large payload for logging."""
        try:
            t = (event_type or "generic").lower()
            
            if t == "llm_generation":
                req = payload.get("request", {})
                return {
                    "request": {
                        "model": str(req.get("model", ""))[:200],
                        "provider": str(req.get("provider", ""))[:200],
                        "messages": "truncated"
                    },
                    "response": {"output": "truncated"}
                }
            elif t == "function_call":
                return {
                    "function_name": str(payload.get("function_name", ""))[:200],
                    "arguments": "truncated"
                }
            else:
                return {"details": "preview_unavailable"}
                
        except Exception:
            return {"details": "preview_error"}