"""Enhanced EventQueue with parallel event processing and dependency management.

This module extends the original EventQueue with:
- Parallel event sending using ThreadPoolExecutor
- Smart dependency grouping to maintain parent-child relationships
- Improved connection pooling and retry logic
- Configurable concurrency limits
"""

import gzip
import io
import json
import logging
import os
import queue
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"
VERBOSE = os.getenv("LUCIDIC_VERBOSE", "False") == "True"


class ParallelEventQueue:
    """Enhanced EventQueue with parallel processing capabilities."""
    
    def __init__(self, client):
        # Configuration
        self.max_queue_size: int = int(os.getenv("LUCIDIC_MAX_QUEUE_SIZE", 100000))
        self.flush_interval_ms: int = int(os.getenv("LUCIDIC_FLUSH_INTERVAL", 100))
        self.flush_at_count: int = int(os.getenv("LUCIDIC_FLUSH_AT", 100))
        self.blob_threshold: int = int(os.getenv("LUCIDIC_BLOB_THRESHOLD", 64 * 1024))
        self._daemon_mode = os.getenv("LUCIDIC_DAEMON_QUEUE", "true").lower() == "true"
        
        # Parallel processing configuration
        self.max_parallel_sends: int = int(os.getenv("LUCIDIC_MAX_PARALLEL", 10))
        self.retry_failed: bool = os.getenv("LUCIDIC_RETRY_FAILED", "true").lower() == "true"
        self.connection_pool_size: int = int(os.getenv("LUCIDIC_CONNECTION_POOL_SIZE", 20))
        
        # Runtime state
        self._client = client
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._stopped = threading.Event()
        self._flush_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._sent_ids: Set[str] = set()
        self._deferred_queue: List[Dict[str, Any]] = []
        self._deferred_lock = threading.Lock()
        
        # Thread pool for parallel sending
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_parallel_sends,
            thread_name_prefix="LucidicSender"
        )
        
        # Thread safety for flush operations
        self._flush_lock = threading.Lock()
        self._processing_count = 0
        self._processing_lock = threading.Lock()
        self._flush_complete = threading.Event()
        
        # Error suppression
        self.suppress_errors = os.getenv("LUCIDIC_SUPPRESS_ERRORS", "true").lower() == "true"
        
        # Start background worker
        self._start_worker()
        
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Initialized with {self.max_parallel_sends} parallel workers")

    # --- Public API (same as original) ---
    def queue_event(self, event_request: Dict[str, Any]) -> None:
        """Enqueue an event for background processing."""
        if "defer_count" not in event_request:
            event_request["defer_count"] = 0
        
        try:
            self._queue.put(event_request, block=True, timeout=0.001)
            
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Queued event {event_request.get('client_event_id')}, queue size: {self._queue.qsize()}")
            
            # Wake worker if batch large enough
            if self._queue.qsize() >= self.flush_at_count:
                self._flush_event.set()
                
        except queue.Full:
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Queue at max size {self.max_queue_size}, dropping event")

    def force_flush(self, timeout_seconds: float = 5.0) -> None:
        """Flush current queue synchronously (best-effort). Thread-safe."""
        with self._flush_lock:
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Force flush requested, queue size: {self._queue.qsize()}")
            
            # Signal the worker to flush immediately
            self._flush_event.set()
            
            # Wait for the queue to be processed
            end_time = time.time() + timeout_seconds
            last_size = -1
            stable_count = 0
            
            while time.time() < end_time:
                current_size = self._queue.qsize()
                
                # Check if we're making progress
                if current_size == 0 and self._processing_count == 0:
                    # Queue is empty and nothing being processed
                    if stable_count >= 2:  # Wait for 2 cycles to ensure stability
                        if DEBUG:
                            logger.debug("[ParallelEventQueue] Force flush complete - queue empty")
                        return
                    stable_count += 1
                else:
                    stable_count = 0
                
                # If size hasn't changed, we might be stuck
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 10:  # 0.5 seconds of no progress
                        if DEBUG:
                            logger.debug(f"[ParallelEventQueue] Force flush timeout - queue stuck at {current_size}")
                        break
                else:
                    stable_count = 0
                    last_size = current_size
                
                # Signal flush again in case worker missed it
                self._flush_event.set()
                time.sleep(0.05)
            
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Force flush ended, remaining: {self._queue.qsize()}")

    def is_empty(self) -> bool:
        """Check if queue is completely empty and no events are being processed."""
        with self._processing_lock:
            queue_empty = self._queue.empty()
            not_processing = self._processing_count == 0
        with self._deferred_lock:
            deferred_empty = len(self._deferred_queue) == 0
        return queue_empty and not_processing and deferred_empty

    def shutdown(self, timeout: float = 5.0) -> None:
        """Enhanced shutdown with better flushing."""
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Shutdown requested, queue size: {self._queue.qsize()}")
        
        # First try to flush remaining events
        self.force_flush(timeout_seconds=timeout)
        
        # Wait for queue to be truly empty
        wait_start = time.time()
        while not self.is_empty() and (time.time() - wait_start < 2.0):
            time.sleep(0.01)
        
        # Shutdown executor
        self._executor.shutdown(wait=True, timeout=timeout)
        
        # Then signal stop
        self._stopped.set()
        self._flush_event.set()  # Wake up worker
        
        # Wait for worker with timeout
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=timeout)
            if self._worker.is_alive() and DEBUG:
                logger.debug("[ParallelEventQueue] Worker thread did not terminate in time")

    # --- Internal Implementation ---
    def _start_worker(self) -> None:
        """Start the background worker thread."""
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_loop, 
            name="LucidicParallelEventQueue", 
            daemon=self._daemon_mode
        )
        self._worker.start()
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Started worker thread (daemon={self._daemon_mode})")

    def _run_loop(self) -> None:
        """Main worker loop with parallel batch processing."""
        while not self._stopped.is_set():
            batch: List[Dict[str, Any]] = []
            deadline = time.time() + (self.flush_interval_ms / 1000.0)
            force_flush = False
            
            # Collect batch
            while True:
                if self._flush_event.is_set():
                    force_flush = True
                    self._flush_event.clear()
                
                if force_flush:
                    # Drain entire queue when flushing
                    while not self._queue.empty():
                        try:
                            item = self._queue.get_nowait()
                            batch.append(item)
                        except queue.Empty:
                            break
                    if batch:
                        break
                else:
                    # Normal batching logic
                    if len(batch) >= self.flush_at_count:
                        break
                    
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        timeout = min(remaining_time, 0.05)
                        item = self._queue.get(block=True, timeout=timeout)
                        batch.append(item)
                    except queue.Empty:
                        if self._stopped.is_set():
                            while not self._queue.empty():
                                try:
                                    batch.append(self._queue.get_nowait())
                                except queue.Empty:
                                    break
                            break
                        if batch and time.time() >= deadline:
                            break

            # Process batch if we have events
            if batch:
                with self._processing_lock:
                    self._processing_count = len(batch)
                try:
                    self._process_batch_parallel(batch)
                except Exception as e:
                    if DEBUG:
                        logger.debug(f"[ParallelEventQueue] Batch processing error: {e}")
                finally:
                    with self._processing_lock:
                        self._processing_count = 0

    def _process_batch_parallel(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of events with parallel sending."""
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Processing batch of {len(batch)} events in parallel")
        
        # Add any deferred events back to the batch
        with self._deferred_lock:
            if self._deferred_queue:
                batch.extend(self._deferred_queue)
                self._deferred_queue.clear()
        
        # Group events by dependency levels
        dependency_groups = self._group_by_dependencies(batch)
        
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Organized into {len(dependency_groups)} dependency groups")
        
        # Process each dependency level in parallel
        for group_index, group in enumerate(dependency_groups):
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Processing group {group_index + 1}/{len(dependency_groups)} with {len(group)} events")
            
            # Submit all events in this group for parallel processing
            futures_to_event = {}
            for event in group:
                future = self._executor.submit(self._send_event_safe, event)
                futures_to_event[future] = event
            
            # Wait for all in group to complete
            succeeded = []
            failed = []
            
            for future in as_completed(futures_to_event):
                event = futures_to_event[future]
                try:
                    success = future.result(timeout=30)
                    if success:
                        succeeded.append(event)
                        if event_id := event.get("client_event_id"):
                            self._sent_ids.add(event_id)
                    else:
                        failed.append(event)
                except Exception as e:
                    if not self.suppress_errors:
                        logger.error(f"[ParallelEventQueue] Event send failed: {e}")
                    failed.append(event)
            
            if DEBUG and succeeded:
                logger.debug(f"[ParallelEventQueue] Successfully sent {len(succeeded)} events in group {group_index + 1}")
            
            # Retry failed events if configured
            if failed and self.retry_failed:
                self._retry_failed_events(failed)

    def _group_by_dependencies(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group events by dependency levels for parallel processing.
        
        Events in the same group have no dependencies on each other and can be sent in parallel.
        """
        groups = []
        remaining = list(events)
        processed_ids = set(self._sent_ids)
        
        safety_counter = 0
        max_iterations = len(events) * 2 if events else 1
        
        while remaining and safety_counter < max_iterations:
            safety_counter += 1
            current_group = []
            next_remaining = []
            
            for event in remaining:
                parent_id = event.get("client_parent_event_id")
                
                # Can send if no parent or parent already processed
                if not parent_id or parent_id in processed_ids:
                    current_group.append(event)
                    # Mark as will be processed
                    if event_id := event.get("client_event_id"):
                        processed_ids.add(event_id)
                else:
                    next_remaining.append(event)
            
            if current_group:
                groups.append(current_group)
                remaining = next_remaining
            elif remaining:
                # Circular dependency or orphaned events - send anyway
                if DEBUG:
                    logger.warning(f"[ParallelEventQueue] Found {len(remaining)} events with unresolved dependencies")
                groups.append(remaining)
                break
        
        return groups

    def _send_event_safe(self, event_request: Dict[str, Any]) -> bool:
        """Send a single event with error suppression."""
        if self.suppress_errors:
            try:
                return self._send_event_impl(event_request)
            except Exception as e:
                if DEBUG:
                    logger.debug(f"[ParallelEventQueue] Suppressed send error: {e}")
                return False
        else:
            return self._send_event_impl(event_request)

    def _send_event_impl(self, event_request: Dict[str, Any]) -> bool:
        """Send event implementation with retry logic."""
        # Check for parent dependency
        parent_id = event_request.get("client_parent_event_id")
        if parent_id and parent_id not in self._sent_ids:
            # Defer if parent not sent yet (up to 5 times)
            if event_request.get("defer_count", 0) < 5:
                event_request["defer_count"] = event_request.get("defer_count", 0) + 1
                with self._deferred_lock:
                    self._deferred_queue.append(event_request)
                return True
        
        # Retry logic
        attempt = 0
        backoff = 0.25
        while attempt < 3:
            try:
                # Check for blob offloading
                payload = event_request.get("payload", {})
                raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                should_offload = len(raw_bytes) > self.blob_threshold
                
                send_body: Dict[str, Any] = dict(event_request)
                if should_offload:
                    send_body["needs_blob"] = True
                    send_body["payload"] = self._to_preview(send_body.get("type"), payload)
                else:
                    send_body["needs_blob"] = False
                
                # POST /events
                response = self._client.make_request("events", "POST", send_body)
                
                # If offloading, upload compressed payload
                if should_offload:
                    blob_url = response.get("blob_url")
                    if blob_url:
                        compressed = self._compress_json(payload)
                        self._upload_blob(blob_url, compressed)
                    else:
                        logger.error("[ParallelEventQueue] No blob_url received for large payload")
                        return False
                
                return True
                
            except Exception as e:
                attempt += 1
                if DEBUG:
                    logger.debug(f"[ParallelEventQueue] Send attempt {attempt}/3 failed: {e}")
                if attempt >= 3:
                    return False
                time.sleep(backoff)
                backoff *= 2
        
        return False

    def _retry_failed_events(self, failed_events: List[Dict[str, Any]]) -> None:
        """Retry failed events with exponential backoff."""
        if not failed_events:
            return
        
        if DEBUG:
            logger.debug(f"[ParallelEventQueue] Retrying {len(failed_events)} failed events")
        
        # Wait a bit before retry
        time.sleep(1.0)
        
        # Re-queue failed events for another attempt
        for event in failed_events:
            # Increment retry count
            event["retry_count"] = event.get("retry_count", 0) + 1
            
            # Only retry up to 3 times total
            if event["retry_count"] <= 3:
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    if DEBUG:
                        logger.debug("[ParallelEventQueue] Queue full, cannot retry event")

    # --- Helper methods (same as original) ---
    @staticmethod
    def _compress_json(payload: Dict[str, Any]) -> bytes:
        """Compress JSON payload using gzip."""
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(raw)
        return buf.getvalue()

    def _upload_blob(self, blob_url: str, data: bytes) -> None:
        """Upload blob with proper error handling."""
        try:
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Uploading blob, size: {len(data)} bytes")
            
            headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
            resp = requests.put(blob_url, data=data, headers=headers)
            resp.raise_for_status()
            
            if DEBUG:
                logger.debug(f"[ParallelEventQueue] Blob upload successful")
                
        except Exception as e:
            logger.error(f"[ParallelEventQueue] Blob upload failed: {e}")
            raise

    @staticmethod
    def _to_preview(event_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create preview of large payload."""
        t = (event_type or "generic").lower()
        try:
            if t == "llm_generation":
                req = payload.get("request", {})
                usage = payload.get("usage", {})
                messages = req.get("messages", [])[:5]
                output = payload.get("response", {}).get("output", {})
                compressed_messages = []
                for i, m in enumerate(messages):
                    compressed_message_item = {}
                    for k, v in m.items():
                        compressed_message_item[k] = str(v)[:200] if v else None
                    compressed_messages.append(compressed_message_item)
                return {
                    "request": {
                        "model": req.get("model")[:200] if req.get("model") else None,
                        "provider": req.get("provider")[:200] if req.get("provider") else None,
                        "messages": compressed_messages,
                    },
                    "usage": {
                        k: usage.get(k) for k in ("input_tokens", "output_tokens", "cost") if k in usage
                    },
                    "response": {
                        "output": str(output)[:200] if output else None,
                    }
                }
            elif t == "function_call":
                args = payload.get("arguments")
                truncated_args = (
                    {k: (str(v)[:200] if v is not None else None) for k, v in args.items()}
                    if isinstance(args, dict)
                    else (str(args)[:200] if args is not None else None)
                )
                return {
                    "function_name": (payload.get("function_name")[:200] if payload.get("function_name") else None),
                    "arguments": truncated_args,
                }
            elif t == "error_traceback":
                return {
                    "error": (payload.get("error")[:200] if payload.get("error") else None),
                }
            elif t == "generic":
                return {
                    "details": (payload.get("details")[:200] if payload.get("details") else None),
                }
        except Exception:
            pass
        return {"details": "preview_unavailable"}