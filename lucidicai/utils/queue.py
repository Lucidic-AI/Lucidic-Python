"""Parallel event queue for efficient event processing.

This module provides a high-performance event queue that processes events
in parallel while respecting parent-child dependencies.
"""
import gzip
import io
import json
import queue
import threading
import time
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.config import get_config
from ..utils.logger import debug, info, warning, error, truncate_id, truncate_data


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
        # Removed deferred queue - no longer needed since backend handles any order
        
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
        
        debug(f"[EventQueue] Initialized with {self.max_workers} parallel workers, batch size: {self.flush_at_count}, flush interval: {self.flush_interval_ms}ms")

    def queue_event(self, event_request: Dict[str, Any]) -> None:
        """Enqueue an event for background processing."""
        if "defer_count" not in event_request:
            event_request["defer_count"] = 0
        
        try:
            self._queue.put(event_request, block=True, timeout=0.001)
            
            event_id = event_request.get('client_event_id', 'unknown')
            parent_id = event_request.get('client_parent_event_id')
            debug(f"[EventQueue] Queued event {truncate_id(event_id)} (parent: {truncate_id(parent_id)}), queue size: {self._queue.qsize()}")
            
            # Wake worker if batch large enough
            if self._queue.qsize() >= self.flush_at_count:
                self._flush_event.set()
                
        except queue.Full:
            warning(f"[EventQueue] Queue at max size {self.max_queue_size}, dropping event")

    def force_flush(self, timeout_seconds: float = 5.0) -> None:
        """Flush current queue synchronously (best-effort)."""
        with self._flush_lock:
            debug(f"[EventQueue] Force flush requested, queue size: {self._queue.qsize()}")
            
            # Signal the worker to flush immediately
            self._flush_event.set()
            
            # Wait for the queue to be processed
            end_time = time.time() + timeout_seconds
            last_size = -1
            stable_count = 0
            
            debug(f"[EventQueue] Force flush: entering wait loop, timeout={timeout_seconds}s")
            iterations = 0
            start_time = time.time()
            while time.time() < end_time:
                iterations += 1
                if iterations % 20 == 1:  # Log every second (20 * 0.05s)
                    debug(f"[EventQueue] Force flush: iteration {iterations}, time left: {end_time - time.time():.1f}s")
                
                current_size = self._queue.qsize()
                
                with self._processing_lock:
                    processing = self._processing_count
                
                # Check if we're done
                if current_size == 0 and processing == 0:
                    if stable_count >= 2:
                        debug("[EventQueue] Force flush complete")
                        return
                    stable_count += 1
                    debug(f"[EventQueue] Force flush: queue empty, stable_count={stable_count}")
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
                
                # Safety check to prevent infinite loop
                if time.time() - start_time > timeout_seconds + 1:
                    warning(f"[EventQueue] Force flush: exceeded timeout by >1s, breaking")
                    break
            
            debug(f"[EventQueue] Force flush: exited wait loop after {time.time() - start_time:.1f}s")

    def is_empty(self) -> bool:
        """Check if queue is completely empty."""
        with self._processing_lock:
            queue_empty = self._queue.empty()
            not_processing = self._processing_count == 0
        # No deferred queue to check anymore
        return queue_empty and not_processing

    def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown the event queue."""
        info(f"[EventQueue] Shutting down with {self._queue.qsize()} events in queue")
        
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
                    error(f"[EventQueue] Batch processing error: {e}")
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
        debug(f"[EventQueue] Processing batch of {len(batch)} events")
        
        # No need to handle deferred events - we don't defer anymore
        
        # Group for parallel processing
        dependency_groups = self._group_by_dependencies(batch)
        
        # Process each group in parallel
        for group_index, group in enumerate(dependency_groups):
            debug(f"[EventQueue] Processing dependency group {group_index + 1}/{len(dependency_groups)} with {len(group)} events in parallel")
            
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
                    debug(f"[EventQueue] Failed to send event: {e}")
                    if self.retry_failed:
                        self._retry_event(event)

    def _group_by_dependencies(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group events for parallel processing.
        
        Since the backend handles events in any order using client-side event IDs,
        we don't need to check dependencies. Just return all events in one group
        for maximum parallel processing.
        """
        if not events:
            return []
        
        # Mark all event IDs as sent for tracking
        for event in events:
            if event_id := event.get("client_event_id"):
                self._sent_ids.add(event_id)
        
        # Return all events in a single group for parallel processing
        return [events]

    def _send_event_safe(self, event_request: Dict[str, Any]) -> bool:
        """Send event with error suppression if configured."""
        if self.config.error_handling.suppress_errors:
            try:
                return self._send_event(event_request)
            except Exception as e:
                warning(f"[EventQueue] Suppressed send error: {e}")
                return False
        else:
            return self._send_event(event_request)

    def _send_event(self, event_request: Dict[str, Any]) -> bool:
        """Send a single event to the backend."""
        # No dependency checking needed - backend handles events in any order
        
        # Check for blob offloading
        payload = event_request.get("payload", {})
        raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        should_offload = len(raw_bytes) > self.blob_threshold
        
        if should_offload:
            event_id = event_request.get('client_event_id', 'unknown')
            debug(f"[EventQueue] Event {truncate_id(event_id)} needs blob storage ({len(raw_bytes)} bytes > {self.blob_threshold} threshold)")
        
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
                    debug(f"[EventQueue] Blob uploaded for event {truncate_id(event_request.get('client_event_id'))}")
                else:
                    error("[EventQueue] No blob_url received for large payload")
                    return False
            
            return True
            
        except Exception as e:
            debug(f"[EventQueue] Failed to send event {truncate_id(event_request.get('client_event_id'))}: {e}")
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
        resp = httpx.put(blob_url, content=data, headers=headers)
        resp.raise_for_status()

    @staticmethod
    def _create_preview(event_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create preview of large payload for logging."""
        try:
            t = (event_type or "generic").lower()
            
            if t == "llm_generation":

                req = payload.get("request", {})
                usage = payload.get("usage", {})
                messages = req.get("messages", [])[:5]
                output = payload.get("response", {}).get("output", {})
                compressed_messages = []
                for i, m in enumerate(messages):
                    compressed_message_item = {}
                    for k, v in messages[i].items():
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
                    "function_name": payload.get("function_name")[:200] if payload.get("function_name") else None,
                    "arguments": truncated_args,
                }

            elif t == "error_traceback":
                return {
                    "error": payload.get("error")[:200] if payload.get("error") else None,
                }

            elif t == "generic":
                return {
                    "details": payload.get("details")[:200] if payload.get("details") else None,
                }
            else:
                return {"details": "preview_unavailable"}
                
        except Exception:
            return {"details": "preview_error"}