"""Asynchronous, non-blocking event queue with client-side UUIDs and blob handling.

This module implements the TypeScript-style EventQueue for the Python SDK:
- Immediate return of client_event_id (UUID) on event creation
- Background batching and retries
- Client-side blob size detection, preview generation, and gzip upload
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"
VERBOSE = os.getenv("LUCIDIC_VERBOSE", "False") == "True"


class EventQueue:
    def __init__(self, client):
        # Configuration
        self.max_queue_size: int = int(os.getenv("LUCIDIC_MAX_QUEUE_SIZE", 100000))
        self.flush_interval_ms: int = int(os.getenv("LUCIDIC_FLUSH_INTERVAL", 100))
        self.flush_at_count: int = int(os.getenv("LUCIDIC_FLUSH_AT", 100))
        self.blob_threshold: int = int(os.getenv("LUCIDIC_BLOB_THRESHOLD", 64 * 1024))
        self._daemon_mode = os.getenv("LUCIDIC_DAEMON_QUEUE", "true").lower() == "true"

        # Runtime state
        self._client = client
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._stopped = threading.Event()
        self._flush_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._sent_ids: set[str] = set()
        self._deferred_queue: List[Dict[str, Any]] = []
        self._deferred_lock = threading.Lock()
        
        # Thread safety for flush operations
        self._flush_lock = threading.Lock()
        self._processing_count = 0
        self._processing_lock = threading.Lock()
        self._flush_complete = threading.Event()

        # Start background worker
        self._start_worker()

    # --- Public API ---
    def queue_event(self, event_request: Dict[str, Any]) -> None:
        """Enqueue an event for background processing.

        event_request must include:
          - session_id
          - client_event_id (client-side uuid)
          - type
          - payload (typed payload)
          - occurred_at (ISO string)
          - Optional: duration, tags, metadata, client_parent_event_id
        """
        # Ensure a defer counter exists for parent-order deferrals
        if "defer_count" not in event_request:
            event_request["defer_count"] = 0
        
        try:
            # Try to put with a small timeout to handle full queue
            self._queue.put(event_request, block=True, timeout=0.001)
            
            if DEBUG:
                logger.debug(f"[EventQueue] Queued event {event_request.get('client_event_id')}, queue size: {self._queue.qsize()}")
            if VERBOSE:
                logger.debug(f"[EventQueue] Event payload: {json.dumps(event_request, indent=2)}")
            
            # Wake worker if batch large enough
            if self._queue.qsize() >= self.flush_at_count:
                self._flush_event.set()
                
        except queue.Full:
            if DEBUG:
                logger.debug(f"[EventQueue] Queue at max size {self.max_queue_size}, dropping event")
            # In the original implementation, oldest was dropped. With Queue, we drop the new one.
            # To match original behavior exactly, we'd need a deque, but this is simpler.

    def force_flush(self, timeout_seconds: float = 5.0) -> None:
        """Flush current queue synchronously (best-effort). Thread-safe."""
        with self._flush_lock:
            if DEBUG:
                logger.debug(f"[EventQueue] Force flush requested, queue size: {self._queue.qsize()}")
            
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
                            logger.debug("[EventQueue] Force flush complete - queue empty")
                        return
                    stable_count += 1
                else:
                    stable_count = 0
                
                # If size hasn't changed, we might be stuck
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 10:  # 0.5 seconds of no progress
                        if DEBUG:
                            logger.debug(f"[EventQueue] Force flush timeout - queue stuck at {current_size}")
                        break
                else:
                    stable_count = 0
                    last_size = current_size
                
                # Signal flush again in case worker missed it
                self._flush_event.set()
                time.sleep(0.05)
            
            if DEBUG:
                logger.debug(f"[EventQueue] Force flush ended, remaining: {self._queue.qsize()}")

    def is_empty(self) -> bool:
        """Check if queue is completely empty and no events are being processed."""
        with self._processing_lock:
            queue_empty = self._queue.empty()
            not_processing = self._processing_count == 0
        with self._deferred_lock:
            deferred_empty = len(self._deferred_queue) == 0
        return queue_empty and not_processing and deferred_empty

    def shutdown(self) -> None:
        """Enhanced shutdown with better flushing."""
        if DEBUG:
            logger.debug(f"[EventQueue] Shutdown requested, queue size: {self._queue.qsize()}")
        
        # First try to flush remaining events
        self.force_flush(timeout_seconds=2.0)
        
        # Wait for queue to be truly empty
        wait_start = time.time()
        while not self.is_empty() and (time.time() - wait_start < 2.0):
            time.sleep(0.01)
            
        if not self.is_empty() and DEBUG:
            logger.debug(f"[EventQueue] Not empty after wait: queue={self._queue.qsize()}, processing={self._processing_count}, deferred={len(self._deferred_queue)}")
        
        # Then signal stop
        self._stopped.set()
        self._flush_event.set()  # Wake up worker
        
        # Wait for worker with timeout
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5.0)  # Increased timeout
            if self._worker.is_alive() and DEBUG:
                logger.debug("[EventQueue] Worker thread did not terminate in time")

    # --- Internals ---
    def _start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        # Use configurable daemon mode
        self._worker = threading.Thread(
            target=self._run_loop, 
            name="LucidicEventQueue", 
            daemon=self._daemon_mode
        )
        self._worker.start()
        if DEBUG:
            logger.debug(f"[EventQueue] Started worker thread (daemon={self._daemon_mode})")

    def _run_loop(self) -> None:
        """Main worker loop using queue.Queue for simpler implementation."""
        while not self._stopped.is_set():
            batch: List[Dict[str, Any]] = []
            deadline = time.time() + (self.flush_interval_ms / 1000.0)
            force_flush = False
            
            # Collect batch up to flush_at_count or until deadline
            while True:
                # Check if flush was requested
                if self._flush_event.is_set():
                    force_flush = True
                    self._flush_event.clear()
                
                # During force flush, get ALL events
                if force_flush:
                    # Drain entire queue when flushing
                    while not self._queue.empty():
                        try:
                            item = self._queue.get_nowait()
                            batch.append(item)
                        except queue.Empty:
                            break
                    # Process what we have
                    if batch:
                        break
                    # If still empty after draining, wait a bit for stragglers
                    if not batch:
                        time.sleep(0.01)
                        continue
                else:
                    # Normal batching logic
                    if len(batch) >= self.flush_at_count:
                        break
                    
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        # Wait for item with timeout
                        timeout = min(remaining_time, 0.05)  # Check more frequently
                        item = self._queue.get(block=True, timeout=timeout)
                        batch.append(item)
                    except queue.Empty:
                        # Check if stopped
                        if self._stopped.is_set():
                            # Drain remaining queue on shutdown
                            while not self._queue.empty():
                                try:
                                    batch.append(self._queue.get_nowait())
                                except queue.Empty:
                                    break
                            break
                        # If we have events and deadline passed, process them
                        if batch and time.time() >= deadline:
                            break

            # Process batch if we have events
            if batch:
                with self._processing_lock:
                    self._processing_count = len(batch)
                try:
                    self._process_batch(batch)
                except Exception as e:
                    if DEBUG:
                        logger.debug(f"[EventQueue] Batch processing error: {e}")
                finally:
                    with self._processing_lock:
                        self._processing_count = 0
                    
        # Final drain on shutdown
        final_batch = []
        while not self._queue.empty():
            try:
                final_batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if final_batch:
            with self._processing_lock:
                self._processing_count = len(final_batch)
            try:
                self._process_batch(final_batch)
            except Exception:
                pass
            finally:
                with self._processing_lock:
                    self._processing_count = 0

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of events with parent-child ordering."""
        if DEBUG:
            logger.debug(f"[EventQueue] Processing batch of {len(batch)} events")
        
        # Add any deferred events back to the batch
        with self._deferred_lock:
            if self._deferred_queue:
                batch.extend(self._deferred_queue)
                self._deferred_queue.clear()
        
        # Reorder within batch to respect parent -> child when both present
        id_to_evt = {e.get("client_event_id"): e for e in batch if e.get("client_event_id")}
        remaining = list(batch)
        ordered: List[Dict[str, Any]] = []

        processed_ids: set[str] = set()
        max_iterations = len(remaining) * 2 if remaining else 0
        iters = 0
        while remaining and iters < max_iterations:
            iters += 1
            progressed = False
            next_remaining: List[Dict[str, Any]] = []
            for ev in remaining:
                parent_id = ev.get("client_parent_event_id")
                if not parent_id or (parent_id not in id_to_evt) or (parent_id in processed_ids) or (parent_id in self._sent_ids):
                    ordered.append(ev)
                    if ev.get("client_event_id"):
                        processed_ids.add(ev["client_event_id"])
                    progressed = True
                else:
                    next_remaining.append(ev)
            remaining = next_remaining if progressed else []
            if not progressed and next_remaining:
                # Break potential cycles by appending the rest
                ordered.extend(next_remaining)
                remaining = []

        for event_request in ordered:
            if DEBUG:
                logger.debug(f"[EventQueue] Sending event {event_request.get('client_event_id')}")
            
            # Retry up to 3 times with exponential backoff
            attempt = 0
            backoff = 0.25
            while attempt < 3:
                try:
                    if self._send_event(event_request):
                        # Mark as sent if it has id
                        ev_id = event_request.get("client_event_id")
                        if ev_id:
                            self._sent_ids.add(ev_id)
                            if DEBUG:
                                logger.debug(f"[EventQueue] Successfully sent event {ev_id}")
                    break
                except Exception as e:
                    attempt += 1
                    if DEBUG:
                        logger.debug(f"[EventQueue] Failed to send event (attempt {attempt}/3): {e}")
                    if attempt >= 3:
                        logger.error(f"[EventQueue] Failed to send event after 3 attempts: {event_request.get('client_event_id')}")
                        break
                    time.sleep(backoff)
                    backoff *= 2

    def _send_event(self, event_request: Dict[str, Any]) -> bool:
        """Send event with enhanced error handling."""
        try:
            # If parent exists and not yet sent, defer up to 5 times
            parent_id = event_request.get("client_parent_event_id")
            if parent_id and parent_id not in self._sent_ids:
                # Defer unless we've tried several times already
                if event_request.get("defer_count", 0) < 5:
                    event_request["defer_count"] = event_request.get("defer_count", 0) + 1
                    # Add to deferred queue for next batch
                    with self._deferred_lock:
                        self._deferred_queue.append(event_request)
                    return True
            
            # Offload large payloads to blob storage
            payload = event_request.get("payload", {})
            raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            should_offload = len(raw_bytes) > self.blob_threshold
            
            if DEBUG:
                logger.debug(f"[EventQueue] Event size: {len(raw_bytes)} bytes, offload: {should_offload}")

            send_body: Dict[str, Any] = dict(event_request)
            if should_offload:
                send_body["needs_blob"] = True
                send_body["payload"] = self._to_preview(send_body.get("type"), payload)
            else:
                send_body["needs_blob"] = False
            
            if VERBOSE and not should_offload:
                logger.debug(f"[EventQueue] Sending body: {json.dumps(send_body, indent=2)}")

            # POST /events
            response = self._client.make_request("events", "POST", send_body)

            # If offloading, synchronously upload compressed payload
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
            logger.error(f"[EventQueue] Failed to send event: {e}")
            raise  # Re-raise for retry logic

    # --- Helpers for blob handling ---
    @staticmethod
    def _compress_json(payload: Dict[str, Any]) -> bytes:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(raw)
        return buf.getvalue()

    def _upload_blob(self, blob_url: str, data: bytes) -> None:
        """Upload blob with proper error handling and logging."""
        try:
            if DEBUG:
                logger.debug(f"[EventQueue] Uploading blob, size: {len(data)} bytes")
            
            headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
            resp = requests.put(blob_url, data=data, headers=headers)
            resp.raise_for_status()
            
            if DEBUG:
                logger.debug(f"[EventQueue] Blob upload successful, status: {resp.status_code}")
                
        except Exception as e:
            # Log error but don't fail silently
            logger.error(f"[EventQueue] Blob upload failed: {e}")
            # Re-raise to trigger retry logic
            raise

    @staticmethod
    def _to_preview(event_type: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
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
            if t == "function_call":
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
            if t == "error_traceback":
                return {
                    "error": (payload.get("error")[:200] if payload.get("error") else None),
                }
            if t == "generic":
                return {
                    "details": (payload.get("details")[:200] if payload.get("details") else None),
                }
        except Exception:
            pass
        return {"details": "preview_unavailable"}