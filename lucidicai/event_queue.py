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
        """Flush current queue synchronously (best-effort)."""
        end_time = time.time() + timeout_seconds
        while time.time() < end_time:
            if self._queue.empty():
                return
            self._flush_event.set()
            time.sleep(0.05)

    def shutdown(self) -> None:
        """Enhanced shutdown with better flushing."""
        if DEBUG:
            logger.debug(f"[EventQueue] Shutdown requested, queue size: {self._queue.qsize()}")
        
        # First try to flush remaining events
        self.force_flush(timeout_seconds=2.0)
        
        # Then signal stop
        self._stopped.set()
        self._flush_event.set()  # Wake up worker
        
        # Wait for worker with timeout
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)
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
            
            # Collect batch up to flush_at_count or until deadline
            while len(batch) < self.flush_at_count:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    break
                    
                try:
                    # Wait for item with timeout
                    timeout = min(remaining_time, 0.1)  # Check stopped flag periodically
                    item = self._queue.get(block=True, timeout=timeout)
                    batch.append(item)
                    
                    # Check if we should flush immediately
                    if self._flush_event.is_set():
                        self._flush_event.clear()
                        # Drain more items if available
                        while len(batch) < self.flush_at_count:
                            try:
                                item = self._queue.get_nowait()
                                batch.append(item)
                            except queue.Empty:
                                break
                        break
                        
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

            # Process batch if we have events
            if batch:
                try:
                    self._process_batch(batch)
                except Exception:
                    # Swallow to keep worker alive
                    pass
                    
        # Final drain on shutdown
        final_batch = []
        while not self._queue.empty():
            try:
                final_batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if final_batch:
            try:
                self._process_batch(final_batch)
            except Exception:
                pass

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
                for i, m in enumerate(messages):
                    messages[i]["content"] = str(m["content"])[:200] if m["content"] else None
                return {
                    "request": {
                        "model": req.get("model")[:200] if req.get("model") else None,
                        "provider": req.get("provider")[:200] if req.get("provider") else None,
                        "messages": messages,
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