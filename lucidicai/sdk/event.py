"""SDK event creation and management."""
import asyncio
import gzip
import io
import json
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, Set
from weakref import WeakSet
import traceback
import httpx

from .context import current_parent_event_id
from ..core.config import get_config
from .event_builder import EventBuilder
from ..utils.logger import debug, warning, error, truncate_id


# Default blob threshold (64KB)
DEFAULT_BLOB_THRESHOLD = 65536

# Track background threads and tasks for flush()
_background_threads: Set[threading.Thread] = WeakSet()
_background_tasks: Set[asyncio.Task] = WeakSet()


def _compress_json(payload: Dict[str, Any]) -> bytes:
    """Compress JSON payload using gzip."""
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw)
    return buf.getvalue()


def _upload_blob_sync(blob_url: str, data: bytes) -> None:
    """Upload compressed blob to presigned URL (synchronous)."""
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    resp = httpx.put(blob_url, content=data, headers=headers)
    resp.raise_for_status()


async def _upload_blob_async(blob_url: str, data: bytes) -> None:
    """Upload compressed blob to presigned URL (asynchronous)."""
    headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
    async with httpx.AsyncClient() as client:
        resp = await client.put(blob_url, content=data, headers=headers)
        resp.raise_for_status()


def _track_background_task(task: asyncio.Task) -> None:
    """Track a background task for flush()."""
    _background_tasks.add(task)


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


def _prepare_event_request(
    type: str,
    event_id: Optional[str],
    session_id: Optional[str],
    blob_threshold: int,
    **kwargs
) -> tuple[Dict[str, Any], bool, Optional[Dict[str, Any]]]:
    """Prepare event request, determining if blob offload is needed.
    
    Returns:
        Tuple of (send_body, needs_blob, original_payload)
    """
    from ..sdk.init import get_session_id

    # Use provided session_id or fall back to context
    if not session_id:
        session_id = get_session_id()

    if not session_id:
        # No active session
        debug("[Event] No active session, returning dummy event ID")
        return None, False, None
    
    # Get parent event ID from context
    parent_event_id = None
    try:
        parent_event_id = current_parent_event_id.get()
    except Exception:
        pass
    
    # Use provided event ID or generate new one
    client_event_id = event_id or str(uuid.uuid4())
    
    # Build parameters for EventBuilder
    params = {
        'type': type,
        'event_id': client_event_id,
        'parent_event_id': parent_event_id,
        'session_id': session_id,
        'occurred_at': kwargs.get('occurred_at') or datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    # Use EventBuilder to create normalized event request
    event_request = EventBuilder.build(params)
    
    debug(f"[Event] Creating {type} event {truncate_id(client_event_id)} (parent: {truncate_id(parent_event_id)}, session: {truncate_id(session_id)})")
    
    # Check for blob offloading
    payload = event_request.get("payload", {})
    raw_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    needs_blob = len(raw_bytes) > blob_threshold
    
    if needs_blob:
        debug(f"[Event] Event {truncate_id(client_event_id)} needs blob storage ({len(raw_bytes)} bytes > {blob_threshold} threshold)")
    
    send_body: Dict[str, Any] = dict(event_request)
    if needs_blob:
        send_body["needs_blob"] = True
        send_body["payload"] = _create_preview(send_body.get("type"), payload)
    else:
        send_body["needs_blob"] = False
    
    return send_body, needs_blob, payload if needs_blob else None


def _get_event_resource():
    """Get an event resource from a registered client.

    Returns:
        Event resource or None if no client is available.
    """
    try:
        from .shutdown_manager import get_shutdown_manager
        manager = get_shutdown_manager()
        with manager._client_lock:
            # Return first available client's event resource
            for client in manager._clients.values():
                if hasattr(client, '_resources') and 'events' in client._resources:
                    return client._resources['events']
    except Exception as e:
        debug(f"[Event] Failed to get event resource from client registry: {e}")
    return None


def create_event(
    type: str = "generic",
    event_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create a new event (synchronous).

    Args:
        type: Event type (llm_generation, function_call, error_traceback, generic)
        event_id: Optional client event ID (will generate if not provided)
        session_id: Optional session ID (will use context if not provided)
        **kwargs: Event-specific fields

    Returns:
        Event ID (client-generated or provided UUID)
    """
    config = get_config()
    blob_threshold = getattr(config, 'blob_threshold', DEFAULT_BLOB_THRESHOLD)

    send_body, needs_blob, original_payload = _prepare_event_request(
        type, event_id, session_id, blob_threshold, **kwargs
    )

    if send_body is None:
        # No active session
        return str(uuid.uuid4())

    client_event_id = send_body.get('client_event_id', str(uuid.uuid4()))

    # Get event resource from client registry
    event_resource = _get_event_resource()
    if not event_resource:
        warning("[Event] No event resource available (no client registered), event not sent")
        return client_event_id

    try:
        response = event_resource.create_event(send_body)

        # Handle blob upload if needed (blocking)
        if needs_blob and original_payload:
            blob_url = response.get("blob_url")
            if blob_url:
                compressed = _compress_json(original_payload)
                _upload_blob_sync(blob_url, compressed)
                debug(f"[Event] Blob uploaded for event {truncate_id(client_event_id)}")
            else:
                error("[Event] No blob_url received for large payload")

        debug(f"[Event] Event {truncate_id(client_event_id)} sent successfully")

    except Exception as e:
        error(f"[Event] Failed to send event {truncate_id(client_event_id)}: {e}")

    return client_event_id


async def acreate_event(
    type: str = "generic",
    event_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create a new event (asynchronous).

    Args:
        type: Event type (llm_generation, function_call, error_traceback, generic)
        event_id: Optional client event ID (will generate if not provided)
        session_id: Optional session ID (will use context if not provided)
        **kwargs: Event-specific fields

    Returns:
        Event ID (client-generated or provided UUID)
    """
    # Check if we're in shutdown - fall back to sync if we are
    if sys.is_finalizing():
        debug(f"[Event] Python is finalizing in acreate_event, falling back to sync")
        return create_event(type, event_id, session_id, **kwargs)

    config = get_config()
    blob_threshold = getattr(config, 'blob_threshold', DEFAULT_BLOB_THRESHOLD)

    send_body, needs_blob, original_payload = _prepare_event_request(
        type, event_id, session_id, blob_threshold, **kwargs
    )

    if send_body is None:
        # No active session
        return str(uuid.uuid4())

    client_event_id = send_body.get('client_event_id', str(uuid.uuid4()))

    # Get event resource from client registry
    event_resource = _get_event_resource()
    if not event_resource:
        warning("[Event] No event resource available (no client registered), event not sent")
        return client_event_id

    try:
        # Try async first, fall back to sync if we get shutdown errors
        try:
            response = await event_resource.acreate_event(send_body)
        except RuntimeError as e:
            if "cannot schedule new futures after interpreter shutdown" in str(e).lower():
                debug(f"[Event] Detected shutdown in acreate_event, falling back to sync")
                response = event_resource.create_event(send_body)
            else:
                raise

        # Handle blob upload if needed (background task)
        if needs_blob and original_payload:
            blob_url = response.get("blob_url")
            if blob_url:
                compressed = _compress_json(original_payload)
                try:
                    # Try to create background task
                    task = asyncio.create_task(_upload_blob_async(blob_url, compressed))
                    _track_background_task(task)
                    debug(f"[Event] Blob upload started in background for event {truncate_id(client_event_id)}")
                except RuntimeError as e:
                    if "cannot schedule new futures" in str(e).lower() or sys.is_finalizing():
                        # Can't create tasks, do it synchronously
                        debug(f"[Event] Cannot create background task, uploading blob synchronously")
                        _upload_blob_sync(blob_url, compressed)
                        debug(f"[Event] Blob uploaded synchronously for event {truncate_id(client_event_id)}")
                    else:
                        raise
            else:
                error("[Event] No blob_url received for large payload")

        debug(f"[Event] Event {truncate_id(client_event_id)} sent successfully")

    except Exception as e:
        error(f"[Event] Failed to send event {truncate_id(client_event_id)}: {e}")

    return client_event_id


def create_error_event(
    error: Union[str, Exception],
    parent_event_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create an error traceback event (synchronous).
    
    This is a convenience function for creating error events with proper
    traceback information.
    
    Args:
        error: The error message or exception object
        parent_event_id: Optional parent event ID for nesting
        **kwargs: Additional event parameters
        
    Returns:
        Event ID of the created error event
    """
    import traceback
    
    if isinstance(error, Exception):
        error_str = str(error)
        traceback_str = traceback.format_exc()
    else:
        error_str = str(error)
        traceback_str = kwargs.pop('traceback', '')
    
    return create_event(
        type="error_traceback",
        error=error_str,
        traceback=traceback_str,
        parent_event_id=parent_event_id,
        **kwargs
    )


async def acreate_error_event(
    error: Union[str, Exception],
    parent_event_id: Optional[str] = None,
    **kwargs
) -> str:
    """Create an error traceback event (asynchronous).
    
    This is a convenience function for creating error events with proper
    traceback information.
    
    Args:
        error: The error message or exception object
        parent_event_id: Optional parent event ID for nesting
        **kwargs: Additional event parameters
        
    Returns:
        Event ID of the created error event
    """
    import traceback
    
    if isinstance(error, Exception):
        error_str = str(error)
        traceback_str = traceback.format_exc()
    else:
        error_str = str(error)
        traceback_str = kwargs.pop('traceback', '')
    
    return await acreate_event(
        type="error_traceback",
        error=error_str,
        traceback=traceback_str,
        parent_event_id=parent_event_id,
        **kwargs
    )


def emit_event(
    type: str = "generic",
    event_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> str:
    """Fire-and-forget event creation that returns instantly.
    
    This function returns immediately with an event ID, while the actual
    event creation and any blob uploads happen in a background thread.
    Perfect for hot path telemetry where latency is critical.
    
    During shutdown, falls back to synchronous event creation to avoid
    "cannot schedule new futures after interpreter shutdown" errors.
    
    Args:
        type: Event type (llm_generation, function_call, error_traceback, generic)
        event_id: Optional client event ID (will generate if not provided)
        session_id: Optional session ID (will use context if not provided)
        **kwargs: Event-specific fields
        
    Returns:
        Event ID (client-generated or provided UUID) - returned immediately
    """
    from ..sdk.init import get_session_id
    from .context import current_session_id
    from .shutdown_manager import get_shutdown_manager
    
    # Pre-generate event ID for instant return
    client_event_id = event_id or str(uuid.uuid4())
    
    # Capture context variables BEFORE creating the thread
    # This preserves the context chain across thread boundaries
    captured_parent_id = kwargs.get('parent_event_id')
    if captured_parent_id is None:
        try:
            captured_parent_id = current_parent_event_id.get()
        except Exception:
            captured_parent_id = None
    
    # Capture session from context if not provided
    if not session_id:
        try:
            # Try context variable first (most specific)
            session_id = current_session_id.get()
        except Exception:
            pass
        
        # Fall back to get_session_id if still None
        if not session_id:
            session_id = get_session_id()
    
    if not session_id:
        debug("[Event] No active session for emit_event, returning dummy event ID")
        return client_event_id
    
    # Update kwargs with captured context
    if captured_parent_id is not None:
        kwargs['parent_event_id'] = captured_parent_id
    
    # Check if Python interpreter is shutting down
    if sys.is_finalizing():
        debug(f"[Event] Python is finalizing, using synchronous event creation for {truncate_id(client_event_id)}")
        try:
            return create_event(type, client_event_id, session_id, **kwargs)
        except Exception as e:
            error(f"[Event] Failed to create event during finalization: {e}")
            return client_event_id
    
    # Check if shutdown manager thinks we're shutting down
    try:
        from .shutdown_manager import get_shutdown_manager
        shutdown_manager = get_shutdown_manager()
        if shutdown_manager.is_shutting_down:
            debug(f"[Event] ShutdownManager indicates shutdown, using synchronous event creation for {truncate_id(client_event_id)}")
            try:
                return create_event(type, client_event_id, session_id, **kwargs)
            except Exception as e:
                error(f"[Event] Failed to create event during shutdown: {e}")
                return client_event_id
    except Exception:
        pass  # ShutdownManager not available
    
    # Try to create and start thread - fall back to sync if it fails
    try:
        # Normal path: Run async function in background thread
        def _run():
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        acreate_event(type, client_event_id, session_id, **kwargs)
                    )
                finally:
                    loop.close()
            except RuntimeError as e:
                if "cannot schedule new futures after interpreter shutdown" in str(e).lower():
                    # Interpreter is shutting down, can't use async
                    debug(f"[Event] Detected interpreter shutdown in thread, falling back to sync")
                    create_event(type, client_event_id, session_id, **kwargs)
                else:
                    error(f"[Event] Background emit failed for {truncate_id(client_event_id)}: {e}")
            except Exception as e:
                error(f"[Event] Background emit failed for {truncate_id(client_event_id)}: {e}")
        
        thread = threading.Thread(target=_run, daemon=True, name=f"emit-{truncate_id(client_event_id)}")
        _background_threads.add(thread)
        thread.start()
    except (RuntimeError, SystemError) as e:
        # Can't create threads during shutdown
        debug(f"[Event] Cannot create thread (likely shutdown): {e}. Using synchronous fallback.")
        try:
            return create_event(type, client_event_id, session_id, **kwargs)
        except Exception as e2:
            error(f"[Event] Synchronous fallback also failed: {e2}")
            return client_event_id
    
    debug(f"[Event] Emitted {type} event {truncate_id(client_event_id)} (fire-and-forget)")
    return client_event_id


def emit_error_event(
    error: Union[str, Exception],
    parent_event_id: Optional[str] = None,
    **kwargs
) -> str:
    """Fire-and-forget error event creation that returns instantly.
    
    This is a convenience function for creating error events with proper
    traceback information, returning immediately while processing happens
    in the background.
    
    Args:
        error: The error message or exception object
        parent_event_id: Optional parent event ID for nesting
        **kwargs: Additional event parameters
        
    Returns:
        Event ID of the created error event - returned immediately
    """
    import traceback
    
    if isinstance(error, Exception):
        error_str = str(error)
        traceback_str = traceback.format_exc()
    else:
        error_str = str(error)
        traceback_str = kwargs.pop('traceback', '')
    
    # Note: emit_event already handles context capture for both
    # parent_event_id and session_id, so we just pass through
    return emit_event(
        type="error_traceback",
        error=error_str,
        traceback=traceback_str,
        parent_event_id=parent_event_id,
        **kwargs
    )


def flush(timeout: float = 5.0) -> None:
    """Wait for all background operations to complete.
    
    This includes:
    - Event creation HTTP requests
    - S3 blob uploads for large payloads
    - Session creation requests
    - Any other background telemetry operations
    
    Useful before program exit or when you need to ensure all telemetry
    has been sent.
    
    Args:
        timeout: Maximum time to wait in seconds (default: 5.0)
    """
    import time
    
    start_time = time.time()
    
    # Flush sessions first
    from .session import flush_sessions as _flush_sessions
    remaining = timeout - (time.time() - start_time)
    if remaining > 0:
        _flush_sessions(timeout=remaining)
    
    # Wait for event background threads
    threads = list(_background_threads)
    for thread in threads:
        if thread.is_alive():
            remaining = timeout - (time.time() - start_time)
            if remaining > 0:
                thread.join(timeout=remaining)
                if thread.is_alive():
                    warning(f"[SDK] Thread {thread.name} did not complete within timeout")
    
    # Wait for async tasks if in async context
    try:
        loop = asyncio.get_running_loop()
        tasks = [t for t in _background_tasks if not t.done()]
        if tasks:
            remaining = timeout - (time.time() - start_time)
            if remaining > 0:
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=remaining
                        )
                    )
                except asyncio.TimeoutError:
                    warning(f"[SDK] {len(tasks)} async tasks did not complete within timeout")
    except RuntimeError:
        # Not in async context, skip async task flushing
        pass
    
    debug(f"[SDK] Flush completed in {time.time() - start_time:.2f}s")
