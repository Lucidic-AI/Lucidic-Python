"""SDK event creation and management."""
import asyncio
import gzip
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import httpx

from .context import current_parent_event_id
from ..core.config import get_config
from .event_builder import EventBuilder
from ..utils.logger import debug, warning, error, truncate_id


# Default blob threshold (64KB)
DEFAULT_BLOB_THRESHOLD = 65536


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
    from ..sdk.init import get_resources
    
    config = get_config()
    blob_threshold = getattr(config, 'blob_threshold', DEFAULT_BLOB_THRESHOLD)
    
    send_body, needs_blob, original_payload = _prepare_event_request(
        type, event_id, session_id, blob_threshold, **kwargs
    )
    
    if send_body is None:
        # No active session
        return str(uuid.uuid4())
    
    client_event_id = send_body.get('client_event_id', str(uuid.uuid4()))
    
    # Get resources and send event
    resources = get_resources()
    if not resources or 'events' not in resources:
        warning("[Event] No event resource available, event not sent")
        return client_event_id
    
    try:
        response = resources['events'].create_event(send_body)
        
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
    from ..sdk.init import get_resources
    
    config = get_config()
    blob_threshold = getattr(config, 'blob_threshold', DEFAULT_BLOB_THRESHOLD)
    
    send_body, needs_blob, original_payload = _prepare_event_request(
        type, event_id, session_id, blob_threshold, **kwargs
    )
    
    if send_body is None:
        # No active session
        return str(uuid.uuid4())
    
    client_event_id = send_body.get('client_event_id', str(uuid.uuid4()))
    
    # Get resources and send event
    resources = get_resources()
    if not resources or 'events' not in resources:
        warning("[Event] No event resource available, event not sent")
        return client_event_id
    
    try:
        response = await resources['events'].acreate_event(send_body)
        
        # Handle blob upload if needed (background task)
        if needs_blob and original_payload:
            blob_url = response.get("blob_url")
            if blob_url:
                compressed = _compress_json(original_payload)
                # Fire and forget - upload in background
                asyncio.create_task(_upload_blob_async(blob_url, compressed))
                debug(f"[Event] Blob upload started in background for event {truncate_id(client_event_id)}")
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
