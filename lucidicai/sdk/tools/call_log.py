"""Optional per-call mock-dispatch log — a real-time view of mocking.

When the ``LUCIDIC_MOCK_LOG`` environment variable points at a file path,
every tool call routed through the mock transport
(``emit_call_through_backend`` / ``aemit_call_through_backend``) appends one
human-readable line, flushed immediately so ``tail -f`` shows mocking as it
happens. Unset → no-op with negligible overhead (one ``getenv``).

This is a developer aid, not a telemetry contract: it never raises into the
dispatch path (every write is guarded), and it's intentionally separate from
the structured events the backend already records per mock_call.

Line shape (columns are space-padded for eyeballing a tailed file)::

    2026-05-28T19:52:22.044Z  session=293ebcb0  slow_query   MOCKED   tier=PYTHON  args={'query': '...'}  -> [[1, 'olivia...'], ...]  (153ms)
    2026-05-28T19:52:22.144Z  session=293ebcb0  get_weather  PASS_THRU tier=PASS_THROUGH  args={'city': 'SF'}  (ran real fn)  (100ms)
    2026-05-28T12:50:55.899Z  session=160e0007  add_numbers  DRIFT->local  args={'a': 2, 'b': 3}  session_hash=5ba0..  current_hash=0612..  (78ms)
    2026-05-28T19:52:22.400Z  session=293ebcb0  slow_query   ERROR    args={'query': '...'}  !! impl_error: DuckDB rejected SQL: ...  (153ms)
"""
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("Lucidic")

ENV_VAR = "LUCIDIC_MOCK_LOG"

_lock = threading.Lock()
_handle = None
_handle_path: Optional[str] = None


def _preview(value: Any, limit: int = 240) -> str:
    """``repr`` a value, single-lined and length-capped, never raising."""
    try:
        s = repr(value)
    except Exception:
        s = f"<unreprable {type(value).__name__}>"
    s = s.replace("\n", "\\n").replace("\r", "")
    if len(s) > limit:
        s = s[:limit] + f"...(+{len(s) - limit} chars)"
    return s


def _get_handle():
    """Return the append handle for the current ``LUCIDIC_MOCK_LOG`` path.

    Returns ``None`` when the env var is unset (logging disabled). Reopens
    if the path changed between calls (tests flip it); caches otherwise so
    the hot path doesn't reopen the file per call.
    """
    global _handle, _handle_path
    path = os.getenv(ENV_VAR)
    if not path:
        return None
    if _handle is not None and _handle_path == path:
        return _handle
    if _handle is not None:
        try:
            _handle.close()
        except Exception:
            pass
        _handle = None
        _handle_path = None
    try:
        # buffering=1 → line-buffered text mode; combined with the explicit
        # flush below this gives a true real-time tail.
        _handle = open(path, "a", buffering=1, encoding="utf-8")
        _handle_path = path
    except OSError as exc:
        logger.warning("[mock_call] could not open %s=%s: %s", ENV_VAR, path, exc)
        _handle = None
        _handle_path = None
    return _handle


def record_call(
    *,
    session_id: str,
    tool_name: str,
    kwargs: dict,
    outcome: str,
    tier: Optional[str] = None,
    result: Any = None,
    has_result: bool = False,
    error: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    extra: Optional[str] = None,
) -> None:
    """Append one line describing a single mock-dispatch outcome.

    No-op unless ``LUCIDIC_MOCK_LOG`` is set. Guaranteed not to raise — a
    logging aid must never break the dispatch it observes.

    Args:
        outcome: one of ``MOCKED`` / ``PASS_THROUGH`` / ``DRIFT->local`` /
            ``ERROR`` (free-form; only used as a display column).
        has_result: when True, the ``result`` value is rendered (used for
            ``MOCKED`` so the mocked return is visible; omitted for paths
            where the value is the user's own local function output).
    """
    try:
        handle = _get_handle()
        if handle is None:
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        sid = session_id[:8] if session_id else "--------"
        parts = [ts, f"session={sid}", f"{tool_name:<22}", f"{outcome:<12}"]
        if tier:
            parts.append(f"tier={tier}")
        parts.append(f"args={_preview(kwargs)}")
        if has_result:
            parts.append(f"-> {_preview(result)}")
        if error:
            parts.append(f"!! {_preview(error, 400)}")
        if extra:
            parts.append(extra)
        if elapsed_ms is not None:
            parts.append(f"({elapsed_ms}ms)")
        line = "  ".join(parts) + "\n"
        with _lock:
            handle.write(line)
            handle.flush()
    except Exception as exc:  # never let logging break dispatch
        logger.debug("[mock_call] mock-log write skipped: %s", exc)
