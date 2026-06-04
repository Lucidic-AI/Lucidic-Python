"""bridge between LiteLLM's CustomLogger and Lucidic's telemetry system.

reads litellm's normalized StandardLoggingPayload (kwargs["standard_logging_object"]) as the
primary source — the same contract every mature litellm integration (datadog/arize/langfuse)
uses — and falls back to parsing response_obj when it is absent (older litellm / build failure).
this captures tool calls, streaming output, reasoning, full usage, and litellm's own cost.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from litellm import CustomLogger
except ImportError:
    # dummy base if litellm is not installed
    class CustomLogger:
        def __init__(self, **kwargs):
            pass

from lucidicai.sdk.event import emit_event
from lucidicai.sdk.init import get_session_id
from lucidicai.sdk.context import current_parent_event_id
from lucidicai.telemetry.extract import _format_tool_calls
from lucidicai.telemetry.utils.model_pricing import calculate_cost
from lucidicai.telemetry.utils.provider import detect_provider

logger = logging.getLogger("Lucidic")
DEBUG = os.getenv("LUCIDIC_DEBUG", "False") == "True"

# chat-style call types that carry choices[].message; everything else is non-chat.
# (litellm normalizes the SLO call_type to "completion" for most chat calls; the async/text
# variants are kept defensively in case a version does not normalize.)
_CHAT_CALL_TYPES = {"completion", "acompletion", "text_completion", "atext_completion"}

# litellm custom_llm_provider values that don't normalize cleanly via model name
_LITELLM_PROVIDER_MAP = {"vertex_ai": "google", "vertex_ai_beta": "google"}


# --------------------------------------------------------------------------- #
# extraction helpers (operate on the dict shapes litellm produces)
# --------------------------------------------------------------------------- #

def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _response_dict(slo: Optional[Dict], response_obj: Any) -> Optional[Dict[str, Any]]:
    """the response as a dict: slo["response"] (already model_dump'd) or response_obj.model_dump()."""
    if isinstance(slo, dict):
        resp = _as_dict(slo.get("response"))
        if resp is not None:
            return resp
    if response_obj is not None and hasattr(response_obj, "model_dump"):
        try:
            dumped = response_obj.model_dump()
            return dumped if isinstance(dumped, dict) else None
        except Exception:
            return None
    return None


def _first_message(resp: Optional[Dict]) -> Dict[str, Any]:
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            return choices[0].get("message") or {}
    return {}


def _extract_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """tool calls as [{"name", "arguments"}]. tolerates raw-dict function + missing id/type.

    litellm's `function.arguments` is a JSON string; we keep it as-is (matches the OTel path).
    """
    calls: List[Dict[str, Any]] = []
    for tc in (message.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
    # legacy function_call (single)
    if not calls:
        fc = message.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            calls.append({"name": fc.get("name"), "arguments": fc.get("arguments")})
    return calls


def _extract_output_text(message: Dict[str, Any]) -> Optional[str]:
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):  # multimodal content blocks
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(p for p in parts if p)
        if joined:
            return joined
    return None


def _normalize_history(messages: Any) -> Any:
    """render the input message history into readable {role, content} (mirrors the OTel path).

    litellm passes raw OpenAI-format messages: an assistant tool-call turn has content=None plus
    a `tool_calls` field (which a content-only renderer shows as null), and tool results are
    role=tool (OpenAI) or tool_use/tool_result content blocks (Anthropic). We fold tool calls and
    tool results into the content string so the history isn't blank.
    """
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role", "user")
        content = m.get("content")

        # list content: multimodal text + anthropic tool_use / tool_result blocks
        if isinstance(content, list):
            texts, tool_uses, tool_results = [], [], []
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    texts.append(b.get("text", ""))
                elif t == "tool_use":
                    tool_uses.append({"name": b.get("name"), "arguments": b.get("input")})
                elif t == "tool_result":
                    r = b.get("content")
                    tool_results.append(r if isinstance(r, str) else json.dumps(r))
            content = "\n".join(t for t in texts if t)
            if not content and tool_uses:
                content = _format_tool_calls(tool_uses)
            if not content and tool_results:
                content = "Tool Result:\n" + "\n".join(tool_results)
                role = "tool"

        # OpenAI-format assistant tool call with empty content -> render the tool call(s)
        if content in (None, "") and m.get("tool_calls"):
            tcs = [{"name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments")}
                   for tc in m.get("tool_calls") if isinstance(tc, dict)]
            if tcs:
                content = _format_tool_calls(tcs)

        # OpenAI-format tool result -> prefix for consistency with the OTel path
        if role == "tool" and isinstance(content, str) and content and not content.startswith("Tool Result"):
            content = "Tool Result:\n" + content

        out.append({"role": role, "content": content if content is not None else ""})
    return out


def _extract_usage(resp: Optional[Dict], slo: Optional[Dict]) -> Dict[str, Any]:
    """flat tokens + cache(read/creation) + reasoning, defensively, from the dumped usage."""
    usage = _as_dict((resp or {}).get("usage")) or {}
    # metadata.usage_object mirrors response.usage and survives when response is absent
    if not usage and isinstance(slo, dict):
        usage = _as_dict((slo.get("metadata") or {}).get("usage_object")) or {}

    input_tokens = (slo or {}).get("prompt_tokens") if isinstance(slo, dict) else None
    output_tokens = (slo or {}).get("completion_tokens") if isinstance(slo, dict) else None
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")

    ptd = _as_dict(usage.get("prompt_tokens_details")) or {}
    ctd = _as_dict(usage.get("completion_tokens_details")) or {}
    cache_read = ptd.get("cached_tokens")
    cache_creation = ptd.get("cache_creation_tokens")
    # cache-creation is a Pydantic PrivateAttr excluded from response.usage model_dump, so
    # for Anthropic it only survives in the raw provider usage (metadata.usage_object).
    if cache_creation is None and isinstance(slo, dict):
        raw_usage = _as_dict((slo.get("metadata") or {}).get("usage_object")) or {}
        cache_creation = raw_usage.get("cache_creation_input_tokens")
        if cache_read is None:
            cache_read = raw_usage.get("cache_read_input_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_creation": cache_creation,
        "reasoning_tokens": ctd.get("reasoning_tokens"),
    }


def _resolve_cost(slo: Optional[Dict], kwargs: Dict[str, Any], model: str, usage: Dict[str, Any]) -> Optional[float]:
    """prefer litellm's precomputed response_cost; recompute locally only when it is None/0."""
    cost = None
    if isinstance(slo, dict):
        cost = slo.get("response_cost")
    if cost is None:
        cost = kwargs.get("response_cost")
    # only recompute when litellm gave us nothing; a real 0.0 (free/cached model) is kept.
    if cost is None:
        try:
            normalized = model.split("/", 1)[1] if "/" in (model or "") else model
            recomputed = calculate_cost(normalized, {
                "prompt_tokens": usage.get("input_tokens") or 0,
                "completion_tokens": usage.get("output_tokens") or 0,
                "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            })
            if recomputed:
                return recomputed
        except Exception:
            pass
    return cost


def _resolve_provider(slo: Optional[Dict], kwargs: Dict[str, Any], model: str) -> str:
    """provider from litellm's custom_llm_provider (normalized), falling back to model name."""
    cp = kwargs.get("custom_llm_provider")
    if not cp and isinstance(slo, dict):
        cp = slo.get("custom_llm_provider") or (slo.get("metadata") or {}).get("custom_llm_provider")
    if cp:
        cp = _LITELLM_PROVIDER_MAP.get(cp, cp)
    # detect_provider normalizes (e.g. bare "azure" -> resolve via model -> "openai")
    return detect_provider(model=model, attributes={"gen_ai.provider.name": cp} if cp else None)


# --------------------------------------------------------------------------- #
# callback
# --------------------------------------------------------------------------- #

class LucidicLiteLLMCallback(CustomLogger):
    """LiteLLM CustomLogger that emits Lucidic llm_generation / error events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_events: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._missing_slo_warnings = 0

    # -- pre-call: capture session/parent in the CALLER's context. litellm may run the
    #    success/failure callback in a worker thread where get_session_id() can't see the
    #    session (thread-local has no fallback), so we bind it here and carry it forward. --
    def log_pre_api_call(self, model, messages, kwargs):
        try:
            call_id = (kwargs or {}).get("litellm_call_id")
            if not call_id:
                return  # no stable key to correlate success/failure; avoid an un-poppable leak
            with self._lock:
                self._active_events[call_id] = {
                    "model": model,
                    "messages": messages,
                    "session_id": get_session_id(),
                    "parent_id": self._parent_id(),
                }
        except Exception as e:
            logger.error(f"[LiteLLM] pre_api_call error: {e}")

    async def async_log_pre_api_call(self, model, messages, kwargs):
        self.log_pre_api_call(model, messages, kwargs)

    # -- success --
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # emit_event is non-blocking (fire-and-forget background thread), safe in the event loop.
        self._handle_success(kwargs, response_obj, start_time, end_time)

    # -- failure --
    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    # -- streaming: litellm assembles chunks itself and fires log_success_event ONCE at
    #    stream end with the complete response (in standard_logging_object), so there is
    #    nothing to do per-chunk here. --
    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    # -- core --
    def _pop_active(self, kwargs) -> Dict[str, Any]:
        call_id = (kwargs or {}).get("litellm_call_id")
        if not call_id:
            return {}
        with self._lock:
            return self._active_events.pop(call_id, {})

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        # pop pre-call state first so it never leaks, even when there is no session
        pre = self._pop_active(kwargs)
        try:
            # prefer the session captured at pre-call time (caller context); the callback
            # may run in a worker thread where get_session_id() returns None.
            session_id = pre.get("session_id") or get_session_id()
            if not session_id:
                return

            slo = _as_dict(kwargs.get("standard_logging_object"))
            warn_missing = False
            if slo is None:
                with self._lock:
                    if self._missing_slo_warnings < 3:
                        self._missing_slo_warnings += 1
                        warn_missing = True
            if warn_missing:
                logger.warning(
                    "[LiteLLM] standard_logging_object missing; falling back to response parsing. "
                    "Upgrade litellm for full fidelity (tool calls / cost / usage detail)."
                )

            call_type = (slo or {}).get("call_type") or kwargs.get("call_type") or "completion"
            model = (slo or {}).get("model") or kwargs.get("model") or pre.get("model") or "unknown"
            messages = _normalize_history(
                (slo or {}).get("messages") or kwargs.get("messages") or pre.get("messages") or [])
            provider = _resolve_provider(slo, kwargs, model)

            resp = _response_dict(slo, response_obj)
            usage = _extract_usage(resp, slo)
            cost = _resolve_cost(slo, kwargs, model, usage)

            output_text = None
            tool_calls: List[Dict[str, Any]] = []
            thinking = None
            if call_type in _CHAT_CALL_TYPES:
                message = _first_message(resp)
                output_text = _extract_output_text(message)
                tool_calls = _extract_tool_calls(message)
                thinking = message.get("reasoning_content") or message.get("thinking_blocks")
            else:
                # non-chat (embedding/image/rerank/transcription): no choices/message
                output_text = f"<{call_type} response>"

            # fold tool calls into output when there is no assistant text (mirrors OTel path)
            if tool_calls and not output_text:
                output_text = _format_tool_calls(tool_calls)

            if output_text is None and not tool_calls:
                logger.warning(f"[LiteLLM] no output/tool_calls extracted for {model} ({call_type})")
                output_text = ""

            cache = {}
            if usage.get("cache_read"):
                cache["read_input_tokens"] = usage["cache_read"]
            if usage.get("cache_creation"):
                cache["creation_input_tokens"] = usage["cache_creation"]

            metadata = {"litellm": True, "call_type": call_type}
            if usage.get("reasoning_tokens"):
                metadata["reasoning_tokens"] = usage["reasoning_tokens"]

            occurred_at = start_time.isoformat() if isinstance(start_time, datetime) else None
            duration = (end_time - start_time).total_seconds() if (
                isinstance(start_time, datetime) and isinstance(end_time, datetime)) else None

            event_kwargs: Dict[str, Any] = {
                "type": "llm_generation",
                "session_id": session_id,
                "provider": provider,
                "model": model,
                "messages": messages,
                "output": output_text,
                "input_tokens": usage.get("input_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or 0,
                "cost": cost,
                "parent_event_id": pre.get("parent_id") or self._parent_id(),
                "occurred_at": occurred_at,
                "duration": duration,
                "metadata": metadata,
            }
            if tool_calls:
                event_kwargs["tool_calls"] = [
                    {"name": tc.get("name"), "arguments": tc.get("arguments")} for tc in tool_calls
                ]
            if thinking:
                event_kwargs["thinking"] = thinking
            if cache:
                event_kwargs["cache"] = cache

            emit_event(**event_kwargs)
            if DEBUG:
                logger.info(f"[LiteLLM] emitted llm_generation for {model} "
                            f"(tool_calls={len(tool_calls)}, cost={cost})")
        except Exception as e:
            logger.error(f"[LiteLLM] success handler error: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        pre = self._pop_active(kwargs)
        try:
            session_id = pre.get("session_id") or get_session_id()
            if not session_id:
                return
            slo = _as_dict(kwargs.get("standard_logging_object"))
            model = (slo or {}).get("model") or kwargs.get("model") or "unknown"
            provider = _resolve_provider(slo, kwargs, model)
            error_msg = (slo or {}).get("error_str") or (str(response_obj) if response_obj else "Unknown error")
            occurred_at = start_time.isoformat() if isinstance(start_time, datetime) else None
            duration = (end_time - start_time).total_seconds() if (
                isinstance(start_time, datetime) and isinstance(end_time, datetime)) else None
            emit_event(
                type="error_traceback",
                session_id=session_id,
                error=error_msg,
                traceback="",
                parent_event_id=pre.get("parent_id") or self._parent_id(),
                occurred_at=occurred_at,
                duration=duration,
                metadata={"provider": provider, "litellm": True, "model": model},
            )
        except Exception as e:
            logger.error(f"[LiteLLM] failure handler error: {e}")

    @staticmethod
    def _parent_id() -> Optional[str]:
        try:
            return current_parent_event_id.get(None)
        except Exception:
            return None


def setup_litellm_callback():
    """register LucidicLiteLLMCallback on litellm.callbacks (idempotent)."""
    try:
        import litellm
    except ImportError:
        logger.info("[LiteLLM] litellm not installed, skipping callback setup")
        return

    if not getattr(litellm, "callbacks", None):
        litellm.callbacks = []

    for existing in litellm.callbacks:
        if isinstance(existing, LucidicLiteLLMCallback):
            if DEBUG:
                logger.debug("[LiteLLM] callback already registered")
            return

    try:
        litellm.callbacks.append(LucidicLiteLLMCallback())
        logger.info("[LiteLLM] registered Lucidic callback for event tracking")
    except Exception as e:
        logger.error(f"[LiteLLM] failed to register callback: {e}")
