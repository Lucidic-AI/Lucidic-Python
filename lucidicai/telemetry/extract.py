"""extraction utilities for LLM span attributes across OTel GenAI semconv shapes.

three shapes are read, newest-first and richness-gated (LUC-667):
  1. new JSON attributes  - gen_ai.input.messages / gen_ai.output.messages
                            (+ gen_ai.system_instructions); parts carry text and
                            tool_call/tool_call_response items. (OTel semconv >= v1.37,
                            openllmetry >= 0.55)
  2. new span events      - gen_ai.{system,user,assistant,tool}.message / gen_ai.choice
  3. legacy flat-indexed  - gen_ai.prompt.{i}.* / gen_ai.completion.{i}.*
                            (openllmetry <= 0.53.x)

"newest-first, richness-gated" means a newer shape is used only if it yields a
non-empty result; otherwise we fall through. the shapes are mutually exclusive on a
given span in practice, so this never double-counts.
"""
import json
from typing import List, Dict, Any, Optional
from ..utils.logger import debug, info, warning, error, verbose, truncate_id


# new-shape attribute keys
NEW_INPUT_KEY = "gen_ai.input.messages"
NEW_OUTPUT_KEY = "gen_ai.output.messages"
NEW_SYSTEM_KEY = "gen_ai.system_instructions"

# new-shape event names
INPUT_EVENT_NAMES = {
    "gen_ai.system.message",
    "gen_ai.user.message",
    "gen_ai.assistant.message",
    "gen_ai.tool.message",
}
OUTPUT_EVENT_NAME = "gen_ai.choice"


def detect_is_llm_span(span) -> bool:
    """check if span is LLM-related - matches TypeScript logic."""
    name = (span.name or "").lower()
    patterns = ['openai', 'anthropic', 'chat', 'completion', 'embedding', 'llm',
                'gemini', 'claude', 'bedrock', 'vertex', 'cohere', 'groq']

    if any(p in name for p in patterns):
        return True

    if hasattr(span, 'attributes') and span.attributes:
        for key in span.attributes:
            if isinstance(key, str) and (key.startswith('gen_ai.') or key.startswith('llm.')):
                return True

    return False


# --------------------------------------------------------------------------- #
# new-shape helpers (gen_ai.input.messages / gen_ai.output.messages)
# --------------------------------------------------------------------------- #

def _loads(raw: Any) -> Any:
    """parse a value that may already be a list/dict or a JSON string."""
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    return None


def _flatten_text_parts(parts: Any) -> str:
    """join the text content of a new-shape message's parts."""
    out = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") == "text":
            # the new semconv part carries text under "content"
            text = p.get("content", p.get("text", ""))
            if text:
                out.append(str(text))
    return " ".join(out)


def _tool_calls_from_parts(parts: Any) -> List[Dict[str, Any]]:
    """extract tool_call parts from a new-shape message."""
    calls = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") == "tool_call":
            calls.append({
                "name": p.get("name"),
                "arguments": p.get("arguments"),
            })
    return calls


def _tool_responses_from_parts(parts: Any) -> str:
    """render tool_call_response parts (the tool RESULTS fed back to the model).

    the new semconv represents a tool result as a part {type: "tool_call_response",
    id, response}. without this, role=tool / tool_result history messages flatten to
    empty content.
    """
    out = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") == "tool_call_response":
            resp = p.get("response", p.get("content", ""))
            if not isinstance(resp, str):
                try:
                    resp = json.dumps(resp)
                except (ValueError, TypeError):
                    resp = str(resp)
            if resp:
                out.append(resp)
    if not out:
        return ""
    header = "Tool Result:" if len(out) == 1 else "Tool Results:"
    return header + "\n" + "\n".join(out)


def _messages_from_new_attr(attrs: Dict[str, Any], key: str) -> Optional[List[Dict]]:
    """build [{role, content}] from a new-shape gen_ai.*.messages JSON attribute.

    richness gate: returns None unless at least one message has non-empty text
    or a tool_call part (so a present-but-empty attribute falls through).
    """
    parsed = _loads(attrs.get(key))
    if not isinstance(parsed, list):
        return None

    messages = []
    had_content = False
    for m in parsed:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        parts = m.get("parts")
        content = _flatten_text_parts(parts)
        tool_calls = _tool_calls_from_parts(parts)
        if not content and tool_calls:
            content = _format_tool_calls(tool_calls)
        if not content:
            # tool RESULT messages (role=tool) carry a tool_call_response part, not text
            content = _tool_responses_from_parts(parts)
        if content:
            had_content = True
        messages.append({"role": role, "content": content})

    return messages if (messages and had_content) else None


def _messages_from_events(span, names: set) -> Optional[List[Dict]]:
    """build [{role, content}] from new-shape span events."""
    events = getattr(span, "events", None) or []
    messages = []
    had_content = False
    for ev in events:
        if ev.name not in names:
            continue
        role = ev.name.removeprefix("gen_ai.").removesuffix(".message")
        body_raw = (ev.attributes or {}).get("body") or (ev.attributes or {}).get("content")
        body = _loads(body_raw)
        if isinstance(body, dict):
            content = _flatten_text_parts(body.get("parts")) or str(body.get("content", ""))
        elif isinstance(body, list):
            content = _flatten_text_parts(body)
        else:
            content = "" if body_raw is None else str(body_raw)
        if content:
            had_content = True
        messages.append({"role": role, "content": content})
    return messages if (messages and had_content) else None


def _format_tool_calls(tool_calls: List[Dict]) -> str:
    """render tool calls as a readable string (matches legacy formatting)."""
    header = 'Tool Calls:' if len(tool_calls) > 1 else 'Tool Call:'
    body = ""
    for k, tc in enumerate(tool_calls):
        body += f'\n{k + 1}) {json.dumps(tc, indent=4)}'
    return header + body


# --------------------------------------------------------------------------- #
# legacy flat-indexed helpers
# --------------------------------------------------------------------------- #

def _legacy_prompts(attrs: Dict[str, Any]) -> Optional[List[Dict]]:
    """extract prompts from the legacy flat-indexed gen_ai.prompt.{i}.* shape."""
    messages = []

    for i in range(50):
        role_key = f"gen_ai.prompt.{i}.role"
        content_key = f"gen_ai.prompt.{i}.content"

        if role_key not in attrs and content_key not in attrs:
            break

        role = attrs.get(role_key, "user")
        content = attrs.get(content_key, "")

        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                pass

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            if text_parts:
                content = " ".join(text_parts)

        # no content => likely a tool-call-only history entry
        if not content:
            j = 0
            tool_calls = []
            while True:
                tool_key_name = f"gen_ai.prompt.{i}.tool_calls.{j}.name"
                tool_key_arguments = f"gen_ai.prompt.{i}.tool_calls.{j}.arguments"
                if tool_key_name not in attrs:
                    break
                tool_calls.append({"name": attrs[tool_key_name], "arguments": attrs.get(tool_key_arguments)})
                j += 1
            if tool_calls:
                content = _format_tool_calls(tool_calls)

        messages.append({"role": role, "content": content})

    if messages:
        return messages

    # direct message list fallback
    prompt_list = attrs.get("gen_ai.prompt") or attrs.get("gen_ai.messages")
    if isinstance(prompt_list, list):
        return prompt_list

    # Vercel AI SDK format
    ai_prompt = attrs.get("ai.prompt.messages")
    if isinstance(ai_prompt, str):
        parsed = _loads(ai_prompt)
        if isinstance(parsed, list):
            return parsed

    return None


# --------------------------------------------------------------------------- #
# public extractors (newest-first, richness-gated fall-through)
# --------------------------------------------------------------------------- #

def extract_prompts(span, attrs: Dict[str, Any]) -> Optional[List[Dict]]:
    """extract prompt messages as [{"role": ..., "content": ...}].

    tries new JSON attrs -> new span events -> legacy flat, returning the first
    non-empty result.
    """
    # 1. new JSON attribute (optionally prefixed with system_instructions)
    new_msgs = _messages_from_new_attr(attrs, NEW_INPUT_KEY)
    if new_msgs is not None:
        sys_instr = _system_instructions(attrs)
        return (sys_instr + new_msgs) if sys_instr else new_msgs

    # 2. new span events
    ev_msgs = _messages_from_events(span, INPUT_EVENT_NAMES)
    if ev_msgs is not None:
        return ev_msgs

    # 3. legacy flat-indexed
    return _legacy_prompts(attrs)


def _system_instructions(attrs: Dict[str, Any]) -> Optional[List[Dict]]:
    """read gen_ai.system_instructions (new shape carries system prompt separately)."""
    raw = attrs.get(NEW_SYSTEM_KEY)
    if raw is None:
        return None
    parsed = _loads(raw)
    if isinstance(parsed, list):
        content = _flatten_text_parts(parsed)
    elif isinstance(parsed, dict):
        content = _flatten_text_parts(parsed.get("parts"))
    else:
        content = str(raw)
    return [{"role": "system", "content": content}] if content else None


def extract_completions(span, attrs: Dict[str, Any]) -> Optional[str]:
    """extract completion/response text. new JSON output -> events -> legacy -> error."""
    # 1. new JSON output messages (text only; tool calls handled by extract_tool_calls)
    parsed = _loads(attrs.get(NEW_OUTPUT_KEY))
    if isinstance(parsed, list):
        texts = []
        for m in parsed:
            if isinstance(m, dict):
                t = _flatten_text_parts(m.get("parts"))
                if t:
                    texts.append(t)
        if texts:
            return "\n".join(texts)

    # 2. new span events (gen_ai.choice)
    for ev in (getattr(span, "events", None) or []):
        if ev.name == OUTPUT_EVENT_NAME:
            body = _loads((ev.attributes or {}).get("body"))
            if isinstance(body, dict):
                t = _flatten_text_parts((body.get("message") or {}).get("parts")) or str(body.get("content", ""))
                if t:
                    return t

    # 3. legacy flat-indexed
    completions = []
    i = 0
    while True:
        key = f"gen_ai.completion.{i}.content"
        if key not in attrs:
            break
        content = attrs[key]
        if isinstance(content, str):
            completions.append(content)
        else:
            try:
                completions.append(json.dumps(content))
            except (ValueError, TypeError):
                completions.append(str(content))
        i += 1
    if completions:
        return "\n".join(completions)

    completion = attrs.get("gen_ai.completion") or attrs.get("llm.completions")
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list) and completion:
        return "\n".join(str(c) for c in completion)

    ai_completion = attrs.get("ai.response.text")
    if isinstance(ai_completion, str):
        return ai_completion

    # error status fallback
    if hasattr(span, 'status'):
        from opentelemetry.trace import StatusCode
        if span.status.status_code == StatusCode.ERROR:
            return f"Error: {span.status.description or 'Unknown error'}"

    return None


def extract_tool_calls(span, attrs: Dict[str, Any]) -> Optional[str]:
    """extract tool calls as a readable string. detects by PRESENCE, not finish_reason.

    tries new JSON output tool_call parts -> legacy flat tool_calls.
    """
    debug("[Telemetry] Extracting tool calls from span")

    # 1. new JSON output messages with tool_call parts
    parsed = _loads(attrs.get(NEW_OUTPUT_KEY))
    if isinstance(parsed, list):
        tool_calls = []
        for m in parsed:
            if isinstance(m, dict):
                tool_calls.extend(_tool_calls_from_parts(m.get("parts")))
        if tool_calls:
            return "\n".join(json.dumps(tc, indent=4) for tc in tool_calls)

    # 2. legacy flat-indexed (presence-based; no finish_reason gate - LUC-667)
    tool_calls = []
    i = 0
    while True:
        key_name = f"gen_ai.completion.0.tool_calls.{i}.name"
        key_arguments = f"gen_ai.completion.0.tool_calls.{i}.arguments"
        if key_name not in attrs:
            break
        name = attrs[key_name]
        arguments = attrs.get(key_arguments)
        debug(f"[Telemetry] Extracted tool call {name} with arguments: {arguments}")
        tool_calls.append({"name": name, "arguments": arguments})
        i += 1

    if tool_calls:
        return "\n".join(json.dumps(tc, indent=4) for tc in tool_calls)

    debug(f"[Telemetry] No tool calls found for span {span.name}")
    return None


def extract_model(attrs: Dict[str, Any]) -> Optional[str]:
    """extract model name from span attributes."""
    return (
        attrs.get("gen_ai.response.model") or
        attrs.get("gen_ai.request.model") or
        attrs.get("llm.response.model") or
        attrs.get("llm.request.model") or
        attrs.get("ai.model.id") or
        attrs.get("ai.model.name")
    )
