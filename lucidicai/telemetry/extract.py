"""Extraction utilities matching TypeScript SDK for span attribute processing."""
import json
from typing import List, Dict, Any, Optional
from ..utils.logger import debug, info, warning, error, verbose, truncate_id


def detect_is_llm_span(span) -> bool:
    """Check if span is LLM-related - matches TypeScript logic."""
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


def extract_prompts(attrs: Dict[str, Any]) -> Optional[List[Dict]]:
    """Extract prompts as message list from span attributes.
    
    Returns messages in format: [{"role": "user", "content": "..."}]
    """
    messages = []
    
    # Check indexed format (gen_ai.prompt.{i}.role/content)
    for i in range(50):
        role_key = f"gen_ai.prompt.{i}.role"
        content_key = f"gen_ai.prompt.{i}.content"
        
        if role_key not in attrs and content_key not in attrs:
            break
            
        role = attrs.get(role_key, "user")
        content = attrs.get(content_key, "")
        
        # Parse content if it's JSON
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                content = parsed
            except (ValueError, TypeError):
                pass
        
        # Format content
        if isinstance(content, list):
            # Content array format (with text/image items)
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            if text_parts:
                content = " ".join(text_parts)

        # if we have no content here then that means we have a tool call
        # NOTE: for now, I am assumign that tools call history only shows up if there is no content
        # based on my testing of otel spans in different cases. Should be revisited if this is not the case.
        if not content:
            # look for tool calls in the attributes
            j = 0
            tool_calls = []
            while True:
                tool_key_name = f"gen_ai.prompt.{i}.tool_calls.{j}.name"
                tool_key_arguments = f"gen_ai.prompt.{i}.tool_calls.{j}.arguments"
                if tool_key_name not in attrs:
                    break
                name = attrs[tool_key_name]
                arguments = attrs[tool_key_arguments]
                tool_calls.append({"name": name, "arguments": arguments})
                j += 1

            # for now, just make content as "Tool Calls:\n 1) <tool call 1> \n 2) <tool call 2> \n ..."
            if tool_calls:
                content = 'Tool Calls:' if len(tool_calls) > 1 else 'Tool Call:'
                for k, tool_call in enumerate(tool_calls):
                    content += f'\n{k + 1}) {json.dumps(tool_call, indent=4)}'
        
        messages.append({"role": role, "content": content})
    
    if messages:
        return messages
    
    # Check for direct message list
    prompt_list = attrs.get("gen_ai.prompt") or attrs.get("gen_ai.messages")
    if isinstance(prompt_list, list):
        return prompt_list
    
    # Check AI SDK format
    ai_prompt = attrs.get("ai.prompt.messages")
    if isinstance(ai_prompt, str):
        try:
            parsed = json.loads(ai_prompt)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, TypeError):
            pass
    
    return None


def extract_completions(span, attrs: Dict[str, Any]) -> Optional[str]:
    """Extract completion/response text from span attributes."""
    completions = []
    
    # Check indexed format (gen_ai.completion.{i}.content)
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
    
    # Check direct completion attribute
    completion = attrs.get("gen_ai.completion") or attrs.get("llm.completions")
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list) and completion:
        return "\n".join(str(c) for c in completion)
    
    # Check AI SDK format
    ai_completion = attrs.get("ai.response.text")
    if isinstance(ai_completion, str):
        return ai_completion
    
    # Check for error status
    if hasattr(span, 'status'):
        from opentelemetry.trace import StatusCode
        if span.status.status_code == StatusCode.ERROR:
            return f"Error: {span.status.description or 'Unknown error'}"
    
    return None


def extract_tool_calls(span, attrs: Dict[str, Any]) -> Optional[List[Dict]]:
    """Extract tool calls from span attributes."""

    debug(f"[Telemetry] Extracting tool calls from span")

    # check if this is a tool call span
    if not attrs.get("gen_ai.completion.0.finish_reason") == "tool_calls":
        debug(f"[Telemetry] Not a tool call span {span.name}")
        return None

    tool_calls = []
    i = 0
    while True:
        key_name = f"gen_ai.completion.0.tool_calls.{i}.name"
        key_arguments = f"gen_ai.completion.0.tool_calls.{i}.arguments"
        if key_name not in attrs:
            break
        name = attrs[key_name]
        arguments = attrs[key_arguments]
        debug(f"[Telemetry] Extracted tool call {name} with arguments: {arguments}")
        tool_calls.append({"name": name, "arguments": arguments})
        i += 1

    if tool_calls:
        # prettify the tool calls and return as a string
        tool_calls_str = [json.dumps(tool_call, indent=4) for tool_call in tool_calls]
        return "\n".join(tool_calls_str)

    return None

def extract_model(attrs: Dict[str, Any]) -> Optional[str]:
    """Extract model name from span attributes."""
    return (
        attrs.get("gen_ai.response.model") or
        attrs.get("gen_ai.request.model") or
        attrs.get("llm.response.model") or
        attrs.get("llm.request.model") or
        attrs.get("ai.model.id") or
        attrs.get("ai.model.name")
    )