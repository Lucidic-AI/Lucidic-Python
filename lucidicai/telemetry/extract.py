"""Extraction utilities matching TypeScript SDK for span attribute processing."""
import json
from typing import List, Dict, Any, Optional


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


def extract_images(attrs: Dict[str, Any]) -> List[str]:
    """Extract images from span attributes - matches TypeScript logic.
    
    Looks for images in:
    - gen_ai.prompt.{i}.content arrays with image_url items
    - Direct image attributes
    """
    images = []
    
    # Check indexed prompt content for images (gen_ai.prompt.{i}.content)
    for i in range(50):
        key = f"gen_ai.prompt.{i}.content"
        if key in attrs:
            content = attrs[key]
            
            # Parse if JSON string
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except:
                    continue
            
            # Extract images from content array
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                                if url.startswith("data:image"):
                                    images.append(url)
                        elif item.get("type") == "image":
                            # Anthropic format
                            source = item.get("source", {})
                            if isinstance(source, dict):
                                data = source.get("data", "")
                                media_type = source.get("media_type", "image/jpeg")
                                if data:
                                    images.append(f"data:{media_type};base64,{data}")
    
    # Also check direct gen_ai.prompt list
    prompt_list = attrs.get("gen_ai.prompt")
    if isinstance(prompt_list, list):
        for msg in prompt_list:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                                if url.startswith("data:image"):
                                    images.append(url)
    
    return images


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
            except:
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
        except:
            pass
    
    return None


def extract_completions(span, attrs: Dict[str, Any]) -> Optional[str]:
    """Extract completion/response text from span attributes."""
    completions = []
    
    # Check indexed format (gen_ai.completion.{i}.content)
    for i in range(50):
        key = f"gen_ai.completion.{i}.content"
        if key not in attrs:
            break
        content = attrs[key]
        if isinstance(content, str):
            completions.append(content)
        else:
            try:
                completions.append(json.dumps(content))
            except:
                completions.append(str(content))
    
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