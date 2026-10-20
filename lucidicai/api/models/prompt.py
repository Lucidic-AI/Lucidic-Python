"""Typed response models for client.prompts reads (LUC-909).

Distinct from the SDK's grandfathered ``Prompt`` dataclass
(``api/resources/prompt.py``), which is a *rendered* prompt (content + metadata,
returned by ``prompts.get()``). These model the prompt catalog + version history.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import APIModel


@dataclass
class PromptInfo(APIModel):
    """A prompt (unique name per agent) as returned by ``client.prompts.list()``.
    ``preview`` is a short excerpt of its latest content."""

    prompt_id: str
    name: Optional[str] = None
    icon: Optional[str] = None
    preview: Optional[str] = None


@dataclass
class PromptVersion(APIModel):
    """One immutable version of a prompt — its content, the labels pointing at
    it, and metadata. Returned by ``client.prompts.versions(name)``."""

    promptversion_id: str
    prompt_version_number: Optional[int] = None
    prompt_content: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    labels: List[str] = field(default_factory=list)
