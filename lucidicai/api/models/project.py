"""Typed response model for client.projects."""
from dataclasses import dataclass
from typing import Optional

from .base import APIModel


@dataclass
class Project(APIModel):
    """A project — an org-level grouping that agents can be filed under
    (``Agent.project``). Org-scoped: a bound key still sees all of them."""

    project_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
