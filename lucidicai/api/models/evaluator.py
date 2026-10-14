"""Typed response model for client.evaluators reads (LUC-910).

The per-result rows (``evaluators.evals()`` distribution and
``evaluators.result()``) reuse the ``EvalResult`` model from ``models.session``
— both endpoints serialize with the backend's ``EvaluatorResultSerializer``.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import APIModel


@dataclass
class Evaluator(APIModel):
    """An evaluator (LLM / CODE / HUMAN / RUNTIME) — the definition of a scoring
    criterion. ``list()`` returns the light shape; ``get()`` additionally
    populates ``rubric_json`` + ``config`` (default ``None`` in a list result).
    """

    evaluator_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    type: Optional[str] = None
    result_type: Optional[str] = None
    is_default: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    criteria: Optional[Any] = None

    # detail-only (get)
    rubric_json: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None
