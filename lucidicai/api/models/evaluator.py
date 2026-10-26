"""Typed response model for client.evaluators reads.

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
    ``datasetitem_id`` is set only when read as a case's evaluator
    (``client.datasets.case_evaluator``): the case it was read through.
    ``dataset_id`` is set only when read as a Dataset's fallback evaluator
    (``client.datasets.fallback_evaluator`` / ``set_fallback_evaluator``), and
    ``created`` only by ``set_fallback_evaluator``: True when a spec made a new
    evaluator rather than reusing one.
    """

    evaluator_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    type: Optional[str] = None
    result_type: Optional[str] = None
    is_default: bool = False
    is_primary: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    criteria: Optional[Any] = None

    # detail-only (get)
    rubric_json: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None

    # case-evaluator reads only (datasets.case_evaluator)
    datasetitem_id: Optional[str] = None

    # fallback-evaluator reads / writes only (datasets.fallback_evaluator,
    # set_fallback_evaluator); created is set by the write alone
    dataset_id: Optional[str] = None
    created: Optional[bool] = None
