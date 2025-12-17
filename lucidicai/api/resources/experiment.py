"""Experiment resource API operations."""
import logging
from typing import Any, Dict, List, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


class ExperimentResource:
    """Handle experiment-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize experiment resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for experiments
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production

    def create(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        LLM_boolean_evaluators: Optional[List[str]] = None,
        LLM_numeric_evaluators: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create a new experiment.

        Args:
            experiment_name: Name of the experiment.
            description: Optional description.
            tags: Optional tags for filtering.
            LLM_boolean_evaluators: Boolean evaluator names.
            LLM_numeric_evaluators: Numeric evaluator names.

        Returns:
            The experiment ID if created successfully, None otherwise.
        """
        evaluator_names = []
        if LLM_boolean_evaluators:
            evaluator_names.extend(LLM_boolean_evaluators)
        if LLM_numeric_evaluators:
            evaluator_names.extend(LLM_numeric_evaluators)

        try:
            response = self.http.post(
                "createexperiment",
                {
                    "agent_id": self._agent_id,
                    "experiment_name": experiment_name,
                    "description": description or "",
                    "tags": tags or [],
                    "evaluator_names": evaluator_names,
                },
            )
            return response.get("experiment_id")
        except Exception as e:
            if self._production:
                logger.error(f"[ExperimentResource] Failed to create experiment: {e}")
                return None
            raise

    async def acreate(
        self,
        experiment_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        LLM_boolean_evaluators: Optional[List[str]] = None,
        LLM_numeric_evaluators: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Create a new experiment (asynchronous).

        See create() for full documentation.
        """
        evaluator_names = []
        if LLM_boolean_evaluators:
            evaluator_names.extend(LLM_boolean_evaluators)
        if LLM_numeric_evaluators:
            evaluator_names.extend(LLM_numeric_evaluators)

        try:
            response = await self.http.apost(
                "createexperiment",
                {
                    "agent_id": self._agent_id,
                    "experiment_name": experiment_name,
                    "description": description or "",
                    "tags": tags or [],
                    "evaluator_names": evaluator_names,
                },
            )
            return response.get("experiment_id")
        except Exception as e:
            if self._production:
                logger.error(f"[ExperimentResource] Failed to create experiment: {e}")
                return None
            raise
