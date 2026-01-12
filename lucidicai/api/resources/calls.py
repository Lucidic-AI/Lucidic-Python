"""Calls resource API operations for persona generation."""
import logging
from typing import Any, Dict, List, Optional

from ..client import HttpClient

logger = logging.getLogger("Lucidic")


class CallsResource:
    """Handle call-related API operations for persona generation.

    This resource treats a dataset as a collection of call transcripts
    and provides methods to generate personas from them.

    Example:
        # generate personas for multiple calls
        result = client.calls.generate_personas(
            dataset_id="...",
            call_ids=["call-001", "call-002"],
            create_retell_agents=False
        )

        # generate custom persona with overrides
        result = client.calls.generate_custom_persona(
            dataset_id="...",
            base_call_id="call-001",
            confidence="anxious",
            mood="frustrated"
        )

        # generate fully synthetic persona
        result = client.calls.generate_synthetic_persona(
            dataset_id="...",
            verbosity="terse",
            mood="frustrated"
        )
    """

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize calls resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production

    # ==================== Generate Personas ====================

    def generate_personas(
        self,
        dataset_id: str,
        call_ids: List[str],
        create_retell_agents: bool = False,
    ) -> Dict[str, Any]:
        """Generate personas for specified calls with no customization.

        Extracts caller behavior exactly as it appears in the transcripts.

        Args:
            dataset_id: UUID of the dataset containing the transcripts
            call_ids: List of call IDs (DatasetItem.name values) - must be non-empty
            create_retell_agents: If True, creates Retell LLM and Agent for each item

        Returns:
            Dictionary with dataset_id and results list containing persona data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "call_ids": call_ids,
                "create_retell_agents": create_retell_agents,
            }
            return self.http.post("sdk/generate-personas", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate personas: {e}")
                return {"dataset_id": dataset_id, "results": []}
            raise

    async def agenerate_personas(
        self,
        dataset_id: str,
        call_ids: List[str],
        create_retell_agents: bool = False,
    ) -> Dict[str, Any]:
        """Generate personas for specified calls (asynchronous).

        Args:
            dataset_id: UUID of the dataset containing the transcripts
            call_ids: List of call IDs (DatasetItem.name values) - must be non-empty
            create_retell_agents: If True, creates Retell LLM and Agent for each item

        Returns:
            Dictionary with dataset_id and results list containing persona data
        """
        try:
            data = {
                "dataset_id": dataset_id,
                "call_ids": call_ids,
                "create_retell_agents": create_retell_agents,
            }
            return await self.http.apost("sdk/generate-personas", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate personas: {e}")
                return {"dataset_id": dataset_id, "results": []}
            raise

    # ==================== Generate Custom Persona ====================

    def generate_custom_persona(
        self,
        dataset_id: str,
        base_call_id: str,
        # communication style
        verbosity: Optional[str] = None,
        formality: Optional[str] = None,
        language: Optional[str] = None,
        # behavioral traits
        confidence: Optional[str] = None,
        engagement: Optional[str] = None,
        info_sharing: Optional[str] = None,
        decision_style: Optional[str] = None,
        # emotional
        mood: Optional[str] = None,
        # context
        urgency: Optional[str] = None,
        budget: Optional[str] = None,
        # lists
        pain_points: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        # free-form
        custom_instructions: Optional[str] = None,
        errors: Optional[str] = None,
        # behavioral
        patience: Optional[str] = None,
        repeats_questions: Optional[str] = None,
        connection_checks: Optional[str] = None,
        interrupts: Optional[str] = None,
        # retell/row creation
        create_retell_agent: bool = False,
        create_row: bool = False,
    ) -> Dict[str, Any]:
        """Generate a persona from transcript with customization overrides.

        Args:
            dataset_id: UUID of the dataset
            base_call_id: Call ID to use as base transcript
            verbosity: Response length (terse, moderate, verbose)
            formality: Communication style (formal, casual, mixed)
            language: Language preference (hindi-dominant, english-dominant, mixed)
            confidence: Confidence level (confident, uncertain, anxious)
            engagement: Engagement style (cooperative, resistant, distracted)
            info_sharing: Info sharing style (forthcoming, guarded, needs-prompting)
            decision_style: Decision making (quick, deliberate, indecisive)
            mood: Baseline mood (neutral, friendly, frustrated, confused)
            urgency: Urgency level (low, medium, high)
            budget: Budget constraint (free-form string)
            pain_points: List of pain points
            interests: List of interests
            constraints: List of constraints
            custom_instructions: Free-form behavioral instructions
            errors: Error injection types (stutter, bad_connection, stt, language, all)
            patience: Patience level (patient, moderate, impatient)
            repeats_questions: Question repetition (never, sometimes, frequently)
            connection_checks: Connection check frequency (none, occasional, frequent)
            interrupts: Interrupting behavior (never, sometimes, frequently)
            create_retell_agent: If True, creates Retell LLM and agent
            create_row: If True, creates a new DatasetItem for this persona

        Returns:
            Dictionary with metadata and item_id (null if create_row=False)
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "base_call_id": base_call_id,
                "create_retell_agent": create_retell_agent,
                "create_row": create_row,
            }
            # add optional fields only if provided
            if verbosity is not None:
                data["verbosity"] = verbosity
            if formality is not None:
                data["formality"] = formality
            if language is not None:
                data["language"] = language
            if confidence is not None:
                data["confidence"] = confidence
            if engagement is not None:
                data["engagement"] = engagement
            if info_sharing is not None:
                data["info_sharing"] = info_sharing
            if decision_style is not None:
                data["decision_style"] = decision_style
            if mood is not None:
                data["mood"] = mood
            if urgency is not None:
                data["urgency"] = urgency
            if budget is not None:
                data["budget"] = budget
            if pain_points is not None:
                data["pain_points"] = pain_points
            if interests is not None:
                data["interests"] = interests
            if constraints is not None:
                data["constraints"] = constraints
            if custom_instructions is not None:
                data["custom_instructions"] = custom_instructions
            if errors is not None:
                data["errors"] = errors
            if patience is not None:
                data["patience"] = patience
            if repeats_questions is not None:
                data["repeats_questions"] = repeats_questions
            if connection_checks is not None:
                data["connection_checks"] = connection_checks
            if interrupts is not None:
                data["interrupts"] = interrupts

            return self.http.post("sdk/generate-custom-persona", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate custom persona: {e}")
                return {"metadata": {}, "item_id": None}
            raise

    async def agenerate_custom_persona(
        self,
        dataset_id: str,
        base_call_id: str,
        # communication style
        verbosity: Optional[str] = None,
        formality: Optional[str] = None,
        language: Optional[str] = None,
        # behavioral traits
        confidence: Optional[str] = None,
        engagement: Optional[str] = None,
        info_sharing: Optional[str] = None,
        decision_style: Optional[str] = None,
        # emotional
        mood: Optional[str] = None,
        # context
        urgency: Optional[str] = None,
        budget: Optional[str] = None,
        # lists
        pain_points: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        # free-form
        custom_instructions: Optional[str] = None,
        errors: Optional[str] = None,
        # behavioral
        patience: Optional[str] = None,
        repeats_questions: Optional[str] = None,
        connection_checks: Optional[str] = None,
        interrupts: Optional[str] = None,
        # retell/row creation
        create_retell_agent: bool = False,
        create_row: bool = False,
    ) -> Dict[str, Any]:
        """Generate a persona from transcript with customization overrides (asynchronous).

        Args:
            dataset_id: UUID of the dataset
            base_call_id: Call ID to use as base transcript
            verbosity: Response length (terse, moderate, verbose)
            formality: Communication style (formal, casual, mixed)
            language: Language preference (hindi-dominant, english-dominant, mixed)
            confidence: Confidence level (confident, uncertain, anxious)
            engagement: Engagement style (cooperative, resistant, distracted)
            info_sharing: Info sharing style (forthcoming, guarded, needs-prompting)
            decision_style: Decision making (quick, deliberate, indecisive)
            mood: Baseline mood (neutral, friendly, frustrated, confused)
            urgency: Urgency level (low, medium, high)
            budget: Budget constraint (free-form string)
            pain_points: List of pain points
            interests: List of interests
            constraints: List of constraints
            custom_instructions: Free-form behavioral instructions
            errors: Error injection types (stutter, bad_connection, stt, language, all)
            patience: Patience level (patient, moderate, impatient)
            repeats_questions: Question repetition (never, sometimes, frequently)
            connection_checks: Connection check frequency (none, occasional, frequent)
            interrupts: Interrupting behavior (never, sometimes, frequently)
            create_retell_agent: If True, creates Retell LLM and agent
            create_row: If True, creates a new DatasetItem for this persona

        Returns:
            Dictionary with metadata and item_id (null if create_row=False)
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "base_call_id": base_call_id,
                "create_retell_agent": create_retell_agent,
                "create_row": create_row,
            }
            # add optional fields only if provided
            if verbosity is not None:
                data["verbosity"] = verbosity
            if formality is not None:
                data["formality"] = formality
            if language is not None:
                data["language"] = language
            if confidence is not None:
                data["confidence"] = confidence
            if engagement is not None:
                data["engagement"] = engagement
            if info_sharing is not None:
                data["info_sharing"] = info_sharing
            if decision_style is not None:
                data["decision_style"] = decision_style
            if mood is not None:
                data["mood"] = mood
            if urgency is not None:
                data["urgency"] = urgency
            if budget is not None:
                data["budget"] = budget
            if pain_points is not None:
                data["pain_points"] = pain_points
            if interests is not None:
                data["interests"] = interests
            if constraints is not None:
                data["constraints"] = constraints
            if custom_instructions is not None:
                data["custom_instructions"] = custom_instructions
            if errors is not None:
                data["errors"] = errors
            if patience is not None:
                data["patience"] = patience
            if repeats_questions is not None:
                data["repeats_questions"] = repeats_questions
            if connection_checks is not None:
                data["connection_checks"] = connection_checks
            if interrupts is not None:
                data["interrupts"] = interrupts

            return await self.http.apost("sdk/generate-custom-persona", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate custom persona: {e}")
                return {"metadata": {}, "item_id": None}
            raise

    # ==================== Generate Synthetic Persona ====================

    def generate_synthetic_persona(
        self,
        dataset_id: str,
        # communication style
        verbosity: Optional[str] = None,
        formality: Optional[str] = None,
        language: Optional[str] = None,
        # behavioral traits
        confidence: Optional[str] = None,
        engagement: Optional[str] = None,
        info_sharing: Optional[str] = None,
        decision_style: Optional[str] = None,
        # emotional
        mood: Optional[str] = None,
        # context
        urgency: Optional[str] = None,
        budget: Optional[str] = None,
        # lists
        pain_points: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        # free-form
        custom_instructions: Optional[str] = None,
        errors: Optional[str] = None,
        # behavioral
        patience: Optional[str] = None,
        repeats_questions: Optional[str] = None,
        connection_checks: Optional[str] = None,
        interrupts: Optional[str] = None,
        # retell/row creation
        create_retell_agent: bool = False,
        create_row: bool = False,
    ) -> Dict[str, Any]:
        """Generate a fully synthetic persona without any transcript.

        Args:
            dataset_id: UUID of the dataset (for agent context only)
            verbosity: Response length (terse, moderate, verbose)
            formality: Communication style (formal, casual, mixed)
            language: Language preference (hindi-dominant, english-dominant, mixed)
            confidence: Confidence level (confident, uncertain, anxious)
            engagement: Engagement style (cooperative, resistant, distracted)
            info_sharing: Info sharing style (forthcoming, guarded, needs-prompting)
            decision_style: Decision making (quick, deliberate, indecisive)
            mood: Baseline mood (neutral, friendly, frustrated, confused)
            urgency: Urgency level (low, medium, high)
            budget: Budget constraint (free-form string)
            pain_points: List of pain points
            interests: List of interests
            constraints: List of constraints
            custom_instructions: Free-form behavioral instructions
            errors: Error injection types (stutter, bad_connection, stt, language, all)
            patience: Patience level (patient, moderate, impatient)
            repeats_questions: Question repetition (never, sometimes, frequently)
            connection_checks: Connection check frequency (none, occasional, frequent)
            interrupts: Interrupting behavior (never, sometimes, frequently)
            create_retell_agent: If True, creates Retell LLM and agent
            create_row: If True, creates a new DatasetItem for this persona

        Returns:
            Dictionary with metadata and item_id (null if create_row=False)
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "create_retell_agent": create_retell_agent,
                "create_row": create_row,
            }
            # add optional fields only if provided
            if verbosity is not None:
                data["verbosity"] = verbosity
            if formality is not None:
                data["formality"] = formality
            if language is not None:
                data["language"] = language
            if confidence is not None:
                data["confidence"] = confidence
            if engagement is not None:
                data["engagement"] = engagement
            if info_sharing is not None:
                data["info_sharing"] = info_sharing
            if decision_style is not None:
                data["decision_style"] = decision_style
            if mood is not None:
                data["mood"] = mood
            if urgency is not None:
                data["urgency"] = urgency
            if budget is not None:
                data["budget"] = budget
            if pain_points is not None:
                data["pain_points"] = pain_points
            if interests is not None:
                data["interests"] = interests
            if constraints is not None:
                data["constraints"] = constraints
            if custom_instructions is not None:
                data["custom_instructions"] = custom_instructions
            if errors is not None:
                data["errors"] = errors
            if patience is not None:
                data["patience"] = patience
            if repeats_questions is not None:
                data["repeats_questions"] = repeats_questions
            if connection_checks is not None:
                data["connection_checks"] = connection_checks
            if interrupts is not None:
                data["interrupts"] = interrupts

            return self.http.post("sdk/generate-synthetic-persona", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate synthetic persona: {e}")
                return {"metadata": {}, "item_id": None}
            raise

    async def agenerate_synthetic_persona(
        self,
        dataset_id: str,
        # communication style
        verbosity: Optional[str] = None,
        formality: Optional[str] = None,
        language: Optional[str] = None,
        # behavioral traits
        confidence: Optional[str] = None,
        engagement: Optional[str] = None,
        info_sharing: Optional[str] = None,
        decision_style: Optional[str] = None,
        # emotional
        mood: Optional[str] = None,
        # context
        urgency: Optional[str] = None,
        budget: Optional[str] = None,
        # lists
        pain_points: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        # free-form
        custom_instructions: Optional[str] = None,
        errors: Optional[str] = None,
        # behavioral
        patience: Optional[str] = None,
        repeats_questions: Optional[str] = None,
        connection_checks: Optional[str] = None,
        interrupts: Optional[str] = None,
        # retell/row creation
        create_retell_agent: bool = False,
        create_row: bool = False,
    ) -> Dict[str, Any]:
        """Generate a fully synthetic persona without any transcript (asynchronous).

        Args:
            dataset_id: UUID of the dataset (for agent context only)
            verbosity: Response length (terse, moderate, verbose)
            formality: Communication style (formal, casual, mixed)
            language: Language preference (hindi-dominant, english-dominant, mixed)
            confidence: Confidence level (confident, uncertain, anxious)
            engagement: Engagement style (cooperative, resistant, distracted)
            info_sharing: Info sharing style (forthcoming, guarded, needs-prompting)
            decision_style: Decision making (quick, deliberate, indecisive)
            mood: Baseline mood (neutral, friendly, frustrated, confused)
            urgency: Urgency level (low, medium, high)
            budget: Budget constraint (free-form string)
            pain_points: List of pain points
            interests: List of interests
            constraints: List of constraints
            custom_instructions: Free-form behavioral instructions
            errors: Error injection types (stutter, bad_connection, stt, language, all)
            patience: Patience level (patient, moderate, impatient)
            repeats_questions: Question repetition (never, sometimes, frequently)
            connection_checks: Connection check frequency (none, occasional, frequent)
            interrupts: Interrupting behavior (never, sometimes, frequently)
            create_retell_agent: If True, creates Retell LLM and agent
            create_row: If True, creates a new DatasetItem for this persona

        Returns:
            Dictionary with metadata and item_id (null if create_row=False)
        """
        try:
            data: Dict[str, Any] = {
                "dataset_id": dataset_id,
                "create_retell_agent": create_retell_agent,
                "create_row": create_row,
            }
            # add optional fields only if provided
            if verbosity is not None:
                data["verbosity"] = verbosity
            if formality is not None:
                data["formality"] = formality
            if language is not None:
                data["language"] = language
            if confidence is not None:
                data["confidence"] = confidence
            if engagement is not None:
                data["engagement"] = engagement
            if info_sharing is not None:
                data["info_sharing"] = info_sharing
            if decision_style is not None:
                data["decision_style"] = decision_style
            if mood is not None:
                data["mood"] = mood
            if urgency is not None:
                data["urgency"] = urgency
            if budget is not None:
                data["budget"] = budget
            if pain_points is not None:
                data["pain_points"] = pain_points
            if interests is not None:
                data["interests"] = interests
            if constraints is not None:
                data["constraints"] = constraints
            if custom_instructions is not None:
                data["custom_instructions"] = custom_instructions
            if errors is not None:
                data["errors"] = errors
            if patience is not None:
                data["patience"] = patience
            if repeats_questions is not None:
                data["repeats_questions"] = repeats_questions
            if connection_checks is not None:
                data["connection_checks"] = connection_checks
            if interrupts is not None:
                data["interrupts"] = interrupts

            return await self.http.apost("sdk/generate-synthetic-persona", data)
        except Exception as e:
            if self._production:
                logger.error(f"[CallsResource] Failed to generate synthetic persona: {e}")
                return {"metadata": {}, "item_id": None}
            raise
