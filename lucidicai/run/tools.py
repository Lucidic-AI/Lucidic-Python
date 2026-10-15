"""The hosted module tools a run's checkpoint exposes, frozen per run."""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .run import Run


class RunTools:
    """``run.tools``: the checkpoint's module tools, fetched once per run.

    A run without a ``checkpoint_id`` (the baseline) has no module tools and
    makes no request. ``list()`` is provider-neutral; ``openai()`` and
    ``anthropic()`` render the same tools as provider specs. ``call()`` runs
    one inside the open run's session.
    """

    def __init__(self, run: "Run", snapshot: Optional[List[Dict[str, Any]]] = None):
        self._run = run
        self._snapshot = snapshot

    def _ensure(self) -> List[Dict[str, Any]]:
        if self._snapshot is None:
            checkpoint_id = self._run.ids.checkpoint_id
            self._snapshot = [] if checkpoint_id is None else (
                self._run.client.training_modules.tools(checkpoint_id=checkpoint_id)
            )
        return self._snapshot

    async def _aensure(self) -> List[Dict[str, Any]]:
        if self._snapshot is None:
            checkpoint_id = self._run.ids.checkpoint_id
            self._snapshot = [] if checkpoint_id is None else (
                await self._run.client.training_modules.atools(checkpoint_id=checkpoint_id)
            )
        return self._snapshot

    def list(self) -> List[Dict[str, Any]]:
        """``[{"name", "description", "input_schema"}]`` for each module tool."""
        return [
            {
                "name": tool["tool_name"],
                "description": tool.get("tool_description") or "",
                "input_schema": tool.get("input_schema") or {},
            }
            for tool in self._ensure()
            if tool.get("tool_name")
        ]

    def openai(self) -> List[Dict[str, Any]]:
        """The same tools as OpenAI function specs."""
        return self._run.client.training_modules.openai_tools(tools=self._ensure())

    def anthropic(self) -> List[Dict[str, Any]]:
        """The same tools as Anthropic tool specs."""
        return self._run.client.training_modules.anthropic_tools(tools=self._ensure())

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """Run module tool ``name`` and return its result (or the structured soft failure).

        Raises:
            InvalidOperationError: outside ``with run:`` (the session must exist).
        """
        self._run._require_open("run.tools.call")
        return self._run.client.training_modules.call(
            name, args, session_id=self._run.ids.session_id
        )

    async def acall(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        """Async sibling of ``call``."""
        self._run._require_open("run.tools.acall")
        return await self._run.client.training_modules.acall(
            name, args, session_id=self._run.ids.session_id
        )
