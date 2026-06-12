"""EvoSim resource API operations."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..client import HttpClient
from ...core.errors import LucidicError

logger = logging.getLogger("Lucidic")


class EvoSimResource:
    """Handle EvoSim-related API operations."""

    def __init__(
        self,
        http: HttpClient,
        agent_id: Optional[str] = None,
        production: bool = False,
    ):
        """Initialize EvoSim resource.

        Args:
            http: HTTP client instance
            agent_id: Default agent ID for training runs
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._agent_id = agent_id
        self._production = production

    def train(self, config_json_file: Union[str, Path]) -> Dict[str, Any]:
        """Start an EvoSim training run from a JSON config file.

        The file must contain a JSON object; its top-level keys become the
        request payload for the training run. ``agent_id`` is filled in from
        the client config when the file doesn't provide one.

        Args:
            config_json_file: Path to the JSON training config

        Returns:
            Backend response describing the run (successful or failed)
        """
        try:
            payload = _load_training_config(config_json_file)
            if self._agent_id:
                payload.setdefault("agent_id", self._agent_id)
            return self.http.post("sdk/evosim/training-run", payload)
        except Exception as e:
            if self._production:
                logger.error(f"[EvoSimResource] Failed to start training run: {e}")
                return {"status": "failed", "error": str(e)}
            raise

    async def atrain(self, config_json_file: Union[str, Path]) -> Dict[str, Any]:
        """Async sibling of ``train``."""
        try:
            payload = _load_training_config(config_json_file)
            if self._agent_id:
                payload.setdefault("agent_id", self._agent_id)
            return await self.http.apost("sdk/evosim/training-run", payload)
        except Exception as e:
            if self._production:
                logger.error(f"[EvoSimResource] Failed to start training run: {e}")
                return {"status": "failed", "error": str(e)}
            raise


def _load_training_config(config_json_file: Union[str, Path]) -> Dict[str, Any]:
    """Read a training config file and return its keys as a payload dict."""
    path = Path(config_json_file)
    try:
        raw = path.read_text()
    except OSError as e:
        raise LucidicError(f"Cannot read EvoSim config file {path}: {e}") from e

    try:
        config = json.loads(raw)
    except ValueError as e:
        raise LucidicError(f"EvoSim config file {path} is not valid JSON: {e}") from e

    if not isinstance(config, dict):
        raise LucidicError(
            f"EvoSim config file {path} must contain a JSON object, "
            f"got {type(config).__name__}"
        )
    return dict(config)
