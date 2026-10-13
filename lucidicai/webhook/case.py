"""The typed case a Lucidic eval-run webhook delivers.

Payload contract: mvp-build/specs/webhook-signing.md, "Payload (runner contract)".
"""
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

from ..core.errors import LucidicError

_KNOWN_KEYS = frozenset({
    "session_id", "datasetitem_id", "input", "experiment_id", "eval_run_id", "checkpoint_id",
})
_EMPTY: Mapping[str, Any] = MappingProxyType({})


class WebhookError(LucidicError):
    """A webhook request the receiver must reject; ``status_code`` is the reply."""

    status_code = 400


class WebhookSignatureError(WebhookError):
    """Missing or invalid ``X-Lucidic-Signature``. Reply 401 (the backend won't re-send)."""

    status_code = 401


class WebhookPayloadError(WebhookError):
    """Signed body that is not a valid case. Reply 422 (the backend won't re-send)."""

    status_code = 422


def _uuid_str(payload: Mapping[str, Any], key: str, required: bool) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        if required:
            raise WebhookPayloadError(f"{key} is required")
        return None
    if not isinstance(value, str):
        raise WebhookPayloadError(f"{key} must be a uuid string")
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise WebhookPayloadError(f"{key} is not a uuid") from None


@dataclass(frozen=True)
class WebhookCase:
    """One dispatched case. Ids are canonical lowercase uuid strings.

    ``session_id`` is the idempotency key: delivery is at-least-once, so
    dedupe on it. Equality and hashing use the ids only, so cases can go in a
    set even when ``input`` is a dict. Unknown top-level keys (e.g. tuner
    params) are kept in ``extra``, read-only at the top level only.
    ``checkpoint_id`` is None when the key is absent or null.
    """

    session_id: str
    datasetitem_id: Optional[str] = None
    input: Any = field(default=None, compare=False)
    experiment_id: Optional[str] = None
    eval_run_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    extra: Mapping[str, Any] = field(default=_EMPTY, compare=False)

    @classmethod
    def from_payload(cls, payload: Any) -> "WebhookCase":
        """Validate a decoded JSON body. Raises WebhookPayloadError."""
        if not isinstance(payload, dict):
            raise WebhookPayloadError("body must be a JSON object")
        extra = {k: v for k, v in payload.items() if k not in _KNOWN_KEYS}
        return cls(
            session_id=_uuid_str(payload, "session_id", required=True),
            datasetitem_id=_uuid_str(payload, "datasetitem_id", required=False),
            input=payload.get("input"),
            experiment_id=_uuid_str(payload, "experiment_id", required=False),
            eval_run_id=_uuid_str(payload, "eval_run_id", required=False),
            checkpoint_id=_uuid_str(payload, "checkpoint_id", required=False),
            extra=MappingProxyType(extra),
        )

    def session_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``client.sessions.create`` that link the session to this case.

        ``eval_run_id`` is only echoed as an ``eval_run:<id>`` tag; the backend
        keys the run on ``session_id``.
        """
        kwargs: Dict[str, Any] = {"session_id": self.session_id}
        for key in ("datasetitem_id", "experiment_id", "checkpoint_id"):
            value = getattr(self, key)
            if value is not None:
                kwargs[key] = value
        tags: List[str] = []
        if self.eval_run_id is not None:
            tags.append(f"eval_run:{self.eval_run_id}")
        if tags:
            kwargs["tags"] = tags
        return kwargs
