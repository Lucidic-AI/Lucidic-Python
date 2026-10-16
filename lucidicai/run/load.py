"""``lucidicai.load()``: one Run from a webhook delivery, ``$LUCIDIC_RUN`` or ``--lucidic-run``.

Payload: ``session_id`` (required); ``datasetitem_id``, ``experiment_id``,
``checkpoint_id``, ``eval_run_id`` and ``input`` (optional); other keys go to ``params``.
"""
import json
import os
import sys
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from ..core.errors import RunLoadError, RunPayloadError, RunSecretMissingError, RunSignatureError
from ..webhook import SECRET_ENV_VAR, SIGNATURE_HEADER, verify
from ..webhook.signing import Secret, secret_keys
from .run import RUN_ENV_VAR, SNAPSHOT_KEY, Run, RunIds

if TYPE_CHECKING:
    from ..client import LucidicAI

RUN_ARG = "--lucidic-run"
_ID_KEYS = ("session_id", "datasetitem_id", "experiment_id", "checkpoint_id", "eval_run_id")
_KNOWN_KEYS = frozenset(_ID_KEYS + ("input", SNAPSHOT_KEY))
_UNSET: Any = object()


def load(
    body: Union[bytes, str, None] = None,
    signature: Optional[str] = None,
    *,
    client: Optional["LucidicAI"] = None,
    secret: Optional[Secret] = None,
    session_id: Optional[str] = None,
    datasetitem_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    input: Any = _UNSET,
    params: Optional[Mapping[str, Any]] = None,
) -> Run:
    """Return the run to execute. No network I/O, so a webhook handler can still reply 202 fast.

    The first source found wins: a webhook ``body`` (verified against
    ``signature``, the ``X-Lucidic-Signature`` value, before it is parsed),
    else ``$LUCIDIC_RUN``, else ``--lucidic-run`` in ``sys.argv`` (each a JSON
    object or a path to one). Keyword arguments then override single fields;
    with no source they describe the whole run (local runs).

    Args:
        body: The raw webhook request body, never re-serialised JSON.
        signature: The ``X-Lucidic-Signature`` header value.
        client: The client the run uses; default: one built from env, closed on exit.
        secret: Webhook secret, or a list of them while one rotates; default
            ``$LUCIDIC_WEBHOOK_SECRET`` (comma-separated for rotation).
        params: Extra params, merged over the payload's unknown keys.

    A signed delivery verifies again if replayed within the 300 s window, so a
    webhook receiver should dedupe on ``session_id``.

    Raises:
        RunSignatureError: bad or missing signature (reply 401).
        RunPayloadError: not a valid run (reply 422).
        RunSecretMissingError: a webhook body but no secret (reply 412).
        RunLoadError: no run found (500).
    """
    payload = _from_source(body, signature, secret)
    if payload is None and session_id is None:
        raise RunLoadError(
            f"no run found: pass a webhook body, set {RUN_ENV_VAR}, pass {RUN_ARG} or session_id="
        )
    if not isinstance(payload, (dict, type(None))):
        raise RunPayloadError("a run must be a JSON object")
    overrides = {
        key: value for key, value in zip(
            _ID_KEYS, (session_id, datasetitem_id, experiment_id, checkpoint_id, eval_run_id)
        ) if value is not None
    }
    if input is not _UNSET:
        overrides["input"] = input
    return _run_from_payload({**(payload or {}), **overrides}, params or {}, client)


def _from_source(body: Union[bytes, str, None], signature: Optional[str], secret: Optional[Secret]) -> Any:
    if body is not None:
        return _from_webhook(body, signature, secret)
    value = os.environ.get(RUN_ENV_VAR)
    if value:
        return _from_value(value, RUN_ENV_VAR)
    value = _argv_value(sys.argv)
    return None if value is None else _from_value(value, RUN_ARG)


def webhook_keys(secret: Optional[Secret] = None) -> List[str]:
    """The keys in ``secret``, else in ``$LUCIDIC_WEBHOOK_SECRET`` (comma-separated while one rotates)."""
    if secret is None:
        secret = os.environ.get(SECRET_ENV_VAR, "").split(",")
    return secret_keys(secret)


def _from_webhook(body: Union[bytes, str], signature: Optional[str], secret: Optional[Secret]) -> Any:
    raw = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    keys = webhook_keys(secret)
    if not keys:
        raise RunSecretMissingError(f"no webhook secret: pass secret= or set {SECRET_ENV_VAR}")
    if not verify(keys, raw, signature):
        raise RunSignatureError(f"missing or invalid {SIGNATURE_HEADER}")
    return _json(raw, "webhook body")


def _from_value(value: str, source: str) -> Any:
    """``value`` is inline JSON when it starts with ``{``, else a path to a JSON file."""
    if value.lstrip().startswith("{"):
        return _json(value, source)
    try:
        with open(value, "rb") as f:
            return _json(f.read(), f"{source} file {value}")
    except OSError as e:
        raise RunLoadError(f"cannot read {source} file {value!r}: {e}") from None


def _argv_value(argv: List[str]) -> Optional[str]:
    # scanned by hand so the agent's own argument parser is untouched
    for i, arg in enumerate(argv[1:], start=1):
        if arg == RUN_ARG:
            if i + 1 >= len(argv):
                raise RunLoadError(f"{RUN_ARG} needs a value")
            return argv[i + 1]
        if arg.startswith(RUN_ARG + "="):
            return arg[len(RUN_ARG) + 1:]
    return None


def _json(raw: Union[bytes, str], source: str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise RunPayloadError(f"{source} is not JSON: {e}") from None


def _uuid_str(payload: Mapping[str, Any], key: str, required: bool) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        if required:
            raise RunPayloadError(f"{key} is required")
        return None
    if not isinstance(value, str):
        raise RunPayloadError(f"{key} must be a uuid string")
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise RunPayloadError(f"{key} is not a uuid") from None


def _snapshot(payload: Mapping[str, Any]) -> Dict[str, Any]:
    snapshot = payload.get(SNAPSHOT_KEY) or {}
    if not (isinstance(snapshot, dict)
            and isinstance(snapshot.get("prompts") or {}, dict)
            and isinstance(snapshot.get("tools") or [], list)
            and isinstance(snapshot.get("session_id") or "", str)):
        raise RunPayloadError(f"{SNAPSHOT_KEY} is malformed")
    return snapshot


def _run_from_payload(payload: Dict[str, Any], params: Mapping[str, Any], client: Optional["LucidicAI"]) -> Run:
    reserved = _KNOWN_KEYS.intersection(params)
    if reserved:
        raise RunPayloadError(f"params may not set {sorted(reserved)}")
    ids = RunIds(
        session_id=_uuid_str(payload, "session_id", required=True),
        **{key: _uuid_str(payload, key, required=False) for key in _ID_KEYS[1:]},
    )
    extra = {key: value for key, value in payload.items() if key not in _KNOWN_KEYS}
    return Run(
        ids,
        payload.get("input"),
        {**extra, **params},
        client=client,
        snapshot=_snapshot(payload),
    )
