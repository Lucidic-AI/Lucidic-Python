"""lucidicai.load(): webhook body > $LUCIDIC_RUN > --lucidic-run, then explicit overrides.

No network I/O happens in load(), so nothing here stubs HTTP.
"""
import json
import sys

import pytest

import lucidicai as lucidic
from lucidicai import RunLoadError, RunPayloadError, RunSignatureError

from .conftest import (
    CHECKPOINT_ID, EVAL_RUN_ID, EXPERIMENT_ID, ITEM_ID, SECRET, SESSION_ID, full_payload, signed, wire,
)

OTHER_ID = "99999999-8888-7777-6666-555555555555"


def _load_signed(payload, **kwargs):
    body, signature = signed(payload)
    return lucidic.load(body, signature, secret=SECRET, **kwargs)


# ---- webhook source ----------------------------------------------------------

def test_webhook_body_gives_the_run_fields():
    run = _load_signed(full_payload())

    assert run.ids == lucidic.RunIds(
        session_id=SESSION_ID, datasetitem_id=ITEM_ID, experiment_id=EXPERIMENT_ID,
        checkpoint_id=CHECKPOINT_ID, eval_run_id=EVAL_RUN_ID,
    )
    assert run.input == {"task": "corporate-ma/x", "documents": ["a.pdf"]}
    assert dict(run.params) == {}


def test_tampered_body_is_401():
    body, signature = signed(full_payload())
    tampered = body.replace(ITEM_ID.encode(), ITEM_ID[::-1].encode())
    with pytest.raises(RunSignatureError) as exc:
        lucidic.load(tampered, signature, secret=SECRET)
    assert exc.value.status_code == 401


def test_body_signed_with_another_secret_is_401():
    body, signature = signed(full_payload(), secret="someone-else")
    with pytest.raises(RunSignatureError):
        lucidic.load(body, signature, secret=SECRET)


def test_missing_signature_is_401():
    with pytest.raises(RunSignatureError):
        lucidic.load(wire(full_payload()), None, secret=SECRET)


def test_raw_bytes_are_verified_not_a_reserialisation():
    # spaced, unsorted, non-ASCII: re-serialising would change the bytes.
    raw = ('{ "session_id": "%s", "input": {"q": "café"}, "datasetitem_id": null }'
           % SESSION_ID).encode("utf-8")
    assert _load_signed(raw).input == {"q": "café"}


def test_signature_is_checked_before_json():
    # garbage with a bad signature is a 401, not a 422.
    with pytest.raises(RunSignatureError):
        lucidic.load(b"not json", "sha256=00", secret=SECRET)


def test_str_body_is_verified_as_utf8():
    body, signature = signed(full_payload(input="café"))
    assert lucidic.load(body.decode("utf-8"), signature, secret=SECRET).input == "café"


def test_secret_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LUCIDIC_WEBHOOK_SECRET", SECRET)
    body, signature = signed(full_payload())
    assert lucidic.load(body, signature).ids.session_id == SESSION_ID


def test_no_secret_is_a_500_config_error():
    body, signature = signed(full_payload())
    with pytest.raises(RunLoadError) as exc:
        lucidic.load(body, signature)
    assert type(exc.value) is RunLoadError
    assert exc.value.status_code == 500


@pytest.mark.parametrize("body", [
    b"not json",
    b"[1, 2]",
    wire({"datasetitem_id": ITEM_ID}),                    # no session_id
    wire({"session_id": "abc"}),                          # not a uuid
    wire({"session_id": 123}),                            # not a string
    wire(full_payload(datasetitem_id="not-a-uuid")),
])
def test_malformed_case_is_422(body):
    with pytest.raises(RunPayloadError) as exc:
        _load_signed(body)
    assert exc.value.status_code == 422


def test_optional_ids_may_be_null_and_checkpoint_absent():
    run = _load_signed({"session_id": SESSION_ID, "datasetitem_id": None, "input": None,
                        "experiment_id": None, "eval_run_id": None})
    assert run.ids == lucidic.RunIds(session_id=SESSION_ID)
    assert run.input is None


def test_null_checkpoint_id_is_treated_as_absent():
    # a 422 is permanent (no re-send), so a null must not cost the case.
    assert _load_signed(full_payload(checkpoint_id=None)).ids.checkpoint_id is None


def test_ids_are_canonical_lowercase():
    assert _load_signed(full_payload(session_id=SESSION_ID.upper())).ids.session_id == SESSION_ID


def test_unknown_keys_become_read_only_params():
    run = _load_signed(full_payload(style="formal", n_turns=3))
    assert dict(run.params) == {"style": "formal", "n_turns": 3}
    with pytest.raises(TypeError):
        run.params["style"] = "casual"


def test_snapshot_key_is_not_a_param():
    run = _load_signed(full_payload(_lucidic={"prompts": {"system": "pinned"}}))
    assert "_lucidic" not in run.params
    assert run.prompt("system") == "pinned"


# ---- LUCIDIC_RUN / --lucidic-run ----------------------------------------------

def test_env_inline_json(monkeypatch):
    monkeypatch.setenv("LUCIDIC_RUN", json.dumps(full_payload()))
    run = lucidic.load()
    assert run.ids.checkpoint_id == CHECKPOINT_ID
    assert run.input["task"] == "corporate-ma/x"


def test_env_path_to_json(monkeypatch, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(full_payload()))
    monkeypatch.setenv("LUCIDIC_RUN", str(path))
    assert lucidic.load().ids.session_id == SESSION_ID


def test_bad_env_json_is_422(monkeypatch):
    monkeypatch.setenv("LUCIDIC_RUN", "{not json")
    with pytest.raises(RunPayloadError):
        lucidic.load()


def test_missing_env_file_is_a_load_error(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDIC_RUN", str(tmp_path / "absent.json"))
    with pytest.raises(RunLoadError):
        lucidic.load()


@pytest.mark.parametrize("argv", [
    ["agent.py", "--verbose", "--lucidic-run", json.dumps(full_payload())],
    ["agent.py", "--lucidic-run=" + json.dumps(full_payload()), "--verbose"],
])
def test_argv_forms(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    assert lucidic.load().ids.session_id == SESSION_ID


def test_argv_flag_without_value_is_a_load_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agent.py", "--lucidic-run"])
    with pytest.raises(RunLoadError):
        lucidic.load()


def test_webhook_body_wins_over_env(monkeypatch):
    monkeypatch.setenv("LUCIDIC_RUN", json.dumps(full_payload(session_id=OTHER_ID)))
    assert _load_signed(full_payload()).ids.session_id == SESSION_ID


def test_env_wins_over_argv(monkeypatch):
    monkeypatch.setenv("LUCIDIC_RUN", json.dumps(full_payload()))
    monkeypatch.setattr(sys, "argv", ["agent.py", "--lucidic-run", json.dumps(full_payload(session_id=OTHER_ID))])
    assert lucidic.load().ids.session_id == SESSION_ID


# ---- explicit arguments --------------------------------------------------------

def test_explicit_args_override_the_source(monkeypatch):
    monkeypatch.setenv("LUCIDIC_RUN", json.dumps(full_payload(style="formal")))
    run = lucidic.load(checkpoint_id=None, experiment_id=OTHER_ID.upper(),
                       input="local task", params={"style": "casual"})
    assert run.ids.checkpoint_id == CHECKPOINT_ID   # None means "not overridden"
    assert run.ids.experiment_id == OTHER_ID        # validated like the source
    assert run.input == "local task"
    assert dict(run.params) == {"style": "casual"}


def test_explicit_args_alone_describe_a_local_run():
    run = lucidic.load(session_id=SESSION_ID, input={"q": 1})
    assert run.ids == lucidic.RunIds(session_id=SESSION_ID)
    assert run.input == {"q": 1}


@pytest.mark.parametrize("kwargs", [
    {"session_id": "not-a-uuid"},
    {"session_id": SESSION_ID, "params": {"session_id": OTHER_ID}},
])
def test_bad_override_is_422(kwargs):
    with pytest.raises(RunPayloadError):
        lucidic.load(**kwargs)


def test_no_source_and_no_session_id_is_a_load_error():
    with pytest.raises(RunLoadError) as exc:
        lucidic.load()
    assert "no run found" in str(exc.value)
