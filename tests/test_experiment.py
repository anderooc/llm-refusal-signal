"""Tests for the experiment driver."""

from __future__ import annotations

from refusal_signal import DEFAULT_RULESET, RECORD_FIELDS, StubBackend, run_experiment
from refusal_signal.probes import all_probes, single_turn_probes
from refusal_signal.rules import build_system_prompt


def _stub(response: str = "I cannot discuss that.") -> StubBackend:
    return StubBackend(lambda system_prompt, user_messages, model: response)


def _run(backend, **kwargs):
    return run_experiment(
        backend=backend,
        models=kwargs.pop("models", ["test-model"]),
        ruleset=kwargs.pop("ruleset", DEFAULT_RULESET),
        request_delay_seconds=0,
        **kwargs,
    )


def test_sweep_covers_every_rule_probe_and_model():
    backend = _stub()
    records = _run(backend, models=["a", "b"])
    assert len(records) == 2 * len(DEFAULT_RULESET) * len(all_probes())


def test_records_use_the_frozen_schema():
    records = _run(_stub())
    assert tuple(records[0]) == RECORD_FIELDS


def test_single_turn_probes_all_run_before_the_why_chains():
    # Matches the order the published raw_responses.csv rows were written in.
    records = _run(_stub())
    types = [record["type"] for record in records]
    assert types == sorted(types, key=lambda t: t != "single_turn")
    assert types.count("single_turn") == 8 * len(DEFAULT_RULESET)


def test_every_call_carries_the_full_hidden_ruleset():
    backend = _stub()
    _run(backend)
    expected = build_system_prompt(DEFAULT_RULESET)
    assert {call[0] for call in backend.calls} == {expected}


def test_multi_turn_probes_send_all_turns_in_one_context():
    backend = _stub()
    _run(backend)
    multi_turn_calls = [call for call in backend.calls if len(call[1]) > 1]
    assert multi_turn_calls
    assert all(len(call[1]) == 3 for call in multi_turn_calls)


def test_delay_is_applied_once_per_call():
    slept: list[float] = []
    run_experiment(
        backend=_stub(),
        models=["m"],
        ruleset=DEFAULT_RULESET[:1],
        probes=single_turn_probes(),
        request_delay_seconds=0.25,
        sleep=slept.append,
    )
    assert slept == [0.25] * 8


def test_progress_callback_reports_the_run():
    messages: list[str] = []
    _run(_stub(), on_progress=messages.append)
    assert any("Completed 33 runs" in message for message in messages)


def test_timestamps_are_naive_utc_iso8601():
    record = _run(_stub())[0]
    from datetime import datetime

    parsed = datetime.fromisoformat(str(record["timestamp"]))
    assert parsed.tzinfo is None
