"""Tests for the chat backends."""

from __future__ import annotations

import pytest

from refusal_signal.backends import ReplayBackend, build_messages


def test_messages_start_with_the_system_prompt_then_user_turns():
    messages = build_messages("sys", ["one", "two"])
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ]


def _records():
    return [
        {"model": "m", "id": "RULE_A", "usrmsgs": "['q']", "response": "first"},
        {"model": "m", "id": "RULE_A", "usrmsgs": "['q']", "response": "second"},
        {"model": "m", "id": "RULE_B", "usrmsgs": "['q']", "response": "other rule"},
    ]


def test_replay_serves_repeated_probes_in_recorded_order():
    backend = ReplayBackend(_records())
    backend.expect_rule("RULE_A")
    assert backend.chat("sys", ["q"], "m") == "first"
    assert backend.chat("sys", ["q"], "m") == "second"


def test_replay_keys_on_the_declared_rule():
    backend = ReplayBackend(_records())
    backend.expect_rule("RULE_B")
    assert backend.chat("sys", ["q"], "m") == "other rule"


def test_replay_refuses_to_guess_without_a_declared_rule():
    backend = ReplayBackend(_records())
    with pytest.raises(RuntimeError, match="expect_rule"):
        backend.chat("sys", ["q"], "m")


def test_replay_raises_rather_than_fabricating_a_response():
    backend = ReplayBackend(_records())
    backend.expect_rule("RULE_A")
    backend.chat("sys", ["q"], "m")
    backend.chat("sys", ["q"], "m")
    with pytest.raises(KeyError):
        backend.chat("sys", ["q"], "m")


def test_replay_parses_multi_turn_message_lists():
    backend = ReplayBackend(
        [{"model": "m", "id": "R", "usrmsgs": "['a', 'b']", "response": "chained"}]
    )
    backend.expect_rule("R")
    assert backend.chat("sys", ["a", "b"], "m") == "chained"


def test_replay_accepts_already_parsed_turn_lists():
    backend = ReplayBackend([{"model": "m", "id": "R", "usrmsgs": ["a"], "response": "ok"}])
    backend.expect_rule("R")
    assert backend.chat("sys", ["a"], "m") == "ok"
