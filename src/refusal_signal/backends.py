"""Chat backends.

The experiment driver only needs one operation -- turn a system prompt plus a
list of user turns into an assistant reply -- so backends are interchangeable.
``OllamaBackend`` talks to a local model, ``ReplayBackend`` re-serves the
responses recorded in ``results/raw_responses.csv`` so the full pipeline can be
reproduced offline, and ``StubBackend`` keeps the test suite hermetic.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatBackend(Protocol):
    """Anything that can answer a system prompt plus a sequence of user turns."""

    def chat(self, system_prompt: str, user_messages: Sequence[str], model: str) -> str: ...


def build_messages(system_prompt: str, user_messages: Sequence[str]) -> list[dict[str, str]]:
    """Assemble the chat payload: one system message followed by the user turns.

    Consecutive user turns are sent without interleaved assistant replies, so the
    model sees the whole escalation at once rather than answering each turn.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": "user", "content": text} for text in user_messages)
    return messages


class OllamaBackend:
    """Queries an instruction-tuned model served by a local Ollama daemon."""

    def __init__(self, host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'ollama' package is required for live generation. "
                "Install it with: pip install 'refusal-signal[generate]'"
            ) from exc
        self._client = ollama.Client(host=host) if host else ollama
        self.host = host

    def chat(self, system_prompt: str, user_messages: Sequence[str], model: str) -> str:
        response = self._client.chat(
            model=model, messages=build_messages(system_prompt, user_messages)
        )
        return response["message"]["content"]


class ReplayBackend:
    """Serves previously recorded responses, keyed by model, rule and probe turns.

    Identical keys are served first-in-first-out, which reproduces the recorded
    run exactly when probes are replayed in their original order. Requests with
    no recorded counterpart raise ``KeyError`` rather than silently fabricating a
    reply.
    """

    def __init__(self, records: Sequence[dict[str, object]]) -> None:
        self._queues: dict[tuple[str, str, tuple[str, ...]], deque[str]] = defaultdict(deque)
        for record in records:
            self._queues[self._key_from_record(record)].append(str(record["response"]))
        self._pending_rule_id: str | None = None

    @classmethod
    def from_csv(cls, path: str | Path) -> ReplayBackend:
        """Load recorded responses from a ``raw_responses.csv`` file."""
        import pandas as pd

        frame = pd.read_csv(path)
        return cls(frame.to_dict("records"))

    @staticmethod
    def _parse_turns(raw: object) -> tuple[str, ...]:
        """Recover the user turns, which round-trip through CSV as a repr'd list."""
        if isinstance(raw, (list, tuple)):
            return tuple(str(turn) for turn in raw)
        parsed = ast.literal_eval(str(raw))
        return tuple(str(turn) for turn in parsed)

    def _key_from_record(self, record: dict[str, object]) -> tuple[str, str, tuple[str, ...]]:
        return (
            str(record["model"]),
            str(record["id"]),
            self._parse_turns(record["usrmsgs"]),
        )

    def expect_rule(self, rule_id: str) -> None:
        """Declare which rule iteration the next ``chat`` calls belong to.

        The recorded schema keys responses by rule, but a backend only sees the
        prompt -- which is identical across rules -- so the driver announces the
        current rule before each block of probes.
        """
        self._pending_rule_id = rule_id

    def chat(self, system_prompt: str, user_messages: Sequence[str], model: str) -> str:
        if self._pending_rule_id is None:
            raise RuntimeError("call expect_rule() before replaying probes")
        key = (model, self._pending_rule_id, tuple(user_messages))
        queue = self._queues.get(key)
        if not queue:
            raise KeyError(f"no recorded response for {key}")
        return queue.popleft()


class StubBackend:
    """Deterministic backend for tests; delegates to a caller-supplied function."""

    def __init__(self, responder: Callable[[str, Sequence[str], str], str]) -> None:
        self._responder = responder
        self.calls: list[tuple[str, tuple[str, ...], str]] = []

    def chat(self, system_prompt: str, user_messages: Sequence[str], model: str) -> str:
        self.calls.append((system_prompt, tuple(user_messages), model))
        return self._responder(system_prompt, user_messages, model)
