"""Experiment driver.

Sweeps every probe against every rule iteration for each model and records one
row per interaction. The record schema is fixed by ``results/raw_responses.csv``
and must not change without invalidating the published results.

Note on the sweep structure: all rules are active in the system prompt for every
call, so iterating over ``ruleset`` produces independent samples of one shared
condition rather than a per-rule manipulation. The loop is preserved to keep the
published run reproducible; see ``docs/limitations.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone

from .backends import ChatBackend
from .probes import Probe, ProbeType, all_probes
from .rules import Rule, build_system_prompt

#: Probes are swept one phase at a time -- every single-turn probe across every
#: rule, then every multi-turn probe -- which is the order the published
#: ``raw_responses.csv`` rows were written in.
PHASE_ORDER: tuple[ProbeType, ...] = ("single_turn", "multi_turn")

#: Column order of a raw results record, matching ``results/raw_responses.csv``.
RECORD_FIELDS: tuple[str, ...] = (
    "timestamp",
    "model",
    "id",
    "cat",
    "type",
    "usrmsgs",
    "response",
)

DEFAULT_REQUEST_DELAY_SECONDS = 0.5


def _utc_timestamp() -> str:
    """Naive UTC ISO-8601 timestamp, matching the recorded format."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def run_experiment(
    backend: ChatBackend,
    models: Sequence[str],
    ruleset: Sequence[Rule],
    probes: Iterable[Probe] | None = None,
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Probe every model under every rule iteration and return raw records.

    Args:
        backend: Chat backend used to generate replies.
        models: Model identifiers to sweep.
        ruleset: Hidden rules; all are active in the system prompt at once.
        probes: Probes to run. Defaults to the full single- and multi-turn set.
        request_delay_seconds: Pause between calls, to avoid saturating the daemon.
        sleep: Injectable sleep, so tests need not wait.
        on_progress: Optional callback for human-readable progress lines.

    Returns:
        One record per interaction, in sweep order.
    """
    probe_list = list(probes) if probes is not None else all_probes()
    system_prompt = build_system_prompt(ruleset)
    records: list[dict[str, object]] = []

    for model in models:
        if on_progress:
            on_progress(f"Probing {model} with {len(probe_list)} probes x {len(ruleset)} rules")

        for phase in PHASE_ORDER:
            phase_probes = [probe for probe in probe_list if probe.probe_type == phase]
            if not phase_probes:
                continue
            if on_progress:
                on_progress(f"  {phase}: {len(phase_probes)} probes per rule")

            for rule in ruleset:
                # Lets a ReplayBackend line recorded responses up with this iteration.
                expect_rule = getattr(backend, "expect_rule", None)
                if callable(expect_rule):
                    expect_rule(rule.rule_id)

                for probe in phase_probes:
                    response = backend.chat(
                        system_prompt=system_prompt,
                        user_messages=list(probe.turns),
                        model=model,
                    )
                    records.append(
                        {
                            "timestamp": _utc_timestamp(),
                            "model": model,
                            "id": rule.rule_id,
                            "cat": probe.category,
                            "type": probe.probe_type,
                            "usrmsgs": list(probe.turns),
                            "response": response,
                        }
                    )
                    if request_delay_seconds:
                        sleep(request_delay_seconds)

    if on_progress:
        on_progress(f"Completed {len(records)} runs")
    return records
