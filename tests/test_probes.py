"""Tests for the probing strategies."""

from __future__ import annotations

import pytest

from refusal_signal import PROBE_CATEGORIES, all_probes, multi_turn_probes, single_turn_probes
from refusal_signal.probes import WHY_CHAIN_FOLLOW_UP, WHY_CHAIN_OPENER, why_chain_turns


def test_eight_single_turn_probes_evenly_split_across_four_framings():
    probes = single_turn_probes()
    assert len(probes) == 8
    counts = {category: 0 for category in PROBE_CATEGORIES if category != "why_chains"}
    for probe in probes:
        counts[probe.category] += 1
    assert set(counts.values()) == {2}


def test_single_turn_probes_have_exactly_one_turn():
    assert all(len(probe.turns) == 1 for probe in single_turn_probes())
    assert all(probe.probe_type == "single_turn" for probe in single_turn_probes())


def test_why_chain_escalates_from_opener_to_repeated_follow_ups():
    assert why_chain_turns(3) == (WHY_CHAIN_OPENER, WHY_CHAIN_FOLLOW_UP, WHY_CHAIN_FOLLOW_UP)
    assert why_chain_turns(1) == (WHY_CHAIN_OPENER,)


def test_why_chain_rejects_empty_conversations():
    with pytest.raises(ValueError):
        why_chain_turns(0)


def test_multi_turn_probes_are_three_identical_chains():
    probes = multi_turn_probes()
    assert len(probes) == 3
    assert {probe.turns for probe in probes} == {why_chain_turns()}
    assert all(probe.probe_type == "multi_turn" for probe in probes)


def test_full_probe_set_matches_the_published_run():
    probes = all_probes()
    assert len(probes) == 11
    assert {probe.category for probe in probes} == set(PROBE_CATEGORIES)


def test_probe_text_exposes_the_opening_turn():
    probe = single_turn_probes()[0]
    assert probe.text == probe.turns[0]


def test_probes_never_ask_the_model_to_break_its_rules():
    # The threat model depends on every probe being benign: no jailbreak
    # phrasing, no instruction to ignore the system prompt.
    import re

    banned = ("ignore", "disregard", "pretend you have no", "override", "jailbreak", "DAN")
    pattern = re.compile("|".join(rf"\b{re.escape(marker)}\b" for marker in banned), re.IGNORECASE)
    for probe in all_probes():
        for turn in probe.turns:
            assert not pattern.search(turn), turn
