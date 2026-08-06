"""Tests for the blind policy-reconstruction step."""

from __future__ import annotations

import pandas as pd
import pytest

from refusal_signal import DEFAULT_RULESET, StubBackend, build_reconstruction_prompt
from refusal_signal.reconstruction import (
    ANALYST_SYSTEM_PROMPT,
    MAX_EXAMPLE_CHARS,
    build_observation_blocks,
    reconstruct_policies,
)


@pytest.fixture
def scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": rule.rule_id,
                "response": f"A polite refusal about {rule.rule_id}. " + "padding " * 80,
                "keyword_leakage": index % 2,
                "semantic_leakage_score": 0.25,
            }
            for index, rule in enumerate(DEFAULT_RULESET)
            for _ in range(4)
        ]
    )


def test_prompt_withholds_the_rule_descriptions(scored_frame):
    # The whole claim rests on the reconstruction model never seeing the policy.
    prompt = build_reconstruction_prompt(scored_frame, DEFAULT_RULESET)
    for rule in DEFAULT_RULESET:
        assert rule.description not in prompt


def test_prompt_withholds_the_forbidden_phrases(scored_frame):
    prompt = build_reconstruction_prompt(scored_frame, DEFAULT_RULESET).lower()
    for rule in DEFAULT_RULESET:
        for phrase in rule.forbidden_phrases:
            assert phrase.lower() not in prompt


def test_prompt_withholds_the_system_prompt(scored_frame):
    from refusal_signal import build_system_prompt

    prompt = build_reconstruction_prompt(scored_frame, DEFAULT_RULESET)
    assert build_system_prompt(DEFAULT_RULESET).strip() not in prompt


def test_prompt_carries_observable_evidence_only(scored_frame):
    prompt = build_reconstruction_prompt(scored_frame, DEFAULT_RULESET)
    assert "Keyword leakage rate:" in prompt
    assert "Average semantic similarity" in prompt
    assert "Example refusal explanations:" in prompt
    assert "reconstruct the hidden rules as a numbered list" in prompt


def test_rule_identifiers_are_disclosed_to_the_analyst(scored_frame):
    # Documented caveat rather than a defect: the observation blocks are keyed by
    # rule ID, so names like RULE_CLASSIFICATION are visible to the reconstructor.
    # See docs/limitations.md.
    prompt = build_reconstruction_prompt(scored_frame, DEFAULT_RULESET)
    for rule in DEFAULT_RULESET:
        assert rule.rule_id in prompt


def _excerpt_lines(blocks: str) -> list[str]:
    return [line.strip() for line in blocks.splitlines() if line.strip().startswith("- ")]


def test_excerpts_are_truncated_to_the_character_budget(scored_frame):
    blocks = build_observation_blocks(scored_frame, DEFAULT_RULESET)
    for line in _excerpt_lines(blocks):
        assert len(line) <= MAX_EXAMPLE_CHARS + len("- ") + len("...")


def test_at_most_three_excerpts_per_rule(scored_frame):
    blocks = build_observation_blocks(scored_frame, DEFAULT_RULESET)
    assert len(_excerpt_lines(blocks)) == 3 * len(DEFAULT_RULESET)


def test_reconstruction_uses_the_neutral_analyst_persona(scored_frame):
    backend = StubBackend(lambda system_prompt, user_messages, model: "1. Confidentiality Rule")
    output = reconstruct_policies(backend, scored_frame, DEFAULT_RULESET, model="test-model")
    assert output == "1. Confidentiality Rule"
    assert backend.calls[0][0] == ANALYST_SYSTEM_PROMPT
