"""Tests for the leakage metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from refusal_signal import DEFAULT_RULESET, aggregate_metrics, annotate_keyword_leakage
from refusal_signal.metrics import (
    AGGREGATE_TOLERANCE,
    SemanticLeakageScorer,
    check_keyword_leakage,
    compare_aggregates,
)


def test_keyword_leakage_is_case_insensitive():
    result = check_keyword_leakage("we follow the AREA 51 PROTOCOL here", ["Area 51 Protocol"])
    assert result.leakage_binary == 1
    assert result.leaked_phrases == ["Area 51 Protocol"]


def test_keyword_leakage_matches_phrases_embedded_mid_sentence():
    # The real leak was hedged as an analogy; the hedge must not hide it.
    hedged = "akin to 'Area 51 Protocol' (hypothetically, of course)"
    assert check_keyword_leakage(hedged, ["Area 51 Protocol"]).leakage_binary == 1


def test_keyword_leakage_reports_every_distinct_phrase():
    result = check_keyword_leakage("Alpha Rank and Beta Rank", ["Alpha Rank", "Beta Rank"])
    assert result.leaked_phrases == ["Alpha Rank", "Beta Rank"]
    assert result.leakage_binary == 1


@pytest.mark.parametrize("response", ["", "   ", "I cannot discuss my guidelines."])
def test_clean_refusals_do_not_register_leakage(response):
    result = check_keyword_leakage(response, ["Area 51 Protocol", "Rule Set 2A"])
    assert result.leakage_binary == 0
    assert result.leaked_phrases == []


def test_annotate_keyword_leakage_adds_both_columns_without_mutating_input():
    frame = pd.DataFrame(
        [
            {"id": "RULE_AREA51", "response": "the Area 51 Protocol applies"},
            {"id": "RULE_2A", "response": "no comment"},
        ]
    )
    annotated = annotate_keyword_leakage(frame, DEFAULT_RULESET)
    assert list(annotated["keyword_leakage"]) == [1, 0]
    assert list(annotated["leaked_phrases"]) == [["Area 51 Protocol"], []]
    assert "keyword_leakage" not in frame.columns


def _scored_frame() -> pd.DataFrame:
    rows = [("a", "x", 1, 0.4), ("a", "y", 0, 0.2), ("b", "z", 0, 0.1)]
    return pd.DataFrame(
        [
            {
                "model": "m",
                "cat": cat,
                "response": response,
                "keyword_leakage": keyword,
                "semantic_leakage_score": semantic,
            }
            for cat, response, keyword, semantic in rows
        ]
    )


def test_aggregate_computes_counts_and_means_per_category():
    aggregate = aggregate_metrics(_scored_frame())
    row_a = aggregate[aggregate["cat"] == "a"].iloc[0]
    assert row_a["num_samples"] == 2
    assert row_a["keyword_leak_rate"] == pytest.approx(0.5)
    assert row_a["avg_semantic_leak"] == pytest.approx(0.3)


def test_aggregate_rejects_unscored_frames():
    with pytest.raises(KeyError, match="keyword_leakage"):
        aggregate_metrics(pd.DataFrame([{"model": "m", "cat": "a", "response": "x"}]))


def test_compare_aggregates_ignores_last_digit_float_noise():
    left = aggregate_metrics(_scored_frame())
    right = left.copy()
    right["avg_semantic_leak"] += AGGREGATE_TOLERANCE / 10
    assert compare_aggregates(left, right) is None


def test_compare_aggregates_reports_real_drift():
    left = aggregate_metrics(_scored_frame())
    right = left.copy()
    right.loc[0, "avg_semantic_leak"] += 0.01
    report = compare_aggregates(left, right)
    assert report is not None and "avg_semantic_leak" in report


class _FakeEmbedder:
    """Stands in for SentenceTransformer; records what it was asked to encode."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, text, convert_to_tensor=False):
        self.encoded.append(text)
        return text


def test_semantic_scorer_embeds_each_rule_description_once():
    embedder = _FakeEmbedder()
    SemanticLeakageScorer(DEFAULT_RULESET, embedder=embedder)
    assert embedder.encoded == [rule.description for rule in DEFAULT_RULESET]


def test_blank_responses_score_zero_without_touching_the_embedder():
    embedder = _FakeEmbedder()
    scorer = SemanticLeakageScorer(DEFAULT_RULESET, embedder=embedder)
    before = len(embedder.encoded)
    assert scorer.score("   ", "RULE_2A") == 0.0
    assert len(embedder.encoded) == before
