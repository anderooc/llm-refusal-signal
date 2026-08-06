"""Regression tests pinning the published results.

The refactor from notebook to package must not move a single reported number.
These tests recompute the paper's metrics from the committed responses and fail
if anything drifts, which is what makes the metric definitions safe to refactor
around.
"""

from __future__ import annotations

import pandas as pd
import pytest

from refusal_signal import (
    DEFAULT_RULESET,
    ReplayBackend,
    aggregate_metrics,
    annotate_keyword_leakage,
    records_to_frame,
    run_experiment,
)
from refusal_signal.metrics import compare_aggregates

PUBLISHED_MODEL = "llama3.1"
PUBLISHED_SAMPLE_COUNT = 33
PUBLISHED_KEYWORD_LEAKS = 2


def test_recorded_run_has_the_published_shape(raw_responses):
    assert len(raw_responses) == PUBLISHED_SAMPLE_COUNT
    assert set(raw_responses["model"]) == {PUBLISHED_MODEL}
    assert set(raw_responses["id"]) == {rule.rule_id for rule in DEFAULT_RULESET}


def test_keyword_leakage_recomputes_to_two_of_thirty_three(raw_responses):
    scored = annotate_keyword_leakage(raw_responses, DEFAULT_RULESET)
    assert scored["keyword_leakage"].sum() == PUBLISHED_KEYWORD_LEAKS


def test_verbatim_leaks_only_occur_under_hypothetical_or_moral_framing(raw_responses):
    scored = annotate_keyword_leakage(raw_responses, DEFAULT_RULESET)
    leaked = scored[scored["keyword_leakage"] == 1]
    assert set(leaked["cat"]) == {"hypothetical_questions", "moral_dilemmas"}


def test_pipeline_reproduces_the_published_aggregate_metrics(raw_responses, published_aggregate):
    scored = annotate_keyword_leakage(raw_responses, DEFAULT_RULESET)
    drift = compare_aggregates(aggregate_metrics(scored), published_aggregate)
    assert drift is None, drift


def test_every_probing_category_leaks_semantically(published_aggregate):
    assert (published_aggregate["avg_semantic_leak"] > 0).all()


def test_why_chains_leak_least_semantically(published_aggregate):
    ordered = published_aggregate.sort_values("avg_semantic_leak")
    assert ordered.iloc[0]["cat"] == "why_chains"


def test_boundary_questions_leak_most_semantically(published_aggregate):
    ordered = published_aggregate.sort_values("avg_semantic_leak", ascending=False)
    assert ordered.iloc[0]["cat"] == "boundary_questions"


def test_replay_reproduces_the_recorded_sweep_row_for_row(results_dir, raw_responses):
    backend = ReplayBackend.from_csv(results_dir / "raw_responses.csv")
    records = run_experiment(
        backend=backend,
        models=[PUBLISHED_MODEL],
        ruleset=DEFAULT_RULESET,
        request_delay_seconds=0,
    )
    replayed = records_to_frame(records)
    for column in ("model", "id", "cat", "type", "response"):
        pd.testing.assert_series_equal(replayed[column], raw_responses[column], check_names=False)
    assert [str(turns) for turns in replayed["usrmsgs"]] == list(raw_responses["usrmsgs"])


def test_reconstructed_policies_are_committed(results_dir):
    text = (results_dir / "reconstructed_policies.txt").read_text(encoding="utf-8")
    assert "reconstructed" in text.lower()
    assert text.count("**") >= 6  # one bolded rule name per inferred policy


@pytest.mark.parametrize("filename", ["raw_responses.csv", "aggregate_metrics.csv"])
def test_result_artefacts_are_readable(results_dir, filename):
    assert not pd.read_csv(results_dir / filename).empty
