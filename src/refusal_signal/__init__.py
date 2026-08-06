"""Refusal Signal Extraction: measuring semantic leakage through LLM refusals.

Safety-aligned models refuse politely and explain themselves. This package
measures how much of a hidden system policy those explanations give away, and
whether the leaked signal is enough for a second model to reconstruct the policy
it never saw.

Typical offline use, reproducing the published metrics from committed responses::

    from refusal_signal import DEFAULT_RULESET, aggregate_metrics, load_raw_responses
    from refusal_signal import annotate_keyword_leakage

    frame = annotate_keyword_leakage(load_raw_responses("results/raw_responses.csv"),
                                     DEFAULT_RULESET)
    print(aggregate_metrics(frame))
"""

from __future__ import annotations

from .backends import ChatBackend, OllamaBackend, ReplayBackend, StubBackend
from .config import ExperimentConfig
from .experiment import RECORD_FIELDS, run_experiment
from .io import (
    load_aggregate_metrics,
    load_raw_responses,
    records_to_frame,
    write_results,
)
from .metrics import (
    AGGREGATE_COLUMNS,
    KeywordLeakage,
    SemanticLeakageScorer,
    aggregate_metrics,
    annotate_keyword_leakage,
    check_keyword_leakage,
)
from .probes import (
    PROBE_CATEGORIES,
    Probe,
    all_probes,
    multi_turn_probes,
    single_turn_probes,
)
from .reconstruction import build_reconstruction_prompt, reconstruct_policies
from .rules import DEFAULT_RULESET, Rule, build_system_prompt, rule_lookup

__version__ = "1.0.0"

__all__ = [
    "AGGREGATE_COLUMNS",
    "DEFAULT_RULESET",
    "PROBE_CATEGORIES",
    "RECORD_FIELDS",
    "ChatBackend",
    "ExperimentConfig",
    "KeywordLeakage",
    "OllamaBackend",
    "Probe",
    "ReplayBackend",
    "Rule",
    "SemanticLeakageScorer",
    "StubBackend",
    "__version__",
    "aggregate_metrics",
    "all_probes",
    "annotate_keyword_leakage",
    "build_reconstruction_prompt",
    "build_system_prompt",
    "check_keyword_leakage",
    "load_aggregate_metrics",
    "load_raw_responses",
    "multi_turn_probes",
    "reconstruct_policies",
    "records_to_frame",
    "rule_lookup",
    "run_experiment",
    "single_turn_probes",
    "write_results",
]
