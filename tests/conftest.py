"""Shared fixtures: paths to the committed result artefacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS_DIR


@pytest.fixture(scope="session")
def raw_responses() -> pd.DataFrame:
    """The 33 refusal responses recorded in the published run."""
    return pd.read_csv(RESULTS_DIR / "raw_responses.csv")


@pytest.fixture(scope="session")
def published_aggregate() -> pd.DataFrame:
    """The per-category leakage metrics reported in the paper."""
    return pd.read_csv(RESULTS_DIR / "aggregate_metrics.csv")
