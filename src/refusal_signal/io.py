"""Reading and writing the result artefacts in ``results/``.

Kept separate from the metrics so the on-disk schema has exactly one owner.
``usrmsgs`` and ``leaked_phrases`` round-trip as Python list literals, which is
how the published CSVs were written.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

RAW_RESPONSES_FILENAME = "raw_responses.csv"
AGGREGATE_METRICS_FILENAME = "aggregate_metrics.csv"
RECONSTRUCTED_POLICIES_FILENAME = "reconstructed_policies.txt"

#: Columns stored as ``repr``'d Python lists rather than scalars.
LIST_COLUMNS: tuple[str, ...] = ("usrmsgs", "leaked_phrases")


def _parse_list_column(value: object) -> object:
    """Best-effort recovery of a list literal, leaving unparseable values alone."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    return parsed if isinstance(parsed, list) else value


def load_raw_responses(path: str | Path, parse_lists: bool = False) -> pd.DataFrame:
    """Load per-response records.

    Args:
        path: Path to a ``raw_responses.csv`` file.
        parse_lists: If true, decode ``usrmsgs``/``leaked_phrases`` into real
            lists. Left off by default so round-tripping is byte-stable.
    """
    frame = pd.read_csv(path)
    if parse_lists:
        for column in LIST_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].map(_parse_list_column)
    return frame


def load_aggregate_metrics(path: str | Path) -> pd.DataFrame:
    """Load per-(model, category) aggregate leakage metrics."""
    return pd.read_csv(path)


def records_to_frame(records: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Turn raw experiment records into a DataFrame."""
    return pd.DataFrame(list(records))


def write_results(
    results_dir: str | Path,
    raw: pd.DataFrame | None = None,
    aggregate: pd.DataFrame | None = None,
    reconstructed_policies: str | None = None,
) -> list[Path]:
    """Write whichever artefacts were supplied and return the paths touched."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if raw is not None:
        path = directory / RAW_RESPONSES_FILENAME
        raw.to_csv(path, index=False)
        written.append(path)

    if aggregate is not None:
        path = directory / AGGREGATE_METRICS_FILENAME
        aggregate.to_csv(path, index=False)
        written.append(path)

    if reconstructed_policies is not None:
        path = directory / RECONSTRUCTED_POLICIES_FILENAME
        path.write_text(reconstructed_policies, encoding="utf-8")
        written.append(path)

    return written
