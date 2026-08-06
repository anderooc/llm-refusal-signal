"""Figure generation.

Renders the two leakage figures used in the paper. Styling is deliberately
conservative -- flat bars, one accent colour, horizontal reference grid, direct
value labels -- so the figures stay legible at single-column width and in
greyscale print.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def use_headless_backend() -> None:
    """Switch to Agg for CLI and CI rendering.

    Deliberately not called at import time: forcing a backend inside a library
    breaks ``%matplotlib inline`` for anyone importing it from a notebook.
    """
    matplotlib.use("Agg")


INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e3e3e6"

#: Semantic leakage is the always-on signal; keyword leakage is the rare breach.
SEMANTIC_COLOR = "#2f6f9f"
KEYWORD_COLOR = "#c0392b"

#: Human-readable axis labels and titles per metric column.
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "keyword_leak_rate": ("Keyword leakage rate", "Verbatim disclosure of forbidden phrases"),
    "avg_semantic_leak": ("Mean cosine similarity", "Semantic alignment with hidden rules"),
}

METRIC_COLORS: dict[str, str] = {
    "keyword_leak_rate": KEYWORD_COLOR,
    "avg_semantic_leak": SEMANTIC_COLOR,
}


def prettify_category(category: str, samples: int | None = None) -> str:
    """``why_chains`` -> ``Why\\nchains``, so long labels stay upright.

    Sample counts ride along in the tick label rather than inside the bar, which
    keeps them legible when a bar has zero height.
    """
    label = category.replace("_", "\n").capitalize()
    return f"{label}\nn={samples}" if samples is not None else label


def apply_style() -> None:
    """Install the shared rcParams for every figure in the project."""
    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 3.8),
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )


def plot_metric_by_category(
    aggregate: pd.DataFrame,
    metric: str,
    model: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw one metric across probing categories for a single model."""
    if metric not in METRIC_LABELS:
        raise KeyError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_LABELS)}")

    subset = aggregate[aggregate["model"] == model].sort_values(metric, ascending=False)
    if subset.empty:
        raise ValueError(f"no rows for model {model!r}")

    if ax is None:
        _, ax = plt.subplots()

    ylabel, subtitle = METRIC_LABELS[metric]
    is_rate = metric == "keyword_leak_rate"
    values = subset[metric].to_numpy()
    labels = [
        prettify_category(category, samples)
        for category, samples in zip(subset["cat"], subset["num_samples"], strict=True)
    ]

    bars = ax.bar(labels, values, color=METRIC_COLORS[metric], width=0.62, zorder=2)

    headroom = max(values.max(), 1e-9) * 1.3
    ax.set_ylim(0, headroom)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(f"{subtitle}\n{model}", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", length=0)

    if is_rate:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

    for bar, value in zip(bars, values, strict=True):
        centre = bar.get_x() + bar.get_width() / 2
        # A measured zero is a result, not missing data -- mark the baseline so
        # the reader can tell the two apart.
        if value == 0:
            ax.plot(
                [bar.get_x(), bar.get_x() + bar.get_width()],
                [0, 0],
                color=METRIC_COLORS[metric],
                linewidth=2.4,
                solid_capstyle="butt",
                zorder=3,
            )
        ax.annotate(
            f"{value:.0%}" if is_rate else f"{value:.3f}",
            (centre, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color=INK if value > 0 else MUTED,
            fontweight="bold" if value > 0 else "normal",
        )

    return ax


def save_figures(
    aggregate: pd.DataFrame,
    output_dir: str | Path,
    metrics: Iterable[str] = ("keyword_leak_rate", "avg_semantic_leak"),
) -> list[Path]:
    """Render one figure per (metric, model) pair and return the files written."""
    apply_style()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for metric in metrics:
        for model in aggregate["model"].unique():
            fig, ax = plt.subplots()
            plot_metric_by_category(aggregate, metric=metric, model=model, ax=ax)
            slug = f"{metric}_{str(model).replace('.', '-').replace(':', '-')}"
            path = directory / f"{slug}.png"
            fig.savefig(path)
            fig.savefig(path.with_suffix(".pdf"))
            plt.close(fig)
            written.extend([path, path.with_suffix(".pdf")])

    return written
