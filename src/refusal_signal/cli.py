"""Command-line entry point: ``refusal-signal <command>``.

Commands
--------
``generate``     Probe a live model via Ollama and write ``raw_responses.csv``.
``score``        Annotate leakage metrics and write ``aggregate_metrics.csv``.
``figures``      Render the leakage figures from ``aggregate_metrics.csv``.
``reconstruct``  Infer hidden rules from leakage observations.
``replay``       Re-run scoring and figures from the committed responses, offline.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import io as results_io
from .config import ExperimentConfig
from .metrics import (
    AGGREGATE_TOLERANCE,
    SEMANTIC_COLUMN,
    aggregate_metrics,
    annotate_keyword_leakage,
    compare_aggregates,
)
from .plots import save_figures, use_headless_backend
from .rules import DEFAULT_RULESET, build_system_prompt


def _load_config(path: str | None) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(path) if path else ExperimentConfig()


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _raw_path(config: ExperimentConfig, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(config.results_dir) / results_io.RAW_RESPONSES_FILENAME


def _aggregate_path(config: ExperimentConfig, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(config.results_dir) / results_io.AGGREGATE_METRICS_FILENAME


def cmd_generate(args: argparse.Namespace) -> int:
    """Probe a live model and record raw refusal responses."""
    from .backends import OllamaBackend
    from .experiment import run_experiment

    config = _load_config(args.config)
    backend = OllamaBackend(host=config.ollama_host)

    records = run_experiment(
        backend=backend,
        models=config.models,
        ruleset=DEFAULT_RULESET,
        request_delay_seconds=config.request_delay_seconds,
        on_progress=_log,
    )
    frame = results_io.records_to_frame(records)
    written = results_io.write_results(config.results_dir, raw=frame)
    for path in written:
        _log(f"wrote {path}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Annotate leakage metrics on recorded responses and aggregate them."""
    config = _load_config(args.config)
    raw_path = _raw_path(config, args.raw)
    frame = results_io.load_raw_responses(raw_path)

    frame = annotate_keyword_leakage(frame, DEFAULT_RULESET)

    if args.reuse_semantic_scores and SEMANTIC_COLUMN in frame.columns:
        _log(f"reusing recorded {SEMANTIC_COLUMN} (no embedding model loaded)")
    else:
        from .metrics import SemanticLeakageScorer

        scorer = SemanticLeakageScorer(DEFAULT_RULESET, model_name=config.embedding_model)
        frame = scorer.annotate(frame)

    aggregate = aggregate_metrics(frame)

    if args.check:
        published_path = _aggregate_path(config, None)
        published = results_io.load_aggregate_metrics(published_path)
        drift = compare_aggregates(aggregate, published)
        if drift is not None:
            _log(f"FAIL: recomputed metrics differ from {published_path}\n{drift}")
            return 1
        _log(f"OK: recomputed metrics match {published_path} within {AGGREGATE_TOLERANCE:g}")
    else:
        written = results_io.write_results(
            config.results_dir,
            raw=frame if args.write_raw else None,
            aggregate=aggregate,
        )
        for path in written:
            _log(f"wrote {path}")

    print(aggregate.to_string(index=False))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    """Render leakage figures from aggregate metrics."""
    use_headless_backend()
    config = _load_config(args.config)
    aggregate = results_io.load_aggregate_metrics(_aggregate_path(config, args.aggregate))
    for path in save_figures(aggregate, args.output or config.figures_dir):
        _log(f"wrote {path}")
    return 0


def cmd_reconstruct(args: argparse.Namespace) -> int:
    """Ask a system-prompt-blind model to infer the hidden rules."""
    from .backends import OllamaBackend
    from .reconstruction import build_reconstruction_prompt, reconstruct_policies

    config = _load_config(args.config)
    frame = results_io.load_raw_responses(_raw_path(config, args.raw))

    if args.print_prompt:
        print(build_reconstruction_prompt(frame, DEFAULT_RULESET))
        return 0

    backend = OllamaBackend(host=config.ollama_host)
    policies = reconstruct_policies(
        backend=backend,
        frame=frame,
        ruleset=DEFAULT_RULESET,
        model=config.reconstruction_model,
    )
    for path in results_io.write_results(config.results_dir, reconstructed_policies=policies):
        _log(f"wrote {path}")
    print(policies)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Recompute metrics from committed responses and rebuild the figures.

    Verifies rather than overwrites: the published metrics are an artefact of a
    specific run and should only change when the science does.
    """
    config = _load_config(args.config)
    score_args = argparse.Namespace(
        config=args.config, raw=None, reuse_semantic_scores=True, write_raw=False, check=True
    )
    if cmd_score(score_args) != 0:
        return 1
    figure_args = argparse.Namespace(config=args.config, aggregate=None, output=None)
    cmd_figures(figure_args)
    _log(f"replayed pipeline from {config.results_dir}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    """Print the system prompt containing the hidden rules."""
    print(build_system_prompt(DEFAULT_RULESET))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refusal-signal",
        description="Measure how much hidden system policy leaks through LLM refusals.",
    )
    parser.add_argument("-c", "--config", help="path to a YAML experiment config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="probe a live model via Ollama")
    generate.set_defaults(func=cmd_generate)

    score = subparsers.add_parser("score", help="compute leakage metrics")
    score.add_argument("--raw", help="path to raw_responses.csv")
    score.add_argument(
        "--reuse-semantic-scores",
        action="store_true",
        help="reuse recorded semantic scores instead of loading the embedding model",
    )
    score.add_argument(
        "--write-raw", action="store_true", help="overwrite raw_responses.csv with annotations"
    )
    score.add_argument(
        "--check",
        action="store_true",
        help="verify against the published metrics instead of overwriting them",
    )
    score.set_defaults(func=cmd_score)

    figures = subparsers.add_parser("figures", help="render leakage figures")
    figures.add_argument("--aggregate", help="path to aggregate_metrics.csv")
    figures.add_argument("-o", "--output", help="output directory for figures")
    figures.set_defaults(func=cmd_figures)

    reconstruct = subparsers.add_parser("reconstruct", help="infer hidden rules from leakage")
    reconstruct.add_argument("--raw", help="path to raw_responses.csv")
    reconstruct.add_argument(
        "--print-prompt",
        action="store_true",
        help="print the reconstruction prompt without querying a model",
    )
    reconstruct.set_defaults(func=cmd_reconstruct)

    replay = subparsers.add_parser("replay", help="rebuild metrics and figures offline")
    replay.set_defaults(func=cmd_replay)

    prompt = subparsers.add_parser("prompt", help="print the hidden-rule system prompt")
    prompt.set_defaults(func=cmd_prompt)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
