"""Experiment configuration loaded from YAML.

Everything has a default matching the published run, so an empty config file
reproduces the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .experiment import DEFAULT_REQUEST_DELAY_SECONDS
from .metrics import DEFAULT_EMBEDDING_MODEL


@dataclass
class ExperimentConfig:
    """Settings for one end-to-end run."""

    #: Models to probe, as Ollama tags.
    models: list[str] = field(default_factory=lambda: ["llama3.1"])
    #: Model asked to reconstruct the hidden rules from leakage observations.
    reconstruction_model: str = "llama3.1"
    #: Sentence embedding model backing the semantic leakage score.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    #: Seconds to wait between generation calls.
    request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS
    #: Where artefacts are written.
    results_dir: str = "results"
    #: Where figures are written.
    figures_dir: str = "results/figures"
    #: Optional Ollama host override, e.g. ``http://localhost:11434``.
    ollama_host: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ExperimentConfig:
        """Build a config from a mapping, rejecting unknown keys."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown config keys: {sorted(unknown)}; expected any of {sorted(known)}"
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file."""
        import yaml

        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))
