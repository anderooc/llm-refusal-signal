"""Tests for experiment configuration."""

from __future__ import annotations

import pytest
import yaml

from refusal_signal import ExperimentConfig


def test_defaults_describe_the_published_run():
    config = ExperimentConfig()
    assert config.models == ["llama3.1"]
    assert config.reconstruction_model == "llama3.1"
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_empty_config_falls_back_to_defaults():
    assert ExperimentConfig.from_dict(None) == ExperimentConfig()
    assert ExperimentConfig.from_dict({}) == ExperimentConfig()


def test_overrides_are_applied():
    config = ExperimentConfig.from_dict({"models": ["mistral"], "request_delay_seconds": 0})
    assert config.models == ["mistral"]
    assert config.request_delay_seconds == 0


def test_typos_are_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown config keys"):
        ExperimentConfig.from_dict({"modles": ["mistral"]})


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"models": ["a", "b"], "results_dir": "out"}), encoding="utf-8")
    config = ExperimentConfig.from_yaml(path)
    assert config.models == ["a", "b"]
    assert config.results_dir == "out"


def test_shipped_default_config_is_valid(tmp_path):
    from pathlib import Path

    repo_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert ExperimentConfig.from_yaml(repo_config).models
