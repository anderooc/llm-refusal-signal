"""Tests for the command-line interface."""

from __future__ import annotations

import pytest

from refusal_signal.cli import main


def test_prompt_command_prints_the_hidden_rules(capsys):
    assert main(["prompt"]) == 0
    assert "RULE_AREA51" in capsys.readouterr().out


def test_score_check_passes_against_the_published_metrics(capsys):
    assert main(["score", "--reuse-semantic-scores", "--check"]) == 0
    assert "OK: recomputed metrics match" in capsys.readouterr().err


def test_score_check_fails_when_the_inputs_change(tmp_path, capsys, raw_responses, results_dir):
    import yaml

    tampered = raw_responses.copy()
    tampered.loc[0, "semantic_leakage_score"] = 0.99
    (tmp_path / "raw_responses.csv").write_text(tampered.to_csv(index=False), encoding="utf-8")
    (tmp_path / "aggregate_metrics.csv").write_text(
        (results_dir / "aggregate_metrics.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"results_dir": str(tmp_path)}), encoding="utf-8")

    assert main(["-c", str(config), "score", "--reuse-semantic-scores", "--check"]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_figures_command_writes_png_and_pdf(tmp_path):
    assert main(["figures", "-o", str(tmp_path)]) == 0
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".pdf", ".pdf", ".png", ".png"]


def test_reconstruct_can_print_the_prompt_without_a_model(capsys):
    assert main(["reconstruct", "--print-prompt"]) == 0
    assert "You are a security analyst" in capsys.readouterr().out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])
