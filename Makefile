.DEFAULT_GOAL := help
PYTHON ?= python
PAPER  := paper/refusal_signal_extraction

.PHONY: help install install-all test lint format replay figures generate reconstruct paper clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies
	$(PYTHON) -m pip install -e '.[dev]'

install-all: ## Install everything, including embeddings and Ollama support
	$(PYTHON) -m pip install -e '.[all]'

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check formatting and lint rules
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check src tests

format: ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format src tests

replay: ## Verify published metrics and rebuild figures, offline
	$(PYTHON) -m refusal_signal.cli replay

figures: ## Rebuild figures from aggregate_metrics.csv
	$(PYTHON) -m refusal_signal.cli figures

generate: ## Probe a live model via Ollama (requires the daemon)
	$(PYTHON) -m refusal_signal.cli generate

reconstruct: ## Infer hidden rules from leakage observations (requires Ollama)
	$(PYTHON) -m refusal_signal.cli reconstruct

paper: ## Build the paper (requires latexmk)
	latexmk -pdf -cd $(PAPER).tex

clean: ## Remove caches and LaTeX build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	latexmk -C -cd $(PAPER).tex 2>/dev/null || true
