PYTHON ?= .venv/bin/python
RUFF ?= ruff

.PHONY: check lint format test

check: lint test

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) format src tests

test:
	$(PYTHON) -m pytest
