.PHONY: test test-compat install install-dev

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v -k "not compat"

test-compat:
	pytest tests/ -v

test-cov:
	pytest tests/ -k "not compat" --cov=bn_mdp --cov-report=term-missing
