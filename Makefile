# Makefile for FIT Analyzer
# Professional Python library development workflow

.PHONY: help install install-dev test test-sequential test-fast test-unit test-integration lint format type-check clean coverage docs build install-editable

# Python interpreter - always use venv to ensure consistent environment
PYTHON := .venv/bin/python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
FLAKE8 := $(PYTHON) -m flake8
PYLINT := $(PYTHON) -m pylint
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy

help:
	@echo "FIT Analyzer - Professional Python Library"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install production dependencies"
	@echo "  make install-dev       Install development dependencies"
	@echo "  make install-editable  Install package in editable mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run all tests in parallel (196 tests, fast)"
	@echo "  make test-sequential   Run all tests sequentially (for debugging)"
	@echo "  make test-fast         Run fast tests only (skip integration)"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests (requires FIT files, slow)"
	@echo "  make coverage          Generate coverage report"
	@echo "  make validate-csv      Validate CSV schema"
	@echo "  make regenerate-csv    Regenerate CSV with fresh modules"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              Run all linters"
	@echo "  make format            Auto-format code"
	@echo "  make type-check        Run type checking"
	@echo ""
	@echo "Build & Distribution:"
	@echo "  make build             Build distribution packages"
	@echo "  make clean             Remove build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make run-example       Run example script"
	@echo "  make sync              Sync from Garmin Connect and analyze"
	@echo "  make analyze           Analyze all FIT files in data/samples/"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

install-editable:
	$(PIP) install -e .

test:
	$(PYTEST) tests/ -n auto -v

test-sequential:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -n auto -v -m "not slow"

test-unit:
	$(PYTEST) tests/test_parser.py tests/test_sync.py -v

test-integration:
	$(PYTEST) tests/test_integration.py -v

coverage:
	$(PYTEST) tests/ -v --cov=src/fitanalyzer --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated: htmlcov/index.html"

lint:
	@echo "Running flake8..."
	@$(FLAKE8) src/fitanalyzer --max-line-length=100
	@echo ""
	@echo "Running pylint..."
	@$(PYLINT) src/fitanalyzer --rcfile=pyproject.toml
	@echo ""
	@echo "✅ Lint checks complete!"

format:
	@echo "Running black..."
	$(BLACK) --line-length 100 src/fitanalyzer tests/
	@echo ""
	@echo "Running isort..."
	$(ISORT) --profile black --line-length 100 src/fitanalyzer tests/
	@echo ""
	@echo "✅ Code formatted!"

type-check:
	@echo "Running mypy..."
	$(MYPY) src/fitanalyzer --ignore-missing-imports || true

build:
	$(PYTHON) -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned build artifacts"

run-example:
	$(PYTHON) examples/analyze_fit.py

sync:
	@echo "Syncing from Garmin Connect and analyzing..."
	./sync.py

analyze:
	@echo "Analyzing FIT files in data/samples/..."
	./analyze.py data/samples/*.fit --ftp 300 --multisport

validate-csv:
	@echo "Validating CSV schema..."
	$(PYTHON) validate_csv.py

regenerate-csv:
	@echo "Regenerating CSV with fresh modules..."
	$(PYTHON) validate_csv.py --regenerate

all: clean install-dev test lint
