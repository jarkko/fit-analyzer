# Makefile for FIT Analyzer
# Professional Python library development workflow

.PHONY: help install install-dev test test-fast test-unit test-integration lint format type-check clean coverage docs build install-editable

# Python interpreter
PYTHON := .venv/bin/python3
PIP := $(PYTHON) -m pip
PYTEST := .venv/bin/pytest

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
	@echo "  make test              Run all tests (52 tests)"
	@echo "  make test-fast         Run fast tests only (skip integration)"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests (requires FIT files, slow)"
	@echo "  make coverage          Generate coverage report"
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
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -v -m "not slow"

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
	@.venv/bin/flake8 src/fitanalyzer --max-line-length=100
	@echo ""
	@echo "Running pylint..."
	@.venv/bin/pylint src/fitanalyzer --disable=C0111,C0103,R0913,R0914
	@echo ""
	@echo "✅ Lint checks complete!"

format:
	@echo "Running black..."
	.venv/bin/black --line-length 100 src/fitanalyzer tests/
	@echo ""
	@echo "Running isort..."
	.venv/bin/isort --profile black --line-length 100 src/fitanalyzer tests/
	@echo ""
	@echo "✅ Code formatted!"

type-check:
	@echo "Running mypy..."
	.venv/bin/mypy src/fitanalyzer --ignore-missing-imports || true

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

all: clean install-dev test lint
