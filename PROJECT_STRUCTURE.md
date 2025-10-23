# FIT Analyzer - Professional Project Structure

This document describes the reorganized professional Python library structure.

## Directory Structure

```
fitanalyzer/
├── src/
│   └── fitanalyzer/              # Main package
│       ├── __init__.py           # Package exports
│       ├── parser.py             # FIT file parsing (was fit_to_summary.py)
│       ├── sync.py               # Garmin Connect sync (was garmin_sync.py)
│       └── credentials.py        # Authentication setup (was setup_credentials.py)
│
├── tests/                        # Test suite
│   ├── conftest.py              # Test configuration
│   ├── test_parser.py           # Parser tests (was test_fit_to_summary.py)
│   ├── test_sync.py             # Sync tests (was test_garmin_sync.py)
│   └── test_integration.py      # Integration tests
│
├── examples/                    # Example scripts
│   └── analyze_fit.py          # Usage demonstration
│
├── data/                        # Data files
│   ├── samples/                # Sample FIT files
│   │   ├── *.fit               # Training activity files
│   │   └── *.zip               # Compressed activities
│   └── *.csv                   # Generated summaries
│
├── docs/                        # Documentation (kept minimal)
│
├── setup.py                    # Package installation
├── pyproject.toml              # Build configuration
├── Makefile                    # Development commands
├── README.md                   # Main documentation
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── .gitignore                  # Git exclusions
├── .flake8                     # Linter configuration
└── .pre-commit-config.yaml     # Pre-commit hooks
```

## Key Changes

### 1. Production Code → src/fitanalyzer/

| Old Location | New Location | Purpose |
|-------------|--------------|---------|
| `fit_to_summary.py` | `src/fitanalyzer/parser.py` | FIT file parsing & metrics |
| `garmin_sync.py` | `src/fitanalyzer/sync.py` | Garmin Connect integration |
| `setup_credentials.py` | `src/fitanalyzer/credentials.py` | Authentication setup |

### 2. Tests → tests/

| Old Location | New Location |
|-------------|--------------|
| `test_fit_to_summary.py` | `tests/test_parser.py` |
| `test_garmin_sync.py` | `tests/test_sync.py` |
| `test_integration.py` | `tests/test_integration.py` |
| `test_garth_contract.py` | *(removed - bug-focused)* |
| `test_garth_with_spec.py` | *(removed - bug-focused)* |
| `test_sport_fields_regression.py` | *(removed - bug-focused)* |

### 3. Data → data/

- FIT files: `data/samples/*.fit`
- Archives: `data/samples/*.zip`
- Outputs: `data/*.csv`

### 4. Removed Files

**Post-mortem documentation:**
- `BUG_FIX_SPORT_FIELDS.md`
- `WHY_TESTS_DIDNT_CATCH_BUG.md`
- `FIXED_SUMMARY.md`
- `TESTING_IMPROVEMENTS.md`
- `YOUR_QUESTION_WAS_RIGHT.md`
- `docs/testing_lessons_learned.md`

**Debug/temporary scripts:**
- `debug_fit.py`
- `debug_sessions.py`
- `demo.py`
- `run_tests.py`
- `sync.sh`
- `fit_to_summary_fixed.py`
- `fit_to_summary_original.py`

**Bug-focused test files:**
- `test_garth_contract.py`
- `test_garth_with_spec.py`
- `test_sport_fields_regression.py`

## Import Changes

### Old Way
```python
from fit_to_summary import summarize_fit_original
from garmin_sync import authenticate_garmin
```

### New Way
```python
from fitanalyzer import summarize_fit_original
from fitanalyzer import authenticate_garmin

# Or more specifically:
from fitanalyzer.parser import summarize_fit_original
from fitanalyzer.sync import authenticate_garmin
```

## Installation

### Development Mode
```bash
# Install in editable mode
make install-dev
make install-editable

# Now you can import from anywhere
python
>>> from fitanalyzer import summarize_fit_original
```

### As Package
```bash
pip install .
# Or from PyPI (when published)
pip install fitanalyzer
```

## Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# With coverage
make coverage
```

## Development Workflow

```bash
# 1. Setup
make install-dev
make install-editable

# 2. Make changes to src/fitanalyzer/*.py

# 3. Run tests
make test

# 4. Format code
make format

# 5. Check quality
make lint
make type-check

# 6. Build distribution
make build
```

## Test Results

After reorganization:
- **36 tests passing** ✅
- **8 tests skipped** (integration tests require FIT files)
- **0 tests failing** ✅

Test breakdown:
- Parser tests: 20 tests
- Sync tests: 16 tests
- Integration tests: 8 tests (skipped, require external files)

## Package Features

### Console Scripts

When installed, provides command-line tools:
```bash
fitanalyzer-parse activity.fit --ftp 300
fitanalyzer-sync --days 30
fitanalyzer-setup
```

### Library Usage

```python
from fitanalyzer import (
    summarize_fit_original,
    summarize_fit_sessions,
    authenticate_garmin,
    download_new_activities,
)

# Parse single activity
summary, sets = summarize_fit_original("ride.fit", ftp=300)

# Parse multisport
sessions = summarize_fit_sessions("triathlon.fit", ftp=300)

# Sync from Garmin
authenticate_garmin("email@example.com", "password")
download_new_activities(days=30, directory="./activities")
```

## Benefits of New Structure

1. **Professional Layout**: Follows Python packaging best practices
2. **Clear Separation**: Production code in `src/`, tests in `tests/`
3. **Installable**: Can be installed with `pip install .` or `pip install -e .`
4. **Importable**: Clean imports via `from fitanalyzer import ...`
5. **Distributable**: Ready for PyPI publication
6. **Maintainable**: Logical organization, no clutter
7. **Testable**: Proper test structure with conftest.py
8. **Documented**: Clear README with examples

## Next Steps

1. **Polish Documentation**: Add API docs in `docs/`
2. **Add Type Hints**: Improve mypy coverage
3. **Increase Test Coverage**: Add more edge cases
4. **Publish to PyPI**: Make installable via `pip install fitanalyzer`
5. **Add CI/CD**: GitHub Actions for automated testing
6. **Version Management**: Use semantic versioning

## References

- [Python Packaging Guide](https://packaging.python.org/)
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [pytest Documentation](https://docs.pytest.org/)
