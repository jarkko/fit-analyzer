# Testing and Code Quality Guide

## Overview

This project includes a comprehensive test suite following Python best practices:

- **Unit Tests** - Test individual functions and components
- **Integration Tests** - Test end-to-end workflows
- **Linting** - Code quality and style checks
- **Type Checking** - Static type analysis
- **Coverage Reports** - Track test coverage

## Quick Start

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

Or using make:
```bash
make install-dev
```

### Run All Tests

```bash
# Using Python
python run_tests.py

# Using pytest directly
pytest -v

# Using make
make test
```

## Test Structure

### Unit Tests

#### `test_fit_to_summary.py`
Tests for FIT file analysis functionality:
- `TestNormalizedPower` - Normalized power calculations
- `TestTRIMP` - Heart rate training load
- `TestSessionDataProcessing` - Session data processing
- `TestFITFileParsing` - FIT file parsing
- `TestMultisportHandling` - Multisport activity handling
- `TestDataValidation` - Edge cases and validation
- `TestCSVOutput` - CSV output formatting

```bash
pytest test_fit_to_summary.py -v
```

#### `test_garmin_sync.py`
Tests for Garmin Connect sync functionality:
- `TestExistingActivityIDs` - Activity ID detection
- `TestGarminAuthentication` - Authentication flow
- `TestDownloadActivities` - Download logic
- `TestAnalysisExecution` - Analysis script execution
- `TestIdempotency` - Idempotent behavior
- `TestEnvironmentVariables` - Environment variable handling

```bash
pytest test_garmin_sync.py -v
```

### Integration Tests

#### `test_integration.py`
End-to-end workflow tests:
- `TestEndToEndWorkflow` - Complete analysis workflow
- `TestCSVOutputValidation` - CSV data quality
- `TestMultisportHandling` - Multisport processing
- `TestErrorHandling` - Error cases

```bash
pytest test_integration.py -v
```

## Code Quality Tools

### Flake8 (Linting)

Checks code style and quality issues:

```bash
# Run flake8
flake8 fit_to_summary.py garmin_sync.py

# Using make
make lint
```

Configuration in `.flake8`:
- Max line length: 100
- Complexity limit: 15
- Ignores: E203, W503 (black compatibility)

### Black (Formatting)

Automatic code formatting:

```bash
# Check formatting
black --check --line-length 100 *.py

# Auto-format
black --line-length 100 *.py

# Using make
make format
```

### isort (Import Sorting)

Sorts and organizes imports:

```bash
# Check imports
isort --check-only --profile black *.py

# Fix imports
isort --profile black *.py

# Included in make format
make format
```

### MyPy (Type Checking)

Static type analysis:

```bash
# Run type checking
mypy fit_to_summary.py garmin_sync.py --ignore-missing-imports

# Using make
make type-check
```

### Pylint (Advanced Linting)

Comprehensive code analysis:

```bash
# Run pylint
pylint fit_to_summary.py garmin_sync.py

# Using make (with sensible ignores)
make lint
```

## Coverage Reports

### Generate Coverage

```bash
# Run tests with coverage
pytest --cov=. --cov-report=term-missing --cov-report=html

# Using make
make coverage
```

### View Coverage

```bash
# Terminal report
pytest --cov=. --cov-report=term-missing

# HTML report (open in browser)
open htmlcov/index.html
```

Coverage targets:
- **Aim for**: >80% coverage
- **Critical paths**: >90% coverage
- **Unit tests**: Cover all business logic

## Pre-commit Hooks

Automatically run checks before commits:

### Setup

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Or using make
make pre-commit-install
```

### Usage

Hooks run automatically on `git commit`. Manual run:

```bash
# Run on all files
pre-commit run --all-files

# Using make
make pre-commit-run
```

Configured checks:
- Trailing whitespace
- File endings
- YAML/JSON/TOML validation
- Black formatting
- isort imports
- Flake8 linting
- Bandit security checks
- MyPy type checking

## CI/CD Integration

### GitHub Actions (Example)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: make test
      - run: make lint
      - run: make type-check
```

## Make Commands Reference

```bash
make help              # Show all available commands
make install           # Install production dependencies
make install-dev       # Install development dependencies
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only
make coverage          # Generate coverage report
make lint              # Run all linters
make format            # Auto-format code
make type-check        # Run type checking
make clean             # Remove generated files
make all               # Run everything
```

## Writing New Tests

### Unit Test Template

```python
import unittest

class TestNewFeature(unittest.TestCase):
    """Test description"""

    def setUp(self):
        """Set up test fixtures"""
        pass

    def tearDown(self):
        """Clean up after tests"""
        pass

    def test_feature_works(self):
        """Test that feature works correctly"""
        result = my_function(input_data)
        self.assertEqual(result, expected_output)

    def test_feature_handles_edge_case(self):
        """Test edge case handling"""
        result = my_function(edge_case_data)
        self.assertIsNotNone(result)
```

### Integration Test Template

```python
import unittest
import tempfile
from pathlib import Path

class TestIntegrationScenario(unittest.TestCase):
    """Integration test description"""

    def setUp(self):
        """Create test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # Setup
        # Execute
        # Assert
        pass
```

## Best Practices

### Test Naming
- Use descriptive names: `test_function_handles_empty_input`
- Group related tests in classes
- Use docstrings to explain what's being tested

### Test Structure
- **Arrange**: Set up test data
- **Act**: Execute the code being tested
- **Assert**: Verify the results

### Mocking
- Mock external dependencies (Garmin API, file system)
- Use `unittest.mock` for isolation
- Don't mock the code you're testing

### Coverage
- Aim for high coverage, but quality > quantity
- Test edge cases and error conditions
- Don't sacrifice readability for coverage

### Code Quality
- Run linters before committing
- Fix warnings, not just errors
- Use type hints where appropriate
- Keep functions small and focused

## Continuous Improvement

### Adding Tests
1. Write test first (TDD)
2. Make test pass
3. Refactor if needed
4. Run full test suite

### Fixing Bugs
1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify test passes
4. Add regression test

### Code Reviews
- Check test coverage
- Verify new tests are meaningful
- Ensure linting passes
- Review type hints

## Troubleshooting

### Tests Failing
```bash
# Run with verbose output
pytest -vv

# Stop at first failure
pytest -x

# Show print statements
pytest -s
```

### Import Errors
```bash
# Ensure dependencies installed
pip install -r requirements-dev.txt

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Coverage Issues
```bash
# Show which lines are not covered
pytest --cov=. --cov-report=term-missing

# Generate HTML report for detail
pytest --cov=. --cov-report=html
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
