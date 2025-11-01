# Code Quality Checklist

This document outlines all code quality standards and checks for this project. Run `make quality` to execute all checks.

## 🪝 ONE-TIME SETUP: Install Git Hooks

**Run this once after cloning the repository:**

```bash
./scripts/install_hooks.sh
```

This installs the pre-commit hook that automatically:
- Runs tests before each commit
- Updates the test count badge in README.md
- Includes the badge update in your commit

## ⚠️ CRITICAL: PRE-COMMIT/PRE-PUSH CHECKLIST

**THE AUTOMATED WAY (ENFORCED):**

The pre-commit hook automatically runs these checks before EVERY commit:
1. Auto-formats code (`make format`)
2. Runs linting (`make lint`)
3. Runs type checking (`make type-check`)
4. Runs all tests (`make test`)
5. Updates test badge

**If any check fails, the commit is BLOCKED.**

**THE MANUAL WAY (if you need to verify before committing):**

```bash
# Run ALL checks in one command (same as pre-commit hook)
make pre-commit

# Or run individually:
make format          # Auto-format code
make quality         # Run lint + type-check
make test           # Run all tests

# Then commit and push
git add -A
git commit -m "your message"
git push
```

**WHY:** 
- Pre-commit hook prevents bad code from being committed in the first place
- No more "oops, forgot to run tests" moments
- No more breaking CI after push
- No more wasted CI resources
- No more embarrassing reverts

**NOTE:** The hook is in `.git/hooks/pre-commit` and runs automatically. You can't commit without passing all checks.

## ⚠️ CRITICAL: RELEASE PROCESS

**NEVER create a release tag until CI passes:**

```bash
# 1. Make changes and commit
git add -A
git commit -m "your changes"

# 2. Run pre-push checklist (see above)
make format
make quality && make test

# 3. Push to main
git push

# 4. WAIT for CI to pass (check GitHub Actions)
gh run list --limit 1
# Verify status shows ✓ (checkmark) not X or *

# 5. Only after CI passes, bump version
# Update setup.py and pyproject.toml version
git add setup.py pyproject.toml
git commit -m "Bump version to X.Y.Z"
git push

# 6. Generate release notes
./scripts/generate_release_notes.sh X.Y.Z
# Review and edit RELEASE_NOTES_X.Y.Z.md

# 7. Create tag with release notes
git tag -a vX.Y.Z -F RELEASE_NOTES_X.Y.Z.md

# 8. Push tag
git push origin vX.Y.Z

# The release workflow will:
# - Verify CI passed for this commit
# - Fail the release if CI didn't pass
# - Create GitHub release if CI passed
```

**WHY:** The release workflow checks if CI passed before creating a release. This prevents releasing broken code. Release notes are auto-generated from commits since the last tag.

## 1. Code Style & Formatting

### Black (Auto-formatter)
- **Purpose**: Consistent code formatting
- **Standard**: Line length 100
- **Command**: `make format` or `black --check src/ tests/`
- **Status**: ✅ Automated in CI

### isort (Import Sorting)
- **Purpose**: Consistent import organization
- **Standard**: Black-compatible profile
- **Command**: `isort --check-only --profile black src/ tests/`
- **Status**: ✅ Automated in CI

## 2. Linting & Code Quality

### Flake8 (Style Guide Enforcement)
- **Purpose**: PEP 8 compliance, code smells
- **Configuration**: Max line length 100
- **Command**: `flake8 src/ tests/ --max-line-length=100`
- **Checks**:
  - PEP 8 violations
  - Unused imports
  - Undefined names
  - Syntax errors
- **Status**: ✅ Automated in CI

### Pylint (Comprehensive Analysis)
- **Purpose**: Deep code analysis, best practices
- **Target**: 10.00/10 score
- **Command**: `pylint src/fitanalyzer --rcfile=pyproject.toml`
- **Checks**:
  - Code smells
  - Design issues
  - Refactoring opportunities
  - Documentation quality
  - Complexity metrics
- **Status**: ✅ Automated in CI

## 3. Type Checking

### MyPy (Static Type Analysis)
- **Purpose**: Type safety, prevent runtime errors
- **Standard**: Strict mode configuration
- **Command**: `make type-check` or `mypy src/fitanalyzer`
- **Configuration**: See `[tool.mypy]` in `pyproject.toml`
- **Checks**:
  - Type annotations consistency
  - Type errors and mismatches
  - Optional/None handling
  - Return type correctness
  - Generic type parameters
- **Target**: Zero type errors (100% compliant)
- **Status**: ✅ Required for all PRs and releases

## 4. Testing

### Pytest (Test Framework)
- **Coverage Target**: ≥95% (currently 95.25%)
- **Test Count**: 321 tests
- **Command**: `make test` or `pytest tests/ -n auto`
- **Requirements**:
  - All tests must pass
  - Fast execution with pytest-xdist
  - Comprehensive test coverage
- **Status**: ✅ Automated in CI (4 Python versions)

### Coverage (Code Coverage Analysis)
- **Minimum**: 82.5% enforced in CI, 95% recommended
- **Command**: `make coverage`
- **Reports**:
  - Terminal output
  - HTML report (htmlcov/)
  - XML for Codecov
- **Status**: ✅ Automated in CI with Codecov badge

## 5. Security

### Bandit (Security Linter)
- **Purpose**: Detect security vulnerabilities in code
- **Level**: Medium-High severity (-ll flag)
- **Command**: `bandit -r src/ -ll -f screen`
- **Checks**:
  - SQL injection risks
  - Shell injection
  - Insecure random
  - Hard-coded passwords
  - Insecure temp files
- **Status**: ✅ Automated in CI

### Safety (Dependency Security)
- **Purpose**: Check dependencies for known CVEs
- **Command**: `safety check --output=text`
- **Checks**:
  - Known security vulnerabilities in dependencies
  - CVE database lookup
- **Status**: ✅ Automated in CI

## 6. Documentation

### Docstrings
- **Standard**: Google style docstrings
- **Required for**:
  - All public functions
  - All classes
  - All modules
  - Complex private functions
- **Validation**: Via pylint
- **Status**: ✅ Enforced by pylint

### README.md
- **Requirements**:
  - Clear project description
  - Installation instructions
  - Usage examples
  - API documentation links
  - Badge status (auto-updated)
- **Status**: ✅ Maintained

### Type Hints
- **Standard**: PEP 484 type hints
- **Required for**:
  - All function signatures
  - Complex variables
  - Return types
- **Validation**: Via mypy
- **Status**: ⚠️ Partial (improving)

## 7. Project Structure

### Package Structure
- ✅ Proper `__init__.py` files
- ✅ Clear module separation
- ✅ Logical package hierarchy
- ✅ No circular imports

### Configuration Files
- ✅ `pyproject.toml` - Modern Python packaging
- ✅ `setup.py` - Legacy compatibility
- ✅ `requirements.txt` - Production dependencies
- ✅ `requirements-dev.txt` - Development dependencies
- ✅ `.gitignore` - Proper exclusions
- ✅ `Makefile` - Task automation

## 8. Git & Version Control

### Commit Standards
- **Format**: Conventional Commits (feat, fix, docs, etc.)
- **Requirements**:
  - Clear, descriptive messages
  - Reference issues when applicable
  - Atomic commits
- **Status**: ✅ Followed

### Branch Protection
- **Main branch**: Protected
- **Requirements**:
  - CI must pass (6 checks)
  - Code review (optional)
- **Status**: ✅ Enforced

## 9. Continuous Integration

### GitHub Actions
- ✅ **Tests**: 4 Python versions (3.10-3.13)
- ✅ **Code Quality**: Flake8, Pylint, Black
- ✅ **Security**: Bandit, Safety
- ✅ **Coverage**: Codecov integration
- ✅ **Build**: Package building
- ⚠️ **Type Check**: Should add MyPy

### Automated Updates
- ✅ Test count badge
- ✅ Coverage badge (Codecov)
- ✅ CI status badge

## 10. Performance

### Test Performance
- ✅ Parallel execution with pytest-xdist
- ✅ Fast tests (<20 seconds for full suite)
- ✅ Optimized CI with caching

### Code Performance
- ✅ Efficient algorithms
- ✅ Proper use of generators
- ✅ Minimal memory footprint
- ✅ Profiling tools available

## 11. Dependencies

### Dependency Management
- ✅ Pinned versions in requirements.txt
- ✅ Minimal dependency count
- ✅ Regular updates via Dependabot
- ✅ Security scanning

### Version Compatibility
- ✅ Python 3.10+ support
- ✅ Tested on multiple Python versions
- ✅ Clear compatibility documentation

## Quick Reference

### Complete Quality Check
```bash
make quality  # Run all quality checks (recommended before commit)
```

### Individual Checks
```bash
make lint          # Flake8 + Pylint
make format        # Black + isort (auto-fix)
make type-check    # MyPy type checking
make test          # All tests with coverage
make security      # Bandit + Safety
```

### CI Replication
```bash
# Run everything CI runs locally:
make quality
make test
make security
```

## Current Status

| Check | Status | Score/Coverage | CI |
|-------|--------|----------------|-----|
| Flake8 | ✅ | Pass | ✅ |
| Pylint | ✅ | 10.00/10 | ✅ |
| Black | ✅ | Formatted | ✅ |
| isort | ✅ | Organized | ✅ |
| MyPy | ⚠️ | Partial | ❌ |
| Tests | ✅ | 321 tests | ✅ |
| Coverage | ✅ | 95.25% | ✅ |
| Bandit | ✅ | No issues | ✅ |
| Safety | ✅ | No CVEs | ✅ |

## 12. Human-Judgment Quality Checks

> **"Is this code something Guido van Rossum would be proud of?"**

These checks require human analysis and cannot be automated. Review before declaring code ready:

### Pythonic Code Quality
- [ ] **Simplicity over Cleverness**: Is the solution straightforward? Could a junior developer understand it?
- [ ] **Explicit is Better Than Implicit**: Are assumptions clear? No hidden magic?
- [ ] **Readability Counts**: Can you understand the code without comments? Are names self-documenting?
- [ ] **One Obvious Way**: Is this the most natural Python approach, or are there simpler alternatives?
- [ ] **Beautiful is Better Than Ugly**: Does the code flow naturally? Is the structure elegant?

### Architecture & Design
- [ ] **Separation of Concerns**: Are responsibilities clearly divided?
- [ ] **DRY (Don't Repeat Yourself)**: Is there unnecessary duplication?
- [ ] **SOLID Principles**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- [ ] **Appropriate Abstractions**: Not too abstract (overengineering), not too concrete (inflexible)
- [ ] **Module Cohesion**: Do modules have a single, clear purpose?
- [ ] **Loose Coupling**: Can components be tested/changed independently?
- [ ] **Naming Consistency**: Do similar things have similar names? Is terminology consistent across the codebase?

### Code Patterns
- [ ] **Appropriate Data Structures**: Using the right tool for the job (dict vs list vs set vs deque)?
- [ ] **Proper Exception Handling**: Catching specific exceptions, not swallowing errors, meaningful error messages
- [ ] **Resource Management**: Using context managers (with statements) for files, connections
- [ ] **Generators vs Lists**: Using generators where appropriate for memory efficiency
- [ ] **Comprehensions**: List/dict comprehensions for clarity, not complexity
- [ ] **Built-ins First**: Using Python's built-in functions before writing custom solutions

### Test Quality
- [ ] **Test Names as Documentation**: Do test names clearly explain what's being tested?
- [ ] **Arrange-Act-Assert**: Is test structure clear?
- [ ] **One Concept Per Test**: Does each test verify one specific behavior?
- [ ] **Test Coverage Philosophy**: Are we testing behavior, not implementation?
- [ ] **Edge Cases**: Do tests cover boundary conditions, empty inputs, error cases?
- [ ] **Test Independence**: Can tests run in any order? No shared state?
- [ ] **Meaningful Assertions**: Do assertions test the right thing with clear failure messages?
- [ ] **Test Fixtures**: Are test fixtures reusable and well-organized (conftest.py)?

### Code Review Questions
- [ ] **Would You Want to Maintain This?**: Imagine coming back to this code in 6 months
- [ ] **Is It Obvious What Changed?**: Can you understand git diffs easily?
- [ ] **Performance Implications**: Are there obvious performance issues? O(n²) where O(n) would work?
- [ ] **Error Handling Completeness**: What happens when things go wrong? Are edge cases handled?
- [ ] **Security Considerations**: User input validated? SQL injection risks? Path traversal?
- [ ] **Memory Leaks**: Are resources properly cleaned up? Large data structures released?
- [ ] **Thread Safety**: If applicable, are shared resources properly protected?

### Documentation Quality
- [ ] **Self-Documenting Code**: Can you understand the code without reading docstrings?
- [ ] **Docstring Necessity**: Are docstrings explaining *why*, not *what* (which should be obvious)?
- [ ] **API Documentation**: Is the public API clearly documented with examples?
- [ ] **README Accuracy**: Does the README match current functionality?
- [ ] **Change Documentation**: Is CHANGELOG updated with meaningful entries?
- [ ] **Type Hints as Documentation**: Do type hints make the API clearer?

### Python Philosophy (The Zen of Python)
- [ ] **Beautiful is better than ugly**
- [ ] **Explicit is better than implicit**
- [ ] **Simple is better than complex**
- [ ] **Complex is better than complicated**
- [ ] **Flat is better than nested**
- [ ] **Sparse is better than dense**
- [ ] **Readability counts**
- [ ] **Special cases aren't special enough to break the rules**
- [ ] **Practicality beats purity**
- [ ] **Errors should never pass silently**
- [ ] **In the face of ambiguity, refuse the temptation to guess**
- [ ] **There should be one-- and preferably only one --obvious way to do it**

### Code Smells to Watch For
- ❌ Functions longer than ~30 lines
- ❌ Classes with more than ~7-10 methods
- ❌ More than 3 levels of indentation
- ❌ Too many function parameters (>5)
- ❌ Boolean flags as parameters (consider separate functions)
- ❌ Mutable default arguments
- ❌ Catching generic `Exception`
- ❌ Using `eval()` or `exec()`
- ❌ Global variables
- ❌ Magic numbers (use named constants)
- ❌ Commented-out code
- ❌ TODO comments without tickets
- ❌ Import * (except in `__init__.py` for API exposure)

### Final "Guido Test" Questions
1. **Would this be accepted in a CPython PR?**
2. **Does this follow Python community best practices?**
3. **Is this code I'd be proud to show other Python developers?**
4. **Could this code be used as a teaching example?**
5. **Does this embody "Pythonic" principles?**

## Improvement Priorities

1. ✅ **Add MyPy to CI** - Complete type checking automation (DONE)
2. **Increase type hint coverage** - Aim for 100% annotation (42 issues to fix)
3. **Documentation generation** - Sphinx docs from docstrings
4. **Performance benchmarks** - Automated performance tracking
5. **Address human-judgment quality issues** - Ongoing code review with "Guido Test"

## Standards References

### Official Python Standards
- **PEP 8**: Style Guide for Python Code
- **PEP 20**: The Zen of Python (`python -m this`)
- **PEP 257**: Docstring Conventions
- **PEP 484**: Type Hints
- **PEP 518**: pyproject.toml specification

### Community Best Practices
- **Google Style Guide**: Python docstrings
- **Effective Python** by Brett Slatkin: 90 ways to write better Python
- **Clean Code** by Robert C. Martin: General principles
- **The Pragmatic Programmer**: Software craftsmanship
- **Conventional Commits**: Commit message format

### Python Community Resources
- **Real Python**: Best practices and patterns
- **Python Patterns**: Design patterns in Python
- **The Hitchhiker's Guide to Python**: Best practices guide
