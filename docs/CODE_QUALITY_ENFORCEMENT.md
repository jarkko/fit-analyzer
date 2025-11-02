# Code Quality Enforcement

**Version**: 0.4.2+
**Last Updated**: November 2, 2025

This document describes the automated code quality rules enforced by our linting tools. These rules were established after the v0.4.2 refactoring and prevent quality regressions.

---

## Overview

All code quality checks are enforced via:
- **Pylint** - Comprehensive static analysis
- **Flake8** - PEP 8 compliance + complexity checks
- **MyPy** - Static type checking
- **Black** - Code formatting
- **isort** - Import sorting

Run all checks with: `make pre-commit`

---

## Function Complexity Limits

### Function Length
**Enforced by**: Flake8 (flake8-functions plugin)
**Rule**: `max-function-length = 70`
**Error Code**: CFQ001

After v0.4.2 refactoring, the longest function is 63 lines. The limit of 70 lines provides a small buffer while preventing large functions from being added.

**Why**: Long functions are harder to understand, test, and maintain. Extract helper functions when approaching this limit.

### Cyclomatic Complexity
**Enforced by**: Flake8 (McCabe)
**Rule**: `max-complexity = 10`
**Error Code**: C901

Measures the number of independent paths through code (branches, loops, etc.).

**Why**: High complexity indicates code that's difficult to test and prone to bugs.

### Maximum Branches
**Enforced by**: Pylint
**Rule**: `max-branches = 12`
**Error Code**: R0912

Counts if/elif/else branches in a function.

**Why**: Too many branches make code hard to follow. Consider extracting decision logic.

### Maximum Statements
**Enforced by**: Pylint
**Rule**: `max-statements = 50`
**Error Code**: R0915

Counts the total number of statements in a function body.

**Why**: Complements function length limits. Forces decomposition of complex logic.

### Nesting Depth
**Enforced by**: Pylint
**Rule**: `max-nested-blocks = 3`
**Error Code**: R1702

Maximum nesting level (if, for, while, with, try blocks).

**Why**: Deep nesting is hard to read. Use early returns and guard clauses instead.

**Example - Before (depth 4)**:
```python
def process_data(data):
    if data:
        if data.valid:
            if data.ready:
                if data.complete:
                    return process(data)
```

**Example - After (depth 1)**:
```python
def process_data(data):
    if not data:
        return None
    if not data.valid:
        return None
    if not data.ready:
        return None
    if not data.complete:
        return None
    return process(data)
```

---

## Function Parameters

### Maximum Arguments
**Enforced by**: Pylint
**Rule**: `max-args = 5`
**Error Code**: R0913

Maximum number of function arguments (positional + keyword).

**Why**: Too many parameters indicate poor abstraction. Consider:
- Using a configuration object
- Breaking into smaller functions
- Using **kwargs for optional parameters

### Maximum Positional Arguments
**Enforced by**: Pylint
**Rule**: `max-positional-arguments = 5`
**Error Code**: R0917

Maximum number of positional-only arguments.

**Why**: Forces use of keyword arguments for clarity at call sites.

### flake8-functions Parameters
**Enforced by**: Flake8 (flake8-functions plugin)
**Rule**: `max-parameters-amount = 7`
**Error Code**: CFQ002

Slightly more lenient than Pylint to allow helper functions with context.

---

## Return Statements

### Maximum Returns
**Enforced by**: Pylint
**Rule**: `max-returns = 6`
**Error Code**: R0911

Maximum number of return statements in a function.

**Why**: Too many returns can indicate complex control flow. However, early returns are encouraged for guard clauses.

### flake8-functions Returns
**Enforced by**: Flake8 (flake8-functions plugin)
**Rule**: `max-returns-amount = 8`
**Error Code**: CFQ004

More lenient to support early return pattern (guard clauses).

---

## Local Variables

### Maximum Local Variables
**Enforced by**: Pylint
**Rule**: `max-locals = 15`
**Error Code**: R0914

Maximum number of local variables in a function.

**Why**: Too many variables suggest the function is doing too much.

---

## Boolean Expressions

### Maximum Boolean Expressions
**Enforced by**: Pylint
**Rule**: `max-bool-expr = 5`
**Error Code**: R0916

Maximum number of boolean expressions in a single condition.

**Why**: Complex boolean logic is error-prone. Extract to named variables or helper functions.

**Example - Before**:
```python
if user.active and user.verified and not user.banned and user.age >= 18 and user.country == 'US':
```

**Example - After**:
```python
is_eligible_user = (
    user.active and
    user.verified and
    not user.banned and
    user.age >= 18 and
    user.country == 'US'
)
if is_eligible_user:
```

---

## Class Design

### Maximum Attributes
**Enforced by**: Pylint
**Rule**: `max-attributes = 7`
**Error Code**: R0902

Maximum number of instance attributes in a class.

**Why**: Too many attributes suggest the class has too many responsibilities.

### Maximum Public Methods
**Enforced by**: Pylint
**Rule**: `max-public-methods = 20`
**Error Code**: R0904

Maximum number of public methods in a class.

**Why**: Large classes are hard to maintain. Consider splitting responsibilities.

---

## Type Checking

### MyPy Strict Mode
**Enforced by**: MyPy
**Configuration**: See `[tool.mypy]` in `pyproject.toml`

Key rules:
- `disallow_untyped_defs = true` - All functions must have type annotations
- `disallow_incomplete_defs = true` - All parameters and returns must be typed
- `disallow_any_generics = true` - Generic types must specify type parameters
- `warn_return_any = true` - Warn when returning `Any`
- `strict_optional = true` - `Optional[]` must be explicit
- `warn_redundant_casts = true` - Catch unnecessary type casts
- `warn_unused_ignores = true` - Catch outdated `# type: ignore` comments

**Why**: Static typing catches bugs at development time and improves code documentation.

---

## Code Style

### Line Length
**Enforced by**: Black, Flake8, Pylint
**Rule**: `max-line-length = 100`

**Why**: 100 characters balances readability with modern displays.

### Import Sorting
**Enforced by**: isort
**Configuration**: `profile = "black"`

**Why**: Consistent import ordering improves readability and reduces merge conflicts.

---

## Testing Requirements

### Coverage Threshold
**Enforced by**: pytest-cov
**Rule**: `fail_under = 95.0`

Current coverage: **97.72%**

**Why**: High test coverage catches regressions and documents expected behavior.

### Test Execution Speed
**Target**: < 10 seconds (currently ~10s with 490 tests)

Achieved via:
- Parallel execution (`pytest-xdist`)
- Mocked external dependencies
- Focused unit tests

---

## Security

### Bandit
**Enforced by**: `make security`
**Status**: 0 security issues

Checks for common security issues:
- Hardcoded passwords
- SQL injection vulnerabilities
- Use of `eval()` or `exec()`
- Insecure random number generation
- And more...

### Dependency Scanning
**Enforced by**: Safety
**Status**: 0 known CVEs

Scans dependencies for known security vulnerabilities.

---

## Disabling Rules

### When to Disable

Rules should **rarely** be disabled. Valid cases include:

1. **Helper functions with many context parameters**
   ```python
   def _build_result(  # pylint: disable=too-many-arguments
       arg1, arg2, arg3, arg4, arg5, arg6, arg7
   ):
   ```

2. **Functions using early return pattern**
   ```python
   def validate(data):  # pylint: disable=too-many-return-statements
       if not data: return False
       if not data.valid: return False
       # ... more guard clauses
   ```

3. **Test files** - Already have per-file ignores in `.flake8`

### How to Disable

**Inline (preferred)**:
```python
def my_function():  # pylint: disable=rule-name
    # flake8: noqa: CODE
```

**Per-file** (in `.flake8`):
```ini
per-file-ignores =
    test_*.py:E501,F401
```

**Global** (last resort, requires team discussion):
Add to `pyproject.toml` `[tool.pylint.messages_control]` or `.flake8` `ignore`

---

## Continuous Integration

All checks run automatically on:
- **Pre-commit hook** - Locally before commit (via `make pre-commit`)
- **Pull requests** - CI/CD pipeline runs full suite
- **Main branch** - Protected by required status checks

**Builds will fail if**:
- Pylint score < 10.00
- MyPy finds type errors
- Flake8 finds violations
- Test coverage < 95%
- Any tests fail

---

## Making Changes to These Rules

### Process

1. **Propose**: Open an issue explaining why the rule should change
2. **Discuss**: Team reviews impact and alternatives
3. **Update**: Modify configuration files
4. **Document**: Update this file
5. **Verify**: Run `make pre-commit` to ensure no violations
6. **Commit**: Include rationale in commit message

### Configuration Files

- **Pylint**: `pyproject.toml` → `[tool.pylint.*]`
- **Flake8**: `.flake8` → `[flake8]`
- **MyPy**: `pyproject.toml` → `[tool.mypy]`
- **Black**: `pyproject.toml` → `[tool.black]`
- **isort**: `pyproject.toml` → `[tool.isort]`
- **pytest**: `pyproject.toml` → `[tool.pytest.ini_options]`
- **Coverage**: `pyproject.toml` → `[tool.coverage.*]`

---

## Quick Reference

| Metric | Limit | Tool | Code |
|--------|-------|------|------|
| Function length | ≤70 lines | flake8-functions | CFQ001 |
| Cyclomatic complexity | ≤10 | flake8 (McCabe) | C901 |
| Max branches | ≤12 | pylint | R0912 |
| Max statements | ≤50 | pylint | R0915 |
| Nesting depth | ≤3 levels | pylint | R1702 |
| Function arguments | ≤5 (pylint), ≤7 (flake8) | pylint, flake8 | R0913, CFQ002 |
| Return statements | ≤6 (pylint), ≤8 (flake8) | pylint, flake8 | R0911, CFQ004 |
| Local variables | ≤15 | pylint | R0914 |
| Boolean expressions | ≤5 | pylint | R0916 |
| Class attributes | ≤7 | pylint | R0902 |
| Public methods | ≤20 | pylint | R0904 |
| Test coverage | ≥95% | pytest-cov | - |
| Line length | ≤100 chars | black, flake8 | E501 |

---

## Resources

- [Pylint Documentation](https://pylint.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Black Documentation](https://black.readthedocs.io/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

**Remember**: These rules exist to maintain code quality and prevent technical debt. If you find yourself frequently fighting a rule, it may indicate a design issue rather than a configuration issue.
