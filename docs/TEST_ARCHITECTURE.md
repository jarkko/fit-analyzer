# Test Architecture Guidelines

## Philosophy

Tests should be **architecture-driven, not bug-driven**. This means:
- Tests document and enforce function contracts
- Coverage is based on code structure (parameters, branches, edge cases)
- New features come with complete test coverage from day one
- Bug fixes add missing contract tests, not just regression tests

## Test Organization Principles

### 1. Contract-Based Testing

Every public function must have tests that cover its **contract**:

```python
def function_name(param1: Type1, param2: Optional[Type2] = None) -> ReturnType:
    """Function description."""
```

Required tests:
- **Parameter contracts**: Test all meaningful parameter values
  - `None` vs non-`None` for Optional parameters
  - Empty collections (`[]`, `{}`) vs populated
  - Boundary values (0, negative, max)
  - Invalid inputs (wrong type, out of range)
- **Return value contracts**: Verify all possible return values/states
- **Side effects**: Verify all state changes (files, database, etc.)
- **Error contracts**: Test all documented exceptions

### 2. Parameter Matrix Testing

For each parameter, test the complete behavior matrix:

**Example: `run_analysis(directory, output_dir, updated_files, **kwargs)`**

| updated_files | Expected Behavior | Test Name |
|---------------|-------------------|-----------|
| `None` | Analyze all files in directory | `test_updated_files_none_analyzes_all` |
| `[]` | Skip analysis (nothing to do) | `test_updated_files_empty_skips` |
| `["file.fit"]` | Analyze only that file | `test_updated_files_single_file` |
| `["f1.fit", "f2.fit"]` | Analyze both files | `test_updated_files_multiple` |
| `["missing.fit"]` | Handle gracefully | `test_updated_files_nonexistent` |

### 3. Test File Organization

Organize tests by **architectural layer**, not by bug history:

```
tests/
├── unit/                          # Pure function tests
│   ├── test_metrics.py           # Calculation functions
│   └── test_parser.py            # Parsing logic
├── integration/                   # Component integration
│   ├── test_sync_workflow.py    # Full sync pipeline
│   └── test_analysis_pipeline.py # Analysis workflow
├── contract/                      # Public API contracts
│   ├── test_run_analysis_contract.py
│   ├── test_download_contract.py
│   └── test_sync_activities_contract.py
└── edge_cases/                    # Edge cases & error paths
    ├── test_malformed_files.py
    └── test_network_errors.py
```

### 4. Test Naming Convention

Use descriptive names that document the contract:

```python
# ❌ Bad: Bug-driven names
def test_fix_duplicate_bug():
def test_issue_123():

# ✅ Good: Contract-driven names
def test_empty_updated_files_skips_analysis():
def test_none_updated_files_analyzes_all():
def test_duplicate_files_deduped_by_file_column():
```

## Coverage Requirements

### Branch Coverage

Every control flow branch must be tested:

```python
def example(value: Optional[int], flag: bool) -> str:
    if value is None:           # Branch 1: Test with value=None
        return "none"
    elif value < 0:             # Branch 2: Test with value=-1
        return "negative"
    elif flag:                  # Branch 3: Test with value=5, flag=True
        return "flagged"
    else:                       # Branch 4: Test with value=5, flag=False
        return "normal"
```

Required: 4 tests minimum to cover all branches.

### Edge Case Coverage

Test edge cases for every function:

**Numeric parameters:**
- Zero
- Negative values
- Boundary values (min/max)
- Very large values

**Collections:**
- Empty (`[]`, `{}`, `""`)
- Single element
- Many elements
- Duplicates

**Optional parameters:**
- `None`
- Default value
- Explicit value

**Files/Paths:**
- Non-existent
- Empty file
- Malformed content
- Read-only
- Very large files

### Error Path Coverage

Test all error conditions:

```python
def process_file(path: str) -> dict:
    """Process a file.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is malformed
        PermissionError: If file not readable
    """
```

Required tests:
- `test_process_file_not_found` → FileNotFoundError
- `test_process_file_malformed` → ValueError
- `test_process_file_permission_denied` → PermissionError
- `test_process_file_success` → Happy path

## Property-Based Testing

For complex logic, use property-based testing (hypothesis):

```python
from hypothesis import given
from hypothesis.strategies import integers, lists

@given(lists(integers()))
def test_deduplication_preserves_order(items):
    """Deduplication should preserve original order."""
    result = deduplicate(items)
    # Property: result should be a subsequence of items
    assert all(items.index(x) <= items.index(y)
               for x, y in zip(result, result[1:]))
```

## Test Documentation

Every test file should start with a contract specification:

```python
"""Contract tests for run_analysis() function.

Function Signature:
    run_analysis(
        directory: str = ".",
        output_dir: str = "data",
        updated_files: Optional[List[str]] = None,
        **kwargs
    ) -> bool

Parameter Contracts:
    updated_files:
        - None: Analyze all FIT files in directory
        - []: Skip analysis (no files to process)
        - [...files]: Analyze only specified files

Return Value:
    - True: Analysis completed successfully
    - False: Analysis failed (no files found, errors)

Side Effects:
    - Creates CSV files in output_dir
    - Calls CLI module with parsed arguments
"""
```

## Anti-Patterns to Avoid

### ❌ Bug-Driven Test Files
```python
# tests/test_duplicate_bug.py
# tests/test_issue_456.py
# tests/test_hotfix.py
```
These accumulate technical debt. Instead, move tests to contract files.

### ❌ Testing Implementation Details
```python
def test_uses_pandas_dataframe():  # Too specific
    assert isinstance(result, pd.DataFrame)
```
Test behavior, not implementation.

### ❌ Missing Edge Cases
```python
def test_function_works():
    result = function([1, 2, 3])
    assert result == 6
# Missing: empty list, single item, negative numbers, etc.
```

### ❌ Incomplete Parameter Testing
```python
# Only tests default value
def test_run_analysis():
    run_analysis(directory=".")
# Missing: all other parameter combinations
```

## Migration Strategy

For existing test files:

1. **Audit**: Identify missing contract tests using parameter matrix
2. **Add**: Write missing contract tests
3. **Reorganize**: Move tests to appropriate contract files
4. **Deprecate**: Mark bug-specific test files for removal after migration
5. **Document**: Update this document with patterns and examples

## Tools & Metrics

- **Coverage target**: 100% branch coverage
- **Property testing**: Use hypothesis for complex logic
- **Mutation testing**: Consider using mutmut to verify test quality
- **Test review**: Include contract coverage in code review checklist

## Architecture Audit (October 31, 2025)

### Test Suite Status
- **Total Tests**: 245 passing
- **Architecture Compliance**: 100% ✅
- **Bug-Driven Tests**: 0 (removed)

### Audit Findings

**✅ Architecture-Compliant Files (245 tests):**
- `test_sync.py` (44 tests) - Organized by function: authentication, download, analysis
- `test_parser.py` (65 tests) - Organized by calculation: NP, TRIMP, session processing
- `test_sync_programmatic.py` (8 tests) - Contract tests for `sync_activities()`
- `test_metrics.py` (24 tests) - Organized by metric type
- `test_strength_training.py` (30 tests) - Feature-based organization
- `test_api_exercise_names.py` (13 tests) - API operation contracts
- `contract/test_run_analysis_contract.py` (14 tests) - Parameter matrix for `run_analysis()`
- All other test files follow architecture-driven principles

**❌ Removed Bug-Driven Files:**
- `test_sync_duplicate_bug.py` (3 tests) - **REMOVED** on Oct 31, 2025
  - Tests were redundant with contract tests
  - Classic anti-pattern: named after bug, not architecture
  - Functionality now covered by parameter matrix in contract tests

### Key Strengths
1. 89% of tests were already architecture-driven before cleanup
2. Excellent organization by feature/function in most files
3. Good parameter coverage for edge cases
4. Clear separation of unit vs integration tests

### Recommendations Implemented
1. ✅ Created comprehensive TEST_ARCHITECTURE.md guidelines
2. ✅ Implemented example contract tests for `run_analysis()`
3. ✅ Removed bug-driven test file
4. ✅ Updated AI_INSTRUCTIONS.md with architecture principles
5. ✅ Achieved 100% architecture compliance

### Future Considerations
- Consider contract tests for `_save_workout_summary()` deduplication behavior
- Monitor for new bug-driven tests and migrate immediately
- Continue parameter matrix approach for all new functions

## References

- [Contract-Driven Testing](https://www.hillelwayne.com/post/contract-examples/)
- [Property-Based Testing with Hypothesis](https://hypothesis.readthedocs.io/)
- [Test Desiderata](https://kentbeck.github.io/TestDesiderata/)
