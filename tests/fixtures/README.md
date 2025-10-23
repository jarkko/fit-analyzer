# Test Fixtures

This directory contains FIT files used for deterministic integration testing.

## Files

### 20548472357_ACTIVITY.fit
- **Sport:** Volleyball
- **Date:** 2025-09-30
- **Duration:** 134.6 minutes
- **Avg HR:** 118.1 bpm
- **Max HR:** 163 bpm
- **TRIMP:** 112.9
- **Power:** None (HR-only activity)
- **Purpose:** Tests single-sport activity parsing, HR metrics, TRIMP calculation

### 20744294788_ACTIVITY.fit
- **Type:** Multisport activity
- **Sessions:**
  1. **Cycling** (indoor_cycling): 10.0 min, HR 114.7 bpm
  2. **Training** (strength_training): 64.5 min, HR 114.4 bpm
- **Purpose:** Tests multisport session separation, session parsing

### 20747700969_ACTIVITY.fit
- **Sport:** Cycling
- **Date:** 2025-10-20
- **Duration:** 30.0 minutes
- **Avg HR:** 112.7 bpm
- **TRIMP:** 20.4
- **Purpose:** Tests short activity parsing, used in multi-file tests

## Usage

These fixtures are used by `tests/test_integration.py` for:
- Deterministic testing with exact value assertions
- Fast test execution (no dependency on large sample directory)
- Reproducible test results across environments

## Test Parameters

All tests use consistent parameters:
- **FTP:** 300 watts
- **HR Rest:** 50 bpm
- **HR Max:** 190 bpm
- **Timezone:** Europe/Helsinki

## Adding New Fixtures

When adding new test fixtures:
1. Copy the FIT file to this directory
2. Document its characteristics in this README
3. Add corresponding tests with exact value assertions in `test_integration.py`
4. Keep fixtures small (< 200KB) for fast test execution
