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

### 20684859222_ACTIVITY.fit
- **Sport:** Cycling (indoor_cycling)
- **Date:** 2025-10-14
- **Duration:** 30.0 minutes
- **Avg HR:** 116.0 bpm
- **Avg Power:** 184.8 watts
- **Avg Speed:** 7.58 m/s (27.30 kph)
- **Avg Cadence:** 85.7 rpm
- **Distance:** 13,611.8 meters
- **Purpose:** Tests speed/cadence/distance metrics for cycling activities

### 20544585388_ACTIVITY.fit
- **Sport:** Running (trail)
- **Date:** 2025-09-30
- **Duration:** 26.0 minutes
- **Avg HR:** 128.5 bpm
- **Avg Power:** 320.9 watts
- **Avg Speed:** 2.65 m/s (9.54 kph)
- **Elevation:** Ascent 115.0m, Descent 105.2m, Avg 120.7m
- **Distance:** 4,102.9 meters
- **Purpose:** Tests elevation metrics (GPS/altitude data) for running activities

### 20794985860_ACTIVITY.fit
- **Sport:** Training (strength_training)
- **Date:** 2025-10-25
- **Duration:** 47.5 minutes
- **Avg HR:** 92.4 bpm
- **Purpose:** Tests activities without GPS/elevation data (indoor strength training)

## Adding New Fixtures

When adding new test fixtures:
1. Copy the FIT file to this directory
2. Document its characteristics in this README
3. Add corresponding tests with exact value assertions in `test_integration.py`
4. Keep fixtures small (< 200KB) for fast test execution
