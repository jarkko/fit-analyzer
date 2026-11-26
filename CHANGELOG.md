# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5] - 2026-11-26

### Added

- `fitanalyzer.wellness`: daily wellness sync from Garmin Connect
  - `fetch_daily_wellness(start, end)` returns one row per day with overnight HRV
    (last-night and weekly average, status), resting heart rate, sleep duration and sleep score
  - `sync_wellness(csv_path, days)` upserts those rows by date into a CSV. Missing values never
    overwrite stored ones, extra columns are kept, and the file is written atomically (through
    a symlink, if the CSV is one; a new file gets the usual umask-based permissions)
  - A missing day or a failed request for one day yields empty values (logged as a warning)
  - Raises `WellnessSyncError` when a whole source (HRV, resting HR or sleep) fails for every
    day, or when an existing CSV is unreadable; `WellnessAuthError` when the garth session is
    not authenticated or its token refresh is rejected; 429 responses are re-raised. Days
    fetched before the error are written to the CSV first (`fetch_daily_wellness` reports them
    through its `progress` list)
  - Waits 0.3 s between per-day requests (`request_delay`) to avoid rate limiting

## [0.2.0] - 2025-11-01

### Added

- **Training Load Metrics (CTL/ATL/TSB)** - Comprehensive fitness and fatigue tracking based on the Banister Fitness-Fatigue Model
  - CTL (Chronic Training Load): 42-day exponentially weighted fitness metric
  - ATL (Acute Training Load): 7-day exponentially weighted fatigue metric
  - TSB (Training Stress Balance): Form indicator (CTL - ATL)
  - Automatic calculation in workout CSV output
  - Smart fallback: uses TSS when available, TRIMP otherwise
  - Values rounded to 4 decimal places for readability
- New module: `fitanalyzer.training_load` with 100% test coverage
- 21 comprehensive contract tests for training load calculations

### Changed

- Workout summary CSV now includes three new columns: `ctl`, `atl`, `tsb`
- Improved test coverage from 95.13% to 95.25%
- Enhanced CLI to gracefully handle missing training load data

### Technical

- Test suite expanded: 300 → 321 tests (all passing)
- Maintained 10.00/10 pylint score
- Full TDD approach: tests written before implementation
- Scientific basis: Banister et al. (1975), Coggan (2003)

## [0.1.0] - 2025-10-31

### Initial Release

- FIT file parsing and analysis
- Session-based metrics calculation
- Strength training set extraction
- Incremental analysis with file caching
- Garmin Connect synchronization
- Comprehensive test suite with 95%+ coverage
