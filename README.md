# FIT Analyzer

[![CI](https://github.com/jarkko/fit-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/jarkko/fit-analyzer/actions/workflows/ci.yml)
[![Coverage](https://github.com/jarkko/fit-analyzer/actions/workflows/coverage.yml/badge.svg)](https://github.com/jarkko/fit-analyzer/actions/workflows/coverage.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code style: pylint](https://img.shields.io/badge/code%20quality-10.00%2F10-brightgreen)](https://pylint.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional Python library for analyzing Garmin FIT files and calculating training metrics.

## Features

- 📊 **Parse FIT Files**: Extract comprehensive data from Garmin FIT activity files
- 🏃 **Training Metrics**: Calculate NP (Normalized Power), TSS, TRIMP, and IF
- � **Strength Training**: Extract sets, weights, repetitions, and exercise categories
- �🔄 **Garmin Sync**: Automated syncing from Garmin Connect
- 🎯 **Multisport Support**: Handle complex multisport activities
- 📈 **CSV Export**: Generate training summaries for analysis

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/fitanalyzer.git
cd fitanalyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
make install-dev
make install-editable
```

### As a Package

```bash
pip install fitanalyzer
```

## Quick Start

### Daily Workflow: Sync from Garmin Connect

The simplest way to keep your workout summary up to date:

```bash
# Download new activities and analyze them
make sync
```

Or with custom options:

```bash
./sync.py --email your@email.com --days 30 --ftp 300
```

This will:
1. Connect to Garmin Connect and download new activities
2. Save FIT files to `data/samples/`
3. Analyze all activities and update `workout_summary_from_fit.csv`

### Analyze Local FIT Files

If you already have FIT files:

```bash
# Analyze all FIT files in data/samples/
make analyze
```

Or specify files:

```bash
./analyze.py data/samples/*.fit --ftp 300 --multisport
```

### Use as a Python Library

```python
from fitanalyzer import summarize_fit_original

# Analyze a single-session activity
summary, sets = summarize_fit_original("activity.fit", ftp=300)

print(f"Sport: {summary['sport']}")
print(f"Duration: {summary['duration_min']} min")
print(f"Avg HR: {summary['avg_hr']} bpm")
print(f"TSS: {summary['TSS']}")
```

### Handle Multisport Activities

```python
from fitanalyzer import summarize_fit_sessions

# Analyze multisport (triathlon, etc.)
sessions = summarize_fit_sessions("triathlon.fit", ftp=300)

for session in sessions:
    print(f"{session['sport']}: {session['duration_min']} min")
```

### Sync from Garmin Connect

```python
from fitanalyzer import setup_garmin_client, sync_activities

# First time setup
setup_garmin_client()

# Sync new activities
sync_activities(output_dir="./activities", days=30)
```

### Extract Strength Training Data

FIT Analyzer can extract detailed strength training sets including weights, repetitions, and exercise categories:

```bash
# Extract strength sets from FIT files
./analyze.py data/samples/*.fit --ftp 300 --dump-sets
```

This generates two outputs:

1. **Individual CSV files** (one per workout): `ACTIVITY_ID_strength_sets.csv`
   - Contains all sets (active + rest periods)
   - Includes detailed exercise categories and timing

2. **Consolidated summary**: `strength_training_summary.csv`
   - Aggregates all active sets from multiple workouts
   - Columns: `activity_id`, `file`, `date`, `sport`, `sub_sport`, `set_number`, `set_type`, `exercise_name`, `category`, `category_subtype`, `repetitions`, `weight`, `duration`, `timestamp`
   - `exercise_name` uses **two-level naming system**:
     * **Specific names** when your watch records detailed exercise data: "Barbell Power Clean", "Ghd Back Extensions", "Single Arm Neutral Grip Dumbbell Row"
     * **Category names** as fallback when only high-level data available: "Olympic Lift", "Hyperextension", "Row"
     * See [EXERCISE_MAPPINGS.md](EXERCISE_MAPPINGS.md) for complete mapping of 53 categories and 1,846+ specific exercises
   - Perfect for tracking strength training progress over time

Example output:
```
Yhteensä 72 strength training settiä 3 treenikerrasta.

activity_id  date        repetitions  weight  duration
20474406937  2025-09-23           10    14.0    26.207
20474406937  2025-09-23            5     0.0    33.812
20555050352  2025-10-01           10    14.0    27.049
...
```

**Usage in Python:**

```python
from fitanalyzer import summarize_fit_original

# Get both workout summary and strength sets
summary, strength_sets = summarize_fit_original("workout.fit", ftp=300)

if strength_sets is not None:
    print(f"Found {len(strength_sets)} sets")
    print(strength_sets[['repetitions', 'weight', 'duration']])
```

## Project Structure

```
fitanalyzer/
├── src/
│   └── fitanalyzer/          # Main package
│       ├── __init__.py
│       ├── parser.py          # FIT file parsing and metrics
│       ├── sync.py            # Garmin Connect integration
│       └── credentials.py     # Authentication setup
├── tests/                     # Test suite
│   ├── conftest.py
│   ├── test_parser.py
│   ├── test_sync.py
│   └── test_integration.py
├── examples/                  # Example scripts
│   └── analyze_fit.py
├── data/                      # Data files
│   └── samples/              # Sample FIT files
├── docs/                     # Documentation
├── setup.py                  # Package configuration
├── pyproject.toml           # Build configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
└── Makefile                 # Development commands
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration

# Generate coverage report
make coverage
```

### Code Quality

```bash
# Run linters
make lint

# Auto-format code
make format

# Type checking
make type-check
```

### Available Commands

Run `make help` to see all available commands:

```
FIT Analyzer - Professional Python Library
===========================================

Setup:
  make install           Install production dependencies
  make install-dev       Install development dependencies
  make install-editable  Install package in editable mode

Testing:
  make test              Run all tests
  make test-unit         Run unit tests only
  make test-integration  Run integration tests
  make coverage          Generate coverage report

Code Quality:
  make lint              Run all linters
  make format            Auto-format code
  make type-check        Run type checking

Build & Distribution:
  make build             Build distribution packages
  make clean             Remove build artifacts

Examples:
  make run-example       Run example script
```

## Training Metrics

### Normalized Power (NP)

Provides a better representation of the true physiological demands of a workout than average power.

### Intensity Factor (IF)

The ratio of Normalized Power to FTP (Functional Threshold Power):
- IF < 0.75: Recovery rides
- IF 0.75-0.85: Endurance rides
- IF 0.85-0.95: Tempo rides
- IF 0.95-1.05: Threshold rides
- IF > 1.05: VO2max and anaerobic work

### Training Stress Score (TSS)

Quantifies the training load of a workout:
- TSS < 150: Low training load
- TSS 150-300: Medium training load
- TSS 300-450: High training load
- TSS > 450: Very high training load

### TRIMP (Training Impulse)

Heart rate-based training load metric for activities without power data.

## Requirements

- Python 3.8+
- fitparse >= 1.2.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- garth >= 0.5.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [fitparse](https://github.com/dtcooper/python-fitparse) for FIT file parsing
- Uses [garth](https://github.com/matin/garth) for Garmin Connect API
- Inspired by the training analysis tools in the cycling community
