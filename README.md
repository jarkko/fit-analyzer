# FIT Analyzer

A professional Python library for analyzing Garmin FIT files and calculating training metrics.

## Features

- 📊 **Parse FIT Files**: Extract comprehensive data from Garmin FIT activity files
- 🏃 **Training Metrics**: Calculate NP (Normalized Power), TSS, TRIMP, and IF
- 🔄 **Garmin Sync**: Automated syncing from Garmin Connect
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
