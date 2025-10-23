# FIT Analyzer - Quick Start Guide

## What is this?

This tool helps you analyze your workout FIT files from Garmin devices. It can:
- Sync activities directly from Garmin Connect
- Parse FIT files and extract training metrics
- Calculate training stress, normalized power, and heart rate zones
- Handle multisport activities (triathlon, etc.)
- Generate CSV summaries of your workouts

## Quick Usage

### Option 1: Sync from Garmin Connect (Recommended)

Automatically download new activities and analyze them:

```bash
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

### Option 2: Analyze Local FIT Files

If you already have FIT files:

```bash
make analyze
```

Or with specific files:

```bash
./analyze.py data/samples/*.fit --ftp 300 --multisport
```

### Method 2: Direct Python Script

```bash
# Analyze specific files
.venv/bin/python3 -m fitanalyzer.parser data/samples/*.fit --ftp 300 --multisport

# Analyze with custom parameters
.venv/bin/python3 -m fitanalyzer.parser data/samples/*.fit \
    --ftp 300 \
    --hrrest 50 \
    --hrmax 190 \
    --multisport
```

### Method 3: Python Library

Use it as a Python library in your own scripts:

```python
from fitanalyzer import summarize_fit_original, summarize_fit_sessions
import pandas as pd

# For single-sport activities
summary, sets = summarize_fit_original("activity.fit", ftp=300)

# For multisport activities
sessions, _ = summarize_fit_sessions("triathlon.fit", ftp=300)

# Generate CSV
rows = []
for fit_file in fit_files:
    sessions, _ = summarize_fit_sessions(fit_file, ftp=300)
    rows.extend(sessions)

df = pd.DataFrame(rows)
df.to_csv("my_workouts.csv", index=False)
```

## Command-Line Options

```
./analyze.py <FIT_FILES> --ftp <VALUE> [OPTIONS]

Required:
  FIT_FILES          One or more .fit files to analyze
  --ftp VALUE        Your Functional Threshold Power in watts

Optional:
  --multisport       Process multisport activities by session
  --hrrest VALUE     Resting heart rate (default: 50)
  --hrmax VALUE      Maximum heart rate (default: 190)
  --tz TIMEZONE      Timezone name (default: Europe/Helsinki)
  --dump-sets        Save strength training sets to separate CSV files
```

## Common Workflows

### Update Your Workout Summary

```bash
# 1. Put new FIT files in data/samples/
cp ~/Downloads/*.fit data/samples/

# 2. Run analysis
make analyze

# 3. Check the output
cat workout_summary_from_fit.csv
```

### Sync from Garmin Connect

```bash
# First time setup
.venv/bin/python3 -c "from fitanalyzer import setup_credentials; setup_credentials()"

# Sync activities
.venv/bin/python3 -m fitanalyzer.sync --days 30

# Then analyze them
make analyze
```

### Custom Analysis Script

Create your own analysis script:

```python
#!/usr/bin/env python3
from fitanalyzer import summarize_fit_sessions
from pathlib import Path
import pandas as pd

fit_files = Path("data/samples").glob("*.fit")
rows = []

for fit_file in fit_files:
    sessions, _ = summarize_fit_sessions(str(fit_file), ftp=300)
    rows.extend(sessions)

df = pd.DataFrame(rows)

# Filter only cycling activities
cycling = df[df['sport'] == 'cycling']
print(f"Total cycling time: {cycling['duration_min'].sum()} minutes")
print(f"Average TSS: {cycling['TSS'].mean():.1f}")

# Save
cycling.to_csv("cycling_summary.csv", index=False)
```

## Output Format

The generated CSV contains:

- `file`: Activity filename
- `sport`: Sport type (cycling, running, training, etc.)
- `sub_sport`: Sub-sport (indoor_cycling, strength_training, etc.)
- `date`: Activity date
- `start_time`: Start timestamp
- `end_time`: End timestamp
- `duration_min`: Duration in minutes
- `avg_hr`: Average heart rate
- `max_hr`: Maximum heart rate
- `avg_power_w`: Average power (watts)
- `max_power_w`: Maximum power (watts)
- `np_w`: Normalized Power
- `IF`: Intensity Factor
- `TSS`: Training Stress Score
- `TRIMP`: Training Impulse (heart rate based load)

## Troubleshooting

### "Command not found: ./analyze.py"

Make it executable:
```bash
chmod +x analyze.py
```

### "No module named 'fitparse'"

Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

Or install dependencies:
```bash
make install-dev
```

### "No FIT files found"

Make sure your FIT files are in `data/samples/`:
```bash
ls data/samples/*.fit
```

## Getting Help

```bash
# Show help
./analyze.py --help

# Run example
make run-example

# Run tests
make test
```
