# Programmatic Sync Example

## Quick Start

The simplest way to use incremental sync from Python:

```python
from fitanalyzer import sync_activities

# Sync last 7 days (incremental - only downloads new/changed activities)
result = sync_activities(days=7)

if result["success"]:
    print(f"Downloaded {result['new_activities']} new activities")
    print(f"CSV: {result['csv_path']}")
```

## Full Example

```python
from fitanalyzer import sync_activities

result = sync_activities(
    email="your@email.com",  # Or use GARMIN_EMAIL env var
    password="your_password",  # Or use GARMIN_PASSWORD env var
    days=30,  # Sync last 30 days
    directory="./activities",  # Where to store FIT files
    output_dir="./output",  # Where to save CSVs
    ftp=280,  # Your FTP in watts
    hrrest=55,  # Your resting HR
    hrmax=185,  # Your max HR
    multisport=True,  # Handle multisport activities
)

if result["success"]:
    print(f"✅ Success! Downloaded {result['new_activities']} activities")

    # Use the data
    import pandas as pd
    df = pd.read_csv(result["csv_path"])
    print(f"Total workouts: {len(df)}")
else:
    print(f"❌ Error: {result['error']}")
```

## Key Features

- **Incremental by default**: Only downloads new or changed activities
- **Safe to call frequently**: Won't re-download unchanged files
- **Simple return value**: Dict with success status and file paths
- **Environment variable support**: Credentials from GARMIN_EMAIL/GARMIN_PASSWORD

## Running the Example

```bash
# Set credentials (recommended)
export GARMIN_EMAIL="your@email.com"
export GARMIN_PASSWORD="your_password"

# Run the example
python examples/sync_programmatically.py
```

## Return Value

```python
{
    "success": bool,  # Whether sync completed successfully
    "new_activities": int,  # Number of new activities downloaded
    "csv_path": str,  # Path to workout_summary_from_fit.csv
    "strength_csv_path": str,  # Path to strength_training_summary.csv
    "error": str  # Error message (only if success=False)
}
```

## Integration Example

```python
def my_workout_analysis():
    """Integrate sync into your workflow."""
    from fitanalyzer import sync_activities
    import pandas as pd

    # Step 1: Sync from Garmin
    result = sync_activities(days=7)

    if not result["success"]:
        print(f"Sync failed: {result['error']}")
        return

    # Step 2: Load and analyze data
    df = pd.read_csv(result["csv_path"])

    # Your custom analysis here
    recent_runs = df[df['sport'] == 'running'].tail(5)
    print(f"Last 5 runs: {recent_runs[['date', 'distance', 'duration']]}")
```

## Options

### Analyze Only (Skip Download)

```python
result = sync_activities(analyze_only=True)
# Only analyzes existing FIT files, doesn't download new ones
```

### Download Only (Skip Analysis)

```python
result = sync_activities(download_only=True)
# Only downloads FIT files, doesn't generate CSVs
```

## See Also

- Main documentation: `../README.md`
- CLI usage: `fitanalyzer-sync --help`
- Other examples: `analyze_fit.py`
