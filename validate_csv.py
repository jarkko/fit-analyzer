#!/usr/bin/env python3
"""
CSV Schema Validation Script

This script ensures that generated CSV files have the correct schema
and that new features are properly integrated. Run this after any
changes to the data extraction or CSV generation code.
"""
import sys
from pathlib import Path
import os  # Add os import

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd


def validate_csv_schema(csv_path: str = "data/workout_summary_from_fit.csv") -> bool:
    """
    Validate that the CSV has the correct schema with all expected columns.

    Returns:
        True if schema is correct, False otherwise
    """
    # Expected schema after speed/cadence/distance feature
    expected_columns = [
        # Core metadata
        "file",
        "sport",
        "sub_sport",
        "date",
        "start_time",
        "end_time",
        "duration_min",
        # Heart rate metrics
        "avg_hr",
        "max_hr",
        # Power metrics
        "avg_power_w",
        "max_power_w",
        "np_w",
        "IF",
        "TSS",
        # Training load
        "TRIMP",
        # Speed metrics (NEW)
        "avg_speed_mps",
        "max_speed_mps",
        "avg_speed_kph",
        "max_speed_kph",
        # Cadence metrics (NEW)
        "avg_cadence",
        "max_cadence",
        # Distance metrics (NEW)
        "total_distance_m",
        "total_distance_km",
    ]

    try:
        if not Path(csv_path).exists():
            print(f"❌ CSV file not found: {csv_path}")
            return False

        df = pd.read_csv(csv_path)
        actual_columns = list(df.columns)

        print(f"📊 CSV Schema Validation: {csv_path}")
        print(f"   Expected columns: {len(expected_columns)}")
        print(f"   Actual columns: {len(actual_columns)}")

        # Check for missing columns
        missing = set(expected_columns) - set(actual_columns)
        if missing:
            print(f"❌ Missing columns: {sorted(missing)}")
            return False

        # Check for unexpected extra columns
        extra = set(actual_columns) - set(expected_columns)
        if extra:
            print(f"⚠️  Extra columns: {sorted(extra)}")
            # Extra columns are warning, not error

        # Check column order
        for i, expected_col in enumerate(expected_columns):
            if i < len(actual_columns) and actual_columns[i] != expected_col:
                print(
                    f"⚠️  Column order mismatch at position {i}: expected '{expected_col}', got '{actual_columns[i]}'"
                )

        # Validate data in new columns
        new_columns = ["avg_speed_mps", "avg_cadence", "total_distance_m"]
        for col in new_columns:
            if col in df.columns:
                non_empty = (df[col] != "").sum()
                total = len(df)
                print(
                    f"   {col}: {non_empty}/{total} sessions have data ({non_empty/total*100:.1f}%)"
                )

        print("✅ CSV schema validation passed!")
        return True

    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        return False


def regenerate_and_validate() -> bool:
    """
    Regenerate CSV with fresh modules and validate schema.
    This is the recommended way to test new features.
    """
    print("🔄 Regenerating CSV with fresh modules...")

    # Set environment variable to force reload
    os.environ["FITANALYZER_FORCE_RELOAD"] = "1"

    # Import parser module
    from fitanalyzer.parser import summarize_fit_sessions, AnalysisConfig
    import pandas as pd

    config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="UTC")

    # Get all FIT files
    fit_files = list(Path("data/samples").glob("*.fit"))
    if Path("data/downloaded").exists():
        fit_files.extend(list(Path("data/downloaded").glob("*.fit")))

    all_sessions = []

    print(f"   Processing {len(fit_files)} FIT files...")
    for fit_file in fit_files:
        try:
            sessions, _ = summarize_fit_sessions(str(fit_file), config)
            all_sessions.extend(sessions)
        except Exception as e:
            print(f"   Error processing {fit_file.name}: {e}")
            continue

    if not all_sessions:
        print("❌ No sessions found!")
        return False

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_sessions).sort_values(["date", "start_time"])

    # Remove internal columns
    columns_to_remove = [col for col in df.columns if col.startswith("_")]
    df_clean = df.drop(columns=columns_to_remove)

    # Save CSV
    csv_path = Path("data/workout_summary_from_fit.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(csv_path, index=False)

    print(f"   Generated CSV with {len(df_clean)} sessions")

    # Validate the generated CSV
    return validate_csv_schema(str(csv_path))


def main():
    """Main validation script"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate CSV schema and regenerate if needed")
    parser.add_argument(
        "--regenerate", action="store_true", help="Regenerate CSV before validating"
    )
    parser.add_argument(
        "--csv", default="data/workout_summary_from_fit.csv", help="Path to CSV file to validate"
    )

    args = parser.parse_args()

    if args.regenerate:
        success = regenerate_and_validate()
    else:
        success = validate_csv_schema(args.csv)

    if not success:
        print("\n💡 Try running with --regenerate to fix the CSV")
        sys.exit(1)
    else:
        print("\n🎉 All validations passed!")


if __name__ == "__main__":
    main()
