#!/usr/bin/env python3
"""
Example: How to use incremental sync programmatically.

This demonstrates the recommended way to sync Garmin activities from Python code.
The sync is incremental by default - only new/changed activities are processed.
"""

import sys
from pathlib import Path

# Add src to path if running without installation
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from fitanalyzer import sync_activities


def example_simple_sync():
    """Simplest usage - sync last 7 days with defaults."""
    print("Example 1: Simple sync (uses env vars for credentials)")
    print("=" * 60)

    result = sync_activities(days=7)

    if result["success"]:
        print(f"✅ Success!")
        print(f"   New activities: {result['new_activities']}")
        print(f"   Workout CSV: {result['csv_path']}")
        print(f"   Strength CSV: {result['strength_csv_path']}")
    else:
        print(f"❌ Failed: {result['error']}")
    print()


def example_custom_sync():
    """More advanced usage with custom parameters."""
    print("Example 2: Custom sync with explicit parameters")
    print("=" * 60)

    result = sync_activities(
        days=30,  # Last 30 days
        directory="./my_activities",  # Custom directory
        output_dir="./my_output",  # Custom output
        ftp=280,  # Your FTP
        hrrest=55,  # Your resting HR
        hrmax=185,  # Your max HR
        multisport=True,  # Enable multisport handling
    )

    if result["success"]:
        print(f"✅ Success!")
        print(f"   New activities: {result['new_activities']}")
        print(f"   Files saved to: {result['csv_path']}")
    else:
        print(f"❌ Failed: {result['error']}")
    print()


def example_analyze_only():
    """Only analyze existing files without downloading."""
    print("Example 3: Analyze existing files only")
    print("=" * 60)

    result = sync_activities(
        analyze_only=True,  # Skip download
        directory="./activities",
        output_dir="./output",
    )

    if result["success"]:
        print(f"✅ Analysis complete!")
        print(f"   CSV: {result['csv_path']}")
    else:
        print(f"❌ Failed: {result['error']}")
    print()


def example_in_script():
    """Show how to integrate into a larger script."""
    print("Example 4: Integration into larger workflow")
    print("=" * 60)

    # Your app's workflow
    print("1. Syncing activities from Garmin...")
    result = sync_activities(days=7)

    if not result["success"]:
        print(f"❌ Sync failed: {result['error']}")
        return False

    print(f"2. Downloaded {result['new_activities']} activities")

    # Now do something with the data
    import pandas as pd

    try:
        df = pd.read_csv(result["csv_path"])
        print(f"3. Loaded {len(df)} workouts from CSV")

        # Your custom analysis
        if not df.empty:
            print(f"4. Latest workout: {df.iloc[-1]['sport']} on {df.iloc[-1]['date']}")

        strength_df = pd.read_csv(result["strength_csv_path"])
        if not strength_df.empty:
            print(f"5. Strength training: {len(strength_df)} sets recorded")

        return True
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 Fitanalyzer Programmatic Sync Examples\n")

    # Run examples
    example_simple_sync()
    # example_custom_sync()  # Uncomment to try
    # example_analyze_only()  # Uncomment to try
    # example_in_script()  # Uncomment to try

    print("\n💡 Key Points:")
    print("   • Sync is INCREMENTAL by default - only new/changed activities processed")
    print("   • Credentials from GARMIN_EMAIL & GARMIN_PASSWORD env vars")
    print("   • Returns dict with success status and file paths")
    print("   • Safe to call frequently - won't re-download unchanged activities")
