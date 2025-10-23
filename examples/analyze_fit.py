#!/usr/bin/env python3
"""
Example script demonstrating how to use the fitanalyzer library.

This script shows how to:
1. Parse a FIT file
2. Extract training metrics
3. Generate a summary
"""

import sys
from pathlib import Path

# Add src to path if running without installation
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from fitanalyzer import summarize_fit_original, summarize_fit_sessions


def main():
    """Run example FIT file analysis."""
    # Path to sample FIT files
    data_dir = Path(__file__).parent.parent / "data" / "samples"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    # Find first FIT file
    fit_files = list(data_dir.glob("*.fit"))
    if not fit_files:
        print(f"No FIT files found in {data_dir}")
        return 1

    fit_file = fit_files[0]
    print(f"Analyzing: {fit_file.name}")
    print("=" * 60)

    # Try to analyze as single-session activity
    try:
        summary, sets = summarize_fit_original(str(fit_file), ftp=300)

        if summary:
            print("\n📊 Activity Summary:")
            print(f"   Sport: {summary.get('sport', 'N/A')}")
            print(f"   Sub-sport: {summary.get('sub_sport', 'N/A')}")
            print(f"   Date: {summary.get('date', 'N/A')}")
            print(f"   Start: {summary.get('start_time', 'N/A')}")
            print(f"   Duration: {summary.get('duration_min', 'N/A')} min")
            print(f"   Avg HR: {summary.get('avg_hr', 'N/A')} bpm")
            print(f"   Max HR: {summary.get('max_hr', 'N/A')} bpm")
            print(f"   Avg Power: {summary.get('avg_power_w', 'N/A')} W")
            print(f"   NP: {summary.get('np_w', 'N/A')} W")
            print(f"   IF: {summary.get('IF', 'N/A')}")
            print(f"   TSS: {summary.get('TSS', 'N/A')}")
            print(f"   TRIMP: {summary.get('TRIMP', 'N/A')}")

            if not sets.empty:
                print(f"\n   Sets: {len(sets)} strength training sets")
        else:
            # Try as multisport activity
            sessions = summarize_fit_sessions(str(fit_file), ftp=300)
            if sessions:
                print(f"\n📊 Multisport Activity ({len(sessions)} sessions):")
                for i, sess in enumerate(sessions, 1):
                    print(f"\n   Session {i}:")
                    print(f"      Sport: {sess.get('sport', 'N/A')}")
                    print(f"      Duration: {sess.get('duration_min', 'N/A')} min")
                    print(f"      Avg HR: {sess.get('avg_hr', 'N/A')} bpm")
                    print(f"      TSS: {sess.get('TSS', 'N/A')}")
            else:
                print("\n⚠️  Could not parse activity")

    except Exception as e:
        print(f"\n❌ Error analyzing file: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
