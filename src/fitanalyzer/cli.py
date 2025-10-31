"""
Command-line interface for FIT file analysis.

This module provides the CLI entry points and batch processing
functionality for analyzing FIT files from the command line.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from fitparse import FitFile

from fitanalyzer.activities import summarize_fit_original, summarize_fit_sessions
from fitanalyzer.aggregation import aggregate_strength_sets
from fitanalyzer.config import AnalysisConfig
from fitanalyzer.incremental import (
    determine_files_to_process,
    load_existing_analysis,
    load_existing_rows,
)
from fitanalyzer.strength import extract_sets_from_fit, save_strength_sets_csv

__all__ = ["parse_arguments", "main", "main_with_args"]


def parse_arguments(args=None):
    """Parse command line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_files", nargs="+")
    ap.add_argument("--ftp", type=float, required=True)
    ap.add_argument("--hrrest", type=int, default=50)
    ap.add_argument("--hrmax", type=int, default=190)
    ap.add_argument("--tz", type=str, default="Europe/Helsinki")
    ap.add_argument("--dump-sets", action="store_true", help="Save strength training sets to CSV")
    ap.add_argument(
        "--multisport", action="store_true", help="Process multisport activities by session"
    )
    ap.add_argument("--output-dir", type=str, default="data", help="Directory for output CSV files")
    ap.add_argument(
        "--force", action="store_true", help="Force reanalysis of all files (ignore cache)"
    )
    return ap.parse_args(args)


def _process_multisport_file(fit_file, args, processed_sessions):
    """Process a multisport FIT file and return new rows"""
    results, _ = summarize_fit_sessions(
        fit_file, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
    )

    # Get file modification time
    try:
        file_mtime = Path(fit_file).stat().st_mtime
    except (OSError, FileNotFoundError):
        file_mtime = 0.0

    new_rows = []
    for result in results:
        if not result:
            continue

        # Add file modification time for incremental analysis
        result["_file_mtime"] = file_mtime

        # Create a unique key for this session
        session_key = (
            result.get("sport"),
            result.get("start_time"),
            result.get("duration_min"),
            result.get("avg_hr", ""),
            result.get("avg_power_w", ""),
        )

        if session_key not in processed_sessions:
            processed_sessions.add(session_key)
            new_rows.append(result)
        else:
            sport = result["sport"]
            start = result["start_time"]
            print(f"Skipping duplicate session: {sport} at {start}")

    return new_rows


def _process_single_file(fit_file, args):
    """Process a single FIT file and return summary"""
    summary = summarize_fit_original(
        fit_file, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
    )
    if summary:
        # Add file modification time for incremental analysis
        try:
            summary["_file_mtime"] = Path(fit_file).stat().st_mtime
        except (OSError, FileNotFoundError):
            summary["_file_mtime"] = 0.0
    return [summary] if summary else []


def _process_files(
    files_to_process: List[str], args, processed_sessions: set
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Process all files that need analysis."""
    rows = []
    all_sets = []

    for fit_file in files_to_process:
        print(f"📊 Analyzing: {fit_file}")

        # Process based on mode
        if args.multisport:
            new_rows = _process_multisport_file(fit_file, args, processed_sessions)
            rows.extend(new_rows)
        else:
            new_rows = _process_single_file(fit_file, args)
            rows.extend(new_rows)

        # Handle strength sets if requested (only for single-sport files)
        if args.dump_sets and not args.multisport:
            ff = FitFile(fit_file)
            df_sets = extract_sets_from_fit(ff, fit_file_path=fit_file)
            csv_file = save_strength_sets_csv(fit_file, df_sets)
            if csv_file:
                all_sets.append(csv_file)

    return rows, all_sets


def _save_workout_summary(rows: List[Dict[str, Any]], output_dir: str, all_sets: List[str]):
    """Save workout summary CSV and print results."""
    if not rows:
        print("No data to output.")
        return

    out = pd.DataFrame(rows).sort_values(["date", "start_time"])
    csv_path = Path(output_dir) / "workout_summary_from_fit.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV with _file_mtime for incremental analysis
    out.to_csv(csv_path, index=False)
    print(f"\n✅ Created: {csv_path}")

    # Print summary without internal columns
    columns_to_remove = [col for col in out.columns if col.startswith("_")]
    out_display = out.drop(columns=columns_to_remove)
    print(f"📈 Processed {len(out_display)} workout(s)")

    if all_sets:
        print("\n📋 Strength training sets saved to:")
        for p in all_sets:
            print(" -", p)


def main_with_args(args):
    """Main function that takes parsed arguments"""
    processed_sessions = set()
    csv_path = Path(args.output_dir) / "workout_summary_from_fit.csv"

    # Load existing analysis for incremental processing
    existing_analysis = load_existing_analysis(csv_path)
    existing_rows = load_existing_rows(csv_path, existing_analysis)

    # Determine which files need processing
    files_to_process, skipped_count = determine_files_to_process(
        args.fit_files, existing_analysis, args.force
    )

    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count} unchanged file(s) (use --force to reanalyze)")

    # Process files
    new_rows, all_sets = _process_files(files_to_process, args, processed_sessions)

    # Merge with existing data
    if existing_rows:
        processed_files = set(files_to_process)
        kept_rows = [r for r in existing_rows if r.get("file") not in processed_files]
        rows = kept_rows + new_rows
    else:
        rows = new_rows

    # Save results
    _save_workout_summary(rows, args.output_dir, all_sets)

    # Generate consolidated strength training summary if requested
    if args.dump_sets:
        config = AnalysisConfig(
            ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
        )
        df_strength_summary = aggregate_strength_sets(
            args.fit_files,
            config,
            multisport=args.multisport,
        )

        if df_strength_summary is not None and not df_strength_summary.empty:
            csv_path = Path(args.output_dir) / "strength_training_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_strength_summary.to_csv(csv_path, index=False)
            print(f"\n✅ Created: {csv_path}")
            print(
                f"💪 {len(df_strength_summary)} strength training sets "
                f"from {len(df_strength_summary['activity_id'].unique())} workouts"
            )
        else:
            print("\nNo strength training sets found.")

    return 0


def main() -> int:
    """Main entry point for command line"""
    args = parse_arguments()
    return main_with_args(args)


if __name__ == "__main__":
    main()
