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
from fitanalyzer.training_load import calculate_training_load_metrics

__all__ = ["parse_arguments", "main", "main_with_args"]


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_files", nargs="+")
    ap.add_argument("--ftp", type=float, required=True)
    ap.add_argument("--hrrest", type=int, default=50)
    ap.add_argument("--hrmax", type=int, default=190)
    ap.add_argument("--tz", type=str, default="Europe/Helsinki")
    ap.add_argument("--dump-sets", action="store_true", help="Save strength training sets to CSV")
    ap.add_argument("--output-dir", type=str, default="data", help="Directory for output CSV files")
    ap.add_argument(
        "--force", action="store_true", help="Force reanalysis of all files (ignore cache)"
    )
    parsed = ap.parse_args(args)

    # Deduplicate fit_files list (preserve order)
    seen = set()
    unique_files = []
    for f in parsed.fit_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    if len(unique_files) < len(parsed.fit_files):
        duplicates_count = len(parsed.fit_files) - len(unique_files)
        print(f"⚠️  Removed {duplicates_count} duplicate file(s) from input list")

    parsed.fit_files = unique_files
    return parsed


def _process_multisport_file(
    fit_file: str, args: argparse.Namespace, processed_sessions: set[tuple[Any, ...]]
) -> list[dict[str, Any]]:
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
        # Track original file for proper incremental merge
        result["_original_file"] = fit_file

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


def _process_single_file(fit_file: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Process a single-sport FIT file and return a list of summary dicts"""
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


def _is_multisport_file(fit_file: str) -> bool:
    """Check if a FIT file contains multiple sessions (multisport)."""
    # pylint: disable=import-outside-toplevel
    from fitanalyzer.parser import extract_sessions_from_fit

    try:
        ff = FitFile(fit_file)
        sessions = list(extract_sessions_from_fit(ff))
        return len(sessions) > 1
    except (OSError, ValueError, KeyError, AttributeError):
        # If we can't read the file or parse sessions, treat as single sport
        return False


def _process_files(
    files_to_process: List[str], args: argparse.Namespace, processed_sessions: set[tuple[Any, ...]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Process all files that need analysis.

    Automatically detects multisport files and processes them appropriately.
    """
    rows = []
    all_sets = []
    processed_files = set()  # Track which files we've already processed

    for fit_file in files_to_process:
        # Skip if we've already processed this exact file
        if fit_file in processed_files:
            print(f"⏭️  Skipping already processed file: {fit_file}")
            continue

        processed_files.add(fit_file)
        print(f"📊 Analyzing: {fit_file}")

        # Automatically detect multisport files
        is_multisport = _is_multisport_file(fit_file)

        if is_multisport:
            new_rows = _process_multisport_file(fit_file, args, processed_sessions)
            rows.extend(new_rows)
        else:
            new_rows = _process_single_file(fit_file, args)
            rows.extend(new_rows)

        # Handle strength sets if requested (only for single-sport files)
        if args.dump_sets and not is_multisport:
            ff = FitFile(fit_file)
            df_sets = extract_sets_from_fit(ff, fit_file_path=fit_file)
            csv_file = save_strength_sets_csv(fit_file, df_sets)
            if csv_file:
                all_sets.append(csv_file)

    return rows, all_sets


def _save_workout_summary(rows: List[Dict[str, Any]], output_dir: str, all_sets: List[str]) -> None:
    """Save workout summary CSV and print results."""
    if not rows:
        print("No data to output.")
        return

    # Deduplicate based on file identifier (for multisport: use session identifier)
    out = pd.DataFrame(rows)
    # Use 'file' column for deduplication if it exists (unique identifier for each workout/session)
    if not out.empty and "file" in out.columns:
        out = out.drop_duplicates(subset=["file"], keep="last")
    out = out.sort_values(["date", "start_time"])

    # Calculate training load metrics (CTL, ATL, TSB) if TSS or TRIMP data is available
    if not out.empty and "date" in out.columns:
        # Check for TSS first (preferred), then TRIMP as fallback
        if "TSS" in out.columns:
            out = calculate_training_load_metrics(out, load_column="TSS")
        elif "TRIMP" in out.columns:
            out = calculate_training_load_metrics(out, load_column="TRIMP")

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


def _generate_strength_summary(
    args: argparse.Namespace, files_to_process: List[str], existing_strength: pd.DataFrame
) -> pd.DataFrame:
    """Generate strength training summary for processed files.

    Args:
        args: Parsed command-line arguments
        files_to_process: List of files that were processed
        existing_strength: Existing strength data to merge with

    Returns:
        DataFrame with strength training summary
    """
    config = AnalysisConfig(ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz)

    # Merge files to process with API-updated files
    strength_files_to_process = set(files_to_process)
    if hasattr(args, "updated_files") and args.updated_files:
        strength_files_to_process.update(args.updated_files)

    # Only aggregate sets from files that were actually processed or had API updates
    # Note: multisport detection is now automatic in aggregate_strength_sets
    new_strength = aggregate_strength_sets(
        list(strength_files_to_process),
        config,
    )

    # Merge: keep existing rows for unchanged files, add rows for processed files
    if not existing_strength.empty and new_strength is not None:
        # Get activity IDs from processed files
        processed_activity_ids = {Path(f).stem.replace("_ACTIVITY", "") for f in files_to_process}
        # Keep rows for files that weren't processed
        kept_rows = existing_strength[
            ~existing_strength["activity_id"].isin(processed_activity_ids)
        ]
        result = pd.concat([kept_rows, new_strength], ignore_index=True)
        return result.sort_values(["date", "timestamp"], na_position="last")
    if new_strength is not None:
        return new_strength
    return existing_strength


def main_with_args(args: argparse.Namespace) -> int:
    """Main function that takes parsed arguments"""
    processed_sessions: set[tuple[Any, ...]] = set()
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
        # Use _original_file to match multisport sessions correctly
        kept_rows = [
            r
            for r in existing_rows
            if r.get("_original_file", r.get("file")) not in processed_files
        ]
        rows = kept_rows + new_rows
    else:
        rows = new_rows

    # Save results
    _save_workout_summary(rows, args.output_dir, all_sets)

    # Generate consolidated strength training summary if requested
    if args.dump_sets:
        csv_path = Path(args.output_dir) / "strength_training_summary.csv"

        # Load existing strength data
        existing_strength = pd.DataFrame()
        if csv_path.exists() and not args.force:
            try:
                existing_strength = pd.read_csv(csv_path)
            except (OSError, pd.errors.ParserError):
                pass

        # Generate strength summary
        df_strength_summary = _generate_strength_summary(args, files_to_process, existing_strength)

        if df_strength_summary is not None and not df_strength_summary.empty:
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
