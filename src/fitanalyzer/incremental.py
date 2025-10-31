"""
Incremental analysis utilities for tracking and skipping unchanged FIT files.

This module handles file modification tracking and cache management to avoid
re-analyzing files that haven't changed since the last analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def load_existing_analysis(csv_path: Path) -> Dict[str, float]:
    """Load existing analysis results and return file -> mtime mapping.

    Args:
        csv_path: Path to the existing CSV file

    Returns:
        Dictionary mapping FIT file paths to their last analyzed modification time
    """
    if not csv_path.exists():
        return {}

    try:
        df_data = pd.read_csv(csv_path)
        if "file" not in df_data.columns or "_file_mtime" not in df_data.columns:
            return {}

        # Build mapping of file -> mtime
        file_mtimes = {}
        for _, row in df_data.iterrows():
            file_path = row["file"]
            mtime = row.get("_file_mtime")
            if pd.notna(mtime):
                file_mtimes[file_path] = float(mtime)

        return file_mtimes
    except (IOError, pd.errors.ParserError, KeyError, ValueError):
        # If there's any error reading the CSV, start fresh
        return {}


def needs_analysis(fit_file: str, existing_analysis: Dict[str, float], force: bool) -> bool:
    """Check if a FIT file needs (re)analysis.

    Args:
        fit_file: Path to the FIT file
        existing_analysis: Dictionary of file -> mtime from previous analysis
        force: If True, always reanalyze

    Returns:
        True if file needs analysis, False if it's already up-to-date
    """
    if force:
        return True

    # Get file's current modification time
    try:
        current_mtime = Path(fit_file).stat().st_mtime
    except (OSError, FileNotFoundError):
        # File doesn't exist or can't be accessed - skip it
        return False

    # Check if we have analyzed this file before
    analyzed_mtime = existing_analysis.get(fit_file)

    # Needs analysis if:
    # 1. Never analyzed before (not in existing_analysis)
    # 2. File has been modified since last analysis
    if analyzed_mtime is None:
        return True

    return current_mtime > analyzed_mtime


def load_existing_rows(
    csv_path: Path, existing_analysis: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Load existing rows from CSV and restore _file_mtime.

    Args:
        csv_path: Path to the CSV file
        existing_analysis: Dictionary mapping files to their modification times

    Returns:
        List of row dictionaries with _file_mtime restored
    """
    if not csv_path.exists():
        return []

    try:
        existing_df = pd.read_csv(csv_path)
        if "file" not in existing_df.columns:
            return []

        existing_rows = []
        for _, row in existing_df.iterrows():
            row_dict = row.to_dict()
            # Add _file_mtime from existing_analysis
            fit_file = row_dict.get("file")
            if fit_file and fit_file in existing_analysis:
                row_dict["_file_mtime"] = existing_analysis[fit_file]
            existing_rows.append(row_dict)
        return existing_rows
    except (IOError, pd.errors.ParserError, KeyError, ValueError):
        return []


def determine_files_to_process(
    fit_files: List[str], existing_analysis: Dict[str, float], force: bool
) -> Tuple[List[str], int]:
    """Determine which files need processing.

    Args:
        fit_files: List of all FIT files
        existing_analysis: Dictionary of file -> mtime from previous analysis
        force: If True, process all files

    Returns:
        Tuple of (files_to_process, skipped_count)
    """
    files_to_process = []
    skipped_count = 0

    for fit_file in fit_files:
        if needs_analysis(fit_file, existing_analysis, force):
            files_to_process.append(fit_file)
        else:
            skipped_count += 1

    return files_to_process, skipped_count
