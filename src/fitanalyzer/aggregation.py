"""
Aggregation utilities for combining data from multiple FIT files.

This module provides utilities for aggregating data across multiple activities,
specifically for strength training set aggregation.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import tz as dateutil_tz
from fitparse import FitFile

from fitanalyzer.config import SetMetadata
from fitanalyzer.parser import extract_sessions_from_fit, get_sport_names
from fitanalyzer.strength import extract_sets_from_fit

if TYPE_CHECKING:
    from fitanalyzer.config import AnalysisConfig


def process_file_for_sets(
    fit_file: str, config: "AnalysisConfig", multisport: bool
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Process a single file and return sessions and sets for strength training aggregation."""
    # Get FIT file and extract sets
    ff = FitFile(fit_file)
    df_sets = extract_sets_from_fit(ff, fit_file_path=fit_file)

    # Get session data directly from FIT file
    sessions = list(extract_sessions_from_fit(ff))

    # For multisport files, find the strength training session
    if len(sessions) > 1 or multisport:
        strength_sessions = [s for s in sessions if s.get("sub_sport") == "strength_training"]
        if strength_sessions:
            sessions = strength_sessions

    # Get sport names and date from first session
    sport, sub_sport = get_sport_names(sessions)
    date = None
    if sessions:
        start_time = sessions[0].get("start_time")
        if start_time:
            # Convert to local timezone for date
            tz_name = config.tz_name
            local_tz = dateutil_tz.gettz(tz_name)
            local_time = start_time.astimezone(local_tz)
            date = local_time.date().isoformat()

    # Create simple session dict with sport info and date
    df_sessions = {"sport": sport, "sub_sport": sub_sport, "date": date}

    return df_sessions, df_sets


def extract_session_metadata(df_sessions: pd.DataFrame) -> tuple[str, str, Any]:
    """Extract metadata from the first session."""
    sport = "unknown"
    sub_sport = "unknown"
    date = None

    first_session = None
    if isinstance(df_sessions, dict):
        first_session = df_sessions
    elif isinstance(df_sessions, list) and df_sessions:
        first_session = df_sessions[0]
    elif isinstance(df_sessions, pd.DataFrame) and not df_sessions.empty:
        first_session = df_sessions.iloc[0]

    if first_session is not None:
        sport = first_session.get("sport", "unknown")
        sub_sport = first_session.get("sub_sport", "unknown")
        date = first_session.get("date", None)

    return sport, sub_sport, date


def create_set_record(row: pd.Series, idx: int, metadata: SetMetadata) -> Dict[str, Any]:
    """Create a dictionary record for a single set."""
    return {
        "activity_id": metadata.activity_id,
        "file": metadata.file_name,
        "date": metadata.date,
        "sport": metadata.sport,
        "sub_sport": metadata.sub_sport,
        "set_number": idx,
        "set_type": row.get("set_type"),
        "exercise_name": row.get("exercise_name", "Unknown"),
        "category": row.get("category"),
        "category_subtype": row.get("category_subtype"),
        "repetitions": row.get("repetitions"),
        "weight": row.get("weight"),
        "duration": row.get("duration"),
        "timestamp": row.get("timestamp"),
    }


def aggregate_strength_sets(
    fit_files: list[str], config: "AnalysisConfig"
) -> Optional[pd.DataFrame]:
    """
    Aggregate strength training sets from multiple FIT files into a single DataFrame.

    Automatically detects multisport files and handles them appropriately.

    Args:
        fit_files: List of FIT file paths to process
        config: Analysis configuration with ftp, hr_rest, hr_max, tz_name

    Returns:
        DataFrame with columns: activity_id, file, date, sport, sub_sport, set_number,
        set_type, category, category_subtype, repetitions, weight, duration, timestamp
    """
    all_strength_data = []

    for fit_file in fit_files:
        # Auto-detect multisport (parameter removed, always auto-detect)
        df_sessions, df_sets = process_file_for_sets(fit_file, config, multisport=False)

        # Skip if no strength sets found
        if df_sets is None or (isinstance(df_sets, pd.DataFrame) and df_sets.empty):
            continue

        # Extract metadata from first session
        sport, sub_sport, date = extract_session_metadata(df_sessions)

        metadata = SetMetadata(
            Path(fit_file).stem.replace("_ACTIVITY", ""),
            Path(fit_file).name,
            date,
            sport,
            sub_sport,
        )

        # Add metadata to each active set
        for idx, row in df_sets.iterrows():
            if row.get("set_type") == "active":
                all_strength_data.append(create_set_record(row, idx, metadata))

    if not all_strength_data:
        return None

    # Return sorted dataframe
    return pd.DataFrame(all_strength_data).sort_values(["date", "timestamp"], na_position="last")
