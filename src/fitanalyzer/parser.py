"""
FIT file parser and analyzer for Garmin activity files.

This module provides functions to parse FIT files, extract activity data,
and calculate training metrics such as normalized power, TSS, and TRIMP.
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz
from fitparse import FitFile

from fitanalyzer.constants import (
    DEFAULT_FTP,
    DEFAULT_HR_REST,
    DEFAULT_HR_MAX,
    DEFAULT_TIMEZONE,
    SPORT_MAPPING,
    SUB_SPORT_MAPPING,
    EXERCISE_CATEGORY_MAPPING,
)
# Apply monkey patch to fix fitparse deprecation warnings
# This import is only for its side-effect (patching fitparse)
from . import fitparse_fix  # noqa: F401 pylint: disable=unused-import


class _GarminProfileLoader:
    """Singleton loader for Garmin FIT SDK Profile (lazy loading)"""
    _instance = None
    _profile = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_profile(self):
        """Get the Garmin Profile, loading it on first access"""
        if self._profile is None:
            try:
                from garmin_fit_sdk import Profile  # pylint: disable=import-outside-toplevel
                self._profile = Profile
            except ImportError:
                self._profile = {}  # Fallback if SDK not installed
        return self._profile


# Global instance of the loader
_profile_loader = _GarminProfileLoader()


def _load_exercise_sets_from_json(fit_file_path: str) -> Optional[Dict[str, Any]]:
    """Load exercise sets data from JSON file (wrapper to avoid circular import).

    Args:
        fit_file_path: Path to the FIT file

    Returns:
        Exercise sets data, or None if file doesn't exist
    """
    # pylint: disable=import-outside-toplevel
    from fitanalyzer.sync import load_exercise_sets_from_json
    return load_exercise_sets_from_json(fit_file_path)


def _get_garmin_profile():
    """Lazy-load Garmin FIT SDK Profile"""
    return _profile_loader.get_profile()


def get_specific_exercise_name(category_id: int, subtype_id: int) -> str:
    """Get specific exercise name from category and subtype IDs

    Uses the Garmin FIT SDK to look up detailed exercise names like
    "Barbell Power Clean" instead of just "Olympic Lift".

    Args:
        category_id: Exercise category ID (e.g., 18 for Olympic Lift)
        subtype_id: Exercise subtype ID (e.g., 2 for Barbell Power Clean)

    Returns:
        str: Specific exercise name, or None if not found
    """
    profile = _get_garmin_profile()
    if not profile or "types" not in profile:
        return None

    # Get the category name to find the right exercise_name type
    category_types = profile["types"].get("exercise_category", {})
    category_name = category_types.get(str(category_id))

    if not category_name or category_name == "unknown":
        return None

    # Map category name to its exercise_name type
    type_name = f"{category_name}_exercise_name"
    exercise_types = profile["types"].get(type_name, {})

    exercise_name = exercise_types.get(str(subtype_id))
    if exercise_name and exercise_name != "unknown":
        # Convert snake_case to Title Case
        return " ".join(word.capitalize() for word in exercise_name.split("_"))

    return None


def merge_api_exercise_names(
    fit_df: pd.DataFrame, api_data: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Merge exercise names from Garmin API into FIT DataFrame.

    Matches sets by message_index and replaces exercise_name with API data.
    API names are more accurate as they reflect manual corrections in Garmin Connect.

    Args:
        fit_df: DataFrame with FIT file data including message_index and exercise_name
        api_data: Exercise sets data from Garmin API (or None)

    Returns:
        DataFrame with updated exercise names where API data is available
    """
    if api_data is None or not api_data.get('exerciseSets'):
        return fit_df

    # Create a copy to avoid modifying original
    result_df = fit_df.copy()

    # Build a mapping from messageIndex to exercise name
    api_exercise_map = {}
    for ex_set in api_data['exerciseSets']:
        message_index = ex_set.get('messageIndex')
        exercises = ex_set.get('exercises', [])

        if message_index is not None and exercises:
            # Take the first exercise (highest probability)
            exercise_name = exercises[0].get('name', '')
            if exercise_name:
                # Convert from UPPER_SNAKE_CASE to Title Case
                formatted_name = " ".join(
                    word.capitalize() for word in exercise_name.split("_")
                )
                api_exercise_map[message_index] = formatted_name

    # Update exercise names in DataFrame where we have API data
    for msg_idx, api_name in api_exercise_map.items():
        mask = result_df['message_index'] == msg_idx
        if mask.any():
            result_df.loc[mask, 'exercise_name'] = api_name

    return result_df


__all__ = [
    "AnalysisConfig",
    "np_power",
    "trimp_from_hr",
    "summarize_fit_sessions",
    "summarize_fit_original",
    "merge_api_exercise_names",
    "parse_arguments",
    "main",
]


@dataclass
class AnalysisConfig:
    """Configuration for session analysis.

    Attributes:
        ftp: Functional Threshold Power in watts
        hr_rest: Resting heart rate in bpm
        hr_max: Maximum heart rate in bpm
        tz_name: Timezone name (e.g., "Europe/Helsinki")
    """

    ftp: float
    hr_rest: int
    hr_max: int
    tz_name: str


def np_power(power: pd.Series) -> float:
    """Calculate Normalized Power for cycling power data.

    Args:
        power: Series of power values in watts

    Returns:
        Normalized Power value in watts
    """
    if len(power) == 0:
        return np.nan
    s = pd.Series(power, dtype=float)
    # Assume 1s timestamps; 30s rolling average
    ma30 = s.rolling(30, min_periods=1).mean()
    return float((ma30.pow(4).mean()) ** 0.25)


def trimp_from_hr(
    hr: pd.Series, hr_rest: int = DEFAULT_HR_REST, hr_max: int = DEFAULT_HR_MAX
) -> float:
    """Calculate Training Impulse (TRIMP) from heart rate data.

    Args:
        hr: Series of heart rate values in bpm
        hr_rest: Resting heart rate in bpm
        hr_max: Maximum heart rate in bpm

    Returns:
        TRIMP value representing training load
    """
    if len(hr) == 0:
        return 0.0
    s = pd.Series(hr, dtype=float).dropna()
    if s.empty:
        return 0.0
    hrr = (s - hr_rest) / max(1, (hr_max - hr_rest))
    hrr = hrr.clip(lower=0)
    inst = hrr * 0.64 * np.exp(1.92 * hrr)
    minutes = len(inst) / 60.0  # Assume 1s intervals
    return float(inst.mean() * minutes)


def summarize_fit_sessions(
    path: str,
    ftp: float = DEFAULT_FTP,
    hr_rest: int = DEFAULT_HR_REST,
    hr_max: int = DEFAULT_HR_MAX,
    tz_name: str = DEFAULT_TIMEZONE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process each session in a FIT file separately to handle multisport activities correctly
    """
    ff = FitFile(path)

    # Get sessions first
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)

    results = []

    # If no sessions or only one session, fall back to original behavior
    if len(sessions) <= 1:
        result, _ = summarize_fit_original(path, ftp, hr_rest, hr_max, tz_name)
        if result:
            results.append(result)
        return results, []

    # Process each session separately
    for session_idx, session in enumerate(sessions):
        session_start = session.get("start_time")
        session_timer_time = session.get("total_timer_time", 0)  # in seconds

        if session_start and session_timer_time > 0:
            session_end = session_start + timedelta(seconds=session_timer_time)

            # Get records for this session time window
            recs = []
            for m in ff.get_messages("record"):
                d = {d.name: d.value for d in m}
                if "timestamp" in d:
                    record_time = d["timestamp"]
                    # Check if record falls within this session's time window
                    if session_start <= record_time <= session_end:
                        recs.append(
                            {
                                "time": record_time,
                                "hr": d.get("heart_rate", np.nan),
                                "power": d.get("power", np.nan),
                            }
                        )

            # Process this session's data
            if recs:
                df = pd.DataFrame(recs).sort_values("time")
                config = AnalysisConfig(ftp=ftp, hr_rest=hr_rest, hr_max=hr_max, tz_name=tz_name)
                session_summary = process_session_data(df, path, session, session_idx, config)
                if session_summary:
                    results.append(session_summary)

    return results, []


def process_session_data(df, path, session, session_idx, config):
    """Process data for a single session

    Args:
        df: DataFrame with time, hr, power columns
        path: Path to the FIT file
        session: Session metadata dictionary
        session_idx: Session index number
        config: AnalysisConfig with ftp, hr_rest, hr_max, tz_name
    """
    if df.empty:
        return None

    local = tz.gettz(config.tz_name)

    # Handle both timezone-aware and naive timestamps
    start_time = pd.to_datetime(df["time"].iloc[0])
    end_time = pd.to_datetime(df["time"].iloc[-1])

    if start_time.tzinfo is None:
        start_utc = start_time.tz_localize("UTC")
        end_utc = end_time.tz_localize("UTC")
    else:
        start_utc = start_time.tz_convert("UTC") if start_time.tzinfo != tz.UTC else start_time
        end_utc = end_time.tz_convert("UTC") if end_time.tzinfo != tz.UTC else end_time

    start_local = start_utc.astimezone(local)
    end_local = end_utc.astimezone(local)
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1
    dur_hr = dur_sec / 3600.0

    # Resample to 1 second for NP calculation
    time_series = pd.to_datetime(df["time"])
    if time_series.dt.tz is None:
        time_index = time_series.dt.tz_localize("UTC")
    else:
        time_index = time_series.dt.tz_convert("UTC")
    df = df.set_index(time_index).sort_index()
    df = df.resample("1s").ffill()

    avg_hr = float(df["hr"].mean()) if df["hr"].notna().any() else np.nan
    max_hr = float(df["hr"].max()) if df["hr"].notna().any() else np.nan
    avg_p = float(df["power"].mean()) if df["power"].notna().any() else np.nan
    max_p = float(df["power"].max()) if df["power"].notna().any() else np.nan
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    IF = (npw / config.ftp) if np.isfinite(npw) and config.ftp > 0 else np.nan
    TSS = (
        ((dur_hr * npw * IF) / config.ftp * 100)
        if np.all(np.isfinite([dur_hr, npw, IF])) and config.ftp > 0
        else np.nan
    )
    TRIMP = (
        trimp_from_hr(df["hr"].ffill(), hr_rest=config.hr_rest, hr_max=config.hr_max)
        if df["hr"].notna().any()
        else 0.0
    )

    # Create filename with session info
    base_name = Path(path).stem

    # Convert numeric sport codes to names
    raw_sport = session.get("sport", "unknown")
    raw_subsport = session.get("sub_sport", "")

    # Map numeric codes to sport names
    if isinstance(raw_sport, int):
        session_sport = SPORT_MAPPING.get(raw_sport, str(raw_sport))
    else:
        session_sport = raw_sport

    if isinstance(raw_subsport, int):
        session_subsport = SUB_SPORT_MAPPING.get(raw_subsport, str(raw_subsport))
    else:
        session_subsport = raw_subsport

    if session_subsport and session_subsport != "generic":
        file_display = f"{base_name}_session{session_idx}_{session_sport}_{session_subsport}"
    else:
        file_display = f"{base_name}_session{session_idx}_{session_sport}"

    return {
        "file": file_display,
        "sport": session_sport,
        "sub_sport": session_subsport,
        "date": start_local.date().isoformat(),
        "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_local.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(dur_sec / 60.0, 1),
        "avg_hr": round(avg_hr, 1) if np.isfinite(avg_hr) else "",
        "max_hr": int(max_hr) if np.isfinite(max_hr) else "",
        "avg_power_w": round(avg_p, 1) if np.isfinite(avg_p) else "",
        "max_power_w": round(max_p, 1) if np.isfinite(max_p) else "",
        "np_w": round(npw, 1) if np.isfinite(npw) else "",
        "IF": round(IF, 3) if np.isfinite(IF) else "",
        "TSS": round(TSS, 1) if np.isfinite(TSS) else "",
        "TRIMP": round(TRIMP, 1),
        # Keep these for deduplication logic
        "_original_file": path,
        "_session_index": session_idx,
    }


def _extract_sessions_from_fit(ff):
    """Extract session info from FIT file"""
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)
    return sessions


def _get_sport_names(sessions):
    """Get sport and sub-sport names from sessions"""
    raw_sport = sessions[0].get("sport", "") if sessions else ""
    raw_subsport = sessions[0].get("sub_sport", "") if sessions else ""

    # Convert numeric sport codes to names
    if isinstance(raw_sport, int):
        session_sport = SPORT_MAPPING.get(raw_sport, str(raw_sport))
    else:
        session_sport = raw_sport

    if isinstance(raw_subsport, int):
        session_subsport = SUB_SPORT_MAPPING.get(raw_subsport, str(raw_subsport))
    else:
        session_subsport = raw_subsport

    return session_sport, session_subsport


def _extract_records_from_fit(ff):
    """Extract aerobic data records from FIT file"""
    recs = []
    for m in ff.get_messages("record"):
        d = {d.name: d.value for d in m}
        if "timestamp" in d:
            recs.append(
                {
                    "time": d["timestamp"],
                    "hr": d.get("heart_rate", np.nan),
                    "power": d.get("power", np.nan),
                }
            )
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("time")
    return df


def _extract_sets_from_fit(ff, fit_file_path: Optional[str] = None):  # noqa: C901
    """Extract strength training sets from FIT file and add exercise names

    Exercise names are extracted using a two-level system:
    1. Category (e.g., 18 = "Olympic Lift", 13 = "Hyperextension")
    2. Category subtype (e.g., category 18, subtype 2 = "Barbell Power Clean")

    If API exercise data is available (from Garmin Connect), those names are
    preferred as they reflect manual corrections. Otherwise, we use the
    category/subtype mapping from the FIT file.

    Args:
        ff: FitFile object
        fit_file_path: Optional path to FIT file (to load API exercise data)
    """
    sets = []
    for m in ff.get_messages("set"):
        d = {d.name: d.value for d in m}
        sets.append(d)

    if not sets:
        return pd.DataFrame()

    df = pd.DataFrame(sets)

    # Helper functions for exercise name extraction
    def _extract_valid_value(value, invalid_value=65534):
        """Extract first valid value from tuple or return single value."""
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, tuple):
            for v in value:
                if v is not None and v != invalid_value:
                    return v
            return None
        return value if value != invalid_value else None

    def _get_specific_name(cat_val, category_subtype):
        """Try to get specific exercise name from subtype."""
        sub_val = _extract_valid_value(category_subtype)
        if sub_val is not None:
            specific_name = get_specific_exercise_name(cat_val, sub_val)
            if specific_name:
                return specific_name
        return None

    def get_exercise_name(row):
        """Convert category and category_subtype to exercise name

        Args:
            row: DataFrame row with 'category' and 'category_subtype' columns

        Returns:
            str: Human-readable exercise name
        """
        cat_val = _extract_valid_value(row.get("category"))
        if cat_val is None:
            return "Unknown"

        # Try specific exercise name from subtype
        specific_name = _get_specific_name(cat_val, row.get("category_subtype"))
        if specific_name:
            return specific_name

        # Fall back to category-level name
        return EXERCISE_CATEGORY_MAPPING.get(cat_val, f"Exercise {cat_val}")

    df["exercise_name"] = df.apply(get_exercise_name, axis=1)

    # Merge API exercise names if available
    if fit_file_path:
        api_data = _load_exercise_sets_from_json(fit_file_path)
        if api_data:
            df = merge_api_exercise_names(df, api_data)

    return df


# Aggregate strength sets from multiple files
def _get_session_info(fit_file, *, ftp, hr_rest, hr_max, tz_name, multisport):
    """Extract session info and sets from a FIT file.

    Auto-detects multisport activities by checking session count.

    Returns:
        Tuple of (df_sessions, df_sets)
    """
    # Auto-detect multisport: check if file has multiple sessions
    ff = FitFile(fit_file)
    sessions = list(ff.get_messages("session"))
    is_multisport = len(sessions) > 1

    if is_multisport or multisport:
        result = summarize_fit_sessions(
            fit_file, ftp=ftp, hr_rest=hr_rest, hr_max=hr_max, tz_name=tz_name
        )
        df_sessions, df_sets = result
        # multisport mode doesn't extract sets, fall back to original
        if not df_sets or (isinstance(df_sets, list) and not df_sets):
            _, df_sets = summarize_fit_original(
                fit_file, ftp=ftp, hr_rest=hr_rest, hr_max=hr_max, tz_name=tz_name
            )
            # For multisport files, find the strength training session
            if isinstance(df_sessions, list):
                strength_sessions = [
                    s for s in df_sessions if s.get("sub_sport") == "strength_training"
                ]
                if strength_sessions:
                    df_sessions = strength_sessions
    else:
        summary_dict, df_sets = summarize_fit_original(
            fit_file, ftp=ftp, hr_rest=hr_rest, hr_max=hr_max, tz_name=tz_name
        )
        # Convert dict to list for consistency
        df_sessions = [summary_dict] if summary_dict else []

    return df_sessions, df_sets


def _extract_first_session_metadata(df_sessions):
    """Extract sport, sub_sport, and date from first session.

    Returns:
        Tuple of (sport, sub_sport, date)
    """
    sport = "unknown"
    sub_sport = "unknown"
    date = None

    first_session = None
    if isinstance(df_sessions, list) and df_sessions:
        first_session = df_sessions[0]
    # isinstance checks type - false positive from pylint
    elif isinstance(df_sessions, pd.DataFrame) and not df_sessions.empty:
        first_session = df_sessions.iloc[0]

    if first_session:
        sport = first_session.get("sport", "unknown")
        sub_sport = first_session.get("sub_sport", "unknown")
        date = first_session.get("date", None)

    return sport, sub_sport, date


def _create_set_record(row, idx, *, activity_id, file_name, date, sport, sub_sport):
    """Create a set record dictionary from a row."""
    return {
        "activity_id": activity_id,
        "file": file_name,
        "date": date,
        "sport": sport,
        "sub_sport": sub_sport,
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


def _aggregate_strength_sets(
    fit_files, *, ftp, hr_rest, hr_max, tz_name, multisport=False
):
    """
    Aggregate strength training sets from multiple FIT files into a single DataFrame.

    Args:
        fit_files: List of FIT file paths to process
        ftp: Functional Threshold Power
        hr_rest: Resting heart rate
        hr_max: Maximum heart rate
        tz_name: Timezone name
        multisport: Whether to use multisport processing

    Returns:
        DataFrame with columns: activity_id, file, date, sport, sub_sport, set_number,
        set_type, category, category_subtype, repetitions, weight, duration, timestamp
    """
    all_strength_data = []

    for fit_file in fit_files:
        # Extract session info and sets
        df_sessions, df_sets = _get_session_info(
            fit_file, ftp=ftp, hr_rest=hr_rest, hr_max=hr_max,
            tz_name=tz_name, multisport=multisport
        )

        # Skip if no strength sets found
        if df_sets is None or (isinstance(df_sets, pd.DataFrame) and df_sets.empty):
            continue

        # Get activity info
        activity_id = Path(fit_file).stem.replace("_ACTIVITY", "")
        file_name = Path(fit_file).name
        sport, sub_sport, date = _extract_first_session_metadata(df_sessions)

        # Add metadata to each active set
        for idx, row in df_sets.iterrows():
            if row.get("set_type") == "active":
                set_data = _create_set_record(
                    row, idx, activity_id=activity_id, file_name=file_name,
                    date=date, sport=sport, sub_sport=sub_sport
                )
                all_strength_data.append(set_data)

    if not all_strength_data:
        return None

    df_summary = pd.DataFrame(all_strength_data)

    # Sort by date and timestamp
    df_summary = df_summary.sort_values(["date", "timestamp"], na_position="last")

    return df_summary


def _prepare_timezone_aware_index(df):
    """Convert dataframe time column to timezone-aware UTC index"""
    start_time = pd.to_datetime(df["time"].iloc[0])
    end_time = pd.to_datetime(df["time"].iloc[-1])

    # Handle both timezone-aware and naive timestamps
    if start_time.tzinfo is None:
        start_utc = start_time.tz_localize("UTC")
        end_utc = end_time.tz_localize("UTC")
    else:
        start_utc = start_time.tz_convert("UTC") if start_time.tzinfo != tz.UTC else start_time
        end_utc = end_time.tz_convert("UTC") if end_time.tzinfo != tz.UTC else end_time

    # Set index with timezone handling
    time_series = pd.to_datetime(df["time"])
    if time_series.dt.tz is None:
        time_index = time_series.dt.tz_localize("UTC")
    else:
        time_index = time_series.dt.tz_convert("UTC")

    return start_utc, end_utc, time_index


def _calculate_metrics(df, *, ftp, hr_rest, hr_max, start_utc, end_utc):
    """Calculate all training metrics from dataframe"""
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1
    dur_hr = dur_sec / 3600.0

    avg_hr = float(df["hr"].mean()) if df["hr"].notna().any() else np.nan
    max_hr = float(df["hr"].max()) if df["hr"].notna().any() else np.nan
    avg_p = float(df["power"].mean()) if df["power"].notna().any() else np.nan
    max_p = float(df["power"].max()) if df["power"].notna().any() else np.nan
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    IF = (npw / ftp) if np.isfinite(npw) and ftp > 0 else np.nan
    TSS = (
        ((dur_hr * npw * IF) / ftp * 100)
        if np.all(np.isfinite([dur_hr, npw, IF])) and ftp > 0
        else np.nan
    )
    TRIMP = (
        trimp_from_hr(df["hr"].ffill(), hr_rest=hr_rest, hr_max=hr_max)
        if df["hr"].notna().any()
        else 0.0
    )

    return {
        "dur_sec": dur_sec,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "avg_p": avg_p,
        "max_p": max_p,
        "npw": npw,
        "IF": IF,
        "TSS": TSS,
        "TRIMP": TRIMP,
    }


def summarize_fit_original(path, ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"):
    """Original function for single-session activities"""
    ff = FitFile(path)

    # Extract session info and sport names
    sessions = _extract_sessions_from_fit(ff)
    session_sport, session_subsport = _get_sport_names(sessions)

    # Extract aerobic data records
    df = _extract_records_from_fit(ff)

    # Extract strength training sets (passing path for API data)
    df_sets = _extract_sets_from_fit(ff, fit_file_path=path)

    out_summary = None
    if not df.empty:
        # Prepare timezone-aware timestamps and index
        start_utc, end_utc, time_index = _prepare_timezone_aware_index(df)
        df = df.set_index(time_index).sort_index()
        df = df.resample("1s").ffill()

        # Calculate all metrics
        metrics = _calculate_metrics(
            df, ftp=ftp, hr_rest=hr_rest, hr_max=hr_max, start_utc=start_utc, end_utc=end_utc
        )

        # Convert to local timezone for display
        local = tz.gettz(tz_name)
        start_local = start_utc.astimezone(local)
        end_local = end_utc.astimezone(local)

        out_summary = {
            "file": path,
            "sport": session_sport,
            "sub_sport": session_subsport,
            "date": start_local.date().isoformat(),
            "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_local.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": round(metrics["dur_sec"] / 60.0, 1),
            "avg_hr": round(metrics["avg_hr"], 1) if np.isfinite(metrics["avg_hr"]) else "",
            "max_hr": int(metrics["max_hr"]) if np.isfinite(metrics["max_hr"]) else "",
            "avg_power_w": round(metrics["avg_p"], 1) if np.isfinite(metrics["avg_p"]) else "",
            "max_power_w": round(metrics["max_p"], 1) if np.isfinite(metrics["max_p"]) else "",
            "np_w": round(metrics["npw"], 1) if np.isfinite(metrics["npw"]) else "",
            "IF": round(metrics["IF"], 3) if np.isfinite(metrics["IF"]) else "",
            "TSS": round(metrics["TSS"], 1) if np.isfinite(metrics["TSS"]) else "",
            "TRIMP": round(metrics["TRIMP"], 1),
        }
    return out_summary, df_sets


def parse_arguments(args=None):
    """Parse command line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("fit_files", nargs="+")
    ap.add_argument("--ftp", type=float, required=True)
    ap.add_argument("--hrrest", type=int, default=50)
    ap.add_argument("--hrmax", type=int, default=190)
    ap.add_argument("--tz", type=str, default="Europe/Helsinki")
    ap.add_argument(
        "--dump-sets", action="store_true", help="Save strength training sets to CSV"
    )
    ap.add_argument(
        "--multisport", action="store_true", help="Process multisport activities by session"
    )
    ap.add_argument(
        "--output-dir", type=str, default="data", help="Directory for output CSV files"
    )
    return ap.parse_args(args)


def _process_multisport_file(fit_file, args, processed_sessions):
    """Process a multisport FIT file and return new rows"""
    results, _ = summarize_fit_sessions(
        fit_file, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
    )

    new_rows = []
    for result in results:
        if not result:
            continue

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
    summary, _ = summarize_fit_original(
        fit_file, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
    )
    return [summary] if summary else []


def _save_strength_sets(fit_file, df_sets, all_sets):
    """Save strength sets to CSV if present"""
    if df_sets is not None and not df_sets.empty:
        filename = Path(fit_file).with_suffix("").name + "_strength_sets.csv"
        df_sets.to_csv(filename, index=False)
        all_sets.append(filename)


def main_with_args(args):
    """Main function that takes parsed arguments"""
    rows = []
    all_sets = []
    processed_sessions = set()

    for fit_file in args.fit_files:
        # Process based on mode
        if args.multisport:
            new_rows = _process_multisport_file(fit_file, args, processed_sessions)
            rows.extend(new_rows)
        else:
            new_rows = _process_single_file(fit_file, args)
            rows.extend(new_rows)

        # Handle strength sets if requested (only for single-sport files)
        # For multisport, use the consolidated summary instead
        if args.dump_sets and not args.multisport:
            _, df_sets = summarize_fit_original(
                fit_file, ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
            )
            _save_strength_sets(fit_file, df_sets, all_sets)

    if rows:
        out = pd.DataFrame(rows).sort_values(["date", "start_time"])
        # Remove internal columns before saving
        columns_to_remove = [col for col in out.columns if col.startswith("_")]
        out_clean = out.drop(columns=columns_to_remove)
        csv_path = Path(args.output_dir) / "workout_summary_from_fit.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        out_clean.to_csv(csv_path, index=False)
        print(f"\n✅ Created: {csv_path}")
        print(out_clean.to_string(index=False))
        if all_sets:
            print("\n📋 Strength training sets saved to:")
            for p in all_sets:
                print(" -", p)
    else:
        print("No data to output.")

    # Generate consolidated strength training summary if requested
    if args.dump_sets:
        df_strength_summary = _aggregate_strength_sets(
            args.fit_files,
            ftp=args.ftp,
            hr_rest=args.hrrest,
            hr_max=args.hrmax,
            tz_name=args.tz,
            multisport=args.multisport,
        )

        if df_strength_summary is not None and not df_strength_summary.empty:
            csv_path = Path(args.output_dir) / "strength_training_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_strength_summary.to_csv(csv_path, index=False)
            print(f"\n✅ Created: {csv_path}")
            print(
                f"Total: {len(df_strength_summary)} strength training sets "
                f"from {len(df_strength_summary['activity_id'].unique())} workouts."
            )
            print(df_strength_summary.to_string(index=False))
        else:
            print("\nEi strength training settejä löytynyt.")

    return 0


def main():
    """Main entry point for command line"""
    args = parse_arguments()
    return main_with_args(args)


if __name__ == "__main__":
    main()
