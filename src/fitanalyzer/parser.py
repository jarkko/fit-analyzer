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
)
from fitanalyzer.metrics import np_power, trimp_from_hr
from fitanalyzer.strength import (
    merge_api_exercise_names,
    extract_sets_from_fit,
    save_strength_sets_csv,
)

# Apply monkey patch to fix fitparse deprecation warnings
# This import is only for its side-effect (patching fitparse)
from . import fitparse_fix  # noqa: F401 pylint: disable=unused-import


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


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for FIT file analysis."""

    ftp: float
    hr_rest: int
    hr_max: int
    tz_name: str


@dataclass(frozen=True)
class SetMetadata:
    """Metadata for strength training set."""

    activity_id: str
    file_name: str
    date: str
    sport: str
    sub_sport: str


def summarize_fit_sessions(
    path: str, config: AnalysisConfig = None, **kwargs
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process each session in a FIT file separately to handle multisport activities.

    Args:
        path: Absolute or relative path to the FIT file to process.
        config: AnalysisConfig object with ftp, hr_rest, hr_max, tz_name.
        **kwargs: Individual parameters for backwards compatibility.

    Returns:
        A tuple of two lists: (session_summaries, strength_sets)
    """
    config = config or AnalysisConfig(
        ftp=kwargs.get("ftp", DEFAULT_FTP),
        hr_rest=kwargs.get("hr_rest", DEFAULT_HR_REST),
        hr_max=kwargs.get("hr_max", DEFAULT_HR_MAX),
        tz_name=kwargs.get("tz_name", DEFAULT_TIMEZONE),
    )

    ff = FitFile(path)
    sessions = _extract_sessions_from_fit(ff)

    # If no sessions or only one session, fall back to original behavior
    if len(sessions) <= 1:
        result, _ = summarize_fit_original(path, config)
        return ([result] if result else []), []

    # Process each session separately
    results = []

    for session_idx, session in enumerate(sessions):
        if not (session_start := session.get("start_time")):
            continue
        if (session_timer_time := session.get("total_timer_time", 0)) <= 0:
            continue

        # Process this session's data
        if (
            recs := [
                {
                    "time": d["timestamp"],
                    "hr": d.get("heart_rate", np.nan),
                    "power": d.get("power", np.nan),
                }
                for m in ff.get_messages("record")
                if (d := {d.name: d.value for d in m})
                and "timestamp" in d
                and session_start
                <= d["timestamp"]
                <= (session_start + timedelta(seconds=session_timer_time))
            ]
        ) and (
            session_summary := process_session_data(
                pd.DataFrame(recs).sort_values("time"), path, session, session_idx, config
            )
        ):
            results.append(session_summary)

    return results, []


def _calculate_metrics(df: pd.DataFrame, dur_hr: float, config: AnalysisConfig) -> Dict[str, float]:
    """Calculate power and heart rate metrics from session data.

    Args:
        df: Resampled DataFrame with hr and power columns
        dur_hr: Duration in hours
        config: Analysis configuration with ftp, hr_rest, hr_max

    Returns:
        Dictionary with avg_hr, max_hr, avg_p, max_p, npw, intensity_factor, tss, trimp
    """
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    intensity_factor = (npw / config.ftp) if np.isfinite(npw) and config.ftp > 0 else np.nan

    return {
        "avg_hr": float(df["hr"].mean()) if df["hr"].notna().any() else np.nan,
        "max_hr": float(df["hr"].max()) if df["hr"].notna().any() else np.nan,
        "avg_p": float(df["power"].mean()) if df["power"].notna().any() else np.nan,
        "max_p": float(df["power"].max()) if df["power"].notna().any() else np.nan,
        "npw": npw,
        "intensity_factor": intensity_factor,
        "tss": (
            ((dur_hr * npw * intensity_factor) / config.ftp * 100)
            if np.all(np.isfinite([dur_hr, npw, intensity_factor])) and config.ftp > 0
            else np.nan
        ),
        "trimp": (
            trimp_from_hr(df["hr"].ffill(), hr_rest=config.hr_rest, hr_max=config.hr_max)
            if df["hr"].notna().any()
            else 0.0
        ),
    }


def _process_timestamps(df: pd.DataFrame, tz_name: str) -> Dict[str, Any]:
    """Extract and convert timestamps from DataFrame.

    Returns dict with start_utc, end_utc, start_local, end_local, dur_sec, dur_hr
    """
    local = tz.gettz(tz_name)
    start_time = pd.to_datetime(df["time"].iloc[0])
    end_time = pd.to_datetime(df["time"].iloc[-1])

    if start_time.tzinfo is None:
        start_utc = start_time.tz_localize("UTC")
        end_utc = end_time.tz_localize("UTC")
    else:
        start_utc = start_time.tz_convert("UTC") if start_time.tzinfo != tz.UTC else start_time
        end_utc = end_time.tz_convert("UTC") if end_time.tzinfo != tz.UTC else end_time

    dur_sec = int((end_utc - start_utc).total_seconds()) + 1

    return {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_local": start_utc.astimezone(local),
        "end_local": end_utc.astimezone(local),
        "dur_sec": dur_sec,
        "dur_hr": dur_sec / 3600.0,
    }


def _map_sport_names(session: Dict[str, Any]) -> tuple[str, str]:
    """Map numeric sport codes to human-readable names.

    Returns tuple of (sport, sub_sport)
    """
    raw_sport = session.get("sport", "unknown")
    raw_subsport = session.get("sub_sport", "")

    session_sport = (
        SPORT_MAPPING.get(raw_sport, str(raw_sport)) if isinstance(raw_sport, int) else raw_sport
    )
    session_subsport = (
        SUB_SPORT_MAPPING.get(raw_subsport, str(raw_subsport))
        if isinstance(raw_subsport, int)
        else raw_subsport
    )

    return session_sport, session_subsport


def _create_file_display(path: str, session_idx: int, sport: str, subsport: str) -> str:
    """Create display filename for session."""
    base_name = Path(path).stem
    if subsport and subsport != "generic":
        return f"{base_name}_session{session_idx}_{sport}_{subsport}"
    return f"{base_name}_session{session_idx}_{sport}"


def process_session_data(
    df: pd.DataFrame, path: str, session: Dict[str, Any], session_idx: int, config: AnalysisConfig
) -> Optional[Dict[str, Any]]:
    """Process data for a single session and calculate training metrics.

    Takes raw record-level data for one session and computes comprehensive
    training metrics including power, heart rate, duration, and sport identification.
    Handles timezone conversion and data resampling for accurate calculations.

    Args:
        df: DataFrame with columns 'time' (datetime), 'hr' (heart rate), 'power' (watts).
            Should contain one row per second of the session.
        path: Path to the FIT file being processed. Used to construct activity ID
              and filename references.
        session: Dictionary of session metadata from FIT file, containing keys like:
                 'sport', 'sub_sport', 'start_time', 'total_timer_time', etc.
        session_idx: Zero-based index of this session within a multisport activity.
                     Used to differentiate sessions in the output filename.
        config: AnalysisConfig object with attributes:
                - ftp: Functional Threshold Power (watts)
                - hr_rest: Resting heart rate (bpm)
                - hr_max: Maximum heart rate (bpm)
                - tz_name: Timezone name for local time conversion

    Returns:
        Dictionary containing processed session summary with keys:
        - date: ISO format date string
        - start_time, end_time: UTC and local timestamps
        - duration_seconds, duration_hours: Session duration
        - sport, sub_sport: Human-readable sport names
        - avg_hr, max_hr: Heart rate statistics (bpm)
        - avg_power, max_power: Power statistics (watts)
        - normalized_power: Normalized Power (watts)
        - intensity_factor: Ratio of NP to FTP
        - TSS: Training Stress Score
        - TRIMP: Training Impulse
        - file_id, activity_id: File identifiers
        Returns None if DataFrame is empty or processing fails.

    Notes:
        - Resamples data to 1-second intervals using forward-fill
        - Handles both timezone-aware and naive timestamps
        - Maps numeric sport codes to human-readable names
        - Includes session index in multi-sport activities (e.g., "session_1")
    """
    if df.empty:
        return None

    # Extract timestamps and duration
    times = _process_timestamps(df, config.tz_name)

    # Resample to 1 second for NP calculation
    time_series = pd.to_datetime(df["time"])
    time_index = (
        time_series.dt.tz_localize("UTC")
        if time_series.dt.tz is None
        else time_series.dt.tz_convert("UTC")
    )
    df = df.set_index(time_index).sort_index().resample("1s").ffill()

    # Calculate all metrics
    metrics = _calculate_metrics(df, times["dur_hr"], config)

    # Map sport names and create filename
    sport, subsport = _map_sport_names(session)
    file_display = _create_file_display(path, session_idx, sport, subsport)

    return {
        "file": file_display,
        "sport": sport,
        "sub_sport": subsport,
        "date": times["start_local"].date().isoformat(),
        "start_time": times["start_local"].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": times["end_local"].strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(times["dur_sec"] / 60.0, 1),
        "avg_hr": round(metrics["avg_hr"], 1) if np.isfinite(metrics["avg_hr"]) else "",
        "max_hr": int(metrics["max_hr"]) if np.isfinite(metrics["max_hr"]) else "",
        "avg_power_w": round(metrics["avg_p"], 1) if np.isfinite(metrics["avg_p"]) else "",
        "max_power_w": round(metrics["max_p"], 1) if np.isfinite(metrics["max_p"]) else "",
        "np_w": round(metrics["npw"], 1) if np.isfinite(metrics["npw"]) else "",
        "IF": (
            round(metrics["intensity_factor"], 3)
            if np.isfinite(metrics["intensity_factor"])
            else ""
        ),
        "TSS": round(metrics["tss"], 1) if np.isfinite(metrics["tss"]) else "",
        "TRIMP": round(metrics["trimp"], 1),
        # Keep these for deduplication logic
        "_original_file": path,
        "_session_index": session_idx,
    }


def _extract_sessions_from_fit(ff: FitFile) -> List[Dict[str, Any]]:
    """Extract session info from FIT file"""
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)
    return sessions


def _get_sport_names(sessions: List[Dict[str, Any]]) -> Tuple[str, str]:
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


def _extract_records_from_fit(ff: FitFile) -> pd.DataFrame:
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


def _extract_valid_value(value: Any, invalid_value: int = 65534) -> Optional[int]:
    """Extract first valid value from tuple or return single value.

    Args:
        value: Value to extract (int, tuple, or None)
        invalid_value: Value to treat as invalid (default: 65534)

    Returns:
        First valid value, or None if all invalid
    """
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, tuple):
        for v in value:
            if v is not None and v != invalid_value:
                return v
        return None
    return value if value != invalid_value else None


# Aggregate strength sets from multiple files
def _get_session_info(fit_file: str, config: AnalysisConfig, multisport: bool):
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
            fit_file,
            ftp=config.ftp,
            hr_rest=config.hr_rest,
            hr_max=config.hr_max,
            tz_name=config.tz_name,
        )
        df_sessions, df_sets = result
        # multisport mode doesn't extract sets, fall back to original
        if not df_sets or (isinstance(df_sets, list) and not df_sets):
            _, df_sets = summarize_fit_original(
                fit_file,
                ftp=config.ftp,
                hr_rest=config.hr_rest,
                hr_max=config.hr_max,
                tz_name=config.tz_name,
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
            fit_file,
            ftp=config.ftp,
            hr_rest=config.hr_rest,
            hr_max=config.hr_max,
            tz_name=config.tz_name,
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


def _create_set_record(row: Dict[str, Any], idx: int, metadata: SetMetadata) -> Dict[str, Any]:
    """Create a set record dictionary from a row."""
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


def _aggregate_strength_sets(
    fit_files: List[str], config: AnalysisConfig, multisport: bool = False
):
    """
    Aggregate strength training sets from multiple FIT files into a single DataFrame.

    Args:
        fit_files: List of FIT file paths to process
        config: Analysis configuration with ftp, hr_rest, hr_max, tz_name
        multisport: Whether to use multisport processing

    Returns:
        DataFrame with columns: activity_id, file, date, sport, sub_sport, set_number,
        set_type, category, category_subtype, repetitions, weight, duration, timestamp
    """
    all_strength_data = []

    for fit_file in fit_files:
        # Extract session info and sets
        df_sessions, df_sets = _get_session_info(fit_file, config, multisport)

        # Skip if no strength sets found
        if df_sets is None or (isinstance(df_sets, pd.DataFrame) and df_sets.empty):
            continue

        # Process active sets with metadata
        sport, sub_sport, date = _extract_first_session_metadata(df_sessions)
        metadata = SetMetadata(
            Path(fit_file).stem.replace("_ACTIVITY", ""),
            Path(fit_file).name,
            date,
            sport,
            sub_sport,
        )

        # Add metadata to each active set
        all_strength_data.extend(
            [
                _create_set_record(row, idx, metadata)
                for idx, row in df_sets.iterrows()
                if row.get("set_type") == "active"
            ]
        )

    if not all_strength_data:
        return None

    # Return sorted dataframe
    return pd.DataFrame(all_strength_data).sort_values(["date", "timestamp"], na_position="last")


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


def _calculate_metrics_original(df, config: AnalysisConfig, start_utc, end_utc):
    """Calculate all training metrics from dataframe for original function"""
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1
    dur_hr = dur_sec / 3600.0
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    intensity_factor = (npw / config.ftp) if np.isfinite(npw) and config.ftp > 0 else np.nan

    return {
        "dur_sec": dur_sec,
        "avg_hr": float(df["hr"].mean()) if df["hr"].notna().any() else np.nan,
        "max_hr": float(df["hr"].max()) if df["hr"].notna().any() else np.nan,
        "avg_p": float(df["power"].mean()) if df["power"].notna().any() else np.nan,
        "max_p": float(df["power"].max()) if df["power"].notna().any() else np.nan,
        "npw": npw,
        "IF": intensity_factor,
        "TSS": (
            ((dur_hr * npw * intensity_factor) / config.ftp * 100)
            if np.all(np.isfinite([dur_hr, npw, intensity_factor])) and config.ftp > 0
            else np.nan
        ),
        "TRIMP": (
            trimp_from_hr(df["hr"].ffill(), hr_rest=config.hr_rest, hr_max=config.hr_max)
            if df["hr"].notna().any()
            else 0.0
        ),
    }


def summarize_fit_original(
    path: str, config: AnalysisConfig = None, **kwargs
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    """Original function for single-session activities.

    Can accept either a config object or individual parameters for backwards compatibility.
    """
    config = config or AnalysisConfig(
        ftp=kwargs.get("ftp", DEFAULT_FTP),
        hr_rest=kwargs.get("hr_rest", DEFAULT_HR_REST),
        hr_max=kwargs.get("hr_max", DEFAULT_HR_MAX),
        tz_name=kwargs.get("tz_name", DEFAULT_TIMEZONE),
    )

    ff = FitFile(path)
    df = _extract_records_from_fit(ff)
    df_sets = extract_sets_from_fit(ff, fit_file_path=path)

    if df.empty:
        return None, df_sets

    start_utc, end_utc, time_index = _prepare_timezone_aware_index(df)
    metrics = _calculate_metrics_original(
        df.set_index(time_index).sort_index().resample("1s").ffill(), config, start_utc, end_utc
    )

    sport, subsport = _get_sport_names(_extract_sessions_from_fit(ff))
    start_local = start_utc.astimezone(tz.gettz(config.tz_name))

    return {
        "file": path,
        "sport": sport,
        "sub_sport": subsport,
        "date": start_local.date().isoformat(),
        "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_utc.astimezone(tz.gettz(config.tz_name)).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(metrics["dur_sec"] / 60.0, 1),
        "avg_hr": round(metrics["avg_hr"], 1) if np.isfinite(metrics["avg_hr"]) else "",
        "max_hr": int(metrics["max_hr"]) if np.isfinite(metrics["max_hr"]) else "",
        "avg_power_w": round(metrics["avg_p"], 1) if np.isfinite(metrics["avg_p"]) else "",
        "max_power_w": round(metrics["max_p"], 1) if np.isfinite(metrics["max_p"]) else "",
        "np_w": round(metrics["npw"], 1) if np.isfinite(metrics["npw"]) else "",
        "IF": round(metrics["IF"], 3) if np.isfinite(metrics["IF"]) else "",
        "TSS": round(metrics["TSS"], 1) if np.isfinite(metrics["TSS"]) else "",
        "TRIMP": round(metrics["TRIMP"], 1),
    }, df_sets


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
            csv_file = save_strength_sets_csv(fit_file, df_sets)
            if csv_file:
                all_sets.append(csv_file)

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
        config = AnalysisConfig(
            ftp=args.ftp, hr_rest=args.hrrest, hr_max=args.hrmax, tz_name=args.tz
        )
        df_strength_summary = _aggregate_strength_sets(
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
