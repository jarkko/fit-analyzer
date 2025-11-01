"""
Session-level processing for FIT activities.

This module handles processing of individual workout sessions,
including timestamp handling, metric calculation, and data formatting.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dateutil import tz

from fitanalyzer.analysis import calculate_all_data_metrics
from fitanalyzer.config import AnalysisConfig
from fitanalyzer.constants import SPORT_MAPPING, SUB_SPORT_MAPPING
from fitanalyzer.formatting import (
    calculate_basic_hr_power_metrics,
    convert_timestamps_to_utc,
    format_all_data_metrics,
    format_metric_value,
)
from fitanalyzer.metrics import np_power, trimp_from_hr


def process_timestamps(df: pd.DataFrame, tz_name: str) -> Dict[str, Any]:
    """Extract and convert timestamps from DataFrame.

    Returns dict with start_utc, end_utc, start_local, end_local, dur_sec, dur_hr
    """
    local = tz.gettz(tz_name)
    start_utc, end_utc = convert_timestamps_to_utc(df)
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1

    return {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start_local": start_utc.astimezone(local),
        "end_local": end_utc.astimezone(local),
        "dur_sec": dur_sec,
        "dur_hr": dur_sec / 3600.0,
    }


def map_sport_names(session: Dict[str, Any]) -> tuple[str, str]:
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


def create_file_display(path: str, session_idx: int, sport: str, subsport: str) -> str:
    """Create display filename for session."""
    base_name = Path(path).stem
    if subsport and subsport != "generic":
        return f"{base_name}_session{session_idx}_{sport}_{subsport}"
    return f"{base_name}_session{session_idx}_{sport}"


def calculate_session_metrics(
    df: pd.DataFrame, dur_hr: float, config: AnalysisConfig
) -> Dict[str, float]:
    """Calculate power, heart rate, speed, cadence, distance, and elevation metrics.

    Args:
        df: Resampled DataFrame with hr, power, speed, cadence, distance, altitude
        dur_hr: Duration in hours
        config: Analysis configuration with ftp, hr_rest, hr_max

    Returns:
        Dictionary with all metrics: speed, cadence, distance, elevation
    """
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    intensity_factor = (npw / config.ftp) if np.isfinite(npw) and config.ftp > 0 else np.nan

    # Calculate all data metrics using shared function
    metrics = calculate_all_data_metrics(df)

    # Add power and timing metrics
    metrics.update(
        {
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
    )

    # Add basic HR and power metrics
    metrics.update(calculate_basic_hr_power_metrics(df))

    return metrics


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
    times = process_timestamps(df, config.tz_name)

    # Resample to 1 second for NP calculation
    time_series = pd.to_datetime(df["time"])
    time_index = (
        time_series.dt.tz_localize("UTC")
        if time_series.dt.tz is None
        else time_series.dt.tz_convert("UTC")
    )
    df = df.set_index(time_index).sort_index().resample("1s").ffill()

    # Calculate all metrics
    metrics = calculate_session_metrics(df, times["dur_hr"], config)

    # Map sport names and create filename
    sport, subsport = map_sport_names(session)
    file_display = create_file_display(path, session_idx, sport, subsport)

    result = {
        "file": file_display,
        "sport": sport,
        "sub_sport": subsport,
        "date": times["start_local"].date().isoformat(),
        "start_time": times["start_local"].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": times["end_local"].strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(times["dur_sec"] / 60.0, 1),
        "IF": format_metric_value(metrics.get("intensity_factor", np.nan), 3),
        "TSS": format_metric_value(metrics.get("tss", np.nan), 1),
        "TRIMP": round(metrics["trimp"], 1),
        # Keep these for deduplication logic
        "_original_file": path,
        "_session_index": session_idx,
    }

    # Add formatted metrics using shared helper function
    result.update(format_all_data_metrics(metrics))

    return result
