"""
Activity-level processing for FIT files.

This module handles processing complete activities (single-sport and multisport),
combining parsed data with session processing to create activity summaries.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import tz
from fitparse import FitFile

from fitanalyzer.analysis import calculate_all_data_metrics
from fitanalyzer.config import AnalysisConfig
from fitanalyzer.constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST, DEFAULT_TIMEZONE
from fitanalyzer.formatting import (
    calculate_basic_hr_power_metrics,
    convert_timestamps_to_utc,
    format_all_data_metrics,
    format_metric_value,
)
from fitanalyzer.metrics import np_power, trimp_from_hr
from fitanalyzer.parser import (
    create_record_dict,
    extract_records_from_fit,
    extract_sessions_from_fit,
    get_sport_names,
)
from fitanalyzer.sessions import process_session_data

if TYPE_CHECKING:
    from pandas import Series

__all__ = [
    "summarize_fit_sessions",
    "summarize_fit_original",
]


def summarize_fit_sessions(
    path: str, config: AnalysisConfig | None = None, **kwargs: Any
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
    sessions = extract_sessions_from_fit(ff)

    # If no sessions or only one session, fall back to original behavior
    if len(sessions) <= 1:
        result = summarize_fit_original(path, config)
        return ([result] if result else []), []

    # Process each session separately
    results = []

    for session_idx, session in enumerate(sessions):
        if not (session_start := session.get("start_time")):
            continue
        if (session_timer_time := session.get("total_timer_time", 0)) <= 0:
            continue

        # Extract records for this session using shared record creation logic
        if (
            recs := [
                create_record_dict(d)
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


def _prepare_timezone_aware_index(
    df: pd.DataFrame,
) -> Tuple[datetime, datetime, "Series[Any]"]:
    """Prepare timezone-aware time index for the dataframe.

    Returns:
        Tuple of (start_utc, end_utc, time_index)
    """
    start_utc, end_utc = convert_timestamps_to_utc(df)

    # Set index with timezone handling
    time_series = pd.to_datetime(df["time"])
    if time_series.dt.tz is None:
        time_index = time_series.dt.tz_localize("UTC")
    else:
        time_index = time_series.dt.tz_convert("UTC")

    return start_utc, end_utc, time_index


def _calculate_metrics_original(
    df: Any, config: AnalysisConfig, start_utc: Any, end_utc: Any
) -> Dict[str, Any]:
    """Calculate all training metrics from dataframe for original function"""
    dur_sec = int((end_utc - start_utc).total_seconds()) + 1
    dur_hr = dur_sec / 3600.0
    npw = np_power(df["power"].fillna(0)) if df["power"].notna().any() else np.nan
    intensity_factor = (npw / config.ftp) if np.isfinite(npw) and config.ftp > 0 else np.nan

    # Calculate all data metrics using shared function
    metrics = calculate_all_data_metrics(df)

    # Calculate basic HR and power metrics
    basic_metrics = calculate_basic_hr_power_metrics(df)

    metrics.update(
        {
            "dur_sec": dur_sec,
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
    )
    metrics.update(basic_metrics)
    return metrics


def summarize_fit_original(
    path: str, config: AnalysisConfig | None = None, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Original function for single-session activities.

    Can accept either a config object or individual parameters for backwards compatibility.

    Returns:
        Activity summary dictionary, or None if no data

    Note: This function no longer returns strength sets. Use extract_sets_from_fit()
          from the strength module directly if you need strength training data.
    """
    config = config or AnalysisConfig(
        ftp=kwargs.get("ftp", DEFAULT_FTP),
        hr_rest=kwargs.get("hr_rest", DEFAULT_HR_REST),
        hr_max=kwargs.get("hr_max", DEFAULT_HR_MAX),
        tz_name=kwargs.get("tz_name", DEFAULT_TIMEZONE),
    )

    ff = FitFile(path)
    df = extract_records_from_fit(ff)

    if df.empty:
        return None

    start_utc, end_utc, time_index = _prepare_timezone_aware_index(df)
    metrics = _calculate_metrics_original(
        df.set_index(time_index).sort_index().resample("1s").ffill(), config, start_utc, end_utc
    )

    sport, subsport = get_sport_names(extract_sessions_from_fit(ff))
    start_local = start_utc.astimezone(tz.gettz(config.tz_name))

    result = {
        "file": path,
        "sport": sport,
        "sub_sport": subsport,
        "date": start_local.date().isoformat(),
        "start_time": start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_utc.astimezone(tz.gettz(config.tz_name)).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_min": round(metrics["dur_sec"] / 60.0, 1),
        "IF": format_metric_value(metrics["IF"], 3),
        "TSS": format_metric_value(metrics["TSS"], 1),
        "TRIMP": round(metrics["TRIMP"], 1),
    }

    # Add formatted metrics using shared helper function
    result.update(format_all_data_metrics(metrics))

    return result
