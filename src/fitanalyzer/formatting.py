"""
Formatting utilities for activity and session metrics.

This module provides common formatting functions to avoid code duplication
between activities and sessions modules.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from dateutil import tz


def calculate_basic_hr_power_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic heart rate and power metrics from DataFrame.

    Args:
        df: DataFrame with 'hr' and 'power' columns

    Returns:
        Dictionary with avg_hr, max_hr, avg_p, max_p
    """
    return {
        "avg_hr": float(df["hr"].mean()) if df["hr"].notna().any() else np.nan,
        "max_hr": float(df["hr"].max()) if df["hr"].notna().any() else np.nan,
        "avg_p": float(df["power"].mean()) if df["power"].notna().any() else np.nan,
        "max_p": float(df["power"].max()) if df["power"].notna().any() else np.nan,
    }


def convert_timestamps_to_utc(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Convert dataframe timestamps to timezone-aware UTC.

    Args:
        df: DataFrame with 'time' column

    Returns:
        Tuple of (start_utc, end_utc) as timezone-aware timestamps
    """
    start_time = pd.to_datetime(df["time"].iloc[0])
    end_time = pd.to_datetime(df["time"].iloc[-1])

    if start_time.tzinfo is None:
        start_utc = start_time.tz_localize("UTC")
        end_utc = end_time.tz_localize("UTC")
    else:
        start_utc = start_time.tz_convert("UTC") if start_time.tzinfo != tz.UTC else start_time
        end_utc = end_time.tz_convert("UTC") if end_time.tzinfo != tz.UTC else end_time

    return start_utc, end_utc


def format_metric_value(value: float, decimals: int = 1, as_int: bool = False) -> Any:
    """Format a metric value, returning empty string if not finite.

    Args:
        value: The metric value to format
        decimals: Number of decimal places (ignored if as_int=True)
        as_int: Whether to format as integer

    Returns:
        Formatted value or empty string if not finite
    """
    if not np.isfinite(value):
        return ""
    if as_int:
        return int(value)
    return round(value, decimals)


def format_speed_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Format speed-related metrics.

    Args:
        metrics: Dictionary containing speed metrics

    Returns:
        Dictionary with formatted speed metrics
    """
    return {
        "avg_speed_mps": format_metric_value(metrics.get("avg_speed_mps", np.nan), 2),
        "max_speed_mps": format_metric_value(metrics.get("max_speed_mps", np.nan), 2),
        "avg_speed_kph": format_metric_value(metrics.get("avg_speed_kph", np.nan), 2),
        "max_speed_kph": format_metric_value(metrics.get("max_speed_kph", np.nan), 2),
    }


def format_cadence_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Format cadence-related metrics.

    Args:
        metrics: Dictionary containing cadence metrics

    Returns:
        Dictionary with formatted cadence metrics
    """
    return {
        "avg_cadence": format_metric_value(metrics.get("avg_cadence", np.nan), 1),
        "max_cadence": format_metric_value(metrics.get("max_cadence", np.nan), as_int=True),
    }


def format_distance_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Format distance-related metrics.

    Args:
        metrics: Dictionary containing distance metrics

    Returns:
        Dictionary with formatted distance metrics
    """
    return {
        "total_distance_m": format_metric_value(metrics.get("total_distance_m", np.nan), 1),
        "total_distance_km": format_metric_value(metrics.get("total_distance_km", np.nan), 3),
    }


def format_elevation_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Format elevation-related metrics.

    Args:
        metrics: Dictionary containing elevation metrics

    Returns:
        Dictionary with formatted elevation metrics
    """
    return {
        "total_ascent_m": format_metric_value(metrics.get("total_ascent_m", np.nan), 1),
        "total_descent_m": format_metric_value(metrics.get("total_descent_m", np.nan), 1),
        "avg_altitude_m": format_metric_value(metrics.get("avg_altitude_m", np.nan), 1),
        "min_altitude_m": format_metric_value(metrics.get("min_altitude_m", np.nan), 1),
        "max_altitude_m": format_metric_value(metrics.get("max_altitude_m", np.nan), 1),
    }


def format_hr_power_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Format heart rate and power-related metrics.

    Args:
        metrics: Dictionary containing HR and power metrics

    Returns:
        Dictionary with formatted HR and power metrics
    """
    return {
        "avg_hr": format_metric_value(metrics.get("avg_hr", np.nan), 1),
        "max_hr": format_metric_value(metrics.get("max_hr", np.nan), as_int=True),
        "avg_power_w": format_metric_value(metrics.get("avg_p", np.nan), 1),
        "max_power_w": format_metric_value(metrics.get("max_p", np.nan), 1),
        "np_w": format_metric_value(metrics.get("npw", np.nan), 1),
    }
