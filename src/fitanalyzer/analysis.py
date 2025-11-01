"""
Workout analysis and metrics calculation.

This module handles the calculation of training metrics from parsed FIT data,
including speed, cadence, distance, elevation, and training load metrics.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from fitanalyzer.metrics import np_power, trimp_from_hr


def calc_speed_metrics(df: Any) -> Dict[str, float]:
    """Calculate speed metrics from dataframe.

    Args:
        df: DataFrame with 'speed' column (in m/s)

    Returns:
        Dictionary with speed metrics in m/s and km/h
    """
    if "speed" not in df.columns:
        return {
            "avg_speed_mps": np.nan,
            "max_speed_mps": np.nan,
            "avg_speed_kph": np.nan,
            "max_speed_kph": np.nan,
        }

    speed_data = df["speed"].dropna()
    if speed_data.empty:
        return {
            "avg_speed_mps": np.nan,
            "max_speed_mps": np.nan,
            "avg_speed_kph": np.nan,
            "max_speed_kph": np.nan,
        }

    avg_speed_mps = speed_data.mean()
    max_speed_mps = speed_data.max()

    return {
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
        "avg_speed_kph": avg_speed_mps * 3.6,  # Convert m/s to km/h
        "max_speed_kph": max_speed_mps * 3.6,
    }


def calc_cadence_metrics(df: Any) -> Dict[str, float]:
    """Calculate cadence metrics from dataframe.

    Args:
        df: DataFrame with 'cadence' column (in rpm)

    Returns:
        Dictionary with average and maximum cadence
    """
    if "cadence" not in df.columns:
        return {
            "avg_cadence": np.nan,
            "max_cadence": np.nan,
        }

    cadence_data = df["cadence"].dropna()
    if cadence_data.empty:
        return {
            "avg_cadence": np.nan,
            "max_cadence": np.nan,
        }

    return {
        "avg_cadence": cadence_data.mean(),
        "max_cadence": cadence_data.max(),
    }


def calc_distance_metrics(df: Any) -> Dict[str, float]:
    """Calculate distance metrics from dataframe.

    Args:
        df: DataFrame with 'distance' column (in meters)

    Returns:
        Dictionary with total distance in meters and kilometers
    """
    if "distance" not in df.columns:
        return {
            "total_distance_m": np.nan,
            "total_distance_km": np.nan,
        }

    distance_data = df["distance"].dropna()
    if distance_data.empty:
        return {
            "total_distance_m": np.nan,
            "total_distance_km": np.nan,
        }

    # Distance is cumulative, so total is max - min
    total_distance_m = distance_data.max() - distance_data.min()

    return {
        "total_distance_m": total_distance_m,
        "total_distance_km": total_distance_m / 1000.0,
    }


def calc_elevation_metrics(df: Any) -> Dict[str, float]:
    """Calculate elevation metrics from dataframe.

    Args:
        df: DataFrame with 'altitude' column (in meters)

    Returns:
        Dictionary with ascent, descent, and altitude statistics
    """
    if "altitude" not in df.columns:
        return {
            "total_ascent_m": np.nan,
            "total_descent_m": np.nan,
            "avg_altitude_m": np.nan,
            "min_altitude_m": np.nan,
            "max_altitude_m": np.nan,
        }

    altitude_data = df["altitude"].dropna()
    if altitude_data.empty:
        return {
            "total_ascent_m": np.nan,
            "total_descent_m": np.nan,
            "avg_altitude_m": np.nan,
            "min_altitude_m": np.nan,
            "max_altitude_m": np.nan,
        }

    # Calculate elevation changes
    altitude_diff = altitude_data.diff()

    # Sum positive changes for ascent, negative for descent
    total_ascent = altitude_diff[altitude_diff > 0].sum()
    total_descent = abs(altitude_diff[altitude_diff < 0].sum())

    return {
        "total_ascent_m": total_ascent,
        "total_descent_m": total_descent,
        "avg_altitude_m": altitude_data.mean(),
        "min_altitude_m": altitude_data.min(),
        "max_altitude_m": altitude_data.max(),
    }


def _calc_power_metrics(df: Any, ftp: float) -> Dict[str, float]:
    """Calculate power metrics."""
    power_data = df["power"].dropna() if "power" in df.columns else pd.Series(dtype=float)
    if not power_data.empty:
        avg_power = power_data.mean()
        max_power = power_data.max()
        npw = np_power(power_data.values)
        intensity_factor = (npw / ftp) if ftp > 0 else np.nan
        tss = ((len(power_data) * npw * intensity_factor) / (ftp * 36.0)) if ftp > 0 else np.nan
    else:
        avg_power = max_power = npw = intensity_factor = tss = np.nan
    return {
        "avg_power": avg_power,
        "max_power": max_power,
        "normalized_power": npw,
        "intensity_factor": intensity_factor,
        "tss": tss,
    }


def _calc_hr_metrics(df: Any, hr_rest: int, hr_max: int) -> Dict[str, float]:
    """Calculate heart rate metrics."""
    hr_data = df["heart_rate"].dropna() if "heart_rate" in df.columns else pd.Series(dtype=float)
    if not hr_data.empty:
        avg_hr = hr_data.mean()
        max_hr = hr_data.max()
        trimp_val = trimp_from_hr(hr_data.values, hr_rest, hr_max)
    else:
        avg_hr = max_hr = trimp_val = np.nan
    return {
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "trimp": trimp_val,
    }


def calculate_metrics_for_session(df: Any, ftp: float, hr_rest: int, hr_max: int) -> Dict[str, Any]:
    """Calculate all metrics for a workout session.

    Args:
        df: DataFrame with workout data
        ftp: Functional Threshold Power
        hr_rest: Resting heart rate
        hr_max: Maximum heart rate

    Returns:
        Dictionary with all calculated metrics
    """
    power = _calc_power_metrics(df, ftp)
    hr = _calc_hr_metrics(df, hr_rest, hr_max)
    speed = calc_speed_metrics(df)
    cadence = calc_cadence_metrics(df)
    distance = calc_distance_metrics(df)
    elevation = calc_elevation_metrics(df)

    return {
        "avg_power_w": power["avg_power"],
        "max_power_w": power["max_power"],
        "np_w": power["normalized_power"],
        "IF": power["intensity_factor"],
        "TSS": power["tss"],
        "avg_hr": hr["avg_hr"],
        "max_hr": hr["max_hr"],
        "TRIMP": hr["trimp"],
        **speed,
        **cadence,
        **distance,
        **elevation,
    }
