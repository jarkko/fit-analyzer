"""
Training load calculations based on the Fitness-Fatigue Model.

Implements Chronic Training Load (CTL), Acute Training Load (ATL),
and Training Stress Balance (TSB) calculations using exponentially
weighted moving averages.

Scientific Background:
- Banister et al. (1975) - Impulse-response model of training
- Coggan (2003) - Performance Manager Chart (PMC)
- CTL: Long-term fitness (42-day time constant)
- ATL: Short-term fatigue (7-day time constant)
- TSB: Form/readiness (CTL - ATL)
"""

from typing import List

import numpy as np
import pandas as pd


def calculate_ctl(training_loads: List[float], time_constant: int = 42) -> np.ndarray:
    """Calculate Chronic Training Load (fitness) using exponential weighted moving average.

    CTL represents long-term training adaptations. Uses a 42-day time constant
    by default, meaning fitness has a half-life of ~29 days.

    Formula: CTL_today = CTL_yesterday + (Load_today - CTL_yesterday) / time_constant

    Args:
        training_loads: List of daily training load values (TSS or TRIMP)
        time_constant: Number of days for the time constant (default: 42)

    Returns:
        NumPy array of CTL values, one per day
    """
    if not training_loads:
        return np.array([])

    ctl = np.zeros(len(training_loads))
    ctl[0] = training_loads[0] / time_constant  # Start from zero baseline

    for i in range(1, len(training_loads)):
        ctl[i] = ctl[i - 1] + (training_loads[i] - ctl[i - 1]) / time_constant

    return ctl


def calculate_atl(training_loads: List[float], time_constant: int = 7) -> np.ndarray:
    """Calculate Acute Training Load (fatigue) using exponential weighted moving average.

    ATL represents short-term fatigue. Uses a 7-day time constant by default,
    meaning fatigue has a half-life of ~5 days.

    Formula: ATL_today = ATL_yesterday + (Load_today - ATL_yesterday) / time_constant

    Args:
        training_loads: List of daily training load values (TSS or TRIMP)
        time_constant: Number of days for the time constant (default: 7)

    Returns:
        NumPy array of ATL values, one per day
    """
    if not training_loads:
        return np.array([])

    atl = np.zeros(len(training_loads))
    atl[0] = training_loads[0] / time_constant  # Start from zero baseline

    for i in range(1, len(training_loads)):
        atl[i] = atl[i - 1] + (training_loads[i] - atl[i - 1]) / time_constant

    return atl


def calculate_tsb(ctl: np.ndarray, atl: np.ndarray) -> np.ndarray:
    """Calculate Training Stress Balance (form/readiness).

    TSB = CTL - ATL represents the balance between fitness and fatigue.

    Interpretation:
    - Positive TSB: Fresh, peaked, ready to perform
    - Zero TSB: Balanced training and fatigue
    - Negative TSB: Fatigued, in heavy training phase

    Typical race preparation: Build negative TSB (-20 to -40) during training,
    then taper to achieve positive TSB (+5 to +25) for race day.

    Args:
        ctl: Chronic Training Load array
        atl: Acute Training Load array

    Returns:
        NumPy array of TSB values

    Raises:
        ValueError: If CTL and ATL arrays have different lengths
    """
    if len(ctl) != len(atl):
        raise ValueError("CTL and ATL arrays must have the same length")

    return ctl - atl  # type: ignore[no-any-return]


def _create_daily_dataframe(
    df: pd.DataFrame, load_column: str = "tss", date_column: str = "date"
) -> pd.DataFrame:
    """Create a daily DataFrame with rest days filled in as zero load.

    Takes a DataFrame with workouts and creates a new DataFrame with one row
    per day, filling gaps with zero training load for rest days.

    Args:
        df: DataFrame with workout data
        load_column: Name of column containing training load
        date_column: Name of column containing workout dates

    Returns:
        DataFrame with daily rows, rest days have zero load
    """
    if df.empty:
        return df.copy()

    # Sort by date
    df_sorted = df.sort_values(date_column).copy()

    # Convert dates to datetime if they aren't already
    df_sorted[date_column] = pd.to_datetime(df_sorted[date_column])

    # Create date range from first to last workout
    min_date = df_sorted[date_column].min()
    max_date = df_sorted[date_column].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    # Create daily dataframe with all dates
    daily_df = pd.DataFrame({date_column: all_dates})

    # Merge with original data, filling missing values with 0
    daily_df = daily_df.merge(df_sorted[[date_column, load_column]], on=date_column, how="left")
    daily_df[load_column] = daily_df[load_column].fillna(0)

    return daily_df


def _validate_and_prepare_dataframe(
    df: pd.DataFrame, load_column: str, date_column: str
) -> pd.DataFrame:
    """Validate columns and prepare DataFrame for training load calculation.

    Args:
        df: Input DataFrame
        load_column: Name of column containing training load
        date_column: Name of column containing workout dates

    Returns:
        Cleaned and sorted DataFrame ready for processing

    Raises:
        KeyError: If required columns are missing
    """
    if date_column not in df.columns:
        raise KeyError(f"DataFrame must have '{date_column}' column")
    if load_column not in df.columns:
        raise KeyError(f"DataFrame must have '{load_column}' column")

    # Drop existing training load columns to avoid duplication
    df_clean = df.copy()
    for col in ["ctl", "atl", "tsb"]:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Sort by date and convert load column to numeric
    df_sorted = df_clean.sort_values(date_column).copy()
    df_sorted[load_column] = pd.to_numeric(df_sorted[load_column], errors="coerce").fillna(0)

    return df_sorted


def _calculate_daily_metrics(daily_df: pd.DataFrame, load_column: str) -> pd.DataFrame:
    """Calculate CTL, ATL, and TSB for daily training load data.

    Args:
        daily_df: DataFrame with daily training loads (including rest days)
        load_column: Name of column containing training load values

    Returns:
        DataFrame with added 'ctl', 'atl', 'tsb' columns
    """
    training_loads = daily_df[load_column].tolist()

    ctl = calculate_ctl(training_loads)
    atl = calculate_atl(training_loads)
    tsb = calculate_tsb(ctl, atl)

    daily_df["ctl"] = np.round(ctl, 4)
    daily_df["atl"] = np.round(atl, 4)
    daily_df["tsb"] = np.round(tsb, 4)

    return daily_df


def _map_metrics_to_workouts(
    df_workouts: pd.DataFrame, df_metrics: pd.DataFrame, date_column: str
) -> pd.DataFrame:
    """Map daily metrics back to individual workouts by date.

    Args:
        df_workouts: DataFrame with workout data
        df_metrics: DataFrame with daily metrics (CTL, ATL, TSB)
        date_column: Name of date column

    Returns:
        Workout DataFrame with metrics added
    """
    df_workouts[date_column] = pd.to_datetime(df_workouts[date_column])
    df_metrics[date_column] = pd.to_datetime(df_metrics[date_column])

    # Create mapping from date to metrics
    metrics_dict = df_metrics.groupby(date_column)[["ctl", "atl", "tsb"]].last().to_dict("index")

    # Apply metrics to each workout
    df_workouts["ctl"] = df_workouts[date_column].apply(
        lambda d: metrics_dict.get(d, {}).get("ctl", 0)
    )
    df_workouts["atl"] = df_workouts[date_column].apply(
        lambda d: metrics_dict.get(d, {}).get("atl", 0)
    )
    df_workouts["tsb"] = df_workouts[date_column].apply(
        lambda d: metrics_dict.get(d, {}).get("tsb", 0)
    )

    return df_workouts


def calculate_training_load_metrics(
    df: pd.DataFrame, load_column: str = "tss", date_column: str = "date"
) -> pd.DataFrame:
    """Calculate CTL, ATL, and TSB for a DataFrame of workouts.

    Takes a DataFrame with workout data and adds cumulative training load metrics.
    Workouts are sorted by date before calculation. Date gaps are filled with
    zero training load for rest days, allowing proper decay of ATL/CTL.

    IMPORTANT: This function now accounts for rest days! If you have workouts
    separated by multiple days, the ATL and CTL will properly decay during
    the rest period. This fixes the bug where consecutive workouts in the data
    were treated as consecutive days even if they had rest days between them.

    Args:
        df: DataFrame with workout data
        load_column: Name of column containing training load (TSS or TRIMP)
        date_column: Name of column containing workout dates

    Returns:
        DataFrame with added columns: 'ctl', 'atl', 'tsb'
        Note: Returns only workout days (not rest days), but calculations
        include rest day decay.

    Raises:
        KeyError: If required columns are missing
    """
    if df.empty:
        result = df.copy()
        result["ctl"] = []
        result["atl"] = []
        result["tsb"] = []
        return result

    df_sorted = _validate_and_prepare_dataframe(df, load_column, date_column)
    daily_df = _create_daily_dataframe(df_sorted, load_column, date_column)
    daily_df = _calculate_daily_metrics(daily_df, load_column)
    df_result = _map_metrics_to_workouts(df_sorted, daily_df, date_column)

    return df_result
