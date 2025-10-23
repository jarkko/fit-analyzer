"""
Training metrics calculations for FIT file analysis.

This module provides functions to calculate various training metrics including
Normalized Power (NP), Training Impulse (TRIMP), Training Stress Score (TSS),
and Intensity Factor (IF).
"""

import numpy as np
import pandas as pd

from fitanalyzer.constants import DEFAULT_HR_REST, DEFAULT_HR_MAX


def np_power(power: pd.Series) -> float:
    """Calculate Normalized Power (NP) for cycling power data.

    Normalized Power is a weighted average of power output that better represents
    the physiological cost of variable-intensity cycling than simple average power.
    It uses a 30-second rolling average raised to the 4th power to emphasize
    hard efforts, making it more accurate for interval-based workouts.

    The algorithm:
    1. Calculate 30-second rolling average of power
    2. Raise each value to the 4th power
    3. Take the mean of these values
    4. Take the 4th root of the result

    Args:
        power: Series of power values in watts, typically at 1-second intervals.
               Can contain NaN values which will be handled gracefully.

    Returns:
        Normalized Power value in watts as a float. Returns np.nan if:
        - Input series is empty
        - All values are NaN
        - Calculation cannot be completed

    Example:
        >>> import pandas as pd
        >>> power_data = pd.Series([200, 250, 300, 200, 150])
        >>> np_power(power_data)
        223.6  # Example value, actual result may vary

    References:
        TrainingPeaks Normalized Power® concept by Dr. Andrew Coggan
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

    TRIMP (Training Impulse) quantifies cardiovascular training load based on
    heart rate intensity and duration. It uses exponential weighting to emphasize
    time spent at higher intensities, providing a single metric for training stress.

    The algorithm uses Banister's exponential TRIMP formula:
    1. Calculate Heart Rate Reserve (HRR) = (HR - HR_rest) / (HR_max - HR_rest)
    2. Apply exponential weighting: 0.64 * exp(1.92 * HRR)
    3. Multiply by duration in minutes

    Args:
        hr: Series of heart rate values in beats per minute (bpm).
            Assumes 1-second sampling intervals. NaN values are dropped.
        hr_rest: Resting heart rate in bpm. Used to calculate heart rate reserve.
                 Default is DEFAULT_HR_REST from constants.
        hr_max: Maximum heart rate in bpm. Used to calculate heart rate reserve.
                Default is DEFAULT_HR_MAX from constants.

    Returns:
        TRIMP value as a float representing training load. Higher values indicate
        greater training stress. Returns 0.0 if:
        - Input series is empty
        - All values are NaN after cleaning
        - Duration is zero

    Example:
        >>> import pandas as pd
        >>> hr_data = pd.Series([140, 150, 160, 155, 145])
        >>> trimp_from_hr(hr_data, hr_rest=60, hr_max=190)
        45.2  # Example value representing moderate training load

    References:
        Banister, E.W. (1991). Modeling elite athletic performance.
        Physiological Testing of Elite Athletes.
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


def calculate_tss(
    normalized_power: float, intensity_factor: float, duration_hours: float, ftp: float
) -> float:
    """Calculate Training Stress Score (TSS).

    Args:
        normalized_power: Normalized Power in watts
        intensity_factor: Intensity Factor (NP / FTP)
        duration_hours: Duration in hours
        ftp: Functional Threshold Power in watts

    Returns:
        TSS value, or np.nan if inputs are invalid
    """
    if not np.all(np.isfinite([duration_hours, normalized_power, intensity_factor])) or ftp <= 0:
        return np.nan
    return (duration_hours * normalized_power * intensity_factor) / ftp * 100


def calculate_intensity_factor(normalized_power: float, ftp: float) -> float:
    """Calculate Intensity Factor (IF).

    Args:
        normalized_power: Normalized Power in watts
        ftp: Functional Threshold Power in watts

    Returns:
        Intensity Factor (ratio of NP to FTP), or np.nan if invalid
    """
    if not np.isfinite(normalized_power) or ftp <= 0:
        return np.nan
    return normalized_power / ftp
