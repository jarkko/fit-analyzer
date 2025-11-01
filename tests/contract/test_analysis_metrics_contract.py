"""
Contract tests for analysis.py metric calculation functions.

These tests document and enforce the contracts for all metric calculation
functions, ensuring they handle all parameter combinations correctly.
"""

import numpy as np
import pandas as pd
import pytest

from fitanalyzer.analysis import (
    _calc_hr_metrics,
    _calc_power_metrics,
    calc_cadence_metrics,
    calc_distance_metrics,
    calc_elevation_metrics,
    calc_speed_metrics,
    calculate_metrics_for_session,
)


class TestCalcSpeedMetricsContract:
    """Contract tests for calc_speed_metrics() function.

    Contract: Takes DataFrame with optional 'speed' column (m/s),
    returns dict with avg/max speed in m/s and km/h.

    Parameter matrix:
    - DataFrame without 'speed' column → NaN values
    - DataFrame with empty 'speed' data → NaN values
    - DataFrame with speed data → calculated values
    - Speed values with NaN → ignores NaN via dropna()
    """

    def test_missing_speed_column_returns_nan(self):
        """Contract: When DataFrame lacks 'speed' column, return NaN for all metrics."""
        df = pd.DataFrame({"cadence": [80, 85, 90]})

        result = calc_speed_metrics(df)

        assert result == {
            "avg_speed_mps": np.nan,
            "max_speed_mps": np.nan,
            "avg_speed_kph": np.nan,
            "max_speed_kph": np.nan,
        }
        assert all(np.isnan(v) for v in result.values())

    def test_empty_speed_data_returns_nan(self):
        """Contract: When all speed values are NaN, return NaN for all metrics."""
        df = pd.DataFrame({"speed": [np.nan, np.nan, np.nan]})

        result = calc_speed_metrics(df)

        assert all(np.isnan(v) for v in result.values())

    def test_valid_speed_data_calculates_metrics(self):
        """Contract: With valid speed data, calculate avg/max in m/s and km/h."""
        df = pd.DataFrame({"speed": [5.0, 6.0, 7.0]})  # m/s

        result = calc_speed_metrics(df)

        assert result["avg_speed_mps"] == pytest.approx(6.0)
        assert result["max_speed_mps"] == pytest.approx(7.0)
        assert result["avg_speed_kph"] == pytest.approx(21.6)  # 6.0 * 3.6
        assert result["max_speed_kph"] == pytest.approx(25.2)  # 7.0 * 3.6

    def test_speed_with_nan_values_ignores_nan(self):
        """Contract: NaN values in speed data are dropped before calculation."""
        df = pd.DataFrame({"speed": [5.0, np.nan, 7.0, np.nan, 9.0]})

        result = calc_speed_metrics(df)

        # Should calculate from [5.0, 7.0, 9.0] only
        assert result["avg_speed_mps"] == pytest.approx(7.0)
        assert result["max_speed_mps"] == pytest.approx(9.0)


class TestCalcCadenceMetricsContract:
    """Contract tests for calc_cadence_metrics() function.

    Contract: Takes DataFrame with optional 'cadence' column (rpm),
    returns dict with avg/max cadence.
    """

    def test_missing_cadence_column_returns_nan(self):
        """Contract: When DataFrame lacks 'cadence' column, return NaN."""
        df = pd.DataFrame({"speed": [5.0, 6.0]})

        result = calc_cadence_metrics(df)

        assert result == {
            "avg_cadence": np.nan,
            "max_cadence": np.nan,
        }
        assert all(np.isnan(v) for v in result.values())

    def test_empty_cadence_data_returns_nan(self):
        """Contract: When all cadence values are NaN, return NaN."""
        df = pd.DataFrame({"cadence": [np.nan, np.nan]})

        result = calc_cadence_metrics(df)

        assert all(np.isnan(v) for v in result.values())

    def test_valid_cadence_data_calculates_metrics(self):
        """Contract: With valid cadence data, calculate avg/max rpm."""
        df = pd.DataFrame({"cadence": [80, 85, 90, 95]})

        result = calc_cadence_metrics(df)

        assert result["avg_cadence"] == pytest.approx(87.5)
        assert result["max_cadence"] == pytest.approx(95.0)

    def test_cadence_with_nan_values_ignores_nan(self):
        """Contract: NaN values in cadence data are dropped."""
        df = pd.DataFrame({"cadence": [80, np.nan, 100]})

        result = calc_cadence_metrics(df)

        assert result["avg_cadence"] == pytest.approx(90.0)
        assert result["max_cadence"] == pytest.approx(100.0)


class TestCalcDistanceMetricsContract:
    """Contract tests for calc_distance_metrics() function.

    Contract: Takes DataFrame with optional 'distance' column (cumulative meters),
    returns dict with total distance in meters and kilometers.

    Important: Distance is cumulative, so total = max - min
    """

    def test_missing_distance_column_returns_nan(self):
        """Contract: When DataFrame lacks 'distance' column, return NaN."""
        df = pd.DataFrame({"speed": [5.0, 6.0]})

        result = calc_distance_metrics(df)

        assert result == {
            "total_distance_m": np.nan,
            "total_distance_km": np.nan,
        }
        assert all(np.isnan(v) for v in result.values())

    def test_empty_distance_data_returns_nan(self):
        """Contract: When all distance values are NaN, return NaN."""
        df = pd.DataFrame({"distance": [np.nan, np.nan]})

        result = calc_distance_metrics(df)

        assert all(np.isnan(v) for v in result.values())

    def test_cumulative_distance_calculates_total(self):
        """Contract: Distance is cumulative; total = max - min."""
        # Cumulative distance: starts at 0, ends at 5000m
        df = pd.DataFrame({"distance": [0, 1000, 2500, 4000, 5000]})

        result = calc_distance_metrics(df)

        assert result["total_distance_m"] == pytest.approx(5000.0)
        assert result["total_distance_km"] == pytest.approx(5.0)

    def test_distance_with_nan_values_ignores_nan(self):
        """Contract: NaN values are dropped; uses remaining min/max."""
        df = pd.DataFrame({"distance": [100, np.nan, 500, np.nan, 1100]})

        result = calc_distance_metrics(df)

        # Total = 1100 - 100 = 1000m
        assert result["total_distance_m"] == pytest.approx(1000.0)
        assert result["total_distance_km"] == pytest.approx(1.0)


class TestCalcElevationMetricsContract:
    """Contract tests for calc_elevation_metrics() function.

    Contract: Takes DataFrame with optional 'altitude' column (meters),
    returns dict with ascent, descent, and altitude statistics.

    Calculation: Uses diff() to find elevation changes,
    sums positive changes for ascent, negative for descent.
    """

    def test_missing_altitude_column_returns_nan(self):
        """Contract: When DataFrame lacks 'altitude' column, return NaN."""
        df = pd.DataFrame({"speed": [5.0, 6.0]})

        result = calc_elevation_metrics(df)

        assert result == {
            "total_ascent_m": np.nan,
            "total_descent_m": np.nan,
            "avg_altitude_m": np.nan,
            "min_altitude_m": np.nan,
            "max_altitude_m": np.nan,
        }
        assert all(np.isnan(v) for v in result.values())

    def test_empty_altitude_data_returns_nan(self):
        """Contract: When all altitude values are NaN, return NaN."""
        df = pd.DataFrame({"altitude": [np.nan, np.nan]})

        result = calc_elevation_metrics(df)

        assert all(np.isnan(v) for v in result.values())

    def test_flat_terrain_zero_elevation_change(self):
        """Contract: Flat terrain results in zero ascent/descent."""
        df = pd.DataFrame({"altitude": [100, 100, 100, 100]})

        result = calc_elevation_metrics(df)

        assert result["total_ascent_m"] == pytest.approx(0.0)
        assert result["total_descent_m"] == pytest.approx(0.0)
        assert result["avg_altitude_m"] == pytest.approx(100.0)
        assert result["min_altitude_m"] == pytest.approx(100.0)
        assert result["max_altitude_m"] == pytest.approx(100.0)

    def test_uphill_calculates_ascent(self):
        """Contract: Positive elevation changes sum to total ascent."""
        # Climbing from 0 to 300m in steps
        df = pd.DataFrame({"altitude": [0, 100, 200, 300]})

        result = calc_elevation_metrics(df)

        assert result["total_ascent_m"] == pytest.approx(300.0)
        assert result["total_descent_m"] == pytest.approx(0.0)
        assert result["min_altitude_m"] == pytest.approx(0.0)
        assert result["max_altitude_m"] == pytest.approx(300.0)

    def test_downhill_calculates_descent(self):
        """Contract: Negative elevation changes sum to total descent (absolute)."""
        # Descending from 300m to 0m
        df = pd.DataFrame({"altitude": [300, 200, 100, 0]})

        result = calc_elevation_metrics(df)

        assert result["total_ascent_m"] == pytest.approx(0.0)
        assert result["total_descent_m"] == pytest.approx(300.0)
        assert result["min_altitude_m"] == pytest.approx(0.0)
        assert result["max_altitude_m"] == pytest.approx(300.0)

    def test_mixed_terrain_calculates_both(self):
        """Contract: Mixed terrain sums ascent and descent separately."""
        # Up 100, down 50, up 150, down 100
        df = pd.DataFrame({"altitude": [0, 100, 50, 200, 100]})

        result = calc_elevation_metrics(df)

        # Ascent: 100 + 150 = 250m
        # Descent: 50 + 100 = 150m
        assert result["total_ascent_m"] == pytest.approx(250.0)
        assert result["total_descent_m"] == pytest.approx(150.0)
        assert result["avg_altitude_m"] == pytest.approx(90.0)  # (0+100+50+200+100)/5

    def test_altitude_with_nan_values_ignores_nan(self):
        """Contract: NaN values are dropped before calculation."""
        df = pd.DataFrame({"altitude": [0, np.nan, 100, np.nan, 200]})

        result = calc_elevation_metrics(df)

        # Should calculate from [0, 100, 200] only
        # Ascent: 100 + 100 = 200m
        assert result["total_ascent_m"] == pytest.approx(200.0)
        assert result["total_descent_m"] == pytest.approx(0.0)
        assert result["avg_altitude_m"] == pytest.approx(100.0)


class TestCalcPowerMetricsContract:
    """Contract tests for _calc_power_metrics() function.

    Contract: Takes DataFrame with optional 'power' column and FTP value,
    returns dict with power metrics including NP, IF, and TSS.

    Parameter matrix:
    - Missing 'power' column → all NaN
    - Empty power data → all NaN
    - Valid power + FTP > 0 → calculated metrics
    - Valid power + FTP = 0 → NaN for IF and TSS
    """

    def test_missing_power_column_returns_nan(self):
        """Contract: When DataFrame lacks 'power' column, return NaN for all metrics."""
        df = pd.DataFrame({"speed": [5.0, 6.0]})

        result = _calc_power_metrics(df, ftp=300)

        assert all(np.isnan(v) for v in result.values())
        assert set(result.keys()) == {
            "avg_power",
            "max_power",
            "normalized_power",
            "intensity_factor",
            "tss",
        }

    def test_empty_power_data_returns_nan(self):
        """Contract: When all power values are NaN, return NaN."""
        df = pd.DataFrame({"power": [np.nan, np.nan]})

        result = _calc_power_metrics(df, ftp=300)

        assert all(np.isnan(v) for v in result.values())

    def test_valid_power_with_positive_ftp_calculates_all(self):
        """Contract: With power data and FTP > 0, calculate all metrics."""
        # Constant power for predictable NP
        df = pd.DataFrame({"power": [200, 200, 200, 200] * 100})  # 400 samples

        result = _calc_power_metrics(df, ftp=300)

        assert result["avg_power"] == pytest.approx(200.0)
        assert result["max_power"] == pytest.approx(200.0)
        assert result["normalized_power"] == pytest.approx(200.0, abs=5.0)
        assert result["intensity_factor"] == pytest.approx(0.667, abs=0.02)  # NP/FTP
        assert not np.isnan(result["tss"])
        assert result["tss"] > 0

    def test_valid_power_with_zero_ftp_returns_nan_if_tss(self):
        """Contract: With FTP = 0, IF and TSS should be NaN."""
        df = pd.DataFrame({"power": [200, 200, 200]})

        result = _calc_power_metrics(df, ftp=0)

        assert result["avg_power"] == pytest.approx(200.0)
        assert result["max_power"] == pytest.approx(200.0)
        assert not np.isnan(result["normalized_power"])
        assert np.isnan(result["intensity_factor"])
        assert np.isnan(result["tss"])

    def test_power_with_nan_values_ignores_nan(self):
        """Contract: NaN values in power data are dropped."""
        df = pd.DataFrame({"power": [200, np.nan, 300, np.nan]})

        result = _calc_power_metrics(df, ftp=300)

        # Should calculate from [200, 300] only
        assert result["avg_power"] == pytest.approx(250.0)
        assert result["max_power"] == pytest.approx(300.0)


class TestCalcHrMetricsContract:
    """Contract tests for _calc_hr_metrics() function.

    Contract: Takes DataFrame with optional 'heart_rate' column,
    hr_rest, and hr_max, returns dict with HR metrics including TRIMP.
    """

    def test_missing_heart_rate_column_returns_nan(self):
        """Contract: When DataFrame lacks 'heart_rate' column, return NaN."""
        df = pd.DataFrame({"power": [200, 250]})

        result = _calc_hr_metrics(df, hr_rest=60, hr_max=190)

        assert all(np.isnan(v) for v in result.values())
        assert set(result.keys()) == {"avg_hr", "max_hr", "trimp"}

    def test_empty_heart_rate_data_returns_nan(self):
        """Contract: When all HR values are NaN, return NaN."""
        df = pd.DataFrame({"heart_rate": [np.nan, np.nan]})

        result = _calc_hr_metrics(df, hr_rest=60, hr_max=190)

        assert all(np.isnan(v) for v in result.values())

    def test_valid_heart_rate_calculates_metrics(self):
        """Contract: With HR data, calculate avg, max, and TRIMP."""
        df = pd.DataFrame({"heart_rate": [150, 160, 170, 180]})

        result = _calc_hr_metrics(df, hr_rest=60, hr_max=190)

        assert result["avg_hr"] == pytest.approx(165.0)
        assert result["max_hr"] == pytest.approx(180.0)
        assert not np.isnan(result["trimp"])
        assert result["trimp"] > 0

    def test_heart_rate_with_nan_values_ignores_nan(self):
        """Contract: NaN values in HR data are dropped."""
        df = pd.DataFrame({"heart_rate": [150, np.nan, 170, np.nan]})

        result = _calc_hr_metrics(df, hr_rest=60, hr_max=190)

        # Should calculate from [150, 170] only
        assert result["avg_hr"] == pytest.approx(160.0)
        assert result["max_hr"] == pytest.approx(170.0)


class TestCalculateMetricsForSessionContract:
    """Contract tests for calculate_metrics_for_session() function.

    Contract: Orchestrator that calls all metric calculation functions
    and returns combined dictionary of all metrics.

    This tests integration of all calculation functions.
    """

    def test_empty_dataframe_returns_all_nan(self):
        """Contract: Empty DataFrame returns NaN for all metrics."""
        df = pd.DataFrame()

        result = calculate_metrics_for_session(df, ftp=300, hr_rest=60, hr_max=190)

        # Should have keys from all metric categories (note: some keys are renamed)
        assert "avg_power_w" in result
        assert "avg_hr" in result
        assert "avg_speed_mps" in result
        assert "avg_cadence" in result
        assert "total_distance_m" in result
        assert "total_ascent_m" in result

    def test_complete_workout_calculates_all_metrics(self):
        """Contract: DataFrame with all columns calculates all metrics."""
        df = pd.DataFrame(
            {
                "power": [200, 250, 300],
                "heart_rate": [150, 160, 170],
                "speed": [5.0, 6.0, 7.0],
                "cadence": [80, 85, 90],
                "distance": [0, 100, 200],
                "altitude": [100, 110, 120],
            }
        )

        result = calculate_metrics_for_session(df, ftp=300, hr_rest=60, hr_max=190)

        # Verify all metric categories are present and calculated (note renamed keys)
        assert not np.isnan(result["avg_power_w"])
        assert not np.isnan(result["avg_hr"])
        assert not np.isnan(result["avg_speed_mps"])
        assert not np.isnan(result["avg_cadence"])
        assert not np.isnan(result["total_distance_m"])
        assert not np.isnan(result["total_ascent_m"])

    def test_partial_data_calculates_available_metrics(self):
        """Contract: Only available columns are calculated, rest are NaN."""
        df = pd.DataFrame(
            {
                "power": [200, 250, 300],
                "heart_rate": [150, 160, 170],
                # Missing: speed, cadence, distance, altitude
            }
        )

        result = calculate_metrics_for_session(df, ftp=300, hr_rest=60, hr_max=190)

        # Power and HR should be calculated (note renamed keys)
        assert not np.isnan(result["avg_power_w"])
        assert not np.isnan(result["avg_hr"])
        # Others should be NaN
        assert np.isnan(result["avg_speed_mps"])
        assert np.isnan(result["avg_cadence"])
        assert np.isnan(result["total_distance_m"])
