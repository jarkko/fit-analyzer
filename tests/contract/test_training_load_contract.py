"""
Contract tests for training load calculations (Fitness-Fatigue Model).

These tests document and enforce the contracts for CTL, ATL, and TSB calculations
based on the Banister Fitness-Fatigue model.

Scientific References:
- Banister et al. (1975) - Fitness-Fatigue model
- Coggan (2003) - Performance Manager Chart (PMC)
- Time constants: CTL=42 days, ATL=7 days
"""

import numpy as np
import pandas as pd
import pytest

from fitanalyzer.training_load import (
    calculate_atl,
    calculate_ctl,
    calculate_training_load_metrics,
    calculate_tsb,
)


class TestCalculateCtlContract:
    """Contract tests for calculate_ctl() - Chronic Training Load.

    Contract: Calculate fitness (CTL) using exponentially weighted moving average
    with default time constant of 42 days.

    Formula: CTL_today = CTL_yesterday + (TSS_today - CTL_yesterday) / time_constant

    Parameter matrix:
    - Empty training loads → empty result
    - Single workout → CTL equals that load
    - Multiple workouts → exponential decay
    - Different time constants → different decay rates
    """

    def test_empty_training_loads_returns_empty(self):
        """Contract: Empty input returns empty array."""
        training_loads = []

        result = calculate_ctl(training_loads)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_single_workout_ctl_equals_load_divided_by_time_constant(self):
        """Contract: First workout CTL = TSS / time_constant (starting from 0)."""
        training_loads = [100]  # 100 TSS

        result = calculate_ctl(training_loads, time_constant=42)

        # First day: CTL = 0 + (100 - 0) / 42 = 2.38
        assert len(result) == 1
        assert result[0] == pytest.approx(100 / 42, abs=0.01)

    def test_multiple_workouts_exponential_decay(self):
        """Contract: CTL increases with each workout, following exponential curve."""
        # Simulate 10 days of 100 TSS per day
        training_loads = [100] * 10

        result = calculate_ctl(training_loads, time_constant=42)

        assert len(result) == 10
        # CTL should increase each day
        assert result[0] < result[1] < result[2]
        # But at decreasing rate (exponential)
        diff1 = result[1] - result[0]
        diff2 = result[2] - result[1]
        assert diff1 > diff2  # Diminishing returns

    def test_ctl_converges_to_steady_state(self):
        """Contract: With constant load, CTL converges to the load value."""
        # 200 days of constant 100 TSS (way longer than time constant)
        training_loads = [100] * 200

        result = calculate_ctl(training_loads, time_constant=42)

        # After ~4 time constants, should be very close to steady state (100)
        assert result[-1] == pytest.approx(100, abs=1)

    def test_zero_days_dont_decrease_ctl_to_zero_immediately(self):
        """Contract: Rest days cause exponential decay of CTL."""
        # Build up CTL, then rest
        training_loads = [100] * 42 + [0] * 10  # Build up, then 10 rest days

        result = calculate_ctl(training_loads, time_constant=42)

        ctl_before_rest = result[41]
        ctl_after_rest = result[-1]

        # CTL should decrease during rest but not to zero
        assert ctl_after_rest < ctl_before_rest
        assert ctl_after_rest > 0

    def test_custom_time_constant_affects_decay_rate(self):
        """Contract: Shorter time constant = faster response to changes."""
        training_loads = [100] * 20

        ctl_slow = calculate_ctl(training_loads, time_constant=42)
        ctl_fast = calculate_ctl(training_loads, time_constant=7)

        # Faster time constant reaches higher CTL sooner
        assert ctl_fast[-1] > ctl_slow[-1]


class TestCalculateAtlContract:
    """Contract tests for calculate_atl() - Acute Training Load.

    Contract: Calculate fatigue (ATL) using exponentially weighted moving average
    with default time constant of 7 days. Same formula as CTL but shorter window.
    """

    def test_empty_training_loads_returns_empty(self):
        """Contract: Empty input returns empty array."""
        training_loads = []

        result = calculate_atl(training_loads)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_atl_responds_faster_than_ctl(self):
        """Contract: ATL (7-day) responds faster than CTL (42-day)."""
        training_loads = [100] * 10

        ctl = calculate_ctl(training_loads, time_constant=42)
        atl = calculate_atl(training_loads, time_constant=7)

        # ATL should be higher than CTL initially (faster response)
        assert atl[5] > ctl[5]

    def test_atl_decays_faster_during_rest(self):
        """Contract: ATL decreases faster than CTL during rest."""
        # Build up, then rest
        training_loads = [100] * 14 + [0] * 7

        ctl = calculate_ctl(training_loads, time_constant=42)
        atl = calculate_atl(training_loads, time_constant=7)

        # Calculate percentage decrease during rest period
        ctl_decrease_pct = (ctl[13] - ctl[-1]) / ctl[13]
        atl_decrease_pct = (atl[13] - atl[-1]) / atl[13]

        # ATL should decrease more (faster decay)
        assert atl_decrease_pct > ctl_decrease_pct


class TestCalculateTsbContract:
    """Contract tests for calculate_tsb() - Training Stress Balance.

    Contract: TSB = CTL - ATL (Form = Fitness - Fatigue)

    Interpretation:
    - Positive TSB: Fresh, ready to perform
    - Zero TSB: Balanced
    - Negative TSB: Fatigued, training hard
    """

    def test_tsb_is_difference_between_ctl_and_atl(self):
        """Contract: TSB = CTL - ATL."""
        ctl = np.array([50.0, 52.0, 54.0])
        atl = np.array([45.0, 48.0, 51.0])

        result = calculate_tsb(ctl, atl)

        expected = np.array([5.0, 4.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_high_atl_creates_negative_tsb(self):
        """Contract: When ATL > CTL, TSB is negative (fatigue state)."""
        ctl = np.array([50.0])
        atl = np.array([60.0])

        result = calculate_tsb(ctl, atl)

        assert result[0] < 0
        assert result[0] == pytest.approx(-10.0)

    def test_equal_ctl_atl_gives_zero_tsb(self):
        """Contract: When CTL = ATL, TSB = 0 (balanced)."""
        ctl = np.array([50.0])
        atl = np.array([50.0])

        result = calculate_tsb(ctl, atl)

        assert result[0] == pytest.approx(0.0)

    def test_mismatched_array_lengths_raises_error(self):
        """Contract: CTL and ATL arrays must have same length."""
        ctl = np.array([50.0, 52.0])
        atl = np.array([45.0])

        with pytest.raises(ValueError, match="must have the same length"):
            calculate_tsb(ctl, atl)


class TestCalculateTrainingLoadMetricsContract:
    """Contract tests for calculate_training_load_metrics() orchestrator.

    Contract: Takes DataFrame with dates and training loads (TSS or TRIMP),
    returns DataFrame with CTL, ATL, TSB columns added.

    Parameter matrix:
    - Empty DataFrame → empty result
    - Single workout → calculated metrics
    - Multiple workouts → time series
    - Missing date column → error
    - Missing load column → error
    """

    def test_empty_dataframe_returns_empty(self):
        """Contract: Empty DataFrame returns empty DataFrame with expected columns."""
        df = pd.DataFrame()

        result = calculate_training_load_metrics(df)

        assert len(result) == 0
        assert "ctl" in result.columns
        assert "atl" in result.columns
        assert "tsb" in result.columns

    def test_single_workout_calculates_metrics(self):
        """Contract: Single workout gets CTL, ATL, TSB values."""
        df = pd.DataFrame({"date": ["2025-01-01"], "tss": [100.0]})

        result = calculate_training_load_metrics(df, load_column="tss")

        assert len(result) == 1
        assert "ctl" in result.columns
        assert "atl" in result.columns
        assert "tsb" in result.columns
        assert not pd.isna(result["ctl"].iloc[0])

    def test_sorts_by_date_before_calculation(self):
        """Contract: Workouts are sorted by date before calculating cumulative metrics."""
        df = pd.DataFrame(
            {"date": ["2025-01-03", "2025-01-01", "2025-01-02"], "tss": [100.0, 100.0, 100.0]}
        )

        result = calculate_training_load_metrics(df, load_column="tss")

        # Result should be sorted by date (dates are now converted to Timestamp)
        assert str(result["date"].iloc[0].date()) == "2025-01-01"
        assert str(result["date"].iloc[1].date()) == "2025-01-02"
        assert str(result["date"].iloc[2].date()) == "2025-01-03"

    def test_uses_trimp_when_tss_not_available(self):
        """Contract: Falls back to TRIMP if TSS not available."""
        df = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "trimp": [50.0, 60.0]})

        result = calculate_training_load_metrics(df, load_column="trimp")

        assert len(result) == 2
        assert not pd.isna(result["ctl"].iloc[0])

    def test_missing_date_column_raises_error(self):
        """Contract: DataFrame must have 'date' column."""
        df = pd.DataFrame({"tss": [100.0]})

        with pytest.raises(KeyError, match="date"):
            calculate_training_load_metrics(df)

    def test_missing_load_column_raises_error(self):
        """Contract: DataFrame must have specified load column."""
        df = pd.DataFrame({"date": ["2025-01-01"]})

        with pytest.raises(KeyError, match="tss"):
            calculate_training_load_metrics(df, load_column="tss")

    def test_handles_gaps_in_dates(self):
        """Contract: Handles date gaps (assumes zero training on missing days)."""
        df = pd.DataFrame(
            {"date": ["2025-01-01", "2025-01-05"], "tss": [100.0, 100.0]}  # 3-day gap
        )

        result = calculate_training_load_metrics(df, load_column="tss")

        # Should handle gap gracefully
        assert len(result) == 2
        assert not pd.isna(result["ctl"].iloc[1])

    def test_rounds_values_to_four_decimal_places(self):
        """Contract: CTL/ATL/TSB values are rounded to 4 decimal places for readability."""
        df = pd.DataFrame({"date": ["2025-01-01"], "tss": [100.0]})

        result = calculate_training_load_metrics(df, load_column="tss")

        # Check that values are rounded to exactly 4 decimal places
        # CTL = 100/42 = 2.380952380952381... should become 2.381
        ctl_str = str(result["ctl"].iloc[0])
        if "." in ctl_str:
            decimal_places = len(ctl_str.split(".")[1])
            assert decimal_places <= 4, f"CTL has {decimal_places} decimal places, expected <= 4"

        # ATL = 100/7 = 14.285714285714286... should become 14.2857
        atl_str = str(result["atl"].iloc[0])
        if "." in atl_str:
            decimal_places = len(atl_str.split(".")[1])
            assert decimal_places <= 4, f"ATL has {decimal_places} decimal places, expected <= 4"

        # TSB = CTL - ATL should also be rounded
        tsb_str = str(result["tsb"].iloc[0])
        if "." in tsb_str:
            decimal_places = len(tsb_str.split(".")[1])
            assert decimal_places <= 4, f"TSB has {decimal_places} decimal places, expected <= 4"

    def test_recalculation_with_existing_columns_no_duplicates(self):
        """
        REGRESSION TEST: Calculate training load on DataFrame that already has ctl/atl/tsb.

        Critical contract: Function must handle DataFrames that already have training
        load columns (common in incremental analysis when loading from CSV).

        Bug that was missed: When DataFrame has existing ctl/atl/tsb columns,
        pandas merge creates duplicate columns (ctl_x, ctl_y, etc.) instead of
        replacing them.

        Expected behavior: Drop existing columns and recalculate cleanly.
        """
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "tss": [100.0, 80.0],
                "ctl": [2.38, 4.23],  # Existing columns from previous calculation
                "atl": [14.29, 23.67],
                "tsb": [-11.90, -19.44],
            }
        )

        result = calculate_training_load_metrics(df, load_column="tss")

        # CRITICAL: No duplicate columns should be created
        assert "ctl" in result.columns, "ctl column must exist"
        assert "atl" in result.columns, "atl column must exist"
        assert "tsb" in result.columns, "tsb column must exist"
        assert "ctl_x" not in result.columns, "No ctl_x duplicate column"
        assert "ctl_y" not in result.columns, "No ctl_y duplicate column"
        assert "atl_x" not in result.columns, "No atl_x duplicate column"
        assert "atl_y" not in result.columns, "No atl_y duplicate column"
        assert "tsb_x" not in result.columns, "No tsb_x duplicate column"
        assert "tsb_y" not in result.columns, "No tsb_y duplicate column"

        # Values should be recalculated correctly (deterministic)
        assert len(result) == 2
        assert not pd.isna(result["ctl"].iloc[0])
