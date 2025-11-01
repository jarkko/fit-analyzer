"""
Test to verify training load calculations are not duplicated/recalculated incorrectly.

Bug: When new files are added to existing analysis, the training load metrics
(CTL, ATL, TSB) are recalculated for ALL workouts, causing the values to change
even for workouts that haven't changed.
"""

import unittest
from datetime import datetime

import pandas as pd

from fitanalyzer.training_load import calculate_training_load_metrics


class TestTrainingLoadDuplication(unittest.TestCase):
    """Test that training load calculations are stable and not duplicated."""

    def test_training_load_stable_when_recalculated(self):
        """
        Training load values should remain the same when recalculated with the same data.

        Scenario: Calculate training load for a dataset, then recalculate.
        Expected: CTL/ATL/TSB values should be identical.
        """
        # Create sample data
        data = {
            "date": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "tss": [100.0, 80.0, 90.0],
        }
        df = pd.DataFrame(data)

        # Calculate once
        result1 = calculate_training_load_metrics(df, load_column="tss", date_column="date")

        # Calculate again with same data
        result2 = calculate_training_load_metrics(df, load_column="tss", date_column="date")

        # Values should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_incremental_analysis_preserves_old_values(self):
        """
        When adding new workouts, old workout metrics should NOT change.

        This is the KEY bug: Current implementation recalculates ALL workouts
        every time, causing old CTL/ATL values to change.

        Scenario:
        1. Day 1-3: Calculate training load for 3 workouts
        2. Day 4: Add a new workout
        3. Expected: Day 1-3 metrics should remain the same
        4. Actual (BUG): Day 1-3 metrics get recalculated and change
        """
        # First analysis: 3 workouts
        data_initial = {
            "date": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
            ],
            "tss": [100.0, 80.0, 90.0],
        }
        df_initial = pd.DataFrame(data_initial)
        result_initial = calculate_training_load_metrics(
            df_initial, load_column="tss", date_column="date"
        )

        # Save the values for day 3
        day3_ctl_initial = result_initial.iloc[2]["ctl"]
        day3_atl_initial = result_initial.iloc[2]["atl"]
        day3_tsb_initial = result_initial.iloc[2]["tsb"]

        # Second analysis: Add day 4
        data_with_new = {
            "date": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 4),  # NEW
            ],
            "tss": [100.0, 80.0, 90.0, 85.0],  # NEW
        }
        df_with_new = pd.DataFrame(data_with_new)
        result_with_new = calculate_training_load_metrics(
            df_with_new, load_column="tss", date_column="date"
        )

        # Day 3 values should be IDENTICAL to initial calculation
        day3_ctl_new = result_with_new.iloc[2]["ctl"]
        day3_atl_new = result_with_new.iloc[2]["atl"]
        day3_tsb_new = result_with_new.iloc[2]["tsb"]

        print(f"\nDay 3 CTL: Initial={day3_ctl_initial:.2f}, After adding day 4={day3_ctl_new:.2f}")
        print(f"Day 3 ATL: Initial={day3_atl_initial:.2f}, After adding day 4={day3_atl_new:.2f}")
        print(f"Day 3 TSB: Initial={day3_tsb_initial:.2f}, After adding day 4={day3_tsb_new:.2f}")

        # These should be equal
        self.assertAlmostEqual(
            day3_ctl_initial,
            day3_ctl_new,
            places=2,
            msg="CTL for day 3 should not change when day 4 is added",
        )
        self.assertAlmostEqual(
            day3_atl_initial,
            day3_atl_new,
            places=2,
            msg="ATL for day 3 should not change when day 4 is added",
        )
        self.assertAlmostEqual(
            day3_tsb_initial,
            day3_tsb_new,
            places=2,
            msg="TSB for day 3 should not change when day 4 is added",
        )

    def test_recalculation_on_data_with_existing_columns(self):
        """
        REGRESSION TEST: Calculate training load on data that already has ctl/atl/tsb columns.

        This is the critical test that was MISSING and allowed the bug to slip through!

        Bug: When DataFrame already has ctl/atl/tsb columns, pandas merge creates
        duplicate columns (ctl_x, ctl_y, atl_x, atl_y, tsb_x, tsb_y).

        This happens in incremental analysis when:
        1. Load existing CSV (has ctl/atl/tsb)
        2. Merge with new workouts
        3. Recalculate training load on merged data
        """
        # Create data that already has training load columns (like from CSV)
        data_with_existing_columns = {
            "date": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
            "tss": [100.0, 80.0, 90.0],
            "ctl": [2.38, 4.23, 6.27],  # Existing values from previous calculation
            "atl": [14.29, 23.67, 33.15],
            "tsb": [-11.90, -19.44, -26.88],
        }
        df = pd.DataFrame(data_with_existing_columns)

        # Recalculate (this is what happens in incremental analysis)
        result = calculate_training_load_metrics(df, load_column="tss", date_column="date")

        # CRITICAL CHECKS: No duplicate columns should be created
        self.assertIn("ctl", result.columns, "ctl column should exist")
        self.assertIn("atl", result.columns, "atl column should exist")
        self.assertIn("tsb", result.columns, "tsb column should exist")

        # Check for duplicate columns (the bug)
        self.assertNotIn("ctl_x", result.columns, "Should not create ctl_x column")
        self.assertNotIn("ctl_y", result.columns, "Should not create ctl_y column")
        self.assertNotIn("atl_x", result.columns, "Should not create atl_x column")
        self.assertNotIn("atl_y", result.columns, "Should not create atl_y column")
        self.assertNotIn("tsb_x", result.columns, "Should not create tsb_x column")
        self.assertNotIn("tsb_y", result.columns, "Should not create tsb_y column")

        # Values should be recalculated correctly
        self.assertAlmostEqual(result["ctl"].iloc[0], 2.38, places=1)
        self.assertAlmostEqual(result["atl"].iloc[0], 14.29, places=1)

    def test_merge_simulation_shows_bug(self):
        """
        Simulate the bug in cli.py where old rows are merged with new rows,
        then ALL rows get training load recalculated.

        This is what causes the duplication/changing values bug.
        """
        # Simulate first run: analyze days 1-3
        initial_data = {
            "date": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
            "tss": [100.0, 80.0, 90.0],
        }
        df_initial = pd.DataFrame(initial_data)
        result_initial = calculate_training_load_metrics(
            df_initial, load_column="tss", date_column="date"
        )

        # Save as "existing rows" (like CSV would be saved)
        existing_rows = result_initial.to_dict("records")
        day3_ctl_saved = existing_rows[2]["ctl"]

        # Simulate second run: analyze day 4 only
        new_data = {"date": [datetime(2025, 1, 4)], "tss": [85.0]}
        df_new = pd.DataFrame(new_data)
        # New row doesn't have CTL yet (not calculated)
        new_rows = df_new.to_dict("records")

        # BUG: Merge old and new, then recalculate ALL
        merged_rows = existing_rows + new_rows
        df_merged = pd.DataFrame(merged_rows)

        # This is what cli.py does - recalculate training load on merged data
        result_merged = calculate_training_load_metrics(
            df_merged, load_column="tss", date_column="date"
        )

        day3_ctl_recalculated = result_merged.iloc[2]["ctl"]

        # No duplicate columns should exist
        self.assertNotIn("ctl_x", result_merged.columns)
        self.assertNotIn("ctl_y", result_merged.columns)

        # Values should remain stable
        self.assertAlmostEqual(day3_ctl_saved, day3_ctl_recalculated, places=2)


if __name__ == "__main__":
    unittest.main()
