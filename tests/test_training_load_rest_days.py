"""
TDD test to fix training load calculation bug with rest days.

Bug: ATL/CTL calculations don't account for rest days between workouts.
     If you have 2 workouts with 5 rest days between them, the system
     treats them as consecutive days, not allowing proper decay.

Expected: Rest days should be included in calculations with zero load,
          allowing ATL/CTL to decay properly.
"""

import unittest
from datetime import datetime, timedelta

import pandas as pd

from fitanalyzer.training_load import calculate_training_load_metrics


class TestTrainingLoadRestDays(unittest.TestCase):
    """Test that training load calculations properly handle rest days."""

    def test_rest_days_allow_atl_to_decay(self):
        """
        FAILING TEST: ATL should decay during rest days.

        Scenario: Two 100 TSS workouts separated by 7 rest days.
        - Day 1: 100 TSS workout
        - Days 2-8: Rest (no workouts)
        - Day 9: 100 TSS workout

        Expected behavior:
        - After day 1: ATL starts building
        - Days 2-8: ATL should decay (no new load)
        - Day 9: ATL should be lower than if days were consecutive

        Current bug: System treats both workouts as consecutive days,
        so ATL doesn't decay between them.
        """
        # Create data: 2 workouts with 8-day gap
        data = {
            "date": [
                datetime(2025, 1, 1),  # Workout 1
                datetime(2025, 1, 9),  # Workout 2 (8 days later)
            ],
            "tss": [100.0, 100.0],
        }
        df = pd.DataFrame(data)

        result = calculate_training_load_metrics(df, load_column="tss", date_column="date")

        # After first workout (day 1)
        atl_day1 = result.iloc[0]["atl"]
        # ATL = 0 + (100 - 0) / 7 = 14.2857
        self.assertAlmostEqual(atl_day1, 100 / 7, places=2)

        # After second workout (day 9, with 8 rest days)
        atl_day9 = result.iloc[1]["atl"]

        # If days were consecutive (CURRENT BUG):
        # Day 2: ATL = 14.29 + (100 - 14.29) / 7 = 26.53
        consecutive_atl = atl_day1 + (100 - atl_day1) / 7

        # If rest days included (CORRECT):
        # Days 2-8: ATL decays by factor of (6/7) each day
        # Day 2: 14.29 * 6/7 = 12.24
        # Day 3: 12.24 * 6/7 = 10.49
        # ... (8 total decay steps)
        # Day 9: Add new load of 100
        decay_factor = 6 / 7
        decayed_atl = atl_day1
        for _ in range(8):  # 8 rest days
            decayed_atl = decayed_atl * decay_factor
        correct_atl = decayed_atl + (100 - decayed_atl) / 7

        # The current implementation will give consecutive_atl
        # But it should give correct_atl which is much lower
        print(f"\nATL after workout 1: {atl_day1:.2f}")
        print(f"Current (buggy) ATL after workout 2: {atl_day9:.2f}")
        print(f"Expected ATL with consecutive days: {consecutive_atl:.2f}")
        print(f"Expected ATL with rest days: {correct_atl:.2f}")

        # This will FAIL - current code doesn't account for rest days
        self.assertLess(
            atl_day9,
            consecutive_atl - 5,  # Should be at least 5 points lower
            "ATL should be lower when rest days are included",
        )

    def test_daily_dataframe_includes_rest_days(self):
        """Test helper function to create daily dataframe with rest days."""
        # Create data with gap
        data = {
            "date": [datetime(2025, 1, 1), datetime(2025, 1, 5)],
            "tss": [50.0, 100.0],
        }
        df = pd.DataFrame(data)

        # Convert to daily format (this function needs to be created)
        from fitanalyzer.training_load import _create_daily_dataframe

        daily_df = _create_daily_dataframe(df, load_column="tss", date_column="date")

        # Should have 5 rows (days 1-5)
        self.assertEqual(len(daily_df), 5)

        # Check that rest days have zero load
        self.assertEqual(daily_df.iloc[0]["tss"], 50.0)  # Day 1
        self.assertEqual(daily_df.iloc[1]["tss"], 0.0)  # Day 2 (rest)
        self.assertEqual(daily_df.iloc[2]["tss"], 0.0)  # Day 3 (rest)
        self.assertEqual(daily_df.iloc[3]["tss"], 0.0)  # Day 4 (rest)
        self.assertEqual(daily_df.iloc[4]["tss"], 100.0)  # Day 5

    def test_ctl_decays_over_extended_rest(self):
        """Test that CTL (fitness) decays during extended rest period."""
        # Scenario: 1 big workout, then 30 days rest
        data = {
            "date": [datetime(2025, 1, 1), datetime(2025, 1, 31)],
            "tss": [200.0, 50.0],
        }
        df = pd.DataFrame(data)

        result = calculate_training_load_metrics(df, load_column="tss", date_column="date")

        ctl_day1 = result.iloc[0]["ctl"]
        ctl_day31 = result.iloc[1]["ctl"]

        # CTL should decay over 30 days of rest
        # With 42-day time constant, decay factor per day = 41/42
        # After 30 days: ctl_day1 * (41/42)^30
        decay_factor = 41 / 42
        expected_decayed_ctl = ctl_day1 * (decay_factor**30)

        print(f"\nCTL after workout 1: {ctl_day1:.2f}")
        print(f"Current CTL after 30 days: {ctl_day31:.2f}")
        print(f"Expected decayed CTL: {expected_decayed_ctl:.2f}")

        # CTL should be significantly lower after 30 days of rest
        self.assertLess(
            ctl_day31,
            ctl_day1 * 0.8,  # Should be at least 20% lower
            "CTL should decay during extended rest",
        )


if __name__ == "__main__":
    unittest.main()
