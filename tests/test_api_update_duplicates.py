"""Test that API updates don't create duplicate entries in strength CSV."""

from pathlib import Path

import pandas as pd
import pytest

from fitanalyzer.cli import main_with_args, parse_arguments


class TestAPIUpdateNoDuplicates:
    """Test that API updates don't create duplicate entries."""

    def test_api_update_replaces_not_duplicates(self, tmp_path):
        """Test that when API data updates, old entries are replaced, not duplicated.

        This test simulates what happens when:
        1. A strength training workout is analyzed and added to CSV
        2. User changes exercise names in Garmin Connect
        3. Sync detects API change and marks file as updated
        4. Re-analysis should REPLACE old entries, not duplicate them
        """
        import os

        original_dir = os.getcwd()
        test_file = str(Path(original_dir) / "tests/fixtures/20474406937_ACTIVITY.fit")

        os.chdir(tmp_path)

        try:
            # First run: initial analysis
            args1 = parse_arguments(
                [test_file, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )
            main_with_args(args1)

            summary_file = tmp_path / "strength_training_summary.csv"
            df1 = pd.read_csv(summary_file)
            initial_count = len(df1)
            activity_id = df1["activity_id"].iloc[0]
            initial_activity_count = (df1["activity_id"] == activity_id).sum()

            print(
                f"\nInitial analysis: {initial_count} total rows, {initial_activity_count} for activity {activity_id}"
            )

            # Second run: simulate API update for same file
            # This simulates what happens when exercise names change in Garmin API
            args2 = parse_arguments(
                [test_file, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )
            args2.updated_files = [test_file]  # Mark as API-updated
            main_with_args(args2)

            df2 = pd.read_csv(summary_file)
            final_count = len(df2)
            final_activity_count = (df2["activity_id"] == activity_id).sum()

            print(
                f"After API update: {final_count} total rows, {final_activity_count} for activity {activity_id}"
            )

            # Should have same count (replaced, not duplicated)
            assert final_count == initial_count, (
                f"Expected {initial_count} rows, got {final_count}. "
                "API update should replace entries, not duplicate them."
            )

            # Activity should appear same number of times
            assert final_activity_count == initial_activity_count, (
                f"Activity {activity_id} appears {final_activity_count} times, "
                f"expected {initial_activity_count}. Duplicates were created!"
            )

        finally:
            os.chdir(original_dir)

    def test_multiple_activities_only_updated_one_replaced(self, tmp_path):
        """Test that only the API-updated activity is replaced, others remain."""
        import os

        original_dir = os.getcwd()
        file1 = str(Path(original_dir) / "tests/fixtures/20474406937_ACTIVITY.fit")
        file2 = str(Path(original_dir) / "tests/fixtures/20555050352_ACTIVITY.fit")

        os.chdir(tmp_path)

        try:
            # First run: analyze both files
            args1 = parse_arguments(
                [file1, file2, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )
            main_with_args(args1)

            summary_file = tmp_path / "strength_training_summary.csv"
            df1 = pd.read_csv(summary_file)
            initial_total = len(df1)
            activity1_id = Path(file1).stem.replace("_ACTIVITY", "")
            activity2_id = Path(file2).stem.replace("_ACTIVITY", "")

            initial_act1_count = (df1["activity_id"] == activity1_id).sum()
            initial_act2_count = (df1["activity_id"] == activity2_id).sum()

            # Second run: only file1 has API update
            args2 = parse_arguments(
                [file1, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )
            args2.updated_files = [file1]  # Only file1 updated
            main_with_args(args2)

            df2 = pd.read_csv(summary_file)
            final_total = len(df2)
            final_act1_count = (df2["activity_id"] == activity1_id).sum()
            final_act2_count = (df2["activity_id"] == activity2_id).sum()

            # Total should be same (file1 replaced, file2 unchanged)
            assert final_total == initial_total, f"Expected {initial_total} rows, got {final_total}"

            # Activity 1 should have same count (replaced)
            assert (
                final_act1_count == initial_act1_count
            ), f"Activity {activity1_id} should have {initial_act1_count} sets, got {final_act1_count}"

            # Activity 2 should still be present (unchanged)
            assert (
                final_act2_count == initial_act2_count
            ), f"Activity {activity2_id} should still have {initial_act2_count} sets, got {final_act2_count}"

        finally:
            os.chdir(original_dir)
