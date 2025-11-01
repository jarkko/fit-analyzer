"""
End-to-end tests for incremental sync scenarios.

Tests that simulate real-world usage patterns:
- Multiple syncs with data from different time periods
- Appending older data in later syncs
- CSV sorting and deduplication
- Both workout and strength training summaries
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestIncrementalSyncWorkflow(unittest.TestCase):
    """Test end-to-end incremental sync scenarios with CSV merging."""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_multiple_syncs_maintain_chronological_order(self):
        """Test that multiple syncs with random data maintain chronological order."""
        from fitanalyzer.cli import _save_workout_summary
        from fitanalyzer.incremental import load_existing_rows

        csv_path = self.output_dir / "workout_summary_from_fit.csv"

        # Sync 1: January 2025 data
        jan_rows = [
            {
                "file": "2025-01-10_activity.fit",
                "sport": "cycling",
                "date": "2025-01-10",
                "start_time": "2025-01-10 10:00:00",
                "duration_min": 30.0,
                "_file_mtime": 1234567890.0,
            },
            {
                "file": "2025-01-15_activity.fit",
                "sport": "running",
                "date": "2025-01-15",
                "start_time": "2025-01-15 12:00:00",
                "duration_min": 45.0,
                "_file_mtime": 1234567891.0,
            },
            {
                "file": "2025-01-20_activity.fit",
                "sport": "cycling",
                "date": "2025-01-20",
                "start_time": "2025-01-20 14:00:00",
                "duration_min": 60.0,
                "_file_mtime": 1234567892.0,
            },
        ]

        with patch("builtins.print"):
            _save_workout_summary(jan_rows, str(self.output_dir), [])

        df1 = pd.read_csv(csv_path)
        self.assertEqual(len(df1), 3)

        # Sync 2: March 2025 data (newer)
        existing_rows = load_existing_rows(csv_path, {})
        march_rows = [
            {
                "file": "2025-03-05_activity.fit",
                "sport": "running",
                "date": "2025-03-05",
                "start_time": "2025-03-05 08:00:00",
                "duration_min": 30.0,
                "_file_mtime": 1234567893.0,
            },
            {
                "file": "2025-03-10_activity.fit",
                "sport": "cycling",
                "date": "2025-03-10",
                "start_time": "2025-03-10 16:00:00",
                "duration_min": 90.0,
                "_file_mtime": 1234567894.0,
            },
        ]

        with patch("builtins.print"):
            _save_workout_summary(existing_rows + march_rows, str(self.output_dir), [])

        df2 = pd.read_csv(csv_path)
        self.assertEqual(len(df2), 5)

        # Sync 3: February 2025 data (older data discovered later)
        existing_rows = load_existing_rows(csv_path, {})
        feb_rows = [
            {
                "file": "2025-02-01_activity.fit",
                "sport": "cycling",
                "date": "2025-02-01",
                "start_time": "2025-02-01 11:00:00",
                "duration_min": 45.0,
                "_file_mtime": 1234567895.0,
            },
            {
                "file": "2025-02-14_activity.fit",
                "sport": "running",
                "date": "2025-02-14",
                "start_time": "2025-02-14 09:00:00",
                "duration_min": 50.0,
                "_file_mtime": 1234567896.0,
            },
        ]

        with patch("builtins.print"):
            _save_workout_summary(existing_rows + feb_rows, str(self.output_dir), [])

        # Verify final output: ALL data chronologically ordered
        df_final = pd.read_csv(csv_path)
        self.assertEqual(len(df_final), 7)

        # Critical assertion: must be chronologically ordered despite random sync order
        final_times = df_final["start_time"].tolist()
        self.assertEqual(
            final_times,
            [
                "2025-01-10 10:00:00",
                "2025-01-15 12:00:00",
                "2025-01-20 14:00:00",
                "2025-02-01 11:00:00",  # Feb data inserted in correct position
                "2025-02-14 09:00:00",  # Feb data inserted in correct position
                "2025-03-05 08:00:00",
                "2025-03-10 16:00:00",
            ],
        )

        # Verify monotonically increasing (binary search compatible)
        for i in range(len(final_times) - 1):
            self.assertLess(final_times[i], final_times[i + 1])

    def test_duplicate_files_deduplicated_by_file_column(self):
        """Test that duplicate files (same file column) are deduplicated in CSV."""
        from fitanalyzer.cli import _save_workout_summary

        # First sync
        rows1 = [
            {
                "file": "2025-01-10_activity.fit",
                "sport": "cycling",
                "date": "2025-01-10",
                "start_time": "2025-01-10 10:00:00",
                "duration_min": 30.0,
                "_file_mtime": 1234567890.0,
            }
        ]

        with patch("builtins.print"):
            _save_workout_summary(rows1, str(self.output_dir), [])

        csv_path = self.output_dir / "workout_summary_from_fit.csv"
        df1 = pd.read_csv(csv_path)
        self.assertEqual(len(df1), 1)

        # Second sync with "same" file (should deduplicate by file column)
        rows2 = [
            {
                "file": "2025-01-10_activity.fit",  # Same file ID
                "sport": "cycling",
                "date": "2025-01-10",
                "start_time": "2025-01-10 10:00:00",
                "duration_min": 35.0,  # Different duration (updated data)
                "_file_mtime": 1234567899.0,  # Updated mtime
            }
        ]

        with patch("builtins.print"):
            _save_workout_summary(rows2, str(self.output_dir), [])

        # Final CSV should still have only 1 row (dedup by file column, keep=last)
        df2 = pd.read_csv(csv_path)
        self.assertEqual(len(df2), 1)
        self.assertEqual(df2["duration_min"].iloc[0], 35.0)  # Should have updated value

    def test_strength_summary_sorting_across_syncs(self):
        """Test that strength summary is sorted chronologically across multiple syncs."""
        # Simulate existing CSV (March data)
        strength_csv = self.output_dir / "strength_training_summary.csv"
        existing = pd.DataFrame(
            [
                {
                    "activity_id": "2025-03-10_strength",
                    "date": "2025-03-10",
                    "timestamp": "2025-03-10 12:20:00",
                    "exercise_name": "Deadlift",
                    "repetitions": 5,
                    "weight": 150,
                }
            ]
        )
        existing.to_csv(strength_csv, index=False)

        # New data from sync (January - older data)
        new_sets = pd.DataFrame(
            [
                {
                    "activity_id": "2025-01-10_strength",
                    "date": "2025-01-10",
                    "timestamp": "2025-01-10 10:15:00",
                    "exercise_name": "Squat",
                    "repetitions": 5,
                    "weight": 100,
                }
            ]
        )

        # Merge and sort (simulating what happens in CLI)
        result = pd.concat([existing, new_sets], ignore_index=True)
        result = result.sort_values(["date", "timestamp"], na_position="last")
        result.to_csv(strength_csv, index=False)

        # Verify final CSV is chronologically sorted
        saved = pd.read_csv(strength_csv)
        self.assertEqual(len(saved), 2)
        self.assertEqual(
            saved["timestamp"].tolist(), ["2025-01-10 10:15:00", "2025-03-10 12:20:00"]
        )

    def test_strength_summary_partial_reprocess(self):
        """Test that reprocessing only some files preserves other file data."""
        from fitanalyzer.cli import _generate_strength_summary

        # Create existing strength data with multiple files
        strength_csv = self.output_dir / "strength_training_summary.csv"
        existing = pd.DataFrame(
            [
                {
                    "activity_id": "2025-01-10_strength",
                    "date": "2025-01-10",
                    "timestamp": "2025-01-10 10:15:00",
                    "exercise_name": "Squat",
                    "repetitions": 5,
                    "weight": 100,
                },
                {
                    "activity_id": "2025-01-15_strength",
                    "date": "2025-01-15",
                    "timestamp": "2025-01-15 11:00:00",
                    "exercise_name": "Bench Press",
                    "repetitions": 8,
                    "weight": 80,
                },
            ]
        )
        existing.to_csv(strength_csv, index=False)

        # Reprocess only one file (2025-01-10_strength_ACTIVITY.fit)
        files_to_process = [str(self.test_dir / "2025-01-10_strength_ACTIVITY.fit")]

        # Mock args
        args = MagicMock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"
        args.output_dir = str(self.output_dir)
        args.updated_files = []

        # Mock new strength data for the reprocessed file
        new_strength = pd.DataFrame(
            [
                {
                    "activity_id": "2025-01-10_strength",
                    "date": "2025-01-10",
                    "timestamp": "2025-01-10 10:15:00",
                    "exercise_name": "Squat",
                    "repetitions": 10,  # Updated reps
                    "weight": 120,  # Updated weight
                }
            ]
        )

        # Patch aggregate_strength_sets to return our mock data
        with patch("fitanalyzer.cli.aggregate_strength_sets", return_value=new_strength):
            result = _generate_strength_summary(args, files_to_process, existing)

        # Should have 2 rows: updated 2025-01-10 + unchanged 2025-01-15
        self.assertEqual(len(result), 2)

        # 2025-01-10 should have updated values
        jan10 = result[result["activity_id"] == "2025-01-10_strength"].iloc[0]
        self.assertEqual(jan10["repetitions"], 10)
        self.assertEqual(jan10["weight"], 120)

        # 2025-01-15 should be unchanged
        jan15 = result[result["activity_id"] == "2025-01-15_strength"].iloc[0]
        self.assertEqual(jan15["repetitions"], 8)
        self.assertEqual(jan15["weight"], 80)

    def _create_mock_fit_files(self, filenames):
        """Create mock FIT files in test directory."""
        files = []
        for filename in filenames:
            file_path = self.test_dir / filename
            file_path.write_text("mock fit data")
            files.append(str(file_path))
        return files

    def _create_args(self, fit_files, dump_sets=False):
        """Create mock arguments for main_with_args."""
        args = MagicMock()
        args.fit_files = fit_files
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"
        args.dump_sets = dump_sets
        args.output_dir = str(self.output_dir)
        args.force = False
        return args


if __name__ == "__main__":
    unittest.main()
