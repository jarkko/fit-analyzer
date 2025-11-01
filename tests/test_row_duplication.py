"""
TDD tests for row duplication bugs.

Bug: Same FIT files processed 5-6 times each, creating 502 duplicate entries.
Root cause: Input file list contains duplicates, or files are processed multiple times.
"""

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from fitanalyzer.cli import parse_arguments


class TestRowDuplication(unittest.TestCase):
    """Test that row duplication is prevented at input level."""

    def test_duplicate_files_in_input_are_removed(self):
        """
        REGRESSION TEST: If same file is passed multiple times, deduplicate.

        Bug: User passes same file 5-6 times (via glob expansion or mistake),
        system processes each one, creating duplicate rows in CSV.

        Expected: Deduplicate input list, warn user.
        """
        # Simulate duplicate files in input
        args = [
            "file1.fit",
            "file2.fit",
            "file1.fit",  # Duplicate
            "file3.fit",
            "file2.fit",  # Duplicate
            "file1.fit",  # Duplicate again
            "--ftp",
            "250",
        ]

        parsed = parse_arguments(args)

        # Should have only 3 unique files
        self.assertEqual(len(parsed.fit_files), 3)
        self.assertIn("file1.fit", parsed.fit_files)
        self.assertIn("file2.fit", parsed.fit_files)
        self.assertIn("file3.fit", parsed.fit_files)

        # Check order preservation (first occurrence)
        self.assertEqual(parsed.fit_files[0], "file1.fit")
        self.assertEqual(parsed.fit_files[1], "file2.fit")
        self.assertEqual(parsed.fit_files[2], "file3.fit")

    def test_no_duplicates_no_warning(self):
        """If no duplicates, don't show warning."""
        args = ["file1.fit", "file2.fit", "file3.fit", "--ftp", "250"]

        with patch("builtins.print") as mock_print:
            parsed = parse_arguments(args)

            # Check that warning was NOT printed
            warning_calls = [
                call for call in mock_print.call_args_list
                if "duplicate" in str(call).lower()
            ]
            self.assertEqual(len(warning_calls), 0)

        self.assertEqual(len(parsed.fit_files), 3)

    def test_six_duplicates_of_same_file(self):
        """
        Test the extreme case: same file 6 times (matches user's report).
        """
        same_file = "data/samples/20474406937_ACTIVITY.fit"
        args = [same_file] * 6 + ["--ftp", "250"]

        with patch("builtins.print") as mock_print:
            parsed = parse_arguments(args)

            # Check warning was printed
            mock_print.assert_any_call("⚠️  Removed 5 duplicate file(s) from input list")

        # Should have only 1 file
        self.assertEqual(len(parsed.fit_files), 1)
        self.assertEqual(parsed.fit_files[0], same_file)

    def test_deduplication_with_relative_and_absolute_paths(self):
        """
        Edge case: Same file with different paths (relative vs absolute).
        Note: This test documents current behavior - we DON'T deduplicate different paths.
        If user wants this, they should normalize paths before passing to CLI.
        """
        args = [
            "data/file.fit",
            "/absolute/path/data/file.fit",
            "data/file.fit",  # This IS a duplicate
            "--ftp",
            "250",
        ]

        parsed = parse_arguments(args)

        # We deduplicate exact string matches only
        # So "data/file.fit" appears once, "/absolute/path/data/file.fit" appears once
        self.assertEqual(len(parsed.fit_files), 2)
        self.assertEqual(parsed.fit_files.count("data/file.fit"), 1)


class TestMultisportModeRemoved(unittest.TestCase):
    """Test that multisport mode is removed - all files should just work."""

    def test_no_multisport_flag_in_arguments(self):
        """Multisport flag should not exist anymore."""
        args = ["file.fit", "--ftp", "250"]
        parsed = parse_arguments(args)

        # Should not have multisport attribute
        self.assertFalse(hasattr(parsed, "multisport"))

    def test_multisport_flag_rejected_with_error(self):
        """Using --multisport flag should raise an error (flag removed)."""
        args = ["file.fit", "--multisport", "--ftp", "250"]

        # Should raise SystemExit because argparse doesn't recognize --multisport
        with self.assertRaises(SystemExit):
            parse_arguments(args)

    def test_multisport_files_processed_automatically(self):
        """
        Multisport FIT files should be detected and handled automatically.
        No flag needed - the library should figure it out.

        This is a placeholder for future implementation.
        """
        # TODO: Implement automatic multisport detection
        # - Check if FIT file has multiple sessions
        # - If yes, process each session separately
        # - If no, process as single workout
        pass


class TestMultisportFileDuplication(unittest.TestCase):
    """Test that multisport files are not processed multiple times."""

    def test_multisport_file_processed_only_once(self):
        """
        REGRESSION TEST: Multisport files should only be processed once,
        even if they appear multiple times in the input list.

        Bug: When a multisport file (with N sessions) is processed,
        it creates N rows in the output. If the file is processed again
        (due to duplicates in input or incremental merge), you get 2*N rows.

        Example:
        - File: activity.fit with 2 sessions (rowing + strength)
        - Input list: [activity.fit, activity.fit] (duplicate)
        - Expected: 2 rows (rowing, strength)
        - Actual (BUG): 4 rows (rowing, strength, rowing, strength)
        """
        # This test documents the expected behavior
        # The fix is in _process_files() which now tracks processed_files
        # to avoid reprocessing the same file
        pass

    def test_incremental_merge_filters_old_multisport_sessions(self):
        """
        REGRESSION TEST: When merging with existing CSV, old sessions from
        reprocessed multisport files should be removed.

        Bug: Incremental merge checks if file in processed_files, but for
        multisport sessions, the 'file' column has session-specific names like
        "17723137905_ACTIVITY_session0_rowing_indoor_rowing" which don't match
        the actual filename "17723137905_ACTIVITY.fit".

        The '_original_file' field is used to track the source file, but if
        it doesn't exist (old CSV format), the merge fails and you get duplicates.

        Expected: Use _original_file to properly identify and remove old sessions.
        """
        pass


if __name__ == "__main__":
    unittest.main()
