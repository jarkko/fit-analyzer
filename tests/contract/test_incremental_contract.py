"""
Contract tests for incremental.py functions.

These tests document and enforce contracts for incremental analysis functions.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fitanalyzer.incremental import (
    determine_files_to_process,
    load_existing_analysis,
    load_existing_rows,
    needs_analysis,
)


class TestLoadExistingAnalysisContract:
    """Contract tests for load_existing_analysis() function.

    Contract: Extract file modification times from CSV to determine
    which files need reanalysis. Handles multisport files with _original_file column.

    Parameter matrix:
    - CSV doesn't exist → empty dict
    - CSV missing required columns → empty dict
    - CSV with _original_file column → use _original_file when present
    - CSV without _original_file column → use file column
    - _original_file is NaN → fall back to file
    - CSV read errors → empty dict
    """

    def test_nonexistent_csv_returns_empty(self, tmp_path):
        """Contract: When CSV doesn't exist, return empty dict."""
        csv_path = tmp_path / "nonexistent.csv"

        result = load_existing_analysis(csv_path)

        assert result == {}

    def test_csv_missing_file_column_returns_empty(self, tmp_path):
        """Contract: CSV without 'file' column returns empty dict."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"other_col": [1, 2], "_file_mtime": [100.0, 200.0]})
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        assert result == {}

    def test_csv_missing_mtime_column_returns_empty(self, tmp_path):
        """Contract: CSV without '_file_mtime' column returns empty dict."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"file": ["file1.fit", "file2.fit"]})
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        assert result == {}

    def test_basic_csv_extracts_mtimes(self, tmp_path):
        """Contract: CSV with file and _file_mtime returns mapping."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {"file": ["file1.fit", "file2.fit", "file3.fit"], "_file_mtime": [100.5, 200.7, 300.9]}
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        assert result == {"file1.fit": 100.5, "file2.fit": 200.7, "file3.fit": 300.9}

    def test_csv_with_original_file_column_uses_it(self, tmp_path):
        """Contract: When _original_file exists, use it preferentially."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "file": ["20123_ACTIVITY.fit", "20456_ACTIVITY.fit"],
                "_original_file": ["original1.fit", "original2.fit"],
                "_file_mtime": [100.0, 200.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        # Should map using _original_file, not file
        assert result == {"original1.fit": 100.0, "original2.fit": 200.0}

    def test_csv_with_nan_original_file_falls_back(self, tmp_path):
        """Contract: When _original_file is NaN, fall back to file column."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "file": ["file1.fit", "file2.fit", "file3.fit"],
                "_original_file": ["original1.fit", pd.NA, "original3.fit"],
                "_file_mtime": [100.0, 200.0, 300.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        # First and third use _original_file, second falls back to file
        assert result == {"original1.fit": 100.0, "file2.fit": 200.0, "original3.fit": 300.0}

    def test_csv_ignores_nan_mtimes(self, tmp_path):
        """Contract: Rows with NaN _file_mtime are excluded from mapping."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame(
            {"file": ["file1.fit", "file2.fit", "file3.fit"], "_file_mtime": [100.0, pd.NA, 300.0]}
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        # Only files with valid mtime should be included
        assert result == {"file1.fit": 100.0, "file3.fit": 300.0}

    def test_corrupt_csv_returns_empty(self, tmp_path):
        """Contract: Corrupted CSV returns empty dict (exception handling)."""
        csv_path = tmp_path / "corrupt.csv"
        csv_path.write_text("not,valid,csv\ndata")

        result = load_existing_analysis(csv_path)

        # Should handle parse error gracefully
        assert result == {}

    def test_csv_with_invalid_data_types_returns_empty(self, tmp_path):
        """Contract: CSV with data that causes ValueError returns empty dict."""
        csv_path = tmp_path / "bad_types.csv"
        df = pd.DataFrame(
            {
                "file": ["file1.fit"],
                "_file_mtime": ["not_a_number"],  # Invalid: string instead of float
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        # Should handle ValueError gracefully when converting to float
        assert result == {}


class TestLoadExistingRowsContract:
    """Contract tests for load_existing_rows() function.

    Contract: Load existing CSV rows and restore _file_mtime from
    existing_analysis dict.
    """

    def test_nonexistent_csv_returns_empty_list(self, tmp_path):
        """Contract: When CSV doesn't exist, return empty list."""
        csv_path = tmp_path / "nonexistent.csv"

        result = load_existing_rows(csv_path, {})

        assert result == []

    def test_csv_without_file_column_returns_empty(self, tmp_path):
        """Contract: CSV without 'file' column returns empty list."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"other_col": [1, 2]})
        df.to_csv(csv_path, index=False)

        result = load_existing_rows(csv_path, {})

        assert result == []

    def test_loads_rows_and_restores_mtime(self, tmp_path):
        """Contract: Load rows and restore _file_mtime from existing_analysis."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"file": ["file1.fit", "file2.fit"], "sport": ["cycling", "running"]})
        df.to_csv(csv_path, index=False)

        existing_analysis = {"file1.fit": 100.0, "file2.fit": 200.0}

        result = load_existing_rows(csv_path, existing_analysis)

        assert len(result) == 2
        assert result[0]["file"] == "file1.fit"
        assert result[0]["_file_mtime"] == 100.0
        assert result[1]["file"] == "file2.fit"
        assert result[1]["_file_mtime"] == 200.0

    def test_handles_missing_files_in_analysis_dict(self, tmp_path):
        """Contract: Files not in existing_analysis don't get _file_mtime."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"file": ["file1.fit", "file2.fit"], "sport": ["cycling", "running"]})
        df.to_csv(csv_path, index=False)

        existing_analysis = {"file1.fit": 100.0}  # file2.fit missing

        result = load_existing_rows(csv_path, existing_analysis)

        assert len(result) == 2
        assert result[0]["_file_mtime"] == 100.0
        assert "_file_mtime" not in result[1]  # file2 doesn't get mtime

    def test_corrupt_csv_returns_empty_list(self, tmp_path):
        """Contract: Corrupted CSV returns empty list (exception handling)."""
        csv_path = tmp_path / "corrupt.csv"
        csv_path.write_text("not,valid,csv\ndata")

        result = load_existing_rows(csv_path, {})

        # Should handle parse error gracefully
        assert result == []


class TestDetermineFilesToProcessContract:
    """Contract tests for determine_files_to_process() function.

    Contract: Determine which FIT files need analysis based on
    force flag and file modification times.
    """

    def test_force_flag_processes_all_files(self, tmp_path):
        """Contract: When force=True, all files need processing."""
        # Create test files
        file1 = tmp_path / "file1.fit"
        file2 = tmp_path / "file2.fit"
        file1.touch()
        file2.touch()

        fit_files = [str(file1), str(file2)]
        existing_analysis = {str(file1): file1.stat().st_mtime}  # file1 already analyzed

        files_to_process, skipped_count = determine_files_to_process(
            fit_files, existing_analysis, force=True
        )

        assert len(files_to_process) == 2
        assert skipped_count == 0  # No files skipped when force=True

    def test_incremental_only_processes_new_and_modified(self, tmp_path):
        """Contract: When force=False, only process new and modified files."""
        file1 = tmp_path / "file1.fit"
        file2 = tmp_path / "file2.fit"
        file3 = tmp_path / "file3.fit"
        file1.touch()
        file2.touch()
        file3.touch()

        # file1: already analyzed and unchanged
        # file2: new (not in existing_analysis)
        # file3: modified (older mtime in existing_analysis)
        existing_analysis = {
            str(file1): file1.stat().st_mtime,  # Up to date
            str(file3): file3.stat().st_mtime - 10.0,  # Older mtime
        }

        fit_files = [str(file1), str(file2), str(file3)]

        files_to_process, skipped_count = determine_files_to_process(
            fit_files, existing_analysis, force=False
        )

        # Should process file2 (new) and file3 (modified), but not file1
        assert str(file2) in files_to_process
        assert str(file3) in files_to_process
        assert str(file1) not in files_to_process
        assert len(files_to_process) == 2
        assert skipped_count == 1  # file1 was skipped
