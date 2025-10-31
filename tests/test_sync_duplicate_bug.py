"""TDD test to reproduce and fix the duplicate entries bug in sync."""

import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory with real FIT files from fixtures."""
    fit_dir = tmp_path / "fit_files"
    fit_dir.mkdir()

    # Copy real FIT files from fixtures for integration testing
    fixtures_dir = Path(__file__).parent / "fixtures"
    test_files = [
        "20548472357_ACTIVITY.fit",  # Volleyball
        "20747700969_ACTIVITY.fit",  # Cycling
        "20744294788_ACTIVITY.fit",  # Multisport
    ]

    for filename in test_files:
        src = fixtures_dir / filename
        if src.exists():
            shutil.copy(src, fit_dir / filename)

    return tmp_path


def test_sync_activities_no_duplicates_on_second_run(temp_test_dir):
    """Test that running sync twice doesn't create duplicate entries in CSV.

    This test reproduces the bug where run_analysis() analyzes ALL FIT files
    in the directory instead of just the new/updated ones, causing duplicates.
    """
    from fitanalyzer.sync import sync_activities

    fit_dir = temp_test_dir / "fit_files"
    output_dir = temp_test_dir / "output"
    output_dir.mkdir()

    csv_path = output_dir / "workout_summary_from_fit.csv"

    # Get list of actual FIT files
    fit_files = sorted(fit_dir.glob("*_ACTIVITY.fit"))
    assert len(fit_files) == 3, f"Expected 3 FIT files in fixtures, got {len(fit_files)}"

    # Mock authentication and download (but use real analysis)
    with patch('fitanalyzer.sync.check_and_install_garth', return_value=True), \
         patch('fitanalyzer.sync.authenticate_garmin', return_value=True), \
         patch('fitanalyzer.sync.download_new_activities') as mock_download:

        # First sync: "download" only 2 files (simulate partial download)
        mock_download.return_value = (2, [str(fit_files[0]), str(fit_files[1])])

        # First sync - analyze 2 files
        result1 = sync_activities(
            email="test@example.com",
            password="password",
            directory=str(fit_dir),
            output_dir=str(output_dir),
            ftp=300,
            hrrest=50,
            hrmax=190,
        )

        assert result1["success"], f"First sync failed: {result1.get('error')}"
        assert result1["new_activities"] == 2

        # Verify CSV exists and count initial entries
        assert csv_path.exists(), f"CSV not created at {csv_path}"
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            initial_count = len(rows)
            initial_files = {row['file'] for row in rows}
            assert initial_count > 0, f"Expected at least 1 row after first sync, got {initial_count}"

        # Second sync: download the 3rd file
        mock_download.return_value = (1, [str(fit_files[2])])

        # Second sync - should add 1 more file, not re-analyze the first 2
        result2 = sync_activities(
            email="test@example.com",
            password="password",
            directory=str(fit_dir),
            output_dir=str(output_dir),
            ftp=300,
            hrrest=50,
            hrmax=190,
        )

        assert result2["success"], f"Second sync failed: {result2.get('error')}"
        assert result2["new_activities"] == 1

        # BUG: CSV should have initial_count + 1 new entries, NO duplicates
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            final_count = len(rows)
            final_files = {row['file'] for row in rows}

            # Check that we didn't duplicate any files from the first sync
            for file in initial_files:
                count_in_final = sum(1 for r in rows if r['file'] == file)
                assert count_in_final == 1, (
                    f"File {file} appears {count_in_final} times (should be 1). "
                    f"Bug: run_analysis() re-analyzed ALL files instead of just new ones."
                )


def test_sync_activities_with_api_update_no_duplicates(temp_test_dir):
    """Test that API updates don't cause duplicates in CSV.

    When an activity's API data changes (e.g., exercise names updated on Garmin Connect),
    the file should be re-analyzed but should not create duplicate entries.
    """
    from fitanalyzer.sync import sync_activities

    fit_dir = temp_test_dir / "fit_files"
    output_dir = temp_test_dir / "output"
    output_dir.mkdir()

    csv_path = output_dir / "workout_summary_from_fit.csv"

    # Get two files from fixtures
    fit_files = sorted(fit_dir.glob("*_ACTIVITY.fit"))
    assert len(fit_files) >= 2
    file1, file2 = str(fit_files[0]), str(fit_files[1])

    # Mock authentication and download
    with patch('fitanalyzer.sync.check_and_install_garth', return_value=True), \
         patch('fitanalyzer.sync.authenticate_garmin', return_value=True), \
         patch('fitanalyzer.sync.download_new_activities') as mock_download:

        # First sync: "download" 2 files
        mock_download.return_value = (2, [file1, file2])

        # First sync
        result1 = sync_activities(
            email="test@example.com",
            password="password",
            directory=str(fit_dir),
            output_dir=str(output_dir),
            ftp=300,
            hrrest=50,
            hrmax=190,
        )

        assert result1["success"], f"First sync failed: {result1.get('error')}"

        # Verify CSV has entries (multisport files may create multiple rows)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            initial_count = len(rows)
            assert initial_count > 0, f"Expected at least 1 row, got {initial_count}"

        # Second sync: API updated file1 (exercise names changed on Garmin Connect)
        # No new downloads (0), but file1 has API data updated (returned in list)
        mock_download.return_value = (0, [file1])

        # Second sync with API update
        result2 = sync_activities(
            email="test@example.com",
            password="password",
            directory=str(fit_dir),
            output_dir=str(output_dir),
            ftp=300,
            hrrest=50,
            hrmax=190,
        )

        assert result2["success"], f"Second sync failed: {result2.get('error')}"

        # BUG: CSV should still have 2 entries (file1 updated, file2 unchanged)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            final_count = len(rows)
            assert final_count == initial_count, (
                f"Expected {initial_count} rows after API update (no duplicates), got {final_count}"
            )
