"""Tests for activity_download module - FIT file download and management."""

import io
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import Mock, mock_open, patch

from fitanalyzer.activity_download import (
    _extract_fit_from_zip,
    download_single_activity,
    print_download_summary,
    should_download_activity,
)
from fitanalyzer.sync import download_new_activities


class TestExtractFitFromZip(unittest.TestCase):
    """Test _extract_fit_from_zip function."""

    def test_already_fit_file(self) -> None:
        """Test when data is already a FIT file (not zipped)."""
        fit_data = b".FIT file content"  # FIT files don't start with PK
        result = _extract_fit_from_zip(fit_data)
        self.assertEqual(result, fit_data)

    def test_zip_with_fit_file(self) -> None:
        """Test extracting FIT file from ZIP."""
        # Create a ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("activity.fit", b"FIT file content")
        zip_data = zip_buffer.getvalue()

        result = _extract_fit_from_zip(zip_data)
        self.assertEqual(result, b"FIT file content")

    def test_zip_with_uppercase_fit(self) -> None:
        """Test extracting .FIT file with uppercase extension."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("activity.FIT", b"FIT content uppercase")
        zip_data = zip_buffer.getvalue()

        result = _extract_fit_from_zip(zip_data)
        self.assertEqual(result, b"FIT content uppercase")

    def test_zip_with_multiple_files(self) -> None:
        """Test ZIP with multiple files - should extract first FIT."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("readme.txt", b"readme")
            zf.writestr("first.fit", b"first FIT")
            zf.writestr("second.fit", b"second FIT")
        zip_data = zip_buffer.getvalue()

        result = _extract_fit_from_zip(zip_data)
        self.assertEqual(result, b"first FIT")

    def test_zip_with_no_fit_file(self) -> None:
        """Test ZIP with no FIT files."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("data.txt", b"text file")
            zf.writestr("data.json", b'{"json": "data"}')
        zip_data = zip_buffer.getvalue()

        result = _extract_fit_from_zip(zip_data)
        self.assertIsNone(result)

    def test_empty_zip(self) -> None:
        """Test empty ZIP file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            pass  # Empty ZIP
        zip_data = zip_buffer.getvalue()

        result = _extract_fit_from_zip(zip_data)
        self.assertIsNone(result)


class TestShouldDownloadActivity(unittest.TestCase):
    """Test should_download_activity function."""

    def test_new_activity(self) -> None:
        """Test new activity (not in existing)."""
        activity = {"activityId": 12345}
        existing = {}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertTrue(should_dl)
        self.assertFalse(is_update)
        self.assertFalse(check_api)

    def test_existing_no_update_date(self) -> None:
        """Test existing activity with no updateDate."""
        activity = {"activityId": 12345}
        existing = {"12345": 1234567890.0}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertFalse(should_dl)
        self.assertFalse(is_update)
        self.assertTrue(check_api)  # Should still check API for exercise updates

    def test_existing_with_older_garmin_timestamp(self) -> None:
        """Test existing activity where Garmin is older."""
        activity = {
            "activityId": 12345,
            "updateDate": 1234567000000,  # milliseconds
        }
        existing = {"12345": 1234568000.0}  # Local is newer

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertFalse(should_dl)
        self.assertFalse(is_update)
        self.assertTrue(check_api)

    def test_existing_with_newer_garmin_timestamp(self) -> None:
        """Test existing activity where Garmin is newer."""
        activity = {
            "activityId": 12345,
            "updateDate": 1234569000000,  # milliseconds - newer
        }
        existing = {"12345": 1234567000.0}  # Local is older

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertTrue(should_dl)
        self.assertTrue(is_update)
        self.assertFalse(check_api)

    def test_existing_with_lastModified_instead_of_updateDate(self) -> None:
        """Test using lastModified when updateDate not available."""
        activity = {
            "activityId": 12345,
            "lastModified": 1234569000000,  # Use lastModified
        }
        existing = {"12345": 1234567000.0}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertTrue(should_dl)
        self.assertTrue(is_update)
        self.assertFalse(check_api)

    def test_timestamp_within_tolerance(self) -> None:
        """Test timestamp within 1 second tolerance."""
        activity = {
            "activityId": 12345,
            "updateDate": 1234567000500,  # 0.5 seconds newer
        }
        existing = {"12345": 1234567000.0}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        # Should not download - within 1 second tolerance
        self.assertFalse(should_dl)
        self.assertFalse(is_update)
        self.assertTrue(check_api)


class TestDownloadSingleActivity(unittest.TestCase):
    """Test download_single_activity function."""

    @patch("fitanalyzer.activity_download.GARTH_AVAILABLE", False)
    def test_garth_not_available(self) -> None:
        """Test when garth is not available."""
        with self.assertRaises(ImportError) as context:
            download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")
        self.assertIn("garth library not available", str(context.exception))

    @patch("fitanalyzer.activity_download.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.activity_download.save_exercise_sets_to_json")
    @patch("fitanalyzer.activity_download.Path")
    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_successful_download_with_exercise_data(
        self,
        mock_print: Mock,
        mock_file: Mock,
        mock_garth: Mock,
        mock_path: Mock,
        mock_save_json: Mock,
        mock_fetch_api: Mock,
    ) -> None:
        """Test successful download with exercise data."""
        # Create a ZIP with FIT file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("activity.fit", b"FIT data")
        zip_data = zip_buffer.getvalue()

        mock_garth.download.return_value = zip_data
        mock_fetch_api.return_value = {
            "activityId": 12345,
            "exerciseSets": [{"reps": 10}, {"reps": 12}],
        }

        # Mock Path behavior
        mock_filename = Mock()
        mock_path.return_value.__truediv__.return_value = mock_filename

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertTrue(result)
        mock_garth.download.assert_called_once_with("/download-service/files/activity/12345")
        mock_file.assert_called_once()
        mock_save_json.assert_called_once()

    @patch("fitanalyzer.activity_download.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.activity_download.Path")
    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_successful_download_without_exercise_data(
        self,
        mock_print: Mock,
        mock_file: Mock,
        mock_garth: Mock,
        mock_path: Mock,
        mock_fetch_api: Mock,
    ) -> None:
        """Test successful download without exercise data."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("activity.fit", b"FIT data")
        zip_data = zip_buffer.getvalue()

        mock_garth.download.return_value = zip_data
        mock_fetch_api.return_value = None  # No exercise data

        mock_filename = Mock()
        mock_path.return_value.__truediv__.return_value = mock_filename

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertTrue(result)
        mock_garth.download.assert_called_once()

    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.print")
    def test_download_no_fit_in_zip(self, mock_print: Mock, mock_garth: Mock) -> None:
        """Test when ZIP has no FIT file."""
        # Create ZIP with no FIT file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("data.txt", b"text file")
        zip_data = zip_buffer.getvalue()

        mock_garth.download.return_value = zip_data

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertFalse(result)
        self.assertTrue(
            any("No .fit file found" in str(call) for call in mock_print.call_args_list)
        )

    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.print")
    def test_download_os_error(self, mock_print: Mock, mock_garth: Mock) -> None:
        """Test handling of OSError."""
        mock_garth.download.side_effect = OSError("Network error")

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertFalse(result)
        self.assertTrue(any("Error downloading" in str(call) for call in mock_print.call_args_list))

    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.print")
    def test_download_runtime_error(self, mock_print: Mock, mock_garth: Mock) -> None:
        """Test handling of RuntimeError."""
        mock_garth.download.side_effect = RuntimeError("API error")

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertFalse(result)

    @patch("fitanalyzer.activity_download.garth")
    @patch("builtins.print")
    def test_download_value_error(self, mock_print: Mock, mock_garth: Mock) -> None:
        """Test handling of ValueError."""
        mock_garth.download.side_effect = ValueError("Invalid data")

        result = download_single_activity(12345, "Test Activity", "2025-11-01", "/test/dir")

        self.assertFalse(result)


class TestPrintDownloadSummary(unittest.TestCase):
    """Test print_download_summary function."""

    @patch("builtins.print")
    def test_summary_with_all_counts(self, mock_print: Mock) -> None:
        """Test summary with all types of counts."""
        counters = {
            "new_count": 5,
            "updated_count": 3,
            "api_updated_count": 2,
            "skipped_count": 10,
        }

        print_download_summary(counters)

        # Check that all counts are printed
        calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Download complete" in call for call in calls))
        self.assertTrue(any("New activities: 5" in call for call in calls))
        self.assertTrue(any("Updated activities: 3" in call for call in calls))
        self.assertTrue(any("Exercise data updated: 2" in call for call in calls))
        self.assertTrue(any("Skipped" in call and "10" in call for call in calls))

    @patch("builtins.print")
    def test_summary_without_api_updates(self, mock_print: Mock) -> None:
        """Test summary when no API updates."""
        counters = {
            "new_count": 2,
            "updated_count": 1,
            "api_updated_count": 0,
            "skipped_count": 5,
        }

        print_download_summary(counters)

        calls = [str(call) for call in mock_print.call_args_list]
        # Should not print exercise data line when count is 0
        self.assertFalse(any("Exercise data updated" in call for call in calls))

    @patch("builtins.print")
    def test_summary_all_zeros(self, mock_print: Mock) -> None:
        """Test summary with all zero counts."""
        counters = {
            "new_count": 0,
            "updated_count": 0,
            "api_updated_count": 0,
            "skipped_count": 0,
        }

        print_download_summary(counters)

        calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("New activities: 0" in call for call in calls))


class TestDownloadNewActivities(unittest.TestCase):
    """Test download_new_activities function."""

    @patch("fitanalyzer.sync.GARTH_AVAILABLE", False)
    def test_garth_not_available(self) -> None:
        """Test when garth library is not available."""
        with self.assertRaises(ImportError) as context:
            download_new_activities()
        self.assertIn("garth library not available", str(context.exception))

    @patch("fitanalyzer.sync.print_download_summary")
    @patch("fitanalyzer.sync.process_activities")
    @patch("fitanalyzer.sync.identify_multisport_parents")
    @patch("fitanalyzer.sync.filter_recent_activities")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_no_activities_found(
        self,
        mock_print: Mock,
        mock_sync_garth: Mock,
        mock_download_garth: Mock,
        mock_get_existing: Mock,
        mock_filter: Mock,
        mock_identify: Mock,
        mock_process: Mock,
        mock_summary: Mock,
    ) -> None:
        """Test when no activities are returned from API."""
        mock_sync_garth.connectapi.return_value = None
        mock_get_existing.return_value = {}

        count, files = download_new_activities(days=7, directory="/test/dir")

        self.assertEqual(count, 0)
        self.assertEqual(files, [])

    @patch("fitanalyzer.sync.print_download_summary")
    @patch("fitanalyzer.sync.process_activities")
    @patch("fitanalyzer.sync.identify_multisport_parents")
    @patch("fitanalyzer.sync.filter_recent_activities")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_force_mode(
        self,
        mock_print: Mock,
        mock_sync_garth: Mock,
        mock_download_garth: Mock,
        mock_get_existing: Mock,
        mock_filter: Mock,
        mock_identify: Mock,
        mock_process: Mock,
        mock_summary: Mock,
    ) -> None:
        """Test force mode (re-download all)."""
        mock_sync_garth.connectapi.return_value = [
            {
                "activityId": 123,
                "activityName": "Activity 1",
                "startTimeLocal": "2025-11-01T10:00:00.0",
            }
        ]
        mock_download_garth.download.return_value = b"fake_fit_data"
        mock_filter.return_value = [{"activityId": 123}]
        mock_identify.return_value = set()
        mock_process.return_value = (
            {"new_count": 1, "updated_count": 0, "api_updated_count": 0, "skipped_count": 0},
            ["123_ACTIVITY.fit"],
        )

        count, files = download_new_activities(days=7, directory="/test/dir", force=True)

        # Should not call get_existing_activity_ids in force mode
        mock_get_existing.assert_not_called()
        self.assertEqual(count, 1)

    @patch("fitanalyzer.sync.print_download_summary")
    @patch("fitanalyzer.sync.process_activities")
    @patch("fitanalyzer.sync.identify_multisport_parents")
    @patch("fitanalyzer.sync.filter_recent_activities")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_dict_activities_converted_to_list(
        self,
        mock_print: Mock,
        mock_sync_garth: Mock,
        mock_download_garth: Mock,
        mock_get_existing: Mock,
        mock_filter: Mock,
        mock_identify: Mock,
        mock_process: Mock,
        mock_summary: Mock,
    ) -> None:
        """Test when API returns dict instead of list."""
        # Single activity as dict
        mock_sync_garth.connectapi.return_value = {
            "activityId": 123,
            "activityName": "Activity",
            "startTimeLocal": "2025-11-01T10:00:00.0",
        }
        mock_download_garth.download.return_value = b"fake_fit_data"
        mock_get_existing.return_value = {}
        mock_filter.return_value = [{"activityId": 123}]
        mock_identify.return_value = set()
        mock_process.return_value = (
            {"new_count": 1, "updated_count": 0, "api_updated_count": 0, "skipped_count": 0},
            [],
        )

        count, files = download_new_activities(days=7, directory="/test/dir")

        # Should convert dict to list
        mock_filter.assert_called_once()
        self.assertEqual(count, 1)

    @patch("fitanalyzer.sync.print_download_summary")
    @patch("fitanalyzer.sync.process_activities")
    @patch("fitanalyzer.sync.identify_multisport_parents")
    @patch("fitanalyzer.sync.filter_recent_activities")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_with_limit(
        self,
        mock_print: Mock,
        mock_sync_garth: Mock,
        mock_download_garth: Mock,
        mock_get_existing: Mock,
        mock_filter: Mock,
        mock_identify: Mock,
        mock_process: Mock,
        mock_summary: Mock,
    ) -> None:
        """Test with activity limit."""
        mock_sync_garth.connectapi.return_value = []
        mock_get_existing.return_value = {}
        mock_filter.return_value = []
        mock_identify.return_value = set()
        mock_process.return_value = (
            {"new_count": 0, "updated_count": 0, "api_updated_count": 0, "skipped_count": 0},
            [],
        )

        download_new_activities(days=7, limit=50, directory="/test/dir")

        # Check that limit is passed to API
        mock_sync_garth.connectapi.assert_called_once()
        call_args = mock_sync_garth.connectapi.call_args
        self.assertEqual(call_args[1]["params"]["limit"], 50)

    @patch("fitanalyzer.sync.print_download_summary")
    @patch("fitanalyzer.sync.process_activities")
    @patch("fitanalyzer.sync.identify_multisport_parents")
    @patch("fitanalyzer.sync.filter_recent_activities")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_successful_download_flow(
        self,
        mock_print: Mock,
        mock_sync_garth: Mock,
        mock_download_garth: Mock,
        mock_get_existing: Mock,
        mock_filter: Mock,
        mock_identify: Mock,
        mock_process: Mock,
        mock_summary: Mock,
    ) -> None:
        """Test complete successful download flow."""
        mock_sync_garth.connectapi.return_value = [
            {"activityId": 123, "activityName": "Run", "startTimeLocal": "2025-11-01T10:00:00.0"},
            {"activityId": 456, "activityName": "Bike", "startTimeLocal": "2025-11-01T11:00:00.0"},
        ]
        mock_download_garth.download.return_value = b"fake_fit_data"
        mock_get_existing.return_value = {"123": 1234567890.0}
        mock_filter.return_value = [
            {"activityId": 123},
            {"activityId": 456},
        ]
        mock_identify.return_value = set()
        mock_process.return_value = (
            {"new_count": 1, "updated_count": 1, "api_updated_count": 0, "skipped_count": 0},
            ["123_ACTIVITY.fit", "456_ACTIVITY.fit"],
        )

        count, files = download_new_activities(days=7, directory="/test/dir")

        self.assertEqual(count, 2)  # new + updated
        self.assertEqual(len(files), 2)
        mock_summary.assert_called_once()

    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_os_error_during_fetch(
        self, mock_print: Mock, mock_sync_garth: Mock, mock_download_garth: Mock
    ) -> None:
        """Test handling of OSError during activity fetch."""
        mock_sync_garth.connectapi.side_effect = OSError("Network error")

        count, files = download_new_activities(days=7, directory="/test/dir")

        self.assertEqual(count, 0)
        self.assertEqual(files, [])
        self.assertTrue(any("Error fetching" in str(call) for call in mock_print.call_args_list))

    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_runtime_error_during_fetch(
        self, mock_print: Mock, mock_sync_garth: Mock, mock_download_garth: Mock
    ) -> None:
        """Test handling of RuntimeError during activity fetch."""
        mock_sync_garth.connectapi.side_effect = RuntimeError("API error")

        count, files = download_new_activities(days=7, directory="/test/dir")

        self.assertEqual(count, 0)
        self.assertEqual(files, [])

    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    @patch("builtins.print")
    def test_value_error_during_fetch(
        self, mock_print: Mock, mock_sync_garth: Mock, mock_download_garth: Mock
    ) -> None:
        """Test handling of ValueError during activity fetch."""
        mock_sync_garth.connectapi.side_effect = ValueError("Invalid data")

        count, files = download_new_activities(days=7, directory="/test/dir")

        self.assertEqual(count, 0)
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
