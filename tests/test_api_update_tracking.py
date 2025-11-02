"""
Test that API exercise data updates trigger re-aggregation.

TDD approach: Tests written to verify that when JSON exercise data changes
on Garmin Connect API (but FIT file doesn't change), the strength training
summary is properly updated.

Includes multisport activity handling.
Tests are designed to run in parallel safely using separate temp directories.
"""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fitanalyzer.sync import download_new_activities


class TestAPIUpdateTracking(unittest.TestCase):
    """Test that API updates trigger proper re-aggregation"""

    def setUp(self):
        """Create isolated temporary directory for parallel test execution"""
        self.test_dir = tempfile.mkdtemp(prefix="test_api_update_")
        self.output_dir = tempfile.mkdtemp(prefix="test_api_output_")

    def tearDown(self):
        """Clean up temporary directories"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    def test_api_exercise_update_triggers_reaggregation(
        self, mock_download_garth, mock_api_garth, mock_sync_garth
    ):
        """
        Test that updating exercise names via API triggers file tracking.

        Scenario:
        1. Download activity with exercise data (initial)
        2. Verify JSON saved and file tracked
        3. Update exercise names on API (FIT file unchanged)
        4. Re-run sync
        5. Verify JSON updated and file tracked in updated_files

        This ensures API updates flow through to strength aggregation.
        """
        # Step 1: Initial download with exercise data
        activity_id = 20800000001
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activity_date = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")

        initial_exercises = [
            {
                "exercises": [{"category": "BENCH_PRESS", "name": "BARBELL_BENCH_PRESS"}],
                "repetitionCount": 10,
            },
            {
                "exercises": [{"category": "SQUAT", "name": "BACK_SQUAT"}],
                "repetitionCount": 8,
            },
        ]

        mock_activities = [
            {
                "activityId": activity_id,
                "activityName": "Strength Training",
                "startTimeLocal": activity_date,
            }
        ]

        # Mock connectapi to return activity list or exercise data based on URL
        def mock_connectapi_first(url, **_kwargs):
            if "exerciseSets" in url:
                return {"exerciseSets": initial_exercises}
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi_first
        mock_download_garth.connectapi.side_effect = mock_connectapi_first
        mock_api_garth.connectapi.side_effect = mock_connectapi_first
        mock_download_garth.download.return_value = b"fake_fit_data"

        # Download initial activity
        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)
        self.assertEqual(len(updated_files), 1)

        fit_file = Path(self.test_dir) / f"{activity_id}_ACTIVITY.fit"
        json_file = Path(self.test_dir) / f"{activity_id}_ACTIVITY_exercises.json"

        self.assertTrue(fit_file.exists())
        self.assertTrue(json_file.exists())

        # Verify initial JSON content
        with open(json_file, encoding="utf-8") as f:
            json_data = json.load(f)
        self.assertEqual(
            json_data["exerciseSets"][0]["exercises"][0]["name"], "BARBELL_BENCH_PRESS"
        )

        # Step 2: Update exercise names on API (FIT file unchanged)
        updated_exercises = [
            {
                "exercises": [
                    {"category": "BENCH_PRESS", "name": "DUMBBELL_BENCH_PRESS"}
                ],  # Changed!
                "repetitionCount": 10,
            },
            {
                "exercises": [{"category": "SQUAT", "name": "FRONT_SQUAT"}],  # Changed!
                "repetitionCount": 8,
            },
        ]

        # Mock returns no new activities but updated exercise data
        def mock_connectapi_second(url, **_kwargs):
            if "exerciseSets" in url:
                return {"exerciseSets": updated_exercises}
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi_second
        mock_download_garth.connectapi.side_effect = mock_connectapi_second
        mock_api_garth.connectapi.side_effect = mock_connectapi_second

        # Step 3: Re-run sync
        _count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        # The critical assertion: file should be in updated_files for re-aggregation
        self.assertEqual(len(updated_files), 1, "API update should add file to updated_files")

        # Verify the FIT file is in updated_files (critical for re-aggregation)
        expected_fit_file = str(fit_file)
        self.assertEqual(updated_files[0], expected_fit_file)

        # Verify JSON was updated
        with open(json_file, encoding="utf-8") as f:
            updated_json_data = json.load(f)
        self.assertEqual(
            updated_json_data["exerciseSets"][0]["exercises"][0]["name"],
            "DUMBBELL_BENCH_PRESS",  # Should be updated
        )

        # This confirms that when API data changes, the file appears in updated_files
        # which will trigger re-aggregation in the strength summary

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    def test_multisport_api_update_tracking(
        self, mock_download_garth, mock_api_garth, mock_sync_garth
    ):
        """
        Test API updates for multisport activities.

        Multisport activities have session-based identifiers but FIT file tracking.
        Verify that API updates to multisport activities trigger re-aggregation.
        """
        activity_id = 20800000002
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activity_date = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")

        exercises = [
            {
                "category": "DEADLIFT",
                "exerciseName": "BARBELL_DEADLIFT",
                "repetitions": 5,
            }
        ]

        mock_activities = [
            {
                "activityId": activity_id,
                "activityName": "Multisport Workout",
                "startTimeLocal": activity_date,
            }
        ]

        def mock_connectapi(url, **_kwargs):
            if "exerciseSets" in url:
                return {"exerciseSets": exercises}
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.connectapi.side_effect = mock_connectapi
        mock_api_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.download.return_value = b"fake_multisport_fit_data"

        # Download multisport activity
        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)
        self.assertEqual(len(updated_files), 1)

        # Verify file path is tracked (not session identifier)
        updated_file = updated_files[0]
        self.assertTrue(updated_file.endswith("_ACTIVITY.fit"))
        self.assertTrue(Path(updated_file).is_absolute())

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    def test_no_api_update_skips_file(self, mock_download_garth, mock_api_garth, mock_sync_garth):
        """
        Test that files without API updates are not included in updated_files.

        Scenario:
        1. Download activity
        2. Re-sync with identical API data
        3. Verify file is NOT in updated_files list
        """
        activity_id = 20800000003
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activity_date = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")

        exercises = [{"category": "PULL_UP", "exerciseName": "PULL_UP", "repetitions": 12}]

        mock_activities = [
            {
                "activityId": activity_id,
                "activityName": "Pull-ups",
                "startTimeLocal": activity_date,
            }
        ]

        # First download
        def mock_connectapi_first(url, **_kwargs):
            if "exerciseSets" in url:
                return {"exerciseSets": exercises}
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi_first
        mock_download_garth.connectapi.side_effect = mock_connectapi_first
        mock_api_garth.connectapi.side_effect = mock_connectapi_first
        mock_download_garth.download.return_value = b"fake_fit_data"

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)
        self.assertEqual(len(updated_files), 1)

        # Second download with IDENTICAL API data
        mock_download_garth.connectapi.side_effect = (
            mock_connectapi_first  # Same function - identical data
        )
        mock_api_garth.connectapi.side_effect = (
            mock_connectapi_first  # Same function - identical data
        )

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 0)  # No new activities
        self.assertEqual(len(updated_files), 0)  # No updates either

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    def test_fit_file_update_tracked(self, mock_download_garth, mock_api_garth, mock_sync_garth):
        """
        Test that FIT file downloads are tracked in updated_files.

        Verify that when FIT file is downloaded (new or force re-download),
        it appears in updated_files list.
        """
        activity_id = 20800000004
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activity_date = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")

        mock_activities = [
            {
                "activityId": activity_id,
                "activityName": "New Workout",
                "startTimeLocal": activity_date,
            }
        ]

        def mock_connectapi(url, **_kwargs):
            if "exerciseSets" in url:
                return {}  # No exercise data
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.connectapi.side_effect = mock_connectapi
        mock_api_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.download.return_value = b"fake_fit_data"

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)
        self.assertEqual(len(updated_files), 1)

        # Verify the path is correct
        expected_file = str(Path(self.test_dir) / f"{activity_id}_ACTIVITY.fit")
        self.assertEqual(updated_files[0], expected_file)

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    def test_multiple_updates_tracked(self, mock_download_garth, mock_api_garth, mock_sync_garth):
        """
        Test that multiple file updates are all tracked.

        Scenario: Download 3 activities, verify all 3 are in updated_files.
        """
        base_date = datetime.now(timezone.utc) - timedelta(days=1)

        mock_activities = [
            {
                "activityId": 20800000005,
                "activityName": "Workout 1",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            {
                "activityId": 20800000006,
                "activityName": "Workout 2",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            {
                "activityId": 20800000007,
                "activityName": "Workout 3",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        ]

        # Mock API returns activity list or empty exercise data based on URL
        def mock_connectapi(url, **_kwargs):
            if "exerciseSets" in url:
                return {}  # No exercise data for all activities
            return mock_activities

        mock_sync_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.connectapi.side_effect = mock_connectapi
        mock_api_garth.connectapi.side_effect = mock_connectapi
        mock_download_garth.download.return_value = b"fake_fit_data"

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 3)
        self.assertEqual(len(updated_files), 3)

        # Verify all files are unique
        self.assertEqual(len(set(updated_files)), 3)


if __name__ == "__main__":
    unittest.main()
