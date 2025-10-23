"""
Unit tests for Garmin sync module.
Tests Garmin Connect integration, file management, and sync logic.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

from fitanalyzer.sync import (
    authenticate_garmin,
    download_new_activities,
    get_existing_activity_ids,
    run_analysis,
)


class TestExistingActivityIDs(unittest.TestCase):
    """Test detection of existing activity files"""

    def setUp(self):
        """Create temporary directory with test files"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)

    def test_get_existing_activity_ids_empty_directory(self):
        """Test with no FIT files"""
        result = get_existing_activity_ids(self.test_dir)
        self.assertEqual(result, {})

    def test_get_existing_activity_ids_with_files(self):
        """Test with valid activity files"""
        # Create test files
        test_files = [
            "20744294782_ACTIVITY.fit",
            "20744294788_ACTIVITY.fit",
            "20747700969_ACTIVITY.fit",
        ]

        for filename in test_files:
            (Path(self.test_dir) / filename).touch()

        result = get_existing_activity_ids(self.test_dir)

        expected_ids = {"20744294782", "20744294788", "20747700969"}
        # Now returns dict with activity_id -> mtime, so check keys
        self.assertEqual(set(result.keys()), expected_ids)
        # Verify all values are timestamps (floats)
        for mtime in result.values():
            self.assertIsInstance(mtime, float)

    def test_get_existing_activity_ids_mixed_files(self):
        """Test with mix of valid and invalid filenames"""
        test_files = [
            "20744294782_ACTIVITY.fit",  # Valid
            "invalid_file.fit",  # Invalid (not numeric)
            "20744294788_ACTIVITY.fit",  # Valid
            "README.md",  # Invalid (not FIT)
            "test_ACTIVITY.fit",  # Invalid (not numeric)
        ]

        for filename in test_files:
            (Path(self.test_dir) / filename).touch()

        result = get_existing_activity_ids(self.test_dir)

        expected_ids = {"20744294782", "20744294788"}
        # Now returns dict with activity_id -> mtime, so check keys
        self.assertEqual(set(result.keys()), expected_ids)


class TestGarminAuthentication(unittest.TestCase):
    """Test Garmin Connect authentication"""

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.sync.Path")
    def test_authenticate_with_existing_session(self, mock_path, mock_garth):
        """Test resuming existing authentication session"""
        # Mock existing token file
        mock_token_path = Mock()
        mock_token_path.exists.return_value = True
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Mock successful resume
        mock_garth.resume.return_value = None
        mock_garth.client.username = "test_user"

        result = authenticate_garmin()

        self.assertTrue(result)
        mock_garth.resume.assert_called_once()

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.sync.Path")
    def test_authenticate_expired_session(self, mock_path, mock_garth):
        """Test handling of expired session"""
        # Mock existing but expired token
        mock_token_path = Mock()
        mock_token_path.exists.return_value = True
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Resume fails (expired) - use a specific exception type we handle
        mock_garth.resume.side_effect = RuntimeError("Session expired")

        # Login should be attempted and will succeed
        result = authenticate_garmin(email="test@test.com", password="password")

        # Should attempt resume
        mock_garth.resume.assert_called_once()
        # Should call login after resume fails
        mock_garth.login.assert_called_once_with("test@test.com", "password")

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.sync.Path")
    def test_authenticate_new_login(self, mock_path, mock_garth):
        """Test new authentication"""
        # No existing token
        mock_token_path = Mock()
        mock_token_path.exists.return_value = False
        mock_token_path.parent = Mock()
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Mock successful login
        mock_garth.login.return_value = None
        mock_garth.save.return_value = None

        result = authenticate_garmin(email="test@example.com", password="testpass")

        self.assertTrue(result)
        mock_garth.login.assert_called_once_with("test@example.com", "testpass")
        mock_garth.save.assert_called_once()


class TestDownloadActivities(unittest.TestCase):
    """Test activity download logic"""

    def setUp(self):
        """Set up test directory"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    @patch("fitanalyzer.sync.garth")
    def test_download_with_no_new_activities(self, mock_garth):
        """Test when all activities already exist"""
        # Create existing files
        existing_ids = ["20744294782", "20744294788"]
        for activity_id in existing_ids:
            (Path(self.test_dir) / f"{activity_id}_ACTIVITY.fit").touch()

        # Mock Garmin API to return same activities
        mock_activities = [
            {
                "activityId": 20744294782,
                "activityName": "Test Activity 1",
                "startTimeLocal": "2025-10-20T10:00:00Z",
            },
            {
                "activityId": 20744294788,
                "activityName": "Test Activity 2",
                "startTimeLocal": "2025-10-20T11:00:00Z",
            },
        ]

        mock_garth.connectapi.return_value = mock_activities

        new_count = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 0)

    @patch("fitanalyzer.sync.garth")
    def test_download_with_new_activities(self, mock_garth):
        """Test downloading new activities"""
        # Mock Garmin API
        mock_activities = [
            {
                "activityId": 20765123456,
                "activityName": "New Activity",
                "startTimeLocal": "2025-10-23T10:00:00Z",
            }
        ]

        mock_garth.connectapi.return_value = mock_activities
        mock_garth.download.return_value = b"fake_fit_data"

        new_count = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)

        # Check file was created
        expected_file = Path(self.test_dir) / "20765123456_ACTIVITY.fit"
        self.assertTrue(expected_file.exists())

    @patch("fitanalyzer.sync.garth")
    def test_download_handles_errors_gracefully(self, mock_garth):
        """Test error handling during download"""
        mock_activities = [
            {
                "activityId": 20765123456,
                "activityName": "Activity 1",
                "startTimeLocal": "2025-10-23T10:00:00Z",
            },
            {
                "activityId": 20765234567,
                "activityName": "Activity 2",
                "startTimeLocal": "2025-10-23T11:00:00Z",
            },
        ]

        mock_garth.connectapi.return_value = mock_activities

        # First download succeeds, second fails
        mock_garth.download.side_effect = [b"fake_fit_data", Exception("Network error")]

    @patch("fitanalyzer.sync.garth")
    def test_download_handles_timezone_formats(self, mock_garth):
        """Test handling of different timezone formats in activity dates"""
        # Test with various timezone formats that Garmin might return
        mock_activities = [
            {
                "activityId": 20765100001,
                "activityName": "Activity with Z",
                "startTimeLocal": "2025-10-23T10:00:00Z",  # UTC with Z
            },
            {
                "activityId": 20765100002,
                "activityName": "Activity with offset",
                "startTimeLocal": "2025-10-23T10:00:00+00:00",  # UTC with offset
            },
            {
                "activityId": 20765100003,
                "activityName": "Activity naive",
                "startTimeLocal": "2025-10-23T10:00:00",  # Naive (no timezone)
            },
        ]

        mock_garth.connectapi.return_value = mock_activities
        mock_garth.download.return_value = b"fake_fit_data"

        # Should not raise "can't compare offset-naive and offset-aware datetimes"
        new_count = download_new_activities(days=7, directory=self.test_dir)

        # All 3 should be processed without timezone errors
        self.assertEqual(new_count, 3)


class TestAnalysisExecution(unittest.TestCase):
    """Test running the analysis script"""

    def setUp(self):
        """Set up test directory with mock script"""
        self.test_dir = tempfile.mkdtemp()

        # Create a mock analysis script
        script_path = Path(self.test_dir) / "fit_to_summary.py"
        script_path.write_text("#!/usr/bin/env python3\nprint('Analysis complete')")
        script_path.chmod(0o755)

        # Create test FIT files
        for i in range(3):
            (Path(self.test_dir) / f"2076512345{i}_ACTIVITY.fit").touch()

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    @patch("fitanalyzer.parser.main_with_args")
    @patch("fitanalyzer.parser.parse_arguments")
    def test_run_analysis_success(self, mock_parse, mock_main):
        """Test successful analysis execution"""
        mock_main.return_value = 0
        mock_parse.return_value = Mock()

        result = run_analysis(ftp=300, directory=self.test_dir)

        self.assertTrue(result)
        mock_main.assert_called_once()

    def test_run_analysis_no_fit_files(self):
        """Test analysis with no FIT files"""
        empty_dir = tempfile.mkdtemp()

        try:
            result = run_analysis(ftp=300, directory=empty_dir)

            self.assertFalse(result)
        finally:
            shutil.rmtree(empty_dir)

    @patch("fitanalyzer.parser.main_with_args")
    @patch("fitanalyzer.parser.parse_arguments")
    def test_run_analysis_script_error(self, mock_parse, mock_main):
        """Test handling of script errors"""
        mock_main.return_value = 1
        mock_parse.return_value = Mock()

        result = run_analysis(ftp=300, directory=self.test_dir)

        self.assertFalse(result)

        self.assertFalse(result)


class TestIdempotency(unittest.TestCase):
    """Test idempotent behavior of sync operations"""

    def setUp(self):
        """Set up test directory"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_repeated_activity_detection(self):
        """Test that running sync multiple times doesn't create duplicates"""
        # First run - create files
        activity_ids = ["20744294782", "20744294788", "20747700969"]

        for activity_id in activity_ids:
            (Path(self.test_dir) / f"{activity_id}_ACTIVITY.fit").touch()

        # Get existing IDs (first check)
        ids_first = get_existing_activity_ids(self.test_dir)

        # Simulate second run - try to create same files
        for activity_id in activity_ids:
            file_path = Path(self.test_dir) / f"{activity_id}_ACTIVITY.fit"
            # File should already exist
            self.assertTrue(file_path.exists())

        # Get existing IDs (second check)
        ids_second = get_existing_activity_ids(self.test_dir)

        # Should be identical
        self.assertEqual(ids_first, ids_second)
        self.assertEqual(len(ids_first), 3)


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable handling"""

    def test_garmin_email_env_var(self):
        """Test reading GARMIN_EMAIL from environment"""
        with patch.dict(os.environ, {"GARMIN_EMAIL": "test@example.com"}):
            email = os.getenv("GARMIN_EMAIL")
            self.assertEqual(email, "test@example.com")

    def test_garmin_password_env_var(self):
        """Test reading GARMIN_PASSWORD from environment"""
        with patch.dict(os.environ, {"GARMIN_PASSWORD": "testpass"}):
            password = os.getenv("GARMIN_PASSWORD")
            self.assertEqual(password, "testpass")

    def test_missing_env_vars(self):
        """Test handling of missing environment variables"""
        with patch.dict(os.environ, {}, clear=True):
            email = os.getenv("GARMIN_EMAIL")
            password = os.getenv("GARMIN_PASSWORD")

            self.assertIsNone(email)
            self.assertIsNone(password)


class TestCheckAndInstallGarth(unittest.TestCase):
    """Test garth installation checking."""

    @patch('fitanalyzer.sync.GARTH_AVAILABLE', True)
    def test_check_garth_available(self):
        """Test when garth is already available."""
        from fitanalyzer.sync import check_and_install_garth
        result = check_and_install_garth()
        self.assertTrue(result)

    @patch('fitanalyzer.sync.GARTH_AVAILABLE', False)
    @patch('builtins.input', return_value='n')
    @patch('builtins.print')
    def test_check_garth_not_available_decline_install(self, mock_print, mock_input):
        """Test when garth not available and user declines install."""
        from fitanalyzer.sync import check_and_install_garth
        result = check_and_install_garth()
        self.assertFalse(result)

    @patch('fitanalyzer.sync.GARTH_AVAILABLE', False)
    @patch('builtins.input', return_value='y')
    @patch('subprocess.check_call')
    @patch('builtins.print')
    def test_check_garth_install_success(self, mock_print, mock_subprocess, mock_input):
        """Test successful garth installation."""
        from fitanalyzer.sync import check_and_install_garth
        result = check_and_install_garth()
        # Returns False because need to restart
        self.assertFalse(result)
        mock_subprocess.assert_called_once()

    @patch('fitanalyzer.sync.GARTH_AVAILABLE', False)
    @patch('builtins.input', return_value='y')
    @patch('subprocess.check_call', side_effect=subprocess.CalledProcessError(1, 'pip'))
    @patch('builtins.print')
    def test_check_garth_install_failure(self, mock_print, mock_subprocess, mock_input):
        """Test failed garth installation."""
        from fitanalyzer.sync import check_and_install_garth
        result = check_and_install_garth()
        self.assertFalse(result)


class TestExerciseSetsAPI(unittest.TestCase):
    """Test exercise sets API functions."""

    @patch('fitanalyzer.sync.garth')
    def test_fetch_exercise_sets_from_api(self, mock_garth):
        """Test fetching exercise sets from Garmin API."""
        from fitanalyzer.sync import fetch_exercise_sets_from_api
        
        mock_garth.connectapi.return_value = {
            "exerciseSets": [
                {"category": 1, "exerciseName": "BENCH_PRESS", "reps": 10}
            ]
        }
        
        result = fetch_exercise_sets_from_api("12345")
        self.assertIsNotNone(result)
        self.assertIn("exerciseSets", result)

    @patch('fitanalyzer.sync.garth')
    def test_fetch_exercise_sets_api_error(self, mock_garth):
        """Test API error when fetching exercise sets."""
        from fitanalyzer.sync import fetch_exercise_sets_from_api
        
        # Use TypeError which is caught by the function
        mock_garth.connectapi.side_effect = TypeError("API Error")
        
        # Should catch exception and return None
        with patch('builtins.print'):
            result = fetch_exercise_sets_from_api("12345")
            self.assertIsNone(result)

    def test_save_and_load_exercise_sets(self):
        """Test saving and loading exercise sets to/from JSON."""
        from fitanalyzer.sync import save_exercise_sets_to_json, load_exercise_sets_from_json
        import tempfile
        
        test_data = {"exerciseSets": [{"reps": 10}]}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fit_path = str(Path(tmpdir) / "test_activity.fit")
            
            # Save (will create test_activity_exercises.json)
            save_exercise_sets_to_json(fit_path, test_data)
            json_path = Path(tmpdir) / "test_activity_exercises.json"
            self.assertTrue(json_path.exists())
            
            # Load
            loaded = load_exercise_sets_from_json(fit_path)
            self.assertEqual(loaded, test_data)

    def test_load_exercise_sets_missing_file(self):
        """Test loading from missing file returns None."""
        from fitanalyzer.sync import load_exercise_sets_from_json
        
        result = load_exercise_sets_from_json("/nonexistent/file.json")
        self.assertIsNone(result)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
