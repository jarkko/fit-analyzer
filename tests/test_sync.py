"""
Unit tests for Garmin sync module.
Tests Garmin Connect integration, file management, and sync logic.
"""

import io
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
    """Test cases for Garmin Connect authentication."""

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
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

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
    def test_authenticate_expired_session(self, mock_path, mock_garth):
        """Test handling of expired session"""
        # Mock existing but expired token
        mock_token_path = Mock()
        mock_token_path.exists.return_value = True
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Resume fails (expired) - use a specific exception type we handle
        mock_garth.resume.side_effect = RuntimeError("Session expired")

        # Login should be attempted and will succeed
        authenticate_garmin(email="test@test.com", password="password")

        # Should attempt resume
        mock_garth.resume.assert_called_once()
        # Should call login after resume fails
        mock_garth.login.assert_called_once_with("test@test.com", "password")

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
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

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 0)
        self.assertEqual(updated_files, [])

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api", return_value=None)
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    def test_download_with_new_activities(
        self, mock_sync_garth, mock_download_garth, mock_api_garth, mock_fetch_api
    ):
        """Test downloading new activities"""
        # Create a dynamic date that's always within the 7-day range (yesterday)
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activity_date = yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Mock Garmin API
        mock_activities = [
            {
                "activityId": 20765123456,
                "activityName": "New Activity",
                "startTimeLocal": activity_date,
            }
        ]

        mock_sync_garth.connectapi.return_value = mock_activities
        mock_download_garth.download.return_value = b"fake_fit_data"

        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        self.assertEqual(new_count, 1)
        self.assertEqual(len(updated_files), 1)

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

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api", return_value=None)
    @patch("fitanalyzer.garmin_api.garth")
    @patch("fitanalyzer.activity_download.garth")
    @patch("fitanalyzer.sync.garth")
    def test_download_handles_timezone_formats(
        self, mock_sync_garth, mock_download_garth, mock_api_garth, mock_fetch_api
    ):
        """Test handling of different timezone formats in activity dates"""
        # Create dynamic dates that are always within range (3 days ago)
        base_date = datetime.now(timezone.utc) - timedelta(days=3)

        # Test with various timezone formats that Garmin might return
        mock_activities = [
            {
                "activityId": 20765100001,
                "activityName": "Activity with Z",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%SZ"),  # UTC with Z
            },
            {
                "activityId": 20765100002,
                "activityName": "Activity with offset",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),  # UTC with offset
            },
            {
                "activityId": 20765100003,
                "activityName": "Activity naive",
                "startTimeLocal": base_date.strftime("%Y-%m-%dT%H:%M:%S"),  # Naive (no timezone)
            },
        ]

        mock_sync_garth.connectapi.return_value = mock_activities
        mock_download_garth.download.return_value = b"fake_fit_data"

        # Should not raise "can't compare offset-naive and offset-aware datetimes"
        new_count, updated_files = download_new_activities(days=7, directory=self.test_dir)

        # All 3 should be processed without timezone errors
        self.assertEqual(new_count, 3)
        self.assertEqual(len(updated_files), 3)


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

    @patch("fitanalyzer.cli.main_with_args")
    @patch("fitanalyzer.cli.parse_arguments")
    def test_run_analysis_success(self, mock_parse, mock_main):
        """Test successful analysis execution"""
        mock_main.return_value = 0
        mock_parse.return_value = Mock()

        result = run_analysis(directory=self.test_dir, ftp=300)

        self.assertTrue(result)
        mock_main.assert_called_once()

    def test_run_analysis_no_fit_files(self):
        """Test analysis with no FIT files"""
        empty_dir = tempfile.mkdtemp()

        try:
            result = run_analysis(directory=empty_dir, ftp=300)

            self.assertFalse(result)
        finally:
            shutil.rmtree(empty_dir)

    @patch("fitanalyzer.cli.main_with_args")
    @patch("fitanalyzer.cli.parse_arguments")
    def test_run_analysis_script_error(self, mock_parse, mock_main):
        """Test handling of script errors"""
        mock_main.return_value = 1
        mock_parse.return_value = Mock()

        result = run_analysis(directory=self.test_dir, ftp=300)

        self.assertFalse(result)

    @patch("fitanalyzer.cli.main_with_args")
    @patch("fitanalyzer.cli.parse_arguments")
    def test_run_analysis_with_output_dir(self, mock_parse, mock_main):
        """Test analysis with custom output directory"""
        mock_main.return_value = 0
        mock_parsed_args = Mock()
        mock_parse.return_value = mock_parsed_args

        output_dir = tempfile.mkdtemp()
        try:
            result = run_analysis(directory=self.test_dir, output_dir=output_dir, ftp=300)

            self.assertTrue(result)
            mock_main.assert_called_once_with(mock_parsed_args)

            # Check that parse_arguments was called with correct arguments
            call_args = mock_parse.call_args[0][0]
            self.assertIn("--output-dir", call_args)
            self.assertIn(output_dir, call_args)
        finally:
            shutil.rmtree(output_dir)

    @patch("fitanalyzer.cli.main_with_args")
    @patch("fitanalyzer.cli.parse_arguments")
    def test_run_analysis_kwargs_parameters(self, mock_parse, mock_main):
        """Test analysis with parameters passed via kwargs"""
        mock_main.return_value = 0
        mock_parsed_args = Mock()
        mock_parse.return_value = mock_parsed_args

        result = run_analysis(
            directory=self.test_dir,
            output_dir="custom_output",
            ftp=250,
            hrrest=50,
            hrmax=180,
            multisport=False,
        )

        self.assertTrue(result)
        call_args = mock_parse.call_args[0][0]

        # Check all parameters are passed correctly
        self.assertIn("--ftp", call_args)
        self.assertIn("250", call_args)
        self.assertIn("--hrrest", call_args)
        self.assertIn("50", call_args)
        self.assertIn("--hrmax", call_args)
        self.assertIn("180", call_args)
        self.assertIn("--output-dir", call_args)
        self.assertIn("custom_output", call_args)
        self.assertNotIn("--multisport", call_args)  # Should be False

    def test_run_analysis_single_file(self):
        """Test analysis with single FIT file"""
        # Create a single FIT file
        fit_file = Path(self.test_dir) / "single_ACTIVITY.fit"
        fit_file.touch()

        with patch("fitanalyzer.cli.main_with_args") as mock_main, patch(
            "fitanalyzer.cli.parse_arguments"
        ) as mock_parse:
            mock_main.return_value = 0
            mock_parsed_args = Mock()
            mock_parse.return_value = mock_parsed_args

            result = run_analysis(directory=str(fit_file))

            self.assertTrue(result)
            mock_main.assert_called_once_with(mock_parsed_args)

    def test_run_analysis_default_parameters(self):
        """Test analysis with default parameters"""
        with patch("fitanalyzer.cli.main_with_args") as mock_main, patch(
            "fitanalyzer.cli.parse_arguments"
        ) as mock_parse:
            mock_main.return_value = 0
            mock_parsed_args = Mock()
            mock_parse.return_value = mock_parsed_args

            result = run_analysis(directory=self.test_dir)

            self.assertTrue(result)
            call_args = mock_parse.call_args[0][0]

            # Check default values are used
            self.assertIn("--ftp", call_args)
            self.assertIn("300", call_args)  # DEFAULT_FTP
            self.assertIn("--output-dir", call_args)
            self.assertIn("data", call_args)  # default output_dir


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

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", True)
    def test_check_garth_available(self):
        """Test when garth is already available."""
        from fitanalyzer.sync import check_and_install_garth

        result = check_and_install_garth()
        self.assertTrue(result)

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="n")
    @patch("builtins.print")
    def test_check_garth_not_available_decline_install(self, mock_print, mock_input):
        """Test when garth not available and user declines install."""
        from fitanalyzer.sync import check_and_install_garth

        result = check_and_install_garth()
        self.assertFalse(result)

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="y")
    @patch("subprocess.check_call")
    @patch("builtins.print")
    def test_check_garth_install_success(self, mock_print, mock_subprocess, mock_input):
        """Test successful garth installation."""
        from fitanalyzer.sync import check_and_install_garth

        result = check_and_install_garth()
        # Returns False because need to restart
        self.assertFalse(result)
        mock_subprocess.assert_called_once()

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="y")
    @patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip"))
    @patch("builtins.print")
    def test_check_garth_install_failure(self, mock_print, mock_subprocess, mock_input):
        """Test failed garth installation."""
        from fitanalyzer.sync import check_and_install_garth

        result = check_and_install_garth()
        self.assertFalse(result)


class TestExerciseSetsAPI(unittest.TestCase):
    """Test exercise sets API functions."""

    @patch("fitanalyzer.garmin_api.garth")
    def test_fetch_exercise_sets_from_api(self, mock_garth):
        """Test fetching exercise sets from Garmin API."""
        from fitanalyzer.garmin_api import fetch_exercise_sets_from_api

        mock_garth.connectapi.return_value = {
            "exerciseSets": [{"category": 1, "exerciseName": "BENCH_PRESS", "reps": 10}]
        }

        result = fetch_exercise_sets_from_api("12345")
        self.assertIsNotNone(result)
        self.assertIn("exerciseSets", result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_fetch_exercise_sets_api_error(self, mock_garth):
        """Test API error when fetching exercise sets."""
        from fitanalyzer.garmin_api import fetch_exercise_sets_from_api

        # Use TypeError which is caught by the function
        mock_garth.connectapi.side_effect = TypeError("API Error")

        # Should catch exception and return None
        with patch("builtins.print"):
            result = fetch_exercise_sets_from_api("12345")
            self.assertIsNone(result)

    def test_save_and_load_exercise_sets(self):
        """Test saving and loading exercise sets to/from JSON."""
        import tempfile

        from fitanalyzer.sync import load_exercise_sets_from_json, save_exercise_sets_to_json

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


class TestAuthenticationEdgeCases(unittest.TestCase):
    """Test authentication edge cases and error paths"""

    @patch("fitanalyzer.garmin_auth.garth", None)
    def test_authenticate_garth_not_available(self):
        """Test authentication when garth library is not installed (lines 124)"""
        with self.assertRaises(ImportError) as context:
            authenticate_garmin()
        self.assertIn("garth library not available", str(context.exception))

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
    @patch("fitanalyzer.garmin_auth.os.getenv")
    @patch("fitanalyzer.garmin_auth.input")
    def test_authenticate_with_env_email_no_password(
        self, mock_input, mock_getenv, mock_path, mock_garth
    ):
        """Test authentication with email from environment (lines 142-144)"""
        mock_token_path = Mock()
        mock_token_path.exists.return_value = False
        mock_token_path.parent = Mock()
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Email from env, password will be prompted
        def getenv_side_effect(key):
            if key == "GARMIN_EMAIL":
                return "env@example.com"
            return None

        mock_getenv.side_effect = getenv_side_effect
        mock_input.return_value = "password123"  # getpass returns this

        with patch("fitanalyzer.garmin_auth.getpass.getpass", return_value="password123"):
            authenticate_garmin()

        mock_garth.login.assert_called_once_with("env@example.com", "password123")

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
    @patch("fitanalyzer.garmin_auth.os.getenv")
    @patch("fitanalyzer.garmin_auth.input")
    def test_authenticate_prompt_email_env_password(
        self, mock_input, mock_getenv, mock_path, mock_garth
    ):
        """Test authentication with password from environment (lines 147-149)"""
        mock_token_path = Mock()
        mock_token_path.exists.return_value = False
        mock_token_path.parent = Mock()
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Password from env, email will be prompted
        def getenv_side_effect(key):
            if key == "GARMIN_PASSWORD":
                return "env_password"
            return None

        mock_getenv.side_effect = getenv_side_effect
        mock_input.return_value = "prompted@example.com"

        authenticate_garmin()

        mock_garth.login.assert_called_once_with("prompted@example.com", "env_password")

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
    def test_authenticate_login_failure_with_mfa_hint(self, mock_path, mock_garth):
        """Test authentication failure with MFA error (lines 161-167)"""
        mock_token_path = Mock()
        mock_token_path.exists.return_value = False
        mock_token_path.parent = Mock()
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Login fails with MFA error
        mock_garth.login.side_effect = RuntimeError("MFA verification required")

        result = authenticate_garmin(email="test@example.com", password="testpass")

        self.assertFalse(result)

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth.Path")
    def test_authenticate_login_generic_failure(self, mock_path, mock_garth):
        """Test authentication failure without MFA (lines 161-167)"""
        mock_token_path = Mock()
        mock_token_path.exists.return_value = False
        mock_token_path.parent = Mock()
        mock_path.return_value.expanduser.return_value = mock_token_path

        # Login fails with generic error
        mock_garth.login.side_effect = ValueError("Invalid credentials")

        result = authenticate_garmin(email="test@example.com", password="testpass")

        self.assertFalse(result)


class TestDownloadEdgeCases(unittest.TestCase):
    """Test download and file handling edge cases"""

    def setUp(self):
        """Set up test directory"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_extract_fit_from_zip(self):
        """Test extracting FIT file from ZIP (lines 227-233)"""
        import zipfile

        from fitanalyzer.activity_download import _extract_fit_from_zip

        # Create a mock ZIP with a .fit file
        fit_content = b".FIT\x00test_data"
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("test_activity.fit", fit_content)

        zip_data = zip_buffer.getvalue()

        # Extract FIT from ZIP
        result = _extract_fit_from_zip(zip_data)
        self.assertEqual(result, fit_content)

    def test_extract_fit_not_zip(self):
        """Test extracting when input is already a FIT file (lines 227-233)"""
        from fitanalyzer.activity_download import _extract_fit_from_zip

        # FIT file directly (not zipped)
        fit_content = b".FIT\x00test_data"

        result = _extract_fit_from_zip(fit_content)
        self.assertEqual(result, fit_content)

    def testshould_download_activity_new(self):
        """Test download decision for new activity (lines 265-272)"""
        from fitanalyzer.activity_download import should_download_activity

        activity = {"activityId": 12345}
        existing = {}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertTrue(should_dl)
        self.assertFalse(is_update)
        self.assertFalse(check_api)

    def testshould_download_activity_no_update_timestamp(self):
        """Test download decision when no update timestamp (lines 277-284)"""
        import time

        from fitanalyzer.activity_download import should_download_activity

        activity = {"activityId": 12345}  # No updateDate or lastModified
        existing = {"12345": time.time()}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertFalse(should_dl)
        self.assertFalse(is_update)
        self.assertTrue(check_api)  # Should still check API updates

    def testshould_download_activity_updated(self):
        """Test download decision for updated activity (lines 277-284)"""
        import time

        from fitanalyzer.activity_download import should_download_activity

        current_time = time.time()
        older_time = current_time - 3600  # 1 hour ago

        activity = {
            "activityId": 12345,
            "updateDate": int(current_time * 1000),  # Now (in ms)
        }
        existing = {"12345": older_time}

        should_dl, is_update, check_api = should_download_activity(activity, existing)

        self.assertTrue(should_dl)
        self.assertTrue(is_update)
        self.assertFalse(check_api)


class TestOutputDirFunctionality(unittest.TestCase):
    """Test --output-dir argument and path handling"""

    def setUp(self):
        """Set up test directories"""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create test FIT files
        for i in range(2):
            (Path(self.test_dir) / f"test_activity_{i}_ACTIVITY.fit").touch()

    def tearDown(self):
        """Clean up test directories"""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)

    def test_argument_parser_output_dir(self):
        """Test that --output-dir argument is parsed correctly"""
        from fitanalyzer.sync import main

        # Test with minimal patches to avoid complex mocking
        with patch(
            "sys.argv",
            [
                "sync.py",
                "--analyze-only",
                "--directory",
                self.test_dir,
                "--output-dir",
                self.output_dir,
            ],
        ), patch("fitanalyzer.sync.run_analysis") as mock_analysis, patch(
            "builtins.print"
        ):  # Suppress output

            mock_analysis.return_value = True
            result = main()

            # Check that run_analysis was called with correct output_dir
            mock_analysis.assert_called_once()
            call_kwargs = mock_analysis.call_args[1]
            self.assertEqual(call_kwargs["output_dir"], self.output_dir)
            self.assertEqual(result, 0)

    def test_argument_parser_default_output_dir(self):
        """Test that default output directory is 'data'"""
        from fitanalyzer.sync import main

        with patch("sys.argv", ["sync.py", "--analyze-only", "--directory", self.test_dir]), patch(
            "fitanalyzer.sync.run_analysis"
        ) as mock_analysis, patch(
            "builtins.print"
        ):  # Suppress output

            mock_analysis.return_value = True
            result = main()

            # Check that default output_dir is used
            mock_analysis.assert_called_once()
            call_kwargs = mock_analysis.call_args[1]
            self.assertEqual(call_kwargs["output_dir"], "data")
            self.assertEqual(result, 0)

    def test_run_analysis_output_dir_argument_passing(self):
        """Test that output_dir is correctly passed to parser arguments"""
        with patch("fitanalyzer.cli.main_with_args") as mock_main, patch(
            "fitanalyzer.cli.parse_arguments"
        ) as mock_parse:

            mock_main.return_value = 0
            mock_parsed_args = Mock()
            mock_parse.return_value = mock_parsed_args

            custom_output = "/custom/output/path"
            result = run_analysis(directory=self.test_dir, output_dir=custom_output)

            self.assertTrue(result)

            # Verify that --output-dir argument was passed to parser
            call_args = mock_parse.call_args[0][0]
            self.assertIn("--output-dir", call_args)
            output_dir_index = call_args.index("--output-dir")
            self.assertEqual(call_args[output_dir_index + 1], custom_output)

    def test_path_handling_relative_vs_absolute(self):
        """Test that paths are handled correctly (relative to caller)"""
        # This test verifies that Path objects work as expected
        relative_path = "./relative/output"
        absolute_path = "/absolute/output"

        # Test relative path resolution
        rel_path_obj = Path(relative_path)
        abs_path_obj = Path(absolute_path)

        # Relative paths should resolve relative to current working directory
        self.assertFalse(rel_path_obj.is_absolute())
        self.assertTrue(abs_path_obj.is_absolute())

        # Both should work with expanduser() - relative path gets normalized
        rel_expanded = rel_path_obj.expanduser()
        abs_expanded = abs_path_obj.expanduser()

        # Path normalization removes the ./ prefix
        self.assertEqual(str(rel_expanded), "relative/output")
        self.assertEqual(str(abs_expanded), absolute_path)

    def test_single_file_handling_with_output_dir(self):
        """Test that single file analysis works with custom output directory"""
        # Create a single FIT file
        single_file = Path(self.test_dir) / "single_test_ACTIVITY.fit"
        single_file.touch()

        with patch("fitanalyzer.cli.main_with_args") as mock_main, patch(
            "fitanalyzer.cli.parse_arguments"
        ) as mock_parse:

            mock_main.return_value = 0
            mock_parsed_args = Mock()
            mock_parse.return_value = mock_parsed_args

            result = run_analysis(directory=str(single_file), output_dir=self.output_dir)

            self.assertTrue(result)

            # Verify the file was recognized and arguments include output_dir
            call_args = mock_parse.call_args[0][0]
            self.assertIn("--output-dir", call_args)
            self.assertIn(self.output_dir, call_args)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
