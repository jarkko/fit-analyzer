"""
Tests for programmatic sync_activities() function.

TDD tests to ensure sync_activities() works correctly and that CLI uses it.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fitanalyzer.sync import AnalysisParams, SyncConfig, SyncMode, sync_activities


class TestSyncActivitiesProgrammatic(unittest.TestCase):
    """Test the high-level sync_activities() function."""

    def setUp(self):
        """Create isolated temporary directories for each test."""
        self.test_dir = tempfile.mkdtemp(prefix="test_sync_")
        self.output_dir = tempfile.mkdtemp(prefix="test_output_")

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_returns_error_if_garth_not_available(self, mock_check):
        """Test that function returns error dict if garth unavailable."""
        mock_check.return_value = False

        result = sync_activities()

        self.assertFalse(result["success"])
        self.assertEqual(result["new_activities"], 0)
        self.assertIn("garth", result["error"])

    @patch("fitanalyzer.sync.run_analysis")
    @patch("fitanalyzer.sync.download_new_activities")
    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_successful_sync_returns_correct_dict(
        self, mock_check, mock_auth, mock_download, mock_analysis
    ):
        """Test successful sync returns proper result dict."""
        mock_check.return_value = True
        mock_auth.return_value = True
        mock_download.return_value = (5, [])
        mock_analysis.return_value = True

        result = sync_activities(directory=self.test_dir, output_dir=self.output_dir, days=7)

        self.assertTrue(result["success"])
        self.assertEqual(result["new_activities"], 5)
        self.assertIn("csv_path", result)
        self.assertIn("strength_csv_path", result)
        self.assertIn("workout_summary_from_fit.csv", result["csv_path"])

    @patch("fitanalyzer.sync.run_analysis")
    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_analyze_only_skips_download(self, mock_check, mock_auth, mock_analysis):
        """Test analyze_only=True skips authentication and download."""
        mock_check.return_value = True
        mock_analysis.return_value = True

        config = SyncConfig(
            directory=self.test_dir,
            output_dir=self.output_dir,
            mode=SyncMode(analyze_only=True),
        )
        result = sync_activities(config)

        # Should not authenticate if analyze_only
        mock_auth.assert_not_called()
        # But should run analysis
        mock_analysis.assert_called_once()
        self.assertTrue(result["success"])

    @patch("fitanalyzer.sync.download_new_activities")
    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_download_only_skips_analysis(self, mock_check, mock_auth, mock_download):
        """Test download_only=True skips analysis."""
        mock_check.return_value = True
        mock_auth.return_value = True
        mock_download.return_value = (3, [])

        config = SyncConfig(
            directory=self.test_dir,
            output_dir=self.output_dir,
            mode=SyncMode(download_only=True),
        )
        result = sync_activities(config)

        # Should download
        mock_download.assert_called_once()
        # But analysis never called (mock not created)
        self.assertTrue(result["success"])
        self.assertEqual(result["new_activities"], 3)

    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_auth_failure_returns_error(self, mock_check, mock_auth):
        """Test authentication failure returns error dict."""
        mock_check.return_value = True
        mock_auth.return_value = False

        result = sync_activities(directory=self.test_dir)

        self.assertFalse(result["success"])
        self.assertIn("Authentication", result["error"])

    @patch("fitanalyzer.sync.run_analysis")
    @patch("fitanalyzer.sync.download_new_activities")
    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_incremental_by_default(self, mock_check, mock_auth, mock_download, mock_analysis):
        """Test that sync is incremental by default (force=False)."""
        mock_check.return_value = True
        mock_auth.return_value = True
        mock_download.return_value = (0, [])
        mock_analysis.return_value = True

        sync_activities(directory=self.test_dir, output_dir=self.output_dir)

        # Verify download was called with force=False
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        self.assertEqual(call_kwargs["force"], False)

    @patch("fitanalyzer.sync.run_analysis")
    @patch("fitanalyzer.sync.download_new_activities")
    @patch("fitanalyzer.sync.authenticate_garmin")
    @patch("fitanalyzer.sync.check_and_install_garth")
    def test_passes_updated_files_to_analysis(
        self, mock_check, mock_auth, mock_download, mock_analysis
    ):
        """Test that updated_files from download are passed to analysis."""
        mock_check.return_value = True
        mock_auth.return_value = True
        updated_files = ["/path/to/file1.fit", "/path/to/file2.fit"]
        mock_download.return_value = (2, updated_files)
        mock_analysis.return_value = True

        sync_activities(directory=self.test_dir, output_dir=self.output_dir)

        # Verify analysis was called with updated_files
        mock_analysis.assert_called_once()
        call_kwargs = mock_analysis.call_args[1]
        self.assertEqual(call_kwargs["updated_files"], updated_files)

    def test_keyword_only_arguments(self):
        """Test that config must be positional but credentials keyword-only."""
        # This should raise TypeError if email/password are positional
        with self.assertRaises(TypeError):
            sync_activities(None, "email@test.com", "password")  # type: ignore


if __name__ == "__main__":
    unittest.main()
