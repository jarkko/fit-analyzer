"""
Tests for sync configuration dataclasses.

These tests verify the contract for SyncConfig, AnalysisParams, and SyncMode dataclasses.
"""

import unittest

from fitanalyzer.sync_config import AnalysisParams, SyncConfig, SyncMode


class TestAnalysisParams(unittest.TestCase):
    """Test AnalysisParams dataclass."""

    def test_default_values(self):
        """Test that AnalysisParams has correct default values."""
        params = AnalysisParams()
        self.assertEqual(params.ftp, 300)  # DEFAULT_FTP
        self.assertEqual(params.hrrest, 50)  # DEFAULT_HR_REST
        self.assertEqual(params.hrmax, 190)  # DEFAULT_HR_MAX

    def test_custom_values(self):
        """Test AnalysisParams with custom values."""
        params = AnalysisParams(ftp=250, hrrest=60, hrmax=180)
        self.assertEqual(params.ftp, 250)
        self.assertEqual(params.hrrest, 60)
        self.assertEqual(params.hrmax, 180)

    def test_partial_override(self):
        """Test AnalysisParams with partial value override."""
        params = AnalysisParams(ftp=280)
        self.assertEqual(params.ftp, 280)
        self.assertEqual(params.hrrest, 50)  # Still default
        self.assertEqual(params.hrmax, 190)  # Still default


class TestSyncMode(unittest.TestCase):
    """Test SyncMode dataclass."""

    def test_default_mode(self):
        """Test that SyncMode defaults to all False."""
        mode = SyncMode()
        self.assertFalse(mode.analyze_only)
        self.assertFalse(mode.download_only)
        self.assertFalse(mode.force)

    def test_analyze_only_mode(self):
        """Test analyze-only mode."""
        mode = SyncMode(analyze_only=True)
        self.assertTrue(mode.analyze_only)
        self.assertFalse(mode.download_only)
        self.assertFalse(mode.force)

    def test_download_only_mode(self):
        """Test download-only mode."""
        mode = SyncMode(download_only=True)
        self.assertFalse(mode.analyze_only)
        self.assertTrue(mode.download_only)
        self.assertFalse(mode.force)

    def test_force_mode(self):
        """Test force mode."""
        mode = SyncMode(force=True)
        self.assertFalse(mode.analyze_only)
        self.assertFalse(mode.download_only)
        self.assertTrue(mode.force)

    def test_multiple_flags(self):
        """Test multiple flags set simultaneously."""
        mode = SyncMode(download_only=True, force=True)
        self.assertFalse(mode.analyze_only)
        self.assertTrue(mode.download_only)
        self.assertTrue(mode.force)


class TestSyncConfig(unittest.TestCase):
    """Test SyncConfig dataclass."""

    def test_default_config(self):
        """Test SyncConfig with all defaults."""
        config = SyncConfig()
        self.assertEqual(config.directory, ".")
        self.assertEqual(config.output_dir, "data")
        self.assertEqual(config.days, 30)  # DEFAULT_SYNC_DAYS
        self.assertIsNone(config.limit)

        # Nested objects should be auto-initialized
        self.assertIsNotNone(config.analysis)
        self.assertIsInstance(config.analysis, AnalysisParams)
        self.assertIsNotNone(config.mode)
        self.assertIsInstance(config.mode, SyncMode)

    def test_custom_directories(self):
        """Test SyncConfig with custom directories."""
        config = SyncConfig(directory="/tmp/fit", output_dir="/tmp/output")
        self.assertEqual(config.directory, "/tmp/fit")
        self.assertEqual(config.output_dir, "/tmp/output")

    def test_custom_sync_params(self):
        """Test SyncConfig with custom sync parameters."""
        config = SyncConfig(days=7, limit=10)
        self.assertEqual(config.days, 7)
        self.assertEqual(config.limit, 10)

    def test_custom_nested_analysis(self):
        """Test SyncConfig with custom AnalysisParams."""
        custom_analysis = AnalysisParams(ftp=280, hrrest=55)
        config = SyncConfig(analysis=custom_analysis)
        self.assertEqual(config.analysis.ftp, 280)
        self.assertEqual(config.analysis.hrrest, 55)
        self.assertEqual(config.analysis.hrmax, 190)  # Still default

    def test_custom_nested_mode(self):
        """Test SyncConfig with custom SyncMode."""
        custom_mode = SyncMode(force=True)
        config = SyncConfig(mode=custom_mode)
        self.assertTrue(config.mode.force)
        self.assertFalse(config.mode.analyze_only)

    def test_post_init_creates_nested_objects(self):
        """Test that __post_init__ creates nested objects when None."""
        config = SyncConfig()
        # Even though we pass None implicitly, post_init should create them
        self.assertIsNotNone(config.analysis)
        self.assertIsNotNone(config.mode)

    def test_full_custom_config(self):
        """Test fully customized SyncConfig."""
        analysis = AnalysisParams(ftp=250, hrrest=60, hrmax=180)
        mode = SyncMode(analyze_only=True, force=True)
        config = SyncConfig(
            directory="/custom/dir",
            output_dir="/custom/output",
            days=14,
            limit=50,
            analysis=analysis,
            mode=mode,
        )

        self.assertEqual(config.directory, "/custom/dir")
        self.assertEqual(config.output_dir, "/custom/output")
        self.assertEqual(config.days, 14)
        self.assertEqual(config.limit, 50)
        self.assertEqual(config.analysis.ftp, 250)
        self.assertTrue(config.mode.analyze_only)
        self.assertTrue(config.mode.force)


if __name__ == "__main__":
    unittest.main()
