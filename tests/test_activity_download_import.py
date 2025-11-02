"""Tests for activity_download module import fallback."""

import sys
import unittest
from importlib import reload


class TestActivityDownloadImport(unittest.TestCase):
    """Test import fallback for activity_download module."""

    def test_module_imports_successfully(self) -> None:
        """Test that module can be imported and constants are defined."""
        from fitanalyzer import activity_download

        # Module should define these constants
        self.assertTrue(hasattr(activity_download, "GARTH_AVAILABLE"))
        self.assertTrue(hasattr(activity_download, "GarthHTTPError"))

        # In normal environment, garth is available
        # We test the fallback behavior through GARTH_AVAILABLE flag in other tests
        self.assertIsInstance(activity_download.GARTH_AVAILABLE, bool)

    def test_import_fallback_when_garth_unavailable(self) -> None:
        """Test that module handles missing garth gracefully."""
        # Save original modules
        original_garth = sys.modules.get("garth")
        original_garth_utils = sys.modules.get("fitanalyzer.garth_utils")
        original_activity_download = sys.modules.get("fitanalyzer.activity_download")

        try:
            # Block garth import
            sys.modules["garth"] = None  # type: ignore[assignment]

            # Remove garth_utils and activity_download to force reimport
            if "fitanalyzer.garth_utils" in sys.modules:
                del sys.modules["fitanalyzer.garth_utils"]
            if "fitanalyzer.activity_download" in sys.modules:
                del sys.modules["fitanalyzer.activity_download"]

            # Import should not fail
            import fitanalyzer.activity_download

            # Verify fallback behavior
            self.assertFalse(fitanalyzer.activity_download.GARTH_AVAILABLE)
            self.assertIsNone(fitanalyzer.activity_download.garth)
            self.assertEqual(fitanalyzer.activity_download.GarthHTTPError, Exception)

        finally:
            # Restore original state
            if original_garth is not None:
                sys.modules["garth"] = original_garth
            elif "garth" in sys.modules:
                del sys.modules["garth"]

            if original_garth_utils is not None:
                sys.modules["fitanalyzer.garth_utils"] = original_garth_utils
            elif "fitanalyzer.garth_utils" in sys.modules:
                del sys.modules["fitanalyzer.garth_utils"]

            if original_activity_download is not None:
                sys.modules["fitanalyzer.activity_download"] = original_activity_download
            elif "fitanalyzer.activity_download" in sys.modules:
                del sys.modules["fitanalyzer.activity_download"]

            # Reload to restore normal state
            if original_garth_utils is not None:
                reload(original_garth_utils)
            if original_activity_download is not None:
                reload(original_activity_download)


if __name__ == "__main__":
    unittest.main()
