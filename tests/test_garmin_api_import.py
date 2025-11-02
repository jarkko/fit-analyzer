"""Test garmin_api import fallback when garth is not available."""

import sys
import unittest
from unittest.mock import patch


class TestGarthImportFallback(unittest.TestCase):
    """Test import behavior when garth library is unavailable."""

    def test_import_without_garth(self) -> None:
        """Test that module imports successfully even without garth."""
        # Remove garth and related modules
        garth_module = sys.modules.pop("garth", None)
        garth_http_module = sys.modules.pop("garth.http", None)
        garth_utils_module = sys.modules.pop("fitanalyzer.garth_utils", None)
        garmin_api_module = sys.modules.pop("fitanalyzer.garmin_api", None)

        try:
            # Mock the import to raise ImportError
            with patch.dict(sys.modules, {"garth": None, "garth.http": None}):
                # Force a reload by removing from cache
                import importlib

                # Now import the module - it should handle ImportError gracefully
                import fitanalyzer.garmin_api as garmin_api

                # Verify fallback values
                self.assertIsNone(garmin_api.garth)
                self.assertEqual(garmin_api.GarthHTTPError, Exception)

        finally:
            # Restore modules
            if garth_module is not None:
                sys.modules["garth"] = garth_module
            if garth_http_module is not None:
                sys.modules["garth.http"] = garth_http_module
            if garth_utils_module is not None:
                sys.modules["fitanalyzer.garth_utils"] = garth_utils_module
            if garmin_api_module is not None:
                sys.modules["fitanalyzer.garmin_api"] = garmin_api_module


if __name__ == "__main__":
    unittest.main()
