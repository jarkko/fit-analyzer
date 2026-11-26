"""Test garmin_api import fallback when garth is not available."""

import sys
import unittest
from unittest.mock import patch


def _restore_package_attributes(names: tuple[str, ...]) -> None:
    """Point fitanalyzer.<name> back at the module in sys.modules (re-imports rebind it)."""
    import fitanalyzer

    for name in names:
        module = sys.modules.get(f"fitanalyzer.{name}")
        if module is not None:
            setattr(fitanalyzer, name, module)


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
            _restore_package_attributes(("garth_utils", "garmin_api"))

        # The package attribute must point at the module tests patch. On Python 3.10,
        # mock.patch("fitanalyzer.garmin_api.garth") resolves through the attribute,
        # so a stale one sends later patches to the throwaway garth-less module.
        import fitanalyzer

        self.assertIs(fitanalyzer.garmin_api, sys.modules["fitanalyzer.garmin_api"])
        self.assertIs(fitanalyzer.garth_utils, sys.modules["fitanalyzer.garth_utils"])


if __name__ == "__main__":
    unittest.main()
