"""
Integration tests for end-to-end CSV generation with proper column validation.
These tests ensure new features work correctly without caching issues.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fitanalyzer.parser import summarize_fit_sessions, AnalysisConfig
from fitanalyzer.dev_utils import force_reload_fitanalyzer_modules


class TestCSVIntegration(unittest.TestCase):
    """Test complete CSV generation pipeline"""

    def setUp(self):
        """Set up test environment with temp directory"""
        self.test_dir = tempfile.mkdtemp()
        self.config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="UTC")

        # Force fresh modules to avoid caching issues
        force_reload_fitanalyzer_modules()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_csv_has_required_columns(self):
        """Test that generated CSV has all expected columns including new ones"""
        # Expected columns after our speed/cadence/distance feature
        expected_columns = [
            # Original columns
            "file",
            "sport",
            "sub_sport",
            "date",
            "start_time",
            "end_time",
            "duration_min",
            "avg_hr",
            "max_hr",
            "avg_power_w",
            "max_power_w",
            "np_w",
            "IF",
            "TSS",
            "TRIMP",
            # New speed/cadence/distance columns
            "avg_speed_mps",
            "max_speed_mps",
            "avg_speed_kph",
            "max_speed_kph",
            "avg_cadence",
            "max_cadence",
            "total_distance_m",
            "total_distance_km",
        ]

        # Process a real cycling file
        sessions, _ = summarize_fit_sessions("data/samples/20684859222_ACTIVITY.fit", self.config)

        # Verify session data has all columns
        self.assertGreater(len(sessions), 0, "Should have at least one session")
        session = sessions[0]

        for col in expected_columns:
            self.assertIn(col, session, f"Missing column: {col}")

        # Create DataFrame like the real CSV generation does
        df = pd.DataFrame(sessions)

        # Remove internal columns
        columns_to_remove = [col for col in df.columns if col.startswith("_")]
        df_clean = df.drop(columns=columns_to_remove)

        # Verify CSV-ready DataFrame has all expected columns
        actual_columns = list(df_clean.columns)
        for col in expected_columns:
            self.assertIn(col, actual_columns, f"CSV missing column: {col}")

        # Verify we have exactly the expected number of columns (no extras)
        self.assertEqual(
            len(df_clean.columns),
            len(expected_columns),
            f"Expected {len(expected_columns)} columns, got {len(df_clean.columns)}",
        )

    def test_cycling_data_has_speed_cadence_values(self):
        """Test that cycling files produce actual speed/cadence/distance values"""
        sessions, _ = summarize_fit_sessions("data/samples/20684859222_ACTIVITY.fit", self.config)
        session = sessions[0]

        # Cycling file should have real values (not empty strings)
        self.assertNotEqual(session["avg_speed_mps"], "", "Speed should have value")
        self.assertNotEqual(session["avg_cadence"], "", "Cadence should have value")
        self.assertNotEqual(session["total_distance_m"], "", "Distance should have value")

        # Values should be reasonable
        self.assertGreater(session["avg_speed_mps"], 0, "Speed should be positive")
        self.assertGreater(session["avg_cadence"], 30, "Cadence should be reasonable")
        self.assertGreater(session["total_distance_m"], 100, "Distance should be reasonable")

    def test_csv_column_order_stability(self):
        """Test that CSV columns are in a stable, predictable order"""
        sessions, _ = summarize_fit_sessions("data/samples/20684859222_ACTIVITY.fit", self.config)
        df = pd.DataFrame(sessions)

        # Remove internal columns
        columns_to_remove = [col for col in df.columns if col.startswith("_")]
        df_clean = df.drop(columns=columns_to_remove)

        columns = list(df_clean.columns)

        # Verify basic structure: metadata first, then metrics, then new metrics
        self.assertEqual(columns[0], "file")
        self.assertEqual(columns[1], "sport")
        self.assertEqual(columns[2], "sub_sport")

        # Original metrics should be in expected positions
        self.assertIn("avg_hr", columns[:15])
        self.assertIn("TRIMP", columns[:15])

        # New metrics should be at the end
        self.assertIn("avg_speed_mps", columns[15:])
        self.assertIn("avg_cadence", columns[15:])
        self.assertIn("total_distance_m", columns[15:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
