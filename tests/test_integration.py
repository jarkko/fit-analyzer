"""
Integration tests for the FIT analyzer library.
Tests end-to-end scenarios using the library API with deterministic test fixtures.

Test Fixtures:
- 20548472357_ACTIVITY.fit: Volleyball activity (134.6 min, HR 118.1 avg)
- 20744294788_ACTIVITY.fit: Multisport (2 sessions: cycling + strength)
- 20747700969_ACTIVITY.fit: Cycling activity (30.0 min, HR 112.7 avg)
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fitanalyzer import summarize_fit_original, summarize_fit_sessions


class TestLibraryIntegration(unittest.TestCase):
    """Integration tests for the fitanalyzer library API"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.fixtures_dir = Path(__file__).parent / "fixtures"

        # Standard test parameters
        self.ftp = 300
        self.hr_rest = 50
        self.hr_max = 190

    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_parse_volleyball_activity(self):
        """Test parsing a volleyball activity with exact value assertions"""
        fit_file = self.fixtures_dir / "20548472357_ACTIVITY.fit"
        self.assertTrue(fit_file.exists(), f"Test fixture not found: {fit_file}")

        summary, sets = summarize_fit_original(
            str(fit_file), ftp=self.ftp, hr_rest=self.hr_rest, hr_max=self.hr_max
        )

        # Verify it parsed successfully
        self.assertIsNotNone(summary)

        # Test exact values
        self.assertEqual(summary["sport"], "volleyball")
        self.assertEqual(summary["sub_sport"], "generic")
        self.assertEqual(summary["date"], "2025-09-30")
        self.assertAlmostEqual(summary["duration_min"], 134.6, places=1)
        self.assertAlmostEqual(summary["avg_hr"], 118.1, places=1)
        self.assertEqual(summary["max_hr"], 163)
        self.assertAlmostEqual(summary["TRIMP"], 112.9, places=1)

        # No power data for volleyball
        self.assertEqual(summary["avg_power_w"], "")
        self.assertEqual(summary["np_w"], "")
        self.assertEqual(summary["TSS"], "")

        print(f"\n✅ Volleyball activity: {summary['duration_min']} min, HR {summary['avg_hr']}")

    def test_parse_multiple_fit_files(self):
        """Test parsing multiple FIT files with exact counts"""
        fixture_files = [
            self.fixtures_dir / "20548472357_ACTIVITY.fit",  # Volleyball
            self.fixtures_dir / "20747700969_ACTIVITY.fit",  # Cycling
        ]

        for fit_file in fixture_files:
            self.assertTrue(fit_file.exists(), f"Test fixture not found: {fit_file}")

        results = []
        for fit_file in fixture_files:
            summary, _ = summarize_fit_original(
                str(fit_file), ftp=self.ftp, hr_rest=self.hr_rest, hr_max=self.hr_max
            )
            self.assertIsNotNone(summary, f"Failed to parse {fit_file.name}")
            results.append(summary)

        # Should have exactly 2 results
        self.assertEqual(len(results), 2)

        # Verify first is volleyball
        self.assertEqual(results[0]["sport"], "volleyball")
        self.assertAlmostEqual(results[0]["duration_min"], 134.6, places=1)

        # Verify second is cycling
        self.assertEqual(results[1]["sport"], "cycling")
        self.assertAlmostEqual(results[1]["duration_min"], 30.0, places=1)

        print(f"\n✅ Parsed {len(results)} activities successfully")

    def test_generate_csv_output(self):
        """Test generating CSV output from parsed activities"""
        fixture_files = [
            self.fixtures_dir / "20548472357_ACTIVITY.fit",  # Volleyball
            self.fixtures_dir / "20747700969_ACTIVITY.fit",  # Cycling
        ]

        summaries = []
        for fit_file in fixture_files:
            summary, _ = summarize_fit_original(
                str(fit_file), ftp=self.ftp, hr_rest=self.hr_rest, hr_max=self.hr_max
            )
            self.assertIsNotNone(summary)
            summaries.append(summary)

        # Should have exactly 2 summaries
        self.assertEqual(len(summaries), 2)

        # Create DataFrame and save as CSV
        df = pd.DataFrame(summaries)
        csv_path = self.test_dir / "test_summary.csv"
        df.to_csv(csv_path, index=False)

        # Verify CSV was created
        self.assertTrue(csv_path.exists())

        # Verify CSV can be read back
        df_read = pd.read_csv(csv_path)
        self.assertEqual(len(df_read), 2)
        self.assertIn("sport", df_read.columns)
        self.assertIn("duration_min", df_read.columns)

        # Verify specific values in CSV
        self.assertEqual(df_read.iloc[0]["sport"], "volleyball")
        self.assertEqual(df_read.iloc[1]["sport"], "cycling")

        print(f"\n✅ Generated CSV with {len(df_read)} activities")

    def test_multisport_activity_parsing(self):
        """Test parsing multisport activity with exact session assertions"""
        fit_file = self.fixtures_dir / "20744294788_ACTIVITY.fit"
        self.assertTrue(fit_file.exists(), f"Test fixture not found: {fit_file}")

        sessions, _ = summarize_fit_sessions(
            str(fit_file), ftp=self.ftp, hr_rest=self.hr_rest, hr_max=self.hr_max
        )

        # Should have exactly 2 sessions
        self.assertEqual(len(sessions), 2)

        # Verify first session (cycling)
        self.assertEqual(sessions[0]["sport"], "cycling")
        self.assertEqual(sessions[0]["sub_sport"], "indoor_cycling")
        self.assertAlmostEqual(sessions[0]["duration_min"], 10.0, places=1)
        self.assertAlmostEqual(sessions[0]["avg_hr"], 114.7, places=1)

        # Verify second session (strength training)
        self.assertEqual(sessions[1]["sport"], "training")
        self.assertEqual(sessions[1]["sub_sport"], "strength_training")
        self.assertAlmostEqual(sessions[1]["duration_min"], 64.5, places=1)
        self.assertAlmostEqual(sessions[1]["avg_hr"], 114.4, places=1)

        print(f"\n✅ Multisport activity: {len(sessions)} sessions")
        print(f"   Session 1: {sessions[0]['sport']} ({sessions[0]['duration_min']} min)")
        print(f"   Session 2: {sessions[1]['sport']} ({sessions[1]['duration_min']} min)")

    def test_metric_calculations(self):
        """Test that metrics are calculated correctly with exact values"""
        fit_file = self.fixtures_dir / "20548472357_ACTIVITY.fit"
        self.assertTrue(fit_file.exists(), f"Test fixture not found: {fit_file}")

        summary, _ = summarize_fit_original(
            str(fit_file), ftp=self.ftp, hr_rest=self.hr_rest, hr_max=self.hr_max
        )

        self.assertIsNotNone(summary)

        # Verify duration is positive and matches expected
        self.assertAlmostEqual(summary["duration_min"], 134.6, places=1)

        # Verify HR metrics are within valid physiological ranges
        self.assertGreater(summary["avg_hr"], 30)
        self.assertLess(summary["avg_hr"], 220)
        self.assertAlmostEqual(summary["avg_hr"], 118.1, places=1)

        # Verify max HR
        self.assertEqual(summary["max_hr"], 163)
        self.assertLessEqual(summary["max_hr"], self.hr_max)

        # Verify TRIMP calculation
        self.assertGreater(summary["TRIMP"], 0)
        self.assertAlmostEqual(summary["TRIMP"], 112.9, places=1)

        print(f"\n✅ Validated metrics:")
        print(f"   Duration: {summary['duration_min']} min")
        print(f"   Avg HR: {summary['avg_hr']} bpm")
        print(f"   Max HR: {summary['max_hr']} bpm")
        print(f"   TRIMP: {summary['TRIMP']}")


if __name__ == "__main__":
    unittest.main()
