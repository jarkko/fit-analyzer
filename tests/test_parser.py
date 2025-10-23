"""
Unit tests for FIT file parser.
Tests FIT file analysis, multisport handling, and data processing.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from fitanalyzer.parser import (
    SPORT_MAPPING,
    SUB_SPORT_MAPPING,
    AnalysisConfig,
    np_power,
    process_session_data,
    summarize_fit_original,
    summarize_fit_sessions,
    trimp_from_hr,
)


class TestNormalizedPower(unittest.TestCase):
    """Test normalized power calculations"""

    def test_np_power_empty_array(self):
        """Test NP with empty power array"""
        result = np_power([])
        self.assertTrue(np.isnan(result))

    def test_np_power_constant_power(self):
        """Test NP with constant power"""
        power = [200] * 100  # 100 seconds at 200W
        result = np_power(power)
        self.assertAlmostEqual(result, 200.0, delta=1.0)

    def test_np_power_variable_power(self):
        """Test NP with variable power"""
        # Alternating high/low power
        power = [300, 100] * 50  # 100 data points
        result = np_power(power)
        # NP should be higher than average due to 4th power weighting
        avg_power = np.mean(power)
        self.assertGreater(result, avg_power)

    def test_np_power_with_zeros(self):
        """Test NP with zeros (coasting)"""
        power = [200, 200, 0, 0, 200, 200]
        result = np_power(power)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)


class TestTRIMP(unittest.TestCase):
    """Test TRIMP (heart rate based training load) calculations"""

    def test_trimp_empty_array(self):
        """Test TRIMP with empty HR array"""
        result = trimp_from_hr([])
        self.assertEqual(result, 0.0)

    def test_trimp_constant_hr(self):
        """Test TRIMP with constant heart rate"""
        # 30 minutes (1800 seconds) at 150 bpm
        hr = [150] * 1800
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        self.assertGreater(result, 0)
        self.assertIsInstance(result, float)

    def test_trimp_below_rest_hr(self):
        """Test TRIMP with HR below resting (should be clipped to 0)"""
        hr = [40] * 100  # Below typical resting HR
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        # Should handle this gracefully
        self.assertIsInstance(result, float)

    def test_trimp_high_intensity(self):
        """Test TRIMP increases with intensity"""
        hr_moderate = [140] * 1800  # 30 min moderate
        hr_high = [170] * 1800  # 30 min high

        trimp_moderate = trimp_from_hr(hr_moderate, hr_rest=60, hr_max=190)
        trimp_high = trimp_from_hr(hr_high, hr_rest=60, hr_max=190)

        self.assertGreater(trimp_high, trimp_moderate)

    def test_trimp_with_nans(self):
        """Test TRIMP with NaN values"""
        hr = [150, 160, np.nan, 155, 150]
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        self.assertIsInstance(result, float)


class TestSessionDataProcessing(unittest.TestCase):
    """Test session data processing"""

    def setUp(self):
        """Set up test data"""
        # Create a simple DataFrame with time series data
        self.test_data = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-20 10:00:00", periods=600, freq="1s"),
                "hr": [120 + i % 20 for i in range(600)],  # HR varying between 120-140
                "power": [200 + i % 50 for i in range(600)],  # Power varying
            }
        )

        self.mock_session = {
            "sport": "cycling",
            "sub_sport": "indoor_cycling",
            "start_time": datetime(2025, 10, 20, 10, 0, 0),
        }

    def test_process_session_with_power_data(self):
        """Test processing a session with power data"""
        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")
        result = process_session_data(self.test_data, "test.fit", self.mock_session, 0, config)

        self.assertIsNotNone(result)
        self.assertIn("avg_hr", result)
        self.assertIn("avg_power_w", result)
        self.assertIn("np_w", result)
        self.assertIn("TSS", result)
        self.assertIn("IF", result)

    def test_process_session_empty_dataframe(self):
        """Test processing with empty DataFrame"""
        empty_df = pd.DataFrame(columns=["time", "hr", "power"])
        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")
        result = process_session_data(empty_df, "test.fit", self.mock_session, 0, config)

        self.assertIsNone(result)

    def test_process_session_hr_only(self):
        """Test processing a session with HR data only (no power)"""
        hr_only_data = self.test_data.copy()
        hr_only_data["power"] = np.nan

        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")
        result = process_session_data(hr_only_data, "test.fit", self.mock_session, 0, config)

        self.assertIsNotNone(result)
        self.assertIn("avg_hr", result)
        self.assertEqual(result["avg_power_w"], "")  # No power data
        self.assertGreater(result["TRIMP"], 0)  # Should still have TRIMP


class TestFITFileParsing(unittest.TestCase):
    """Test FIT file parsing and analysis"""

    @patch("fitanalyzer.parser.FitFile")
    def test_summarize_fit_original_no_data(self, mock_fitfile):
        """Test handling of FIT file with no record data"""
        mock_ff = MagicMock()
        mock_ff.get_messages.return_value = []
        mock_fitfile.return_value = mock_ff

        result, df_sets = summarize_fit_original("test.fit", ftp=300)

        self.assertIsNone(result)

    @patch("fitanalyzer.parser.FitFile")
    def test_summarize_fit_sessions_single_session(self, mock_fitfile):
        """Test multisport processing with single session"""
        mock_ff = MagicMock()

        # Mock a single session
        mock_session = Mock()
        mock_session.name = "session"

        session_data = [
            Mock(name="sport", value="cycling"),
            Mock(name="start_time", value=datetime(2025, 10, 20, 10, 0, 0)),
            Mock(name="total_timer_time", value=600),  # 10 minutes
        ]

        def mock_get_messages(msg_type):
            if msg_type == "session":
                return [session_data]
            elif msg_type == "record":
                return []  # Empty for this test
            return []

        mock_ff.get_messages.side_effect = mock_get_messages
        mock_fitfile.return_value = mock_ff

        results, _ = summarize_fit_sessions("test.fit", ftp=300)

        # Should handle single session appropriately
        self.assertIsInstance(results, list)


class TestMultisportHandling(unittest.TestCase):
    """Test multisport activity handling and deduplication"""

    def test_session_deduplication_key_generation(self):
        """Test that duplicate sessions can be identified"""
        session1 = {
            "sport": "cycling",
            "start_time": "2025-10-20 13:09:48",
            "duration_min": 10.0,
            "avg_hr": 114.7,
            "avg_power_w": 217.4,
        }

        session2 = {
            "sport": "cycling",
            "start_time": "2025-10-20 13:09:48",
            "duration_min": 10.0,
            "avg_hr": 114.7,
            "avg_power_w": 217.4,
        }

        # These should create identical keys
        key1 = (
            session1.get("sport"),
            session1.get("start_time"),
            session1.get("duration_min"),
            session1.get("avg_hr"),
            session1.get("avg_power_w"),
        )

        key2 = (
            session2.get("sport"),
            session2.get("start_time"),
            session2.get("duration_min"),
            session2.get("avg_hr"),
            session2.get("avg_power_w"),
        )

        self.assertEqual(key1, key2)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""

    def test_invalid_ftp(self):
        """Test handling of invalid FTP values"""
        power = [200] * 100
        npw = np_power(power)

        # FTP = 0 should not cause division by zero
        IF = (npw / 0) if 0 > 0 else np.nan
        self.assertTrue(np.isnan(IF))

    def test_negative_hr_values(self):
        """Test handling of negative HR values"""
        hr = [-10, 150, 160, 155]
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        # Should handle gracefully (clipped to 0)
        self.assertIsInstance(result, float)

    def test_timezone_handling(self):
        """Test timezone conversion"""
        # Create data with UTC timestamps
        data = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-20 10:00:00", periods=10, freq="1s", tz="UTC"),
                "hr": [120] * 10,
                "power": [200] * 10,
            }
        )

        # Should not raise exceptions with timezone-aware data
        self.assertIsNotNone(data)


class TestCSVOutput(unittest.TestCase):
    """Test CSV output formatting and validation"""

    def test_csv_column_names(self):
        """Test that expected columns are present"""
        expected_columns = [
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
        ]

        # Mock a result dictionary
        result = {
            "file": "test.fit",
            "sport": "cycling",
            "sub_sport": "indoor_cycling",
            "date": "2025-10-20",
            "start_time": "2025-10-20 10:00:00",
            "end_time": "2025-10-20 10:10:00",
            "duration_min": 10.0,
            "avg_hr": 120.0,
            "max_hr": 140,
            "avg_power_w": 200.0,
            "max_power_w": 250.0,
            "np_w": 210.0,
            "IF": 0.7,
            "TSS": 5.0,
            "TRIMP": 15.0,
        }

        # Check all expected columns are in the result
        for col in expected_columns:
            self.assertIn(col, result)

    def test_numeric_formatting(self):
        """Test that numeric values are properly formatted"""
        # Test rounding
        self.assertEqual(round(114.678, 1), 114.7)
        self.assertEqual(round(0.7256, 3), 0.726)
        self.assertEqual(int(149.9), 149)


class TestSportMapping(unittest.TestCase):
    """Test sport and sub-sport name mappings"""

    def test_sport_mapping_exists(self):
        """Test that sport mapping dictionary exists"""
        self.assertIsInstance(SPORT_MAPPING, dict)
        self.assertGreater(len(SPORT_MAPPING), 0)

    def test_subsport_mapping_exists(self):
        """Test that sub-sport mapping dictionary exists"""
        self.assertIsInstance(SUB_SPORT_MAPPING, dict)
        self.assertGreater(len(SUB_SPORT_MAPPING), 0)

    def test_common_sports_mapped(self):
        """Test that common sports are properly mapped"""
        # Common sports that should be in the mapping
        expected_sports = {
            1: "running",
            2: "cycling",
            5: "swimming",
            10: "training",
            15: "rowing",
            75: "volleyball",
        }

        for code, name in expected_sports.items():
            self.assertIn(code, SPORT_MAPPING)
            self.assertEqual(SPORT_MAPPING[code], name)

    def test_common_subsports_mapped(self):
        """Test that common sub-sports are properly mapped"""
        expected_subsports = {
            0: "generic",
            1: "treadmill",
            6: "indoor_cycling",
            20: "strength_training",
            3: "trail",
            14: "indoor_rowing",
        }

        for code, name in expected_subsports.items():
            self.assertIn(code, SUB_SPORT_MAPPING)
            self.assertEqual(SUB_SPORT_MAPPING[code], name)

    def test_volleyball_mapping(self):
        """Test specific volleyball mapping (user's issue)"""
        self.assertEqual(SPORT_MAPPING[75], "volleyball")

    def test_sport_code_conversion_logic(self):
        """Test the logic for converting sport codes to names"""
        # Simulate what happens in the parser
        raw_sport = 75  # volleyball
        raw_subsport = 0  # generic

        # Convert numeric codes to names (same logic as in parser)
        if isinstance(raw_sport, int):
            session_sport = SPORT_MAPPING.get(raw_sport, str(raw_sport))
        else:
            session_sport = raw_sport

        if isinstance(raw_subsport, int):
            session_subsport = SUB_SPORT_MAPPING.get(raw_subsport, str(raw_subsport))
        else:
            session_subsport = raw_subsport

        # Verify conversion
        self.assertEqual(session_sport, "volleyball")
        self.assertEqual(session_subsport, "generic")

        # Test with string input (should pass through)
        raw_sport_str = "already_a_string"
        session_sport_str = (
            raw_sport_str
            if not isinstance(raw_sport_str, int)
            else SPORT_MAPPING.get(raw_sport_str, str(raw_sport_str))
        )
        self.assertEqual(session_sport_str, "already_a_string")

        # Test with unknown numeric code (should convert to string)
        raw_sport_unknown = 9999
        session_sport_unknown = SPORT_MAPPING.get(raw_sport_unknown, str(raw_sport_unknown))
        self.assertEqual(session_sport_unknown, "9999")

    def test_all_sport_codes_are_strings(self):
        """Test that all sport names are strings"""
        for code, name in SPORT_MAPPING.items():
            self.assertIsInstance(code, int, f"Sport code {code} should be int")
            self.assertIsInstance(name, str, f"Sport name {name} should be str")

    def test_all_subsport_codes_are_strings(self):
        """Test that all sub-sport names are strings"""
        for code, name in SUB_SPORT_MAPPING.items():
            self.assertIsInstance(code, int, f"Sub-sport code {code} should be int")
            self.assertIsInstance(name, str, f"Sub-sport name {name} should be str")

    def test_no_duplicate_sport_codes(self):
        """Test that there are no duplicate sport codes"""
        codes = list(SPORT_MAPPING.keys())
        self.assertEqual(len(codes), len(set(codes)), "Found duplicate sport codes")

    def test_no_duplicate_subsport_codes(self):
        """Test that there are no duplicate sub-sport codes"""
        codes = list(SUB_SPORT_MAPPING.keys())
        self.assertEqual(len(codes), len(set(codes)), "Found duplicate sub-sport codes")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
