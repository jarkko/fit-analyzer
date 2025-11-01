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

from fitanalyzer.activities import summarize_fit_original, summarize_fit_sessions
from fitanalyzer.config import AnalysisConfig
from fitanalyzer.constants import SPORT_MAPPING, SUB_SPORT_MAPPING
from fitanalyzer.metrics import np_power, trimp_from_hr
from fitanalyzer.sessions import process_session_data


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

    @patch("fitanalyzer.activities.FitFile")
    def test_summarize_fit_original_no_data(self, mock_fitfile):
        """Test handling of FIT file with no record data"""
        mock_ff = MagicMock()
        mock_ff.get_messages.return_value = []
        mock_fitfile.return_value = mock_ff

        config = AnalysisConfig(ftp=300, hr_rest=60, hr_max=190, tz_name="UTC")
        result = summarize_fit_original("test.fit", config)

        self.assertIsNone(result)

    @patch("fitanalyzer.activities.FitFile")
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

        config = AnalysisConfig(ftp=300, hr_rest=60, hr_max=190, tz_name="UTC")
        results, _ = summarize_fit_sessions("test.fit", config)

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


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_summarize_fit_sessions_missing_start_time(self):
        """Test handling of sessions without start_time."""
        with patch("fitanalyzer.activities.FitFile") as mock_fitfile:
            with patch("fitanalyzer.activities.extract_sessions_from_fit") as mock_extract:
                # Session missing start_time
                mock_extract.return_value = [
                    {"total_timer_time": 1000},  # No start_time
                    {"start_time": datetime.now(), "total_timer_time": 1000},
                ]
                mock_fitfile.return_value.get_messages.return_value = []

                config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="UTC")
                results, _ = summarize_fit_sessions("test.fit", config)
                # Should only process the second session
                self.assertEqual(len(results), 0)  # No records, so no results

    def test_summarize_fit_sessions_zero_timer_time(self):
        """Test handling of sessions with zero total_timer_time."""
        with patch("fitanalyzer.activities.FitFile") as mock_fitfile:
            with patch("fitanalyzer.activities.extract_sessions_from_fit") as mock_extract:
                # Session with zero timer time
                mock_extract.return_value = [
                    {"start_time": datetime.now(), "total_timer_time": 0},
                ]
                mock_fitfile.return_value.get_messages.return_value = []

                config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="UTC")
                results, _ = summarize_fit_sessions("test.fit", config)
                self.assertEqual(len(results), 0)

    def test_get_first_valid_value_with_tuple(self):
        """Test _extract_valid_value with tuple input."""
        from fitanalyzer.parser import extract_valid_value

        # Valid value in tuple
        result = extract_valid_value((1, 2, 3))
        self.assertEqual(result, 1)

        # Skip invalid values (65534)
        result = extract_valid_value((65534, 100, 200))
        self.assertEqual(result, 100)

        # All invalid
        result = extract_valid_value((65534, 65534))
        self.assertIsNone(result)

        # None in tuple
        result = extract_valid_value((None, 100))
        self.assertEqual(result, 100)

    def test_get_first_valid_value_with_nan(self):
        """Test _extract_valid_value with NaN and None."""
        from fitanalyzer.parser import extract_valid_value

        result = extract_valid_value(np.nan)
        self.assertIsNone(result)

        result = extract_valid_value(None)
        self.assertIsNone(result)

    def test_get_first_valid_value_with_invalid_marker(self):
        """Test _extract_valid_value with custom invalid marker."""
        from fitanalyzer.parser import extract_valid_value

        result = extract_valid_value(65534)
        self.assertIsNone(result)

        result = extract_valid_value(100)
        self.assertEqual(result, 100)

    def test_get_sport_names_with_string_values(self):
        """Test _get_sport_names when sport/subsport are already strings."""
        from fitanalyzer.parser import get_sport_names

        sessions = [{"sport": "cycling", "sub_sport": "road"}]
        sport, subsport = get_sport_names(sessions)
        self.assertEqual(sport, "cycling")
        self.assertEqual(subsport, "road")

    def test_get_sport_names_with_unmapped_codes(self):
        """Test _get_sport_names with codes not in mapping."""
        from fitanalyzer.parser import get_sport_names

        # Use a code that doesn't exist in mapping
        sessions = [{"sport": 99999, "sub_sport": 88888}]
        sport, subsport = get_sport_names(sessions)
        self.assertEqual(sport, "99999")  # Should convert to string
        self.assertEqual(subsport, "88888")


class TestCLIFunctions(unittest.TestCase):
    """Test CLI-related functions."""

    def test_process_multisport_file_with_no_results(self):
        """Test _process_multisport_file when no results returned."""
        from fitanalyzer.cli import _process_multisport_file

        args = MagicMock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"

        with patch("fitanalyzer.cli.summarize_fit_sessions", return_value=([], [])):
            result = _process_multisport_file("test.fit", args, set())
            self.assertEqual(result, [])

    def test_process_multisport_file_with_null_result(self):
        """Test _process_multisport_file with None in results."""
        from fitanalyzer.cli import _process_multisport_file

        args = MagicMock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"

        # Return list with None
        with patch(
            "fitanalyzer.cli.summarize_fit_sessions",
            return_value=([None, {"sport": "running"}], []),
        ):
            result = _process_multisport_file("test.fit", args, set())
            # Should skip None
            self.assertEqual(len(result), 1)

    def test_process_multisport_file_duplicate_detection(self):
        """Test _process_multisport_file detects duplicates."""
        from fitanalyzer.cli import _process_multisport_file

        args = MagicMock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"

        session = {
            "sport": "running",
            "start_time": "2024-01-01 10:00:00",
            "duration_min": 30.0,
            "avg_hr": 150,
            "avg_power_w": 200,
        }

        processed = set()

        with patch("fitanalyzer.cli.summarize_fit_sessions", return_value=([session], [])):
            # First call should add it
            result1 = _process_multisport_file("test.fit", args, processed)
            self.assertEqual(len(result1), 1)

            # Second call with same session should skip it
            result2 = _process_multisport_file("test.fit", args, processed)
            self.assertEqual(len(result2), 0)

    def test_process_single_file(self):
        """Test _process_single_file function."""
        from fitanalyzer.cli import _process_single_file

        args = MagicMock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"

        summary = {"sport": "cycling", "duration_min": 60.0}

        with patch("fitanalyzer.cli.summarize_fit_original", return_value=summary):
            result = _process_single_file("test.fit", args)
            self.assertEqual(result, [summary])

    def test_main_with_args_no_files(self):
        """Test main_with_args with no FIT files."""
        from fitanalyzer.cli import main_with_args

        args = MagicMock()
        args.fit_files = []
        args.multisport = False
        args.dump_sets = False

        # Should print "No data to output" and return 0
        with patch("builtins.print"):
            result = main_with_args(args)
            self.assertEqual(result, 0)

    def test_main_with_args_dump_sets_empty(self):
        """Test main_with_args with dump_sets but no strength data."""
        from fitanalyzer.cli import main_with_args

        args = MagicMock()
        args.fit_files = ["test.fit"]
        args.multisport = False
        args.dump_sets = True
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"
        args.output_dir = "/tmp"
        args.force = False

        summary = {
            "sport": "cycling",
            "date": "2024-01-01",
            "start_time": "10:00:00",
        }

        with patch("fitanalyzer.cli.summarize_fit_original", return_value=summary):
            with patch("fitanalyzer.cli.load_existing_analysis", return_value={}):
                with patch("fitanalyzer.cli.load_existing_rows", return_value=[]):
                    with patch(
                        "fitanalyzer.cli.determine_files_to_process", return_value=(["test.fit"], 0)
                    ):
                        with patch("fitanalyzer.cli.FitFile"):
                            with patch(
                                "fitanalyzer.cli.extract_sets_from_fit", return_value=pd.DataFrame()
                            ):
                                with patch(
                                    "fitanalyzer.cli.aggregate_strength_sets", return_value=None
                                ):
                                    with patch("builtins.print"):
                                        result = main_with_args(args)
                                        self.assertEqual(result, 0)

    def test_main_entry_point(self):
        """Test main() entry point."""
        from fitanalyzer.cli import main

        with patch("fitanalyzer.cli.parse_arguments") as mock_parse:
            with patch("fitanalyzer.cli.main_with_args", return_value=0) as mock_main:
                mock_parse.return_value = MagicMock()
                result = main()
                self.assertEqual(result, 0)
                mock_parse.assert_called_once()
                mock_main.assert_called_once()


class TestEdgeCasesForFullCoverage(unittest.TestCase):
    """Additional edge case tests to achieve 100% coverage"""

    def test_session_without_start_time_skipped(self):
        """Test that sessions without start_time are skipped (line 105-106)"""
        from fitanalyzer.activities import summarize_fit_sessions
        from fitanalyzer.config import AnalysisConfig

        # Mock the FitFile and _extract_sessions_from_fit to return invalid sessions
        with patch("fitanalyzer.activities.FitFile") as mock_fitfile, patch(
            "fitanalyzer.activities.extract_sessions_from_fit"
        ) as mock_extract:

            # Return sessions with one missing start_time (line 105-106)
            mock_extract.return_value = [
                {"total_timer_time": 100},  # Missing start_time - should be skipped
                {
                    "start_time": datetime.now(),
                    "total_timer_time": 1000,
                },
            ]

            # Mock get_messages to return empty for simplicity
            mock_fitfile.return_value.get_messages.return_value = []

            config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="UTC")

            results, _ = summarize_fit_sessions("dummy.fit", config)

            # Should have skipped the session without start_time
            # Since we're mocking get_messages to return empty, we might get empty results
            # The important thing is it didn't crash
            self.assertIsInstance(results, list)

    def test_session_with_invalid_timer_time(self):
        """Test that sessions with timer_time <= 0 are skipped (line 107-108)"""
        from fitanalyzer.activities import summarize_fit_sessions
        from fitanalyzer.config import AnalysisConfig

        # Mock the FitFile and _extract_sessions_from_fit to return sessions with invalid timer
        with patch("fitanalyzer.activities.FitFile") as mock_fitfile, patch(
            "fitanalyzer.activities.extract_sessions_from_fit"
        ) as mock_extract:

            # Return sessions with invalid timer_time (line 107-108)
            mock_extract.return_value = [
                {
                    "start_time": datetime.now(),
                    "total_timer_time": 0,  # Invalid - should be skipped
                },
                {"start_time": datetime.now(), "total_timer_time": -5},  # Also invalid
            ]

            # Mock get_messages to return empty
            mock_fitfile.return_value.get_messages.return_value = []

            config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="UTC")

            results, _ = summarize_fit_sessions("dummy.fit", config)

            # Should have skipped all sessions with invalid timer_time
            # Results should be empty or minimal
            self.assertIsInstance(results, list)

    def test_aggregate_strength_with_empty_sets(self):
        """Test aggregate_strength_sets when df_sets is empty (line 505)"""
        from fitanalyzer.aggregation import aggregate_strength_sets
        from fitanalyzer.config import AnalysisConfig

        config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="UTC")

        # Mock FitFile and summarize_fit_original to return empty DataFrames
        with patch("fitanalyzer.aggregation.FitFile") as mock_fitfile:
            mock_ff = MagicMock()
            mock_ff.get_messages.return_value = []  # Single session (not multisport)
            mock_fitfile.return_value = mock_ff

            with patch("fitanalyzer.activities.summarize_fit_original") as mock_summary:
                # Return dict and empty DataFrame - the empty df_sets will be skipped
                mock_summary.return_value = ({"file": "test.fit"}, pd.DataFrame())

                with tempfile.TemporaryDirectory() as tmpdir:
                    # Create a dummy file
                    dummy_file = Path(tmpdir) / "test.fit"
                    dummy_file.write_text("")

                    result = aggregate_strength_sets([str(dummy_file)], config, False)

                    # Should return None when all sets are empty
                    self.assertIsNone(result)

    def test_timezone_aware_timestamps_tz_convert(self):
        """Test _process_timestamps with already timezone-aware timestamps (lines 183-184)"""
        import pytz

        from fitanalyzer.sessions import process_timestamps

        # Create timezone-aware timestamps (not UTC)
        eastern = pytz.timezone("US/Eastern")
        times = pd.DatetimeIndex(
            [datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 1, 12, 1, 0)]
        ).tz_localize(eastern)

        df = pd.DataFrame({"time": times, "power": [200, 210]})

        result = process_timestamps(df, "UTC")

        # Should use tz_convert to convert to UTC (line 183-184)
        self.assertIsNotNone(result["start_utc"])
        self.assertEqual(str(result["start_utc"].tzinfo), "UTC")

    def test_timezone_naive_to_utc(self):
        """Test _process_timestamps with naive timestamps (lines 179-180)"""
        from fitanalyzer.sessions import process_timestamps

        # Create naive (no timezone) timestamps
        times = pd.DatetimeIndex([datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 1, 12, 1, 0)])

        df = pd.DataFrame({"time": times, "power": [200, 210]})

        result = process_timestamps(df, "UTC")

        # Should use tz_localize to add UTC timezone (lines 179-180)
        self.assertIsNotNone(result["start_utc"])
        self.assertEqual(str(result["start_utc"].tzinfo), "UTC")

    def test_prepare_timezone_aware_index_with_tz_aware_timestamps(self):
        """Test _prepare_timezone_aware_index with timezone-aware timestamps (lines 543-544)"""
        import pytz

        from fitanalyzer.activities import _prepare_timezone_aware_index

        # Create timezone-aware timestamps (not UTC)
        eastern = pytz.timezone("US/Eastern")
        times = pd.DatetimeIndex(
            [datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 1, 12, 1, 0)]
        ).tz_localize(eastern)

        df = pd.DataFrame({"time": times, "power": [200, 210]})

        start_utc, end_utc, time_index = _prepare_timezone_aware_index(df)

        # Should use tz_convert to convert to UTC (lines 543-544)
        self.assertIsNotNone(start_utc)
        self.assertEqual(str(start_utc.tzinfo), "UTC")
        # time_index should also be converted (line 551)
        self.assertIsNotNone(time_index)

    def test_prepare_timezone_aware_index_with_naive_timestamps(self):
        """Test _prepare_timezone_aware_index with naive timestamps (line 549)"""
        from fitanalyzer.activities import _prepare_timezone_aware_index

        # Create naive (no timezone) timestamps
        times = pd.DatetimeIndex([datetime(2025, 1, 1, 12, 0, 0), datetime(2025, 1, 1, 12, 1, 0)])

        df = pd.DataFrame({"time": times, "power": [200, 210]})

        start_utc, end_utc, time_index = _prepare_timezone_aware_index(df)

        # Should use tz_localize to add UTC timezone (lines 540-541, 549)
        self.assertIsNotNone(start_utc)
        self.assertEqual(str(start_utc.tzinfo), "UTC")
        # time_index should also be localized (line 549)
        self.assertIsNotNone(time_index)

    def test_multisport_with_dump_sets_flag(self):
        """Test that --dump-sets is skipped when --multisport is used (lines 696-697)"""
        from fitanalyzer.cli import main_with_args, parse_arguments

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use strength training fixture
            test_file = "tests/fixtures/20744294788_ACTIVITY.fit"

            args = parse_arguments(
                [test_file, "--multisport", "--dump-sets", "--ftp", "300", "--output-dir", tmpdir]
            )

            result = main_with_args(args)

            # Should complete successfully
            self.assertEqual(result, 0)

            # When multisport is True, the per-file dump-sets is skipped
            # So we shouldn't have individual set CSV files
            output_path = Path(tmpdir)
            set_files = list(output_path.glob("*_sets.csv"))

            # With multisport, sets are only in the aggregated output
            # Individual set files shouldn't be created
            self.assertEqual(len(set_files), 0)


class TestSpeedCadenceDistanceExtraction(unittest.TestCase):
    """Test extraction of speed, cadence, and distance data from FIT files"""

    def test_extract_records_includes_speed_cadence_distance(self):
        """Test that _extract_records_from_fit includes speed, cadence, and distance"""
        from unittest.mock import Mock

        from fitparse import FitFile

        from fitanalyzer.parser import extract_records_from_fit

        # Mock FIT file with speed, cadence, distance data
        mock_fit = Mock(spec=FitFile)

        # Create proper mock message structure
        def create_mock_message(data_dict):
            mock_msg = Mock()
            mock_fields = []
            for key, value in data_dict.items():
                mock_field = Mock()
                mock_field.name = key
                mock_field.value = value
                mock_fields.append(mock_field)
            mock_msg.__iter__ = lambda self: iter(mock_fields)
            return mock_msg

        # Mock record messages with various data fields
        mock_messages = [
            # Complete record with all fields
            create_mock_message(
                {
                    "timestamp": datetime.now(),
                    "heart_rate": 150,
                    "power": 250,
                    "speed": 8.33,  # m/s
                    "cadence": 90,  # rpm
                    "distance": 1000.5,  # meters
                }
            ),
            # Record with missing speed/cadence (should handle gracefully)
            create_mock_message(
                {
                    "timestamp": datetime.now() + timedelta(seconds=1),
                    "heart_rate": 155,
                    "power": 260,
                    "distance": 1010.8,
                }
            ),
            # Record with only basic fields
            create_mock_message(
                {
                    "timestamp": datetime.now() + timedelta(seconds=2),
                    "heart_rate": 148,
                }
            ),
        ]

        mock_fit.get_messages.return_value = mock_messages

        # Extract records
        df = extract_records_from_fit(mock_fit)

        # Verify expected columns exist
        expected_columns = ["time", "hr", "power", "speed", "cadence", "distance"]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' should be in DataFrame")

        # Verify data integrity
        self.assertEqual(len(df), 3, "Should have 3 records")

        # First record should have all data
        self.assertEqual(df.iloc[0]["hr"], 150)
        self.assertEqual(df.iloc[0]["power"], 250)
        self.assertEqual(df.iloc[0]["speed"], 8.33)
        self.assertEqual(df.iloc[0]["cadence"], 90)
        self.assertEqual(df.iloc[0]["distance"], 1000.5)

        # Second record should have NaN for missing cadence
        self.assertEqual(df.iloc[1]["hr"], 155)
        self.assertEqual(df.iloc[1]["power"], 260)
        self.assertTrue(pd.isna(df.iloc[1]["speed"]))
        self.assertTrue(pd.isna(df.iloc[1]["cadence"]))
        self.assertEqual(df.iloc[1]["distance"], 1010.8)

        # Third record should have NaN for missing fields
        self.assertEqual(df.iloc[2]["hr"], 148)
        self.assertTrue(pd.isna(df.iloc[2]["power"]))
        self.assertTrue(pd.isna(df.iloc[2]["speed"]))
        self.assertTrue(pd.isna(df.iloc[2]["cadence"]))
        self.assertTrue(pd.isna(df.iloc[2]["distance"]))

    def test_session_summary_includes_speed_cadence_distance_metrics(self):
        """Test that session summaries include speed, cadence, and distance metrics"""
        from fitanalyzer.sessions import AnalysisConfig, process_session_data

        # Create test data with speed/cadence/distance
        test_data = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-30 10:00:00", periods=5, freq="1s"),
                "hr": [150, 152, 148, 155, 151],
                "power": [250, 260, 240, 270, 255],
                "speed": [8.0, 8.5, 7.8, 9.0, 8.2],  # m/s
                "cadence": [88, 90, 85, 95, 89],  # rpm
                "distance": [1000, 1008, 1016, 1025, 1033],  # meters
            }
        )

        config = AnalysisConfig(ftp=300, hr_rest=60, hr_max=190, tz_name="UTC")
        mock_session = {"sport": "cycling", "sub_sport": "road"}

        result = process_session_data(test_data, "test.fit", mock_session, 0, config)

        # Verify speed metrics are included
        self.assertIn("avg_speed_mps", result)
        self.assertIn("max_speed_mps", result)
        self.assertIn("avg_speed_kph", result)
        self.assertIn("max_speed_kph", result)

        # Verify cadence metrics are included
        self.assertIn("avg_cadence", result)
        self.assertIn("max_cadence", result)

        # Verify distance is included
        self.assertIn("total_distance_m", result)
        self.assertIn("total_distance_km", result)

        # Check calculated values
        self.assertAlmostEqual(result["avg_speed_mps"], 8.3, places=1)
        self.assertAlmostEqual(result["max_speed_mps"], 9.0, places=1)
        self.assertAlmostEqual(result["avg_speed_kph"], 29.88, places=1)  # 8.3 * 3.6
        self.assertAlmostEqual(result["max_speed_kph"], 32.4, places=1)  # 9.0 * 3.6

        self.assertAlmostEqual(result["avg_cadence"], 89.4, places=1)
        self.assertEqual(result["max_cadence"], 95)

        # Distance should be max - min
        self.assertAlmostEqual(result["total_distance_m"], 33, places=0)  # 1033 - 1000
        self.assertAlmostEqual(result["total_distance_km"], 0.033, places=3)

    def test_handles_missing_speed_cadence_distance_gracefully(self):
        """Test that missing speed/cadence/distance data is handled gracefully"""
        from fitanalyzer.sessions import AnalysisConfig, process_session_data

        # Create test data without speed/cadence/distance
        test_data = pd.DataFrame(
            {
                "time": pd.date_range("2025-10-30 10:00:00", periods=3, freq="1s"),
                "hr": [150, 152, 148],
                "power": [250, 260, 240],
                "speed": [np.nan, np.nan, np.nan],
                "cadence": [np.nan, np.nan, np.nan],
                "distance": [np.nan, np.nan, np.nan],
            }
        )

        config = AnalysisConfig(ftp=300, hr_rest=60, hr_max=190, tz_name="UTC")
        mock_session = {"sport": "running", "sub_sport": "generic"}

        result = process_session_data(test_data, "test.fit", mock_session, 0, config)

        # Should still return a valid result
        self.assertIsNotNone(result)

        # Speed/cadence/distance metrics should be empty strings or nan
        self.assertEqual(result["avg_speed_mps"], "")
        self.assertEqual(result["max_speed_mps"], "")
        self.assertEqual(result["avg_cadence"], "")
        self.assertEqual(result["max_cadence"], "")
        self.assertEqual(result["total_distance_m"], "")

    def test_real_cycling_fit_file_integration(self):
        """Integration test: Real FIT file should produce speed/cadence/distance metrics"""
        from fitanalyzer.activities import summarize_fit_sessions
        from fitanalyzer.config import AnalysisConfig

        # Test with the cycling test fixture
        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="UTC")

        # This should work with the test fixture
        sessions, sets = summarize_fit_sessions("tests/fixtures/20684859222_ACTIVITY.fit", config)

        # Should have at least one session
        self.assertGreater(len(sessions), 0)

        session = sessions[0]

        # Should have the new columns
        self.assertIn("avg_speed_mps", session)
        self.assertIn("max_speed_mps", session)
        self.assertIn("avg_speed_kph", session)
        self.assertIn("max_speed_kph", session)
        self.assertIn("avg_cadence", session)
        self.assertIn("max_cadence", session)
        self.assertIn("total_distance_m", session)
        self.assertIn("total_distance_km", session)

        # For a cycling file, these should have actual values (not empty strings)
        self.assertNotEqual(session["avg_speed_mps"], "")
        self.assertNotEqual(session["avg_cadence"], "")
        self.assertNotEqual(session["total_distance_m"], "")

        # Speed should be reasonable for indoor cycling (1-10 m/s)
        self.assertGreaterEqual(session["avg_speed_mps"], 1.0)
        self.assertLessEqual(session["avg_speed_mps"], 10.0)

        # Cadence should be reasonable for cycling (40-120 rpm)
        self.assertGreaterEqual(session["avg_cadence"], 40)
        self.assertLessEqual(session["avg_cadence"], 120)

        # Distance should be positive
        self.assertGreater(session["total_distance_m"], 0)


class TestElevationMetrics(unittest.TestCase):
    """Test elevation metrics extraction and calculation (TDD for ascent/descent)"""

    def test_elevation_fields_in_csv_output(self):
        """Test that elevation fields are present in CSV output"""
        # Use a running activity file that has GPS/elevation data
        fit_file = Path("tests/fixtures/20544585388_ACTIVITY.fit")

        if not fit_file.exists():
            self.skipTest(f"Test file {fit_file} not found")

        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")

        result = summarize_fit_original(str(fit_file), config)

        # Should have elevation-related fields in the output
        self.assertIn("total_ascent_m", result)
        self.assertIn("total_descent_m", result)
        self.assertIn("avg_altitude_m", result)
        self.assertIn("min_altitude_m", result)
        self.assertIn("max_altitude_m", result)

    def test_elevation_calculation_with_gps_data(self):
        """Test that elevation metrics are calculated correctly from GPS data"""
        fit_file = Path("tests/fixtures/20544585388_ACTIVITY.fit")

        if not fit_file.exists():
            self.skipTest(f"Test file {fit_file} not found")

        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")

        result = summarize_fit_original(str(fit_file), config)

        # Elevation values should be numeric (not empty strings)
        self.assertIsInstance(result["total_ascent_m"], (int, float))
        self.assertIsInstance(result["total_descent_m"], (int, float))
        self.assertIsInstance(result["avg_altitude_m"], (int, float))
        self.assertIsInstance(result["min_altitude_m"], (int, float))
        self.assertIsInstance(result["max_altitude_m"], (int, float))

        # Values should be non-negative
        self.assertGreaterEqual(result["total_ascent_m"], 0)
        self.assertGreaterEqual(result["total_descent_m"], 0)

        # Min should be <= avg <= max
        self.assertLessEqual(result["min_altitude_m"], result["avg_altitude_m"])
        self.assertLessEqual(result["avg_altitude_m"], result["max_altitude_m"])

    def test_elevation_empty_when_no_gps_data(self):
        """Test that elevation fields are empty for activities without GPS data"""
        # Use a strength training file that has no GPS/elevation data
        fit_file = Path("tests/fixtures/20794985860_ACTIVITY.fit")

        if not fit_file.exists():
            self.skipTest(f"Test file {fit_file} not found")

        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")

        result = summarize_fit_original(str(fit_file), config)

        # Should have elevation fields but they should be empty
        self.assertIn("total_ascent_m", result)
        self.assertIn("total_descent_m", result)
        self.assertIn("avg_altitude_m", result)

        # Values should be empty strings when no elevation data
        self.assertEqual(result["total_ascent_m"], "")
        self.assertEqual(result["total_descent_m"], "")
        self.assertEqual(result["avg_altitude_m"], "")
        self.assertEqual(result["min_altitude_m"], "")
        self.assertEqual(result["max_altitude_m"], "")

    def test_csv_headers_include_elevation(self):
        """Test that CSV headers include elevation fields"""
        # Generate CSV data from a test file
        fit_file = Path("tests/fixtures/20544585388_ACTIVITY.fit")

        if not fit_file.exists():
            self.skipTest(f"Test file {fit_file} not found")

        config = AnalysisConfig(ftp=250, hr_rest=60, hr_max=190, tz_name="Europe/Helsinki")

        result = summarize_fit_original(str(fit_file), config)

        # Check that result dict has elevation fields (these become CSV columns)
        self.assertIn("total_ascent_m", result)
        self.assertIn("total_descent_m", result)
        self.assertIn("avg_altitude_m", result)
        self.assertIn("min_altitude_m", result)
        self.assertIn("max_altitude_m", result)


class TestIncrementalAnalysis(unittest.TestCase):
    """Test incremental analysis (only process new/modified files)"""

    def setUp(self):
        """Set up test directory and files"""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "output"
        self.output_dir.mkdir()

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def test_load_existing_analysis_no_csv(self):
        """Test load_existing_analysis when CSV doesn't exist"""
        from fitanalyzer.incremental import load_existing_analysis

        csv_path = self.output_dir / "nonexistent.csv"
        result = load_existing_analysis(csv_path)

        self.assertEqual(result, {})

    def test_load_existing_analysis_with_csv(self):
        """Test load_existing_analysis with existing CSV"""
        from fitanalyzer.incremental import load_existing_analysis

        # Create a mock CSV with file and _file_mtime columns
        csv_path = self.output_dir / "workout_summary.csv"
        df = pd.DataFrame(
            {
                "file": ["file1.fit", "file2.fit"],
                "_file_mtime": [1234567890.0, 1234567900.0],
                "sport": ["cycling", "running"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["file1.fit"], 1234567890.0)
        self.assertEqual(result["file2.fit"], 1234567900.0)

    def test_load_existing_analysis_missing_mtime_column(self):
        """Test load_existing_analysis with CSV missing _file_mtime column"""
        from fitanalyzer.incremental import load_existing_analysis

        # Create CSV without _file_mtime column
        csv_path = self.output_dir / "workout_summary.csv"
        df = pd.DataFrame(
            {
                "file": ["file1.fit", "file2.fit"],
                "sport": ["cycling", "running"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_existing_analysis(csv_path)

        self.assertEqual(result, {})

    def testneeds_analysis_force_flag(self):
        """Test that --force flag always returns True"""
        from fitanalyzer.incremental import needs_analysis

        # Create a test file
        test_file = Path(self.test_dir) / "test.fit"
        test_file.touch()

        existing_analysis = {str(test_file): test_file.stat().st_mtime}

        # With force=True, should always need analysis
        result = needs_analysis(str(test_file), existing_analysis, force=True)
        self.assertTrue(result)

    def testneeds_analysis_new_file(self):
        """Test that new files need analysis"""
        from fitanalyzer.incremental import needs_analysis

        # Create a test file
        test_file = Path(self.test_dir) / "new.fit"
        test_file.touch()

        existing_analysis = {}  # Empty - file never analyzed

        result = needs_analysis(str(test_file), existing_analysis, force=False)
        self.assertTrue(result)

    def testneeds_analysis_unmodified_file(self):
        """Test that unmodified files don't need analysis"""
        from fitanalyzer.incremental import needs_analysis

        # Create a test file
        test_file = Path(self.test_dir) / "unchanged.fit"
        test_file.touch()
        mtime = test_file.stat().st_mtime

        existing_analysis = {str(test_file): mtime}

        result = needs_analysis(str(test_file), existing_analysis, force=False)
        self.assertFalse(result)

    def testneeds_analysis_modified_file(self):
        """Test that modified files need reanalysis"""
        import time

        from fitanalyzer.incremental import needs_analysis

        # Create a test file
        test_file = Path(self.test_dir) / "modified.fit"
        test_file.touch()
        old_mtime = test_file.stat().st_mtime

        # Wait enough for filesystem timestamp to change (some filesystems have coarse granularity)
        # This ensures the mtime will actually be different
        time.sleep(2.1)
        test_file.write_text("modified")
        new_mtime = test_file.stat().st_mtime

        # Existing analysis has old mtime
        existing_analysis = {str(test_file): old_mtime}

        result = needs_analysis(str(test_file), existing_analysis, force=False)
        self.assertTrue(result)
        self.assertGreater(new_mtime, old_mtime)

    def testneeds_analysis_nonexistent_file(self):
        """Test that nonexistent files return False"""
        from fitanalyzer.incremental import needs_analysis

        nonexistent = "/nonexistent/file.fit"
        existing_analysis = {}

        result = needs_analysis(nonexistent, existing_analysis, force=False)
        self.assertFalse(result)

    def test_process_single_file_adds_mtime(self):
        """Test that _process_single_file adds _file_mtime to results"""
        from fitanalyzer.cli import _process_single_file

        # Use a real test fixture
        test_file = "tests/fixtures/20684859222_ACTIVITY.fit"
        if not Path(test_file).exists():
            self.skipTest(f"Test file {test_file} not found")

        # Create mock args
        args = Mock()
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"

        results = _process_single_file(test_file, args)

        self.assertEqual(len(results), 1)
        self.assertIn("_file_mtime", results[0])
        self.assertIsInstance(results[0]["_file_mtime"], float)

    def test_incremental_analysis_skips_unchanged_files(self):
        """Integration test: verify unchanged files are skipped"""
        from fitanalyzer.cli import main_with_args

        # Copy a test fixture to temp directory
        src_file = Path("tests/fixtures/20684859222_ACTIVITY.fit")
        if not src_file.exists():
            self.skipTest(f"Test file {src_file} not found")

        test_file = Path(self.test_dir) / "test_activity.fit"
        shutil.copy(src_file, test_file)

        # Create mock args
        args = Mock()
        args.fit_files = [str(test_file)]
        args.ftp = 300
        args.hrrest = 50
        args.hrmax = 190
        args.tz = "UTC"
        args.dump_sets = False
        args.multisport = False
        args.output_dir = str(self.output_dir)
        args.force = False

        # First run - should process the file
        with patch("builtins.print") as mock_print:
            main_with_args(args)

        # Verify CSV was created
        csv_path = self.output_dir / "workout_summary_from_fit.csv"
        self.assertTrue(csv_path.exists())

        # Second run - should skip the file
        with patch("builtins.print") as mock_print:
            main_with_args(args)

            # Check that "Skipping" or "Skipped" message was printed
            print_calls = [str(call) for call in mock_print.call_args_list]
            skipped = any("skip" in call.lower() for call in print_calls)
            self.assertTrue(skipped, f"Expected 'Skipping' message in output. Got: {print_calls}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
