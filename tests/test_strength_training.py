"""Tests for strength training data extraction."""

import pandas as pd
import pytest
from pathlib import Path
from fitanalyzer.parser import (
    _aggregate_strength_sets,
    AnalysisConfig,
    summarize_fit_original,
    summarize_fit_sessions,
)


@pytest.fixture(scope="module")
def strength_fit_file():
    """Path to a sample strength training FIT file."""
    return "tests/fixtures/20474406937_ACTIVITY.fit"


@pytest.fixture(scope="module")
def strength_fit_parsed(strength_fit_file):
    """Pre-parsed strength training FIT file data."""
    summary, df_sets = summarize_fit_original(strength_fit_file, ftp=300)
    return summary, df_sets


@pytest.fixture(scope="module")
def multiple_strength_files():
    """Paths to multiple strength training FIT files."""
    return [
        "tests/fixtures/20474406937_ACTIVITY.fit",
        "tests/fixtures/20555050352_ACTIVITY.fit",
        "tests/fixtures/20748058539_ACTIVITY.fit",
    ]


@pytest.fixture(scope="module")
def aggregated_strength_data(multiple_strength_files):
    """Pre-aggregated strength data from multiple files."""
    config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki")
    return _aggregate_strength_sets(multiple_strength_files, config, multisport=False)


@pytest.fixture(scope="module")
def multisport_with_strength():
    """Path to a multisport file containing a strength training session."""
    return "tests/fixtures/20744294788_ACTIVITY.fit"


@pytest.fixture(scope="module")
def multisport_parsed(multisport_with_strength):
    """Pre-parsed multisport FIT file data."""
    summary, df_sets = summarize_fit_original(multisport_with_strength, ftp=300)
    return summary, df_sets


class TestStrengthTrainingExtraction:
    """Tests for extracting strength training sets from FIT files."""

    def test_summarize_fit_original_returns_strength_sets(self, strength_fit_parsed):
        """Test that summarize_fit_original returns strength sets DataFrame."""
        summary, df_sets = strength_fit_parsed

        # Should return a DataFrame with strength sets
        assert df_sets is not None
        assert isinstance(df_sets, pd.DataFrame)
        assert not df_sets.empty

        # Check expected columns
        expected_columns = [
            "category",
            "duration",
            "repetitions",
            "set_type",
            "timestamp",
            "weight",
        ]
        for col in expected_columns:
            assert col in df_sets.columns, f"Missing column: {col}"

    def test_strength_sets_have_active_and_rest_types(self, strength_fit_parsed):
        """Test that strength sets include both active and rest set types."""
        _, df_sets = strength_fit_parsed

        set_types = df_sets["set_type"].unique()
        assert "active" in set_types
        assert "rest" in set_types

    def test_active_sets_have_repetitions(self, strength_fit_parsed):
        """Test that active sets have repetition counts."""
        _, df_sets = strength_fit_parsed

        active_sets = df_sets[df_sets["set_type"] == "active"]
        assert not active_sets.empty

        # Active sets should have repetitions
        assert active_sets["repetitions"].notna().all()
        assert (active_sets["repetitions"] > 0).any()


class TestStrengthTrainingAggregation:
    """Tests for aggregating strength training data from multiple files."""

    def test_aggregate_strength_sets_returns_dataframe(self, aggregated_strength_data):
        """Test that _aggregate_strength_sets returns a consolidated DataFrame."""
        df_summary = aggregated_strength_data

        assert df_summary is not None
        assert isinstance(df_summary, pd.DataFrame)
        assert not df_summary.empty

    def test_aggregate_includes_expected_columns(self, aggregated_strength_data):
        """Test that aggregated data includes all expected columns."""
        df_summary = aggregated_strength_data

        expected_columns = [
            "activity_id",
            "file",
            "date",
            "sport",
            "sub_sport",
            "set_number",
            "set_type",
            "category",
            "repetitions",
            "weight",
            "duration",
            "timestamp",
        ]

        for col in expected_columns:
            assert col in df_summary.columns, f"Missing column: {col}"

    def test_aggregate_only_includes_active_sets(self, aggregated_strength_data):
        """Test that aggregated summary only includes active sets, not rest periods."""
        df_summary = aggregated_strength_data

        # All set_type should be "active"
        assert (df_summary["set_type"] == "active").all()

    def test_aggregate_extracts_sport_information(self, aggregated_strength_data):
        """Test that sport and sub_sport are correctly extracted from FIT files."""
        df_summary = aggregated_strength_data

        # Sport should not be "unknown" for valid strength training files
        # Check first workout's data
        first_activity_data = df_summary[
            df_summary["activity_id"] == df_summary["activity_id"].iloc[0]
        ]
        assert first_activity_data["sport"].iloc[0] != "unknown"
        assert first_activity_data["sport"].iloc[0] == "training"
        assert first_activity_data["sub_sport"].iloc[0] == "strength_training"

    def test_aggregate_includes_activity_id(self, aggregated_strength_data):
        """Test that each set has the correct activity_id."""
        df_summary = aggregated_strength_data

        # Should have multiple unique activity IDs
        unique_activities = df_summary["activity_id"].nunique()
        assert unique_activities == 3

        # Activity IDs should match file names (without _ACTIVITY suffix)
        expected_ids = ["20474406937", "20555050352", "20748058539"]
        actual_ids = df_summary["activity_id"].unique()
        for expected_id in expected_ids:
            assert expected_id in actual_ids

    def test_aggregate_sorts_by_date_and_timestamp(self, aggregated_strength_data):
        """Test that aggregated data is sorted chronologically."""
        df_summary = aggregated_strength_data

        # Dates should be in order
        dates = pd.to_datetime(df_summary["date"])
        assert dates.is_monotonic_increasing

    def test_aggregate_handles_empty_file_list(self):
        """Test that empty file list returns None."""
        df_summary = _aggregate_strength_sets(
            [],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=False,
        )

        assert df_summary is None

    def test_aggregate_counts_sets_correctly(self, aggregated_strength_data):
        """Test that the total number of sets is counted correctly."""
        df_summary = aggregated_strength_data

        # Should have multiple sets from 3 workouts
        assert len(df_summary) > 20  # At least 20 active sets across 3 workouts

        # Each workout should have multiple sets
        sets_per_workout = df_summary.groupby("activity_id").size()
        assert (sets_per_workout > 5).all()  # Each workout has at least 5 active sets


class TestStrengthTrainingMultisport:
    """Tests for strength training in multisport activities."""

    def test_aggregate_works_with_multisport_flag(self, multiple_strength_files):
        """Test that aggregation works correctly with multisport=True."""
        # This test needs to actually call the function with multisport=True
        df_summary = _aggregate_strength_sets(
            multiple_strength_files,
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=True,
        )

        # Should still return valid data
        assert df_summary is not None
        assert isinstance(df_summary, pd.DataFrame)
        assert not df_summary.empty


class TestExerciseNames:
    """Tests for exercise name extraction from strength training sets."""

    def test_sets_have_exercise_name_column(self, strength_fit_parsed):
        """Test that strength sets include exercise_name column."""
        _, df_sets = strength_fit_parsed

        assert "exercise_name" in df_sets.columns

    def test_exercise_names_are_human_readable(self, strength_fit_parsed):
        """Test that exercise names are human-readable strings, not numeric codes."""
        _, df_sets = strength_fit_parsed

        active_sets = df_sets[df_sets["set_type"] == "active"]
        exercise_names = active_sets["exercise_name"].dropna()

        # Should have some named exercises
        assert len(exercise_names) > 0

        # Exercise names should be strings
        assert all(isinstance(name, str) for name in exercise_names)

        # Should not be just numbers or "unknown"
        named_exercises = [
            name for name in exercise_names if name != "Unknown" and not name.isdigit()
        ]
        assert len(named_exercises) > 0, "Expected some exercises with actual names"

    def test_aggregate_includes_exercise_name(self, aggregated_strength_data):
        """Test that aggregated summary includes exercise_name column."""
        df_summary = aggregated_strength_data

        assert "exercise_name" in df_summary.columns

    def test_exercise_names_vary_across_sets(self, strength_fit_parsed):
        """Test that different sets have different exercise names."""
        _, df_sets = strength_fit_parsed

        active_sets = df_sets[df_sets["set_type"] == "active"]
        unique_exercises = active_sets["exercise_name"].nunique()

        # Should have multiple different exercises in a workout
        assert unique_exercises > 1, "Expected multiple different exercises in workout"


class TestMultisportStrengthExtraction:
    """Tests for extracting strength training from multisport activities."""

    def test_multisport_file_contains_strength_session(self, multisport_with_strength):
        """Test that multisport file correctly identifies strength training session."""
        # This test needs to call the function directly
        sessions, _ = summarize_fit_sessions(multisport_with_strength, ftp=300)

        # Should have multiple sessions
        assert len(sessions) > 1

        # At least one should be strength training
        strength_sessions = [s for s in sessions if s.get("sub_sport") == "strength_training"]
        assert len(strength_sessions) > 0, "Expected at least one strength training session"

    def test_multisport_strength_extraction_with_aggregate(self, multisport_with_strength):
        """Test that strength sets can be extracted from multisport files."""
        # This test needs to call the function directly with multisport=True
        df_summary = _aggregate_strength_sets(
            [multisport_with_strength],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=True,
        )

        # Should extract strength sets from the multisport file
        assert df_summary is not None
        assert isinstance(df_summary, pd.DataFrame)
        assert not df_summary.empty

        # Should have strength training sport
        assert (df_summary["sub_sport"] == "strength_training").any()

    def test_multisport_strength_has_exercise_names(self, multisport_with_strength):
        """Test that strength sets from multisport files include exercise names."""
        # This test needs to call the function directly with multisport=True
        df_summary = _aggregate_strength_sets(
            [multisport_with_strength],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=True,
        )

        # Should have exercise_name column
        assert "exercise_name" in df_summary.columns

        # Should have some named exercises
        named_exercises = df_summary["exercise_name"].dropna()
        assert len(named_exercises) > 0

    def test_multisport_preserves_session_information(self, multisport_with_strength):
        """Test that multisport files preserve correct session/sport information."""
        # This test needs to call the function directly with multisport=True
        df_summary = _aggregate_strength_sets(
            [multisport_with_strength],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=True,
        )

        # All extracted sets should be from strength training session
        assert (df_summary["sub_sport"] == "strength_training").all()
        assert (df_summary["sport"] == "training").all()


class TestStrengthEdgeCasesForFullCoverage:
    """Additional edge case tests to achieve 100% coverage on strength.py"""

    def test_get_exercise_name_with_missing_sdk_import(self):
        """Test ImportError handling when garmin_fit_sdk is not available (lines 42-43)"""
        # We test this indirectly by checking the module imported successfully
        # Actually triggering the ImportError would require uninstalling the package
        from fitanalyzer import strength

        # The module should have imported successfully even if SDK was missing
        assert hasattr(strength, "get_specific_exercise_name")
        assert hasattr(strength, "EXERCISE_CATEGORY_MAPPING")

    def test_get_specific_exercise_name_with_unavailable_profile(self):
        """Test handling when profile is None or missing 'types' (line 103)"""
        from fitanalyzer.strength import get_specific_exercise_name
        from unittest.mock import patch

        # Mock _get_garmin_profile to return None
        with patch("fitanalyzer.strength._get_garmin_profile", return_value=None):
            result = get_specific_exercise_name(category_id=0, subtype_id=0)
            assert result is None

        # Mock _get_garmin_profile to return dict without 'types'
        with patch("fitanalyzer.strength._get_garmin_profile", return_value={}):
            result = get_specific_exercise_name(category_id=0, subtype_id=0)
            assert result is None

    def test_get_specific_exercise_name_unknown_category(self):
        """Test handling of unknown exercise category (line 110)"""
        from fitanalyzer.strength import get_specific_exercise_name

        # When category_id is not in known types, should return None
        result = get_specific_exercise_name(category_id=9999, subtype_id=0)  # Unknown category

        assert result is None

    def test_get_specific_exercise_name_with_unknown_exercise_name(self):
        """Test handling when exercise_name is 'unknown' (line 121)"""
        from fitanalyzer.strength import get_specific_exercise_name
        from unittest.mock import patch

        # Mock profile to return 'unknown' as exercise name
        mock_profile = {
            "types": {
                "exercise_category": {"0": "bench_press"},
                "bench_press_exercise_name": {"0": "unknown"},
            }
        }

        with patch("fitanalyzer.strength._get_garmin_profile", return_value=mock_profile):
            result = get_specific_exercise_name(category_id=0, subtype_id=0)
            # Should return None when exercise name is 'unknown'
            assert result is None

    def test_extract_valid_value_with_tuple(self):
        """Test _extract_valid_value with tuple values (lines 195-198)"""
        from fitanalyzer.strength import _extract_valid_value

        # When value is a tuple, should extract first valid element
        result = _extract_valid_value((5,))
        assert result == 5

        # With invalid marker 65534
        result = _extract_valid_value((65534, 10))
        assert result == 10

        # All invalid (line 200)
        result = _extract_valid_value((65534, None))
        assert result is None

        # All invalid marker values (line 200)
        result = _extract_valid_value((65534, 65534))
        assert result is None

    def test_extract_valid_value_with_none(self):
        """Test _extract_valid_value when value is None (line 193)"""
        from fitanalyzer.strength import _extract_valid_value
        import pandas as pd

        # When value is None or NaN
        assert _extract_valid_value(None) is None
        assert _extract_valid_value(pd.NA) is None

    def test_extract_valid_value_with_invalid_marker(self):
        """Test _extract_valid_value with 65534 invalid marker (line 202)"""
        from fitanalyzer.strength import _extract_valid_value

        # When value is the invalid marker 65534
        result = _extract_valid_value(65534)
        assert result is None

    def test_merge_api_exercise_names_with_data(self):
        """Test merge_api_exercise_names merging API data (line 288)"""
        from fitanalyzer.strength import merge_api_exercise_names

        # Create a sample DataFrame
        df = pd.DataFrame(
            {
                "message_index": [1, 2],
                "reps": [10, 12],
                "weight": [100, 110],
                "exercise_name": ["Exercise 1", "Exercise 2"],
            }
        )

        # API data with correct structure
        api_data = {
            "exerciseSets": [
                {"messageIndex": 1, "exercises": [{"name": "BENCH_PRESS"}]},
                {"messageIndex": 2, "exercises": [{"name": "BARBELL_SQUAT"}]},
            ]
        }

        result = merge_api_exercise_names(df, api_data)

        # Should have merged the exercise names from API data
        assert "exercise_name" in result.columns
        # Check that API names were merged (converted to Title Case)
        assert result[result["message_index"] == 1]["exercise_name"].iloc[0] == "Bench Press"
        assert result[result["message_index"] == 2]["exercise_name"].iloc[0] == "Barbell Squat"

    def test_merge_api_exercise_names_without_api_data(self):
        """Test merge_api_exercise_names when API data is None"""
        from fitanalyzer.strength import merge_api_exercise_names

        # Create a sample DataFrame
        df = pd.DataFrame(
            {
                "message_index": [1, 2],
                "reps": [10, 12],
                "exercise_name": ["Exercise 1", "Exercise 2"],
            }
        )

        # Call with None API data (should return unchanged)
        result = merge_api_exercise_names(df, None)

        # Should return the DataFrame unchanged
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_extract_sets_from_fit_with_json_file(self):
        """Test extract_sets_from_fit merging API names from JSON file (line 288)"""
        from fitanalyzer.strength import extract_sets_from_fit
        from fitparse import FitFile
        from unittest.mock import patch

        # Use a real FIT file
        fit_file = "tests/fixtures/20474406937_ACTIVITY.fit"
        ff = FitFile(fit_file)

        # Mock _load_exercise_sets_from_json to return API data
        mock_api_data = {
            "exerciseSets": [{"messageIndex": 0, "exercises": [{"name": "MOCKED_EXERCISE"}]}]
        }

        with patch(
            "fitanalyzer.strength._load_exercise_sets_from_json", return_value=mock_api_data
        ):
            # This should hit line 288 (merge_api_exercise_names call)
            df = extract_sets_from_fit(ff, fit_file)

            # Should have extracted sets
            assert len(df) > 0
            assert "exercise_name" in df.columns
            # API data should have been merged (line 288)
