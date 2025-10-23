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
        first_activity_data = df_summary[df_summary["activity_id"] == df_summary["activity_id"].iloc[0]]
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
