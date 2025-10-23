"""Tests for strength training summary CSV generation."""

import pytest
import pandas as pd
from pathlib import Path
from fitanalyzer.parser import (
    main_with_args,
    parse_arguments,
    _aggregate_strength_sets,
    AnalysisConfig,
)


@pytest.fixture
def multiple_strength_files():
    """Paths to multiple strength training FIT files."""
    return [
        "tests/fixtures/20474406937_ACTIVITY.fit",
        "tests/fixtures/20555050352_ACTIVITY.fit",
        "tests/fixtures/20748058539_ACTIVITY.fit",
    ]


@pytest.fixture
def multisport_with_strength():
    """Path to multisport activity with strength training session."""
    return "tests/fixtures/20744294788_ACTIVITY.fit"


class TestMultisportStrengthExtraction:
    """Test that strength training from multisport activities has correct sport metadata."""

    def test_multisport_strength_autodetects_and_uses_correct_sport(self, multisport_with_strength):
        """Test that strength sets from multisport activity automatically get correct sport.

        Multisport activities should be detected automatically - no flag needed.
        When extracting strength sets, the sport/subsport should come from the
        strength training session, not from the first session (which might be cycling).
        """

        df_summary = _aggregate_strength_sets(
            [multisport_with_strength],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=False,
        )

        assert df_summary is not None
        assert not df_summary.empty

        # ALL sets should be from strength training session, NOT cycling
        unique_sports = df_summary["sport"].unique()
        unique_subsports = df_summary["sub_sport"].unique()

        assert "cycling" not in unique_sports, (
            f"Bug: Strength sets incorrectly labeled as cycling (from first session). "
            f"Sports found: {unique_sports}. Should auto-detect and use strength session."
        )
        assert "indoor_cycling" not in unique_subsports, (
            f"Bug: Strength sets incorrectly labeled as indoor_cycling (from first session). "
            f"Sub-sports found: {unique_subsports}. Should auto-detect and use strength session."
        )

        # Should be training/strength_training
        assert "training" in unique_sports or "strength_training" in unique_subsports, (
            f"Strength sets should have training or strength_training. "
            f"Found: sport={unique_sports}, sub_sport={unique_subsports}"
        )

    def test_multisport_strength_all_sets_same_sport(self, multisport_with_strength):
        """Test that all extracted sets have consistent sport metadata."""
        df_summary = _aggregate_strength_sets(
            [multisport_with_strength],
            AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="Europe/Helsinki"),
            multisport=False,
        )

        # All sets should have the same sport/subsport (no mixing)
        assert (
            df_summary["sport"].nunique() == 1
        ), f"Expected all sets to have same sport, but found: {df_summary['sport'].unique()}"
        assert (
            df_summary["sub_sport"].nunique() == 1
        ), f"Expected all sets to have same sub_sport, but found: {df_summary['sub_sport'].unique()}"


class TestStrengthSummaryGeneration:
    """Test generating consolidated strength training CSV from multiple FIT files."""

    def test_generates_summary_for_all_fit_files_in_directory(
        self, tmp_path, multiple_strength_files
    ):
        """Test that the script generates a summary for all FIT files in directory."""
        import os

        original_dir = os.getcwd()

        # Convert to absolute paths before changing directory
        abs_files = [str(Path(original_dir) / f) for f in multiple_strength_files]

        os.chdir(tmp_path)

        try:
            args = parse_arguments(
                [*abs_files, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )

            main_with_args(args)

            # Check that strength summary file was created
            summary_file = tmp_path / "strength_training_summary.csv"
            assert summary_file.exists(), "strength_training_summary.csv was not created"

            # Load and check it has data from all files
            df = pd.read_csv(summary_file)

            # Should have data from all 3 files
            unique_files = df["file"].nunique()
            assert unique_files == 3, f"Expected 3 unique files, got {unique_files}"

        finally:
            os.chdir(original_dir)

    def test_summary_not_overwritten_by_last_file(self, tmp_path, multiple_strength_files):
        """Test that each file's data is accumulated, not overwritten."""
        import os

        original_dir = os.getcwd()

        # Convert to absolute paths before changing directory
        abs_files = [str(Path(original_dir) / f) for f in multiple_strength_files]

        os.chdir(tmp_path)

        try:
            args = parse_arguments(
                [*abs_files, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )

            main_with_args(args)

            summary_file = tmp_path / "strength_training_summary.csv"
            df = pd.read_csv(summary_file)

            # Count sets per activity - each should contribute
            sets_per_activity = df.groupby("activity_id").size()

            # All three activities should have sets in the summary
            assert len(sets_per_activity) == 3, (
                f"Only {len(sets_per_activity)} activities in summary, expected 3. "
                f"This suggests data is being overwritten instead of accumulated."
            )

            # Each activity should have a reasonable number of sets
            for activity_id, count in sets_per_activity.items():
                assert count > 0, f"Activity {activity_id} has no sets in summary"

        finally:
            os.chdir(original_dir)

    def test_summary_sorted_by_date_and_time(self, tmp_path, multiple_strength_files):
        """Test that the summary CSV is sorted chronologically."""
        import os

        original_dir = os.getcwd()

        # Convert to absolute paths before changing directory
        abs_files = [str(Path(original_dir) / f) for f in multiple_strength_files]

        os.chdir(tmp_path)

        try:
            args = parse_arguments(
                [*abs_files, "--ftp", "300", "--dump-sets", "--output-dir", str(tmp_path)]
            )

            main_with_args(args)

            summary_file = tmp_path / "strength_training_summary.csv"
            df = pd.read_csv(summary_file)

            # Convert date and timestamp to comparable format
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Check if sorted (allowing for NaT values at end)
            timestamps = df["timestamp"].dropna()
            assert timestamps.is_monotonic_increasing, "Summary should be sorted by timestamp"

        finally:
            os.chdir(original_dir)
