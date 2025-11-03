"""Test that individual sets have unique timestamps."""

from pathlib import Path

import pandas as pd
import pytest

from fitanalyzer.aggregation import aggregate_strength_sets
from fitanalyzer.config import AnalysisConfig


@pytest.fixture
def sample_fit_file():
    """Return path to a sample FIT file with multiple active sets."""
    # Use a file with multiple active sets (has 2 active + 1 rest = 3 total)
    return Path("data/samples/10344918253_ACTIVITY.fit")


def test_aggregate_sets_preserves_unique_timestamps(sample_fit_file):
    """Test that aggregate_strength_sets preserves unique timestamps for each set."""
    if not sample_fit_file.exists():
        pytest.skip(f"Sample file not found: {sample_fit_file}")

    config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="UTC")
    df = aggregate_strength_sets([str(sample_fit_file)], config)

    # Should have some sets
    assert df is not None and len(df) > 0, "No sets returned"

    # Should have timestamp column
    assert "timestamp" in df.columns, "timestamp column missing"

    # Timestamps should not all be the same (this is the bug we're fixing)
    unique_timestamps = df["timestamp"].nunique()
    assert unique_timestamps > 1, (
        f"All {len(df)} sets have the same timestamp! "
        f"Expected unique timestamps, got {unique_timestamps}. "
        f"Timestamps: {df['timestamp'].unique()}"
    )

    # Timestamps should be sequential (later sets have later timestamps)
    timestamps = pd.to_datetime(df["timestamp"])
    assert timestamps.is_monotonic_increasing, "Set timestamps should be in chronological order"


def test_csv_has_unique_timestamps_for_multiple_sets(sample_fit_file, tmp_path):
    """Test that the final CSV written to disk has unique timestamps for sets."""
    if not sample_fit_file.exists():
        pytest.skip(f"Sample file not found: {sample_fit_file}")

    config = AnalysisConfig(ftp=300, hr_rest=50, hr_max=190, tz_name="UTC")

    # Generate strength summary DataFrame
    df = aggregate_strength_sets([str(sample_fit_file)], config)

    # Write to CSV
    csv_path = tmp_path / "strength_training_summary.csv"
    df.to_csv(csv_path, index=False)

    # Read back from CSV and verify
    df_from_csv = pd.read_csv(csv_path)

    # Should have multiple sets
    assert len(df_from_csv) > 0, "No sets in CSV"

    # Timestamps should not all be the same
    unique_timestamps = df_from_csv["timestamp"].nunique()
    assert unique_timestamps > 1, (
        f"All {len(df_from_csv)} sets have the same timestamp in CSV! "
        f"Expected unique timestamps, got {unique_timestamps}. "
        f"Timestamps: {df_from_csv['timestamp'].unique()}"
    )
