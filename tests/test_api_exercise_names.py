"""Tests for fetching and merging exercise names from Garmin API."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fitanalyzer.sync import (
    fetch_exercise_sets_from_api,
    save_exercise_sets_to_json,
    load_exercise_sets_from_json,
)
from fitanalyzer.parser import merge_api_exercise_names


@pytest.fixture
def mock_garth():
    """Mock garth API client."""
    with patch('fitanalyzer.sync.garth') as mock:
        yield mock


@pytest.fixture
def sample_api_exercise_sets():
    """Sample exercise sets data from Garmin API."""
    return {
        "activityId": 20744294788,
        "exerciseSets": [
            {
                "exercises": [
                    {
                        "category": "CORE",
                        "name": "WEIGHTED_GHD_BACK_EXTENSIONS",
                        "probability": 100.0
                    }
                ],
                "duration": 27.781,
                "repetitionCount": 12,
                "weight": 15000.0,  # in grams
                "setType": "ACTIVE",
                "startTime": "2025-10-20T11:22:17.0",
                "messageIndex": 80
            },
            {
                "exercises": [
                    {
                        "category": "TRICEPS_EXTENSION",
                        "name": "DUMBBELL_KICKBACK",
                        "probability": 100.0
                    }
                ],
                "duration": 50.869,
                "repetitionCount": 12,
                "weight": 0.0,
                "setType": "ACTIVE",
                "messageIndex": 48
            }
        ]
    }


@pytest.fixture
def sample_multisport_metadata():
    """Sample metadata for a multisport activity with child IDs."""
    return {
        "activityId": 20744294802,
        "metadataDTO": {
            "childIds": [20744294782, 20744294788],
            "isOriginal": True
        }
    }


class TestFetchExerciseSetsFromAPI:
    """Tests for fetching exercise sets from Garmin Connect API."""

    def test_fetch_exercise_sets_for_regular_activity(self, mock_garth, sample_api_exercise_sets):
        """Test fetching exercise sets for a regular (non-multisport) activity."""
        activity_id = 20744294788

        # Mock API responses
        mock_garth.connectapi.side_effect = [
            {"activityId": activity_id, "metadataDTO": {}},  # No child IDs
            sample_api_exercise_sets  # Exercise sets
        ]

        result = fetch_exercise_sets_from_api(activity_id)

        assert result is not None
        assert result["activityId"] == activity_id
        assert "exerciseSets" in result
        assert len(result["exerciseSets"]) == 2

        # Verify API calls
        assert mock_garth.connectapi.call_count == 2
        mock_garth.connectapi.assert_any_call(f'/activity-service/activity/{activity_id}')
        mock_garth.connectapi.assert_any_call(f'/activity-service/activity/{activity_id}/exerciseSets')

    def test_fetch_exercise_sets_for_multisport_activity(self, mock_garth, sample_multisport_metadata, sample_api_exercise_sets):
        """Test fetching exercise sets from multisport activity's child activities."""
        parent_id = 20744294802
        child_id = 20744294788

        # Mock API responses
        mock_garth.connectapi.side_effect = [
            sample_multisport_metadata,  # Parent with child IDs
            {"activityId": 20744294782, "exerciseSets": None},  # Cycling child (no sets)
            sample_api_exercise_sets  # Strength training child
        ]

        result = fetch_exercise_sets_from_api(parent_id)

        assert result is not None
        assert result["activityId"] == child_id  # Should return the child with exercise sets
        assert len(result["exerciseSets"]) == 2

        # Verify it tried to fetch from child activities
        assert mock_garth.connectapi.call_count == 3

    def test_fetch_returns_none_when_no_exercise_sets(self, mock_garth):
        """Test that function returns None when activity has no exercise sets."""
        activity_id = 12345

        mock_garth.connectapi.side_effect = [
            {"activityId": activity_id, "metadataDTO": {}},
            {"activityId": activity_id, "exerciseSets": None}
        ]

        result = fetch_exercise_sets_from_api(activity_id)

        assert result is None

    def test_fetch_handles_api_errors_gracefully(self, mock_garth):
        """Test that API errors are handled gracefully."""
        activity_id = 12345

        # Simulate API error with KeyError (simpler than GarthHTTPError)
        mock_garth.connectapi.side_effect = KeyError("API Error")

        result = fetch_exercise_sets_from_api(activity_id)

        assert result is None


class TestCacheExerciseSets:
    """Tests for caching exercise sets to/from JSON files."""

    def test_save_exercise_sets_to_json(self, tmp_path, sample_api_exercise_sets):
        """Test saving exercise sets to JSON file."""
        fit_file = tmp_path / "20744294788_ACTIVITY.fit"
        fit_file.touch()

        save_exercise_sets_to_json(str(fit_file), sample_api_exercise_sets)

        json_file = tmp_path / "20744294788_ACTIVITY_exercises.json"
        assert json_file.exists()

        # Verify content
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data["activityId"] == 20744294788
        assert len(data["exerciseSets"]) == 2

    def test_load_exercise_sets_from_json(self, tmp_path, sample_api_exercise_sets):
        """Test loading exercise sets from JSON file."""
        fit_file = tmp_path / "20744294788_ACTIVITY.fit"
        json_file = tmp_path / "20744294788_ACTIVITY_exercises.json"

        # Create JSON file
        with open(json_file, 'w') as f:
            json.dump(sample_api_exercise_sets, f)

        result = load_exercise_sets_from_json(str(fit_file))

        assert result is not None
        assert result["activityId"] == 20744294788
        assert len(result["exerciseSets"]) == 2

    def test_load_returns_none_when_json_missing(self, tmp_path):
        """Test that load returns None when JSON file doesn't exist."""
        fit_file = tmp_path / "12345_ACTIVITY.fit"

        result = load_exercise_sets_from_json(str(fit_file))

        assert result is None

    def test_save_creates_directory_if_missing(self, tmp_path, sample_api_exercise_sets):
        """Test that save creates the directory if it doesn't exist."""
        fit_file = tmp_path / "subdir" / "20744294788_ACTIVITY.fit"

        save_exercise_sets_to_json(str(fit_file), sample_api_exercise_sets)

        json_file = tmp_path / "subdir" / "20744294788_ACTIVITY_exercises.json"
        assert json_file.exists()


class TestMergeAPIExerciseNames:
    """Tests for merging API exercise names with FIT data."""

    def test_merge_replaces_exercise_names_by_message_index(self):
        """Test that API exercise names are merged by matching messageIndex."""
        import pandas as pd

        # FIT data with category-based exercise names
        fit_df = pd.DataFrame({
            'message_index': [80, 48, 35],
            'exercise_name': ['Overhead Bulgarian Split Squat', 'Triceps Extension', 'Barbell Deadlift'],
            'category': [(17, 30, 7), (30, 7, 30), (8, 17, 28)],
            'category_subtype': [(None, 39, None), (None, None, 6), (0, None, None)],
            'repetitions': [12.0, 12.0, 10.0],
            'weight': [15.0, 0.0, 116.0]
        })

        # API data with correct exercise names
        api_data = {
            "activityId": 20744294788,
            "exerciseSets": [
                {
                    "exercises": [{"name": "WEIGHTED_GHD_BACK_EXTENSIONS", "category": "CORE"}],
                    "messageIndex": 80,
                    "repetitionCount": 12,
                    "weight": 15000.0
                },
                {
                    "exercises": [{"name": "DUMBBELL_KICKBACK", "category": "TRICEPS_EXTENSION"}],
                    "messageIndex": 48,
                    "repetitionCount": 12,
                    "weight": 0.0
                }
            ]
        }

        result_df = merge_api_exercise_names(fit_df, api_data)

        # Should replace exercise names for matching message indices
        assert result_df.loc[result_df['message_index'] == 80, 'exercise_name'].iloc[0] == 'Weighted Ghd Back Extensions'
        assert result_df.loc[result_df['message_index'] == 48, 'exercise_name'].iloc[0] == 'Dumbbell Kickback'
        # Should keep original for non-matching
        assert result_df.loc[result_df['message_index'] == 35, 'exercise_name'].iloc[0] == 'Barbell Deadlift'

    def test_merge_handles_missing_api_data(self):
        """Test that merge works when API data is None."""
        import pandas as pd

        fit_df = pd.DataFrame({
            'message_index': [1, 2],
            'exercise_name': ['Squat', 'Bench Press'],
            'repetitions': [10.0, 8.0]
        })

        result_df = merge_api_exercise_names(fit_df, None)

        # Should return original DataFrame unchanged
        assert result_df.equals(fit_df)

    def test_merge_handles_empty_exercise_sets(self):
        """Test that merge handles empty exerciseSets array."""
        import pandas as pd

        fit_df = pd.DataFrame({
            'message_index': [1],
            'exercise_name': ['Squat'],
            'repetitions': [10.0]
        })

        api_data = {"activityId": 12345, "exerciseSets": []}

        result_df = merge_api_exercise_names(fit_df, api_data)

        # Should return original DataFrame unchanged
        assert result_df.equals(fit_df)

    def test_merge_converts_api_name_format(self):
        """Test that API names are converted from UPPER_SNAKE to Title Case."""
        import pandas as pd

        fit_df = pd.DataFrame({
            'message_index': [10],
            'exercise_name': ['Unknown'],
            'repetitions': [8.0]
        })

        api_data = {
            "activityId": 12345,
            "exerciseSets": [
                {
                    "exercises": [{"name": "BARBELL_HANG_POWER_CLEAN", "category": "OLYMPIC_LIFT"}],
                    "messageIndex": 10
                }
            ]
        }

        result_df = merge_api_exercise_names(fit_df, api_data)

        assert result_df.loc[0, 'exercise_name'] == 'Barbell Hang Power Clean'

    def test_merge_handles_multiple_exercises_in_set(self):
        """Test handling sets with multiple possible exercises (takes first one)."""
        import pandas as pd

        fit_df = pd.DataFrame({
            'message_index': [5],
            'exercise_name': ['Unknown'],
            'repetitions': [12.0]
        })

        api_data = {
            "activityId": 12345,
            "exerciseSets": [
                {
                    "exercises": [
                        {"name": "EXERCISE_ONE", "probability": 60.0},
                        {"name": "EXERCISE_TWO", "probability": 40.0}
                    ],
                    "messageIndex": 5
                }
            ]
        }

        result_df = merge_api_exercise_names(fit_df, api_data)

        # Should use the first (highest probability) exercise
        assert result_df.loc[0, 'exercise_name'] == 'Exercise One'
