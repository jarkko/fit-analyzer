"""Tests for garmin_api module - API interaction and exercise data management."""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, call, mock_open, patch

from fitanalyzer.garmin_api import (
    _exercise_names_differ,
    _fetch_exercise_sets_for_activity,
    check_and_update_api_data,
    fetch_exercise_sets_from_api,
    get_child_activity_ids,
)


class TestModuleImports(unittest.TestCase):
    """Test module-level imports and fallbacks."""

    def test_garth_http_error_is_defined(self) -> None:
        """Test that GarthHTTPError is always defined."""
        from fitanalyzer import garmin_api

        # Whether garth is available or not, GarthHTTPError should exist
        self.assertTrue(hasattr(garmin_api, "GarthHTTPError"))
        # It's either the real GarthHTTPError or Exception as fallback
        self.assertTrue(
            isinstance(garmin_api.GarthHTTPError, type)
            and issubclass(garmin_api.GarthHTTPError, Exception)
        )

    def test_module_exports_all_functions(self) -> None:
        """Test that all expected functions are exported."""
        from fitanalyzer import garmin_api

        expected = [
            "fetch_exercise_sets_from_api",
            "check_and_update_api_data",
            "_exercise_names_differ",
            "_fetch_exercise_sets_for_activity",
            "get_child_activity_ids",
        ]
        for func_name in expected:
            self.assertTrue(hasattr(garmin_api, func_name), f"Missing function: {func_name}")


class TestExerciseNamesDiffer(unittest.TestCase):
    """Test _exercise_names_differ function."""

    def test_identical_names(self) -> None:
        """Test that identical exercise names return False."""
        existing = [
            {"exercises": [{"name": "BARBELL_SQUAT"}]},
            {"exercises": [{"name": "BENCH_PRESS"}]},
        ]
        fresh = [
            {"exercises": [{"name": "BARBELL_SQUAT"}]},
            {"exercises": [{"name": "BENCH_PRESS"}]},
        ]
        self.assertFalse(_exercise_names_differ(existing, fresh))

    def test_different_names(self) -> None:
        """Test that different exercise names return True."""
        existing = [
            {"exercises": [{"name": "BARBELL_SQUAT"}]},
            {"exercises": [{"name": "BENCH_PRESS"}]},
        ]
        fresh = [
            {"exercises": [{"name": "BARBELL_SQUAT"}]},
            {"exercises": [{"name": "DEADLIFT"}]},  # Changed
        ]
        self.assertTrue(_exercise_names_differ(existing, fresh))

    def test_empty_exercises_list(self) -> None:
        """Test handling of empty exercises list."""
        existing = [{"exercises": []}]
        fresh = [{"exercises": []}]
        self.assertFalse(_exercise_names_differ(existing, fresh))

    def test_missing_exercises_key(self) -> None:
        """Test handling of missing exercises key."""
        existing = [{}]
        fresh = [{}]
        self.assertFalse(_exercise_names_differ(existing, fresh))

    def test_one_empty_one_with_exercise(self) -> None:
        """Test when one has exercise and other is empty."""
        existing = [{"exercises": [{"name": "SQUAT"}]}]
        fresh = [{"exercises": []}]
        self.assertTrue(_exercise_names_differ(existing, fresh))

    def test_missing_name_key(self) -> None:
        """Test handling of missing name key in exercise."""
        existing = [{"exercises": [{}]}]
        fresh = [{"exercises": [{"name": "SQUAT"}]}]
        self.assertTrue(_exercise_names_differ(existing, fresh))

    def test_empty_lists(self) -> None:
        """Test with completely empty lists."""
        self.assertFalse(_exercise_names_differ([], []))


class TestGetChildActivityIds(unittest.TestCase):
    """Test get_child_activity_ids function."""

    def test_list_input_returns_empty(self) -> None:
        """Test that list input returns empty list."""
        self.assertEqual(get_child_activity_ids([]), [])
        self.assertEqual(get_child_activity_ids([1, 2, 3]), [])

    def test_direct_child_ids(self) -> None:
        """Test extracting childIds from top level."""
        activity_details = {"childIds": [123, 456, 789]}
        self.assertEqual(get_child_activity_ids(activity_details), [123, 456, 789])

    def test_metadata_child_ids(self) -> None:
        """Test extracting childIds from metadataDTO."""
        activity_details = {"metadataDTO": {"childIds": [111, 222]}}
        self.assertEqual(get_child_activity_ids(activity_details), [111, 222])

    def test_direct_takes_precedence(self) -> None:
        """Test that direct childIds takes precedence over metadataDTO."""
        activity_details = {
            "childIds": [123, 456],
            "metadataDTO": {"childIds": [999]},
        }
        self.assertEqual(get_child_activity_ids(activity_details), [123, 456])

    def test_no_child_ids(self) -> None:
        """Test activity with no child IDs."""
        self.assertEqual(get_child_activity_ids({}), [])
        self.assertEqual(get_child_activity_ids({"metadataDTO": {}}), [])

    def test_non_list_child_ids(self) -> None:
        """Test handling of non-list childIds value."""
        activity_details = {"childIds": "not a list"}
        self.assertEqual(get_child_activity_ids(activity_details), [])

    def test_non_list_metadata_child_ids(self) -> None:
        """Test handling of non-list childIds in metadataDTO."""
        activity_details = {"metadataDTO": {"childIds": 123}}
        self.assertEqual(get_child_activity_ids(activity_details), [])


class TestFetchExerciseSetsForActivity(unittest.TestCase):
    """Test _fetch_exercise_sets_for_activity function."""

    @patch("fitanalyzer.garmin_api.garth")
    def test_successful_fetch(self, mock_garth: Mock) -> None:
        """Test successful fetch of exercise sets."""
        expected_data = {
            "activityId": 12345,
            "exerciseSets": [{"setCount": 3, "reps": 10}],
        }
        mock_garth.connectapi.return_value = expected_data

        result = _fetch_exercise_sets_for_activity(12345)

        self.assertEqual(result, expected_data)
        mock_garth.connectapi.assert_called_once_with(
            "/activity-service/activity/12345/exerciseSets"
        )

    @patch("fitanalyzer.garmin_api.garth")
    def test_empty_exercise_sets(self, mock_garth: Mock) -> None:
        """Test when API returns empty exerciseSets."""
        mock_garth.connectapi.return_value = {"activityId": 12345, "exerciseSets": []}
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_missing_exercise_sets_key(self, mock_garth: Mock) -> None:
        """Test when API returns dict without exerciseSets."""
        mock_garth.connectapi.return_value = {"activityId": 12345}
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_non_dict_response(self, mock_garth: Mock) -> None:
        """Test handling of non-dict API response."""
        mock_garth.connectapi.return_value = ["not", "a", "dict"]
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_garth_http_error(self, mock_garth: Mock) -> None:
        """Test handling of GarthHTTPError (using RuntimeError as proxy)."""
        mock_garth.connectapi.side_effect = RuntimeError("API error")
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_key_error(self, mock_garth: Mock) -> None:
        """Test handling of KeyError."""
        mock_garth.connectapi.side_effect = KeyError("missing key")
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.garth")
    def test_type_error(self, mock_garth: Mock) -> None:
        """Test handling of TypeError."""
        mock_garth.connectapi.side_effect = TypeError("type error")
        result = _fetch_exercise_sets_for_activity(12345)
        self.assertIsNone(result)


class TestFetchExerciseSetsFromApi(unittest.TestCase):
    """Test fetch_exercise_sets_from_api function."""

    @patch("fitanalyzer.garmin_api.garth", None)
    def test_garth_not_available(self) -> None:
        """Test when garth library is not available."""
        result = fetch_exercise_sets_from_api(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api._fetch_exercise_sets_for_activity")
    @patch("fitanalyzer.garmin_api.garth")
    def test_main_activity_has_sets(
        self, mock_garth: Mock, mock_fetch: Mock, mock_get_children: Mock
    ) -> None:
        """Test fetching from main activity without children."""
        activity_details = {"activityId": 12345}
        expected_data = {"activityId": 12345, "exerciseSets": [{"reps": 10}]}

        mock_garth.connectapi.return_value = activity_details
        mock_get_children.return_value = []
        mock_fetch.return_value = expected_data

        result = fetch_exercise_sets_from_api(12345)

        self.assertEqual(result, expected_data)
        mock_garth.connectapi.assert_called_once_with("/activity-service/activity/12345")
        mock_fetch.assert_called_once_with(12345)

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api._fetch_exercise_sets_for_activity")
    @patch("fitanalyzer.garmin_api.garth")
    def test_child_activity_has_sets(
        self, mock_garth: Mock, mock_fetch: Mock, mock_get_children: Mock
    ) -> None:
        """Test fetching from child activity in multisport."""
        activity_details = {"activityId": 12345, "childIds": [100, 200]}
        child_data = {"activityId": 100, "exerciseSets": [{"reps": 10}]}

        mock_garth.connectapi.return_value = activity_details
        mock_get_children.return_value = [100, 200]
        mock_fetch.side_effect = [child_data, None]  # First child has data

        result = fetch_exercise_sets_from_api(12345)

        self.assertEqual(result, child_data)
        mock_fetch.assert_called_once_with(100)  # Stops after first child with data

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api._fetch_exercise_sets_for_activity")
    @patch("fitanalyzer.garmin_api.garth")
    def test_fallback_to_main_after_empty_children(
        self, mock_garth: Mock, mock_fetch: Mock, mock_get_children: Mock
    ) -> None:
        """Test falling back to main activity when children have no sets."""
        activity_details = {"childIds": [100, 200]}
        main_data = {"activityId": 12345, "exerciseSets": [{"reps": 10}]}

        mock_garth.connectapi.return_value = activity_details
        mock_get_children.return_value = [100, 200]
        mock_fetch.side_effect = [None, None, main_data]  # Children empty, main has data

        result = fetch_exercise_sets_from_api(12345)

        self.assertEqual(result, main_data)
        self.assertEqual(mock_fetch.call_count, 3)
        mock_fetch.assert_has_calls([call(100), call(200), call(12345)])

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api.garth")
    def test_activity_details_api_error(self, mock_garth: Mock, mock_get_children: Mock) -> None:
        """Test handling of error fetching activity details (using RuntimeError)."""
        mock_garth.connectapi.side_effect = RuntimeError("API error")

        result = fetch_exercise_sets_from_api(12345)

        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api.garth")
    def test_key_error_in_activity_details(self, mock_garth: Mock, mock_get_children: Mock) -> None:
        """Test handling of KeyError."""
        mock_garth.connectapi.side_effect = KeyError("missing")
        result = fetch_exercise_sets_from_api(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api.garth")
    def test_type_error_in_activity_details(
        self, mock_garth: Mock, mock_get_children: Mock
    ) -> None:
        """Test handling of TypeError."""
        mock_garth.connectapi.side_effect = TypeError("type error")
        result = fetch_exercise_sets_from_api(12345)
        self.assertIsNone(result)

    @patch("fitanalyzer.garmin_api.get_child_activity_ids")
    @patch("fitanalyzer.garmin_api._fetch_exercise_sets_for_activity")
    @patch("fitanalyzer.garmin_api.garth")
    def test_none_activity_details(
        self, mock_garth: Mock, mock_fetch: Mock, mock_get_children: Mock
    ) -> None:
        """Test when activity details returns None."""
        mock_garth.connectapi.return_value = None
        mock_get_children.return_value = []
        mock_fetch.return_value = None

        result = fetch_exercise_sets_from_api(12345)

        self.assertIsNone(result)
        mock_get_children.assert_not_called()  # Should not try to get children from None


class TestCheckAndUpdateApiData(unittest.TestCase):
    """Test check_and_update_api_data function."""

    @patch("fitanalyzer.garmin_api.Path")
    def test_file_not_exists(self, mock_path: Mock) -> None:
        """Test when FIT file doesn't exist."""
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_path.return_value.__truediv__.return_value = mock_file

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    def test_no_fresh_data(self, mock_path: Mock, mock_fetch: Mock) -> None:
        """Test when API returns no fresh data."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file
        mock_fetch.return_value = None

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)

    @patch("fitanalyzer.garmin_api.save_exercise_sets_to_json")
    @patch("fitanalyzer.garmin_api.load_exercise_sets_from_json")
    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_no_existing_data(
        self,
        mock_print: Mock,
        mock_path: Mock,
        mock_fetch: Mock,
        mock_load: Mock,
        mock_save: Mock,
    ) -> None:
        """Test update when no existing data."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        fresh_data = {"exerciseSets": [{"reps": 10}]}
        mock_fetch.return_value = fresh_data
        mock_load.return_value = None

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertTrue(result)
        mock_save.assert_called_once()
        mock_print.assert_called_once_with("      └─ Reason: no existing data")

    @patch("fitanalyzer.garmin_api.save_exercise_sets_to_json")
    @patch("fitanalyzer.garmin_api.load_exercise_sets_from_json")
    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_set_count_changed(
        self,
        mock_print: Mock,
        mock_path: Mock,
        mock_fetch: Mock,
        mock_load: Mock,
        mock_save: Mock,
    ) -> None:
        """Test update when set count changes."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        existing_data = {"exerciseSets": [{"reps": 10}, {"reps": 12}]}
        fresh_data = {"exerciseSets": [{"reps": 10}]}

        mock_fetch.return_value = fresh_data
        mock_load.return_value = existing_data

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertTrue(result)
        mock_save.assert_called_once_with(str(mock_file), fresh_data)
        mock_print.assert_called_once_with("      └─ Reason: set count changed (2 → 1)")

    @patch("fitanalyzer.garmin_api._exercise_names_differ")
    @patch("fitanalyzer.garmin_api.save_exercise_sets_to_json")
    @patch("fitanalyzer.garmin_api.load_exercise_sets_from_json")
    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_exercise_names_changed(
        self,
        mock_print: Mock,
        mock_path: Mock,
        mock_fetch: Mock,
        mock_load: Mock,
        mock_save: Mock,
        mock_names_differ: Mock,
    ) -> None:
        """Test update when exercise names change."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        existing_data = {"exerciseSets": [{"reps": 10}]}
        fresh_data = {"exerciseSets": [{"reps": 10}]}

        mock_fetch.return_value = fresh_data
        mock_load.return_value = existing_data
        mock_names_differ.return_value = True

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertTrue(result)
        mock_print.assert_called_once_with("      └─ Reason: exercise names changed")

    @patch("fitanalyzer.garmin_api._exercise_names_differ")
    @patch("fitanalyzer.garmin_api.save_exercise_sets_to_json")
    @patch("fitanalyzer.garmin_api.load_exercise_sets_from_json")
    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_set_values_changed(
        self,
        mock_print: Mock,
        mock_path: Mock,
        mock_fetch: Mock,
        mock_load: Mock,
        mock_save: Mock,
        mock_names_differ: Mock,
    ) -> None:
        """Test update when set values (reps/weight) change."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        existing_data = {"exerciseSets": [{"reps": 10}]}
        fresh_data = {"exerciseSets": [{"reps": 12}]}  # Reps changed

        mock_fetch.return_value = fresh_data
        mock_load.return_value = existing_data
        mock_names_differ.return_value = False

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertTrue(result)
        mock_print.assert_called_once_with("      └─ Reason: set values changed (reps/weight/etc)")

    @patch("fitanalyzer.garmin_api._exercise_names_differ")
    @patch("fitanalyzer.garmin_api.load_exercise_sets_from_json")
    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    def test_no_changes_needed(
        self,
        mock_path: Mock,
        mock_fetch: Mock,
        mock_load: Mock,
        mock_names_differ: Mock,
    ) -> None:
        """Test when data hasn't changed."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file

        same_data = {"exerciseSets": [{"reps": 10}]}

        mock_fetch.return_value = same_data
        mock_load.return_value = same_data
        mock_names_differ.return_value = False

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_os_error(self, mock_print: Mock, mock_path: Mock, mock_fetch: Mock) -> None:
        """Test handling of OSError."""
        mock_file = Mock()
        mock_file.exists.side_effect = OSError("File error")
        mock_path.return_value.__truediv__.return_value = mock_file

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("Error checking API data", mock_print.call_args[0][0])

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_runtime_error(self, mock_print: Mock, mock_path: Mock, mock_fetch: Mock) -> None:
        """Test handling of RuntimeError."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file
        mock_fetch.side_effect = RuntimeError("Runtime error")

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)
        mock_print.assert_called_once()

    @patch("fitanalyzer.garmin_api.fetch_exercise_sets_from_api")
    @patch("fitanalyzer.garmin_api.Path")
    @patch("builtins.print")
    def test_value_error(self, mock_print: Mock, mock_path: Mock, mock_fetch: Mock) -> None:
        """Test handling of ValueError."""
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = mock_file
        mock_fetch.side_effect = ValueError("Value error")

        result = check_and_update_api_data(12345, "/test/dir")

        self.assertFalse(result)
        mock_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
