"""Tests for activity_processor module - activity processing and filtering."""

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, call, patch

from fitanalyzer.activity_processor import (
    ProcessorCallbacks,
    ProcessorContext,
    _parse_activity_date,
    filter_recent_activities,
    get_existing_activity_ids,
    identify_multisport_parents,
    process_activities,
)


class TestGetExistingActivityIds(unittest.TestCase):
    """Test get_existing_activity_ids function."""

    def setUp(self) -> None:
        """Create temporary directory for tests."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_directory(self) -> None:
        """Test with empty directory."""
        result = get_existing_activity_ids(self.test_dir)
        self.assertEqual(result, {})

    def test_single_activity_file(self) -> None:
        """Test with single activity FIT file."""
        activity_file = Path(self.test_dir) / "12345678901_ACTIVITY.fit"
        activity_file.write_bytes(b"test data")

        result = get_existing_activity_ids(self.test_dir)

        self.assertEqual(len(result), 1)
        self.assertIn("12345678901", result)
        self.assertIsInstance(result["12345678901"], float)

    def test_multiple_activity_files(self) -> None:
        """Test with multiple activity files."""
        files = ["12345678901_ACTIVITY.fit", "12345678902_ACTIVITY.fit", "12345678903_ACTIVITY.fit"]
        for filename in files:
            (Path(self.test_dir) / filename).write_bytes(b"test")

        result = get_existing_activity_ids(self.test_dir)

        self.assertEqual(len(result), 3)
        self.assertIn("12345678901", result)
        self.assertIn("12345678902", result)
        self.assertIn("12345678903", result)

    def test_ignores_non_activity_files(self) -> None:
        """Test that non-activity files are ignored."""
        (Path(self.test_dir) / "12345678901_ACTIVITY.fit").write_bytes(b"test")
        (Path(self.test_dir) / "readme.txt").write_bytes(b"readme")
        (Path(self.test_dir) / "data.json").write_bytes(b"json")

        result = get_existing_activity_ids(self.test_dir)

        self.assertEqual(len(result), 1)
        self.assertIn("12345678901", result)

    def test_ignores_invalid_activity_id(self) -> None:
        """Test that files with non-numeric IDs are ignored."""
        (Path(self.test_dir) / "12345678901_ACTIVITY.fit").write_bytes(b"test")
        (Path(self.test_dir) / "invalid_id_ACTIVITY.fit").write_bytes(b"test")
        (Path(self.test_dir) / "abc123_ACTIVITY.fit").write_bytes(b"test")

        result = get_existing_activity_ids(self.test_dir)

        self.assertEqual(len(result), 1)
        self.assertIn("12345678901", result)

    def test_modification_times_are_different(self) -> None:
        """Test that different files have different modification times."""
        import time

        file1 = Path(self.test_dir) / "12345678901_ACTIVITY.fit"
        file1.write_bytes(b"test1")
        time.sleep(0.01)
        file2 = Path(self.test_dir) / "12345678902_ACTIVITY.fit"
        file2.write_bytes(b"test2")

        result = get_existing_activity_ids(self.test_dir)

        # Both should exist with different mtimes
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result["12345678901"], result["12345678902"])


class TestParseActivityDate(unittest.TestCase):
    """Test _parse_activity_date function."""

    def test_utc_with_z_suffix(self) -> None:
        """Test parsing date with Z suffix."""
        activity = {"startTimeLocal": "2025-11-01T10:30:00Z"}
        result = _parse_activity_date(activity)

        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.year, 2025)
        self.assertEqual(result.month, 11)
        self.assertEqual(result.day, 1)

    def test_utc_with_offset(self) -> None:
        """Test parsing date with timezone offset."""
        activity = {"startTimeLocal": "2025-11-01T10:30:00+00:00"}
        result = _parse_activity_date(activity)

        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)

    def test_naive_datetime_becomes_aware(self) -> None:
        """Test that naive datetime is made timezone-aware."""
        activity = {"startTimeLocal": "2025-11-01T10:30:00"}
        result = _parse_activity_date(activity)

        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_with_milliseconds(self) -> None:
        """Test parsing date with milliseconds."""
        activity = {"startTimeLocal": "2025-11-01T10:30:00.123Z"}
        result = _parse_activity_date(activity)

        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)


class TestFilterRecentActivities(unittest.TestCase):
    """Test _filter_recent_activities function."""

    def test_empty_list(self) -> None:
        """Test with empty activity list."""
        result = filter_recent_activities([], days=7)
        self.assertEqual(result, [])

    def test_all_activities_within_range(self) -> None:
        """Test when all activities are within date range."""
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        activities = [
            {"activityId": 1, "startTimeLocal": yesterday.strftime("%Y-%m-%dT%H:%M:%SZ")},
            {
                "activityId": 2,
                "startTimeLocal": (yesterday - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        ]

        result = filter_recent_activities(activities, days=7)

        self.assertEqual(len(result), 2)

    def test_some_activities_outside_range(self) -> None:
        """Test filtering out old activities."""
        recent = datetime.now(timezone.utc) - timedelta(days=3)
        old = datetime.now(timezone.utc) - timedelta(days=10)

        activities = [
            {"activityId": 1, "startTimeLocal": recent.strftime("%Y-%m-%dT%H:%M:%SZ")},
            {"activityId": 2, "startTimeLocal": old.strftime("%Y-%m-%dT%H:%M:%SZ")},
        ]

        result = filter_recent_activities(activities, days=7)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["activityId"], 1)

    def test_activity_exactly_at_cutoff(self) -> None:
        """Test activity exactly at the cutoff date."""
        # Use 6 days 23 hours ago to ensure it's within 7-day window
        # even with minor timing differences between datetime.now() calls
        within_7_days = datetime.now(timezone.utc) - timedelta(days=6, hours=23)
        activities = [
            {"activityId": 1, "startTimeLocal": within_7_days.strftime("%Y-%m-%dT%H:%M:%SZ")},
        ]

        result = filter_recent_activities(activities, days=7)

        # Should be included (>=)
        self.assertEqual(len(result), 1)

    def test_different_days_parameter(self) -> None:
        """Test with different days parameter."""
        date_5_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
        date_15_days_ago = datetime.now(timezone.utc) - timedelta(days=15)

        activities = [
            {"activityId": 1, "startTimeLocal": date_5_days_ago.strftime("%Y-%m-%dT%H:%M:%SZ")},
            {"activityId": 2, "startTimeLocal": date_15_days_ago.strftime("%Y-%m-%dT%H:%M:%SZ")},
        ]

        # With days=7, only first should be included
        result = filter_recent_activities(activities, days=7)
        self.assertEqual(len(result), 1)

        # With days=30, both should be included
        result = filter_recent_activities(activities, days=30)
        self.assertEqual(len(result), 2)


class TestIdentifyMultisportParents(unittest.TestCase):
    """Test _identify_multisport_parents function."""

    def test_no_multisport_activities(self) -> None:
        """Test with no multisport activities."""
        activities = [
            {"activityId": 1, "activityName": "Run"},
            {"activityId": 2, "activityName": "Bike"},
        ]

        result = identify_multisport_parents(activities)

        self.assertEqual(result, set())

    def test_single_multisport_with_childids(self) -> None:
        """Test multisport activity with childIds."""
        activities = [
            {"activityId": 100, "activityName": "Multisport", "childIds": [101, 102]},
            {"activityId": 101, "activityName": "Swim"},
            {"activityId": 102, "activityName": "Bike"},
        ]

        result = identify_multisport_parents(activities)

        self.assertEqual(result, {100})

    def test_multisport_with_metadata_dto(self) -> None:
        """Test multisport with metadataDTO containing childIds."""
        activities = [
            {
                "activityId": 200,
                "activityName": "Triathlon",
                "metadataDTO": {"childIds": [201, 202, 203]},
            },
            {"activityId": 201, "activityName": "Swim"},
        ]

        result = identify_multisport_parents(activities)

        self.assertEqual(result, {200})

    def test_multiple_multisport_parents(self) -> None:
        """Test multiple multisport activities."""
        activities = [
            {"activityId": 100, "childIds": [101, 102]},
            {"activityId": 200, "childIds": [201]},
            {"activityId": 300, "activityName": "Solo run"},
        ]

        result = identify_multisport_parents(activities)

        self.assertEqual(result, {100, 200})

    def test_empty_child_ids_list(self) -> None:
        """Test activity with empty childIds list."""
        activities = [
            {"activityId": 100, "childIds": []},
            {"activityId": 101, "childIds": None},
        ]

        result = identify_multisport_parents(activities)

        self.assertEqual(result, set())


class TestProcessActivities(unittest.TestCase):
    """Test _process_activities function."""

    @patch("fitanalyzer.activity_processor._process_activity")
    def test_empty_activities_list(self, mock_process: Mock) -> None:
        """Test with empty activities list."""
        mock_should = Mock()
        mock_download = Mock()
        callbacks = ProcessorCallbacks(
            should_download_fn=mock_should,
            download_fn=mock_download,
        )
        context = ProcessorContext(
            existing_activities={},
            directory="/tmp/test",
            callbacks=callbacks,
        )
        counters, updated_files = process_activities(
            activities=[],
            context=context,
            parent_activity_ids=set(),
        )

        self.assertEqual(counters["new_count"], 0)
        self.assertEqual(updated_files, [])
        mock_process.assert_not_called()

    @patch("fitanalyzer.activity_processor._process_activity")
    def test_single_activity(self, mock_process: Mock) -> None:
        """Test processing single activity."""
        activities = [{"activityId": 123, "activityName": "Run"}]
        mock_should = Mock()
        mock_download = Mock()
        callbacks = ProcessorCallbacks(
            should_download_fn=mock_should,
            download_fn=mock_download,
        )
        context = ProcessorContext(
            existing_activities={},
            directory="/tmp/test",
            callbacks=callbacks,
        )

        process_activities(
            activities=activities,
            context=context,
            parent_activity_ids=set(),
        )

        mock_process.assert_called_once()

    @patch("fitanalyzer.activity_processor._process_activity")
    def test_skips_multisport_parents(self, mock_process: Mock) -> None:
        """Test that multisport parents are skipped."""
        activities = [
            {"activityId": 100, "activityName": "Multisport"},
            {"activityId": 101, "activityName": "Swim"},
            {"activityId": 102, "activityName": "Bike"},
        ]
        parent_ids = {100}  # Mark 100 as parent
        mock_should = Mock()
        mock_download = Mock()
        callbacks = ProcessorCallbacks(
            should_download_fn=mock_should,
            download_fn=mock_download,
        )
        context = ProcessorContext(
            existing_activities={},
            directory="/tmp/test",
            callbacks=callbacks,
        )

        counters, _ = process_activities(
            activities=activities,
            context=context,
            parent_activity_ids=parent_ids,
        )

        # Should process only 2 activities (skip parent)
        self.assertEqual(mock_process.call_count, 2)
        self.assertEqual(counters["skipped_count"], 1)  # Parent was skipped

    @patch("fitanalyzer.activity_processor._process_activity")
    def test_multiple_activities_with_no_parents(self, mock_process: Mock) -> None:
        """Test processing multiple activities without parents."""
        activities = [
            {"activityId": 1, "activityName": "Activity 1"},
            {"activityId": 2, "activityName": "Activity 2"},
            {"activityId": 3, "activityName": "Activity 3"},
        ]
        mock_should = Mock()
        mock_download = Mock()
        callbacks = ProcessorCallbacks(
            should_download_fn=mock_should,
            download_fn=mock_download,
        )
        context = ProcessorContext(
            existing_activities={},
            directory="/tmp/test",
            callbacks=callbacks,
        )

        process_activities(
            activities=activities,
            context=context,
            parent_activity_ids=set(),
        )

        self.assertEqual(mock_process.call_count, 3)

    @patch("fitanalyzer.activity_processor._process_activity")
    def test_counters_initialization(self, mock_process: Mock) -> None:
        """Test that counters are properly initialized."""
        mock_should = Mock()
        mock_download = Mock()
        callbacks = ProcessorCallbacks(
            should_download_fn=mock_should,
            download_fn=mock_download,
        )
        context = ProcessorContext(
            existing_activities={},
            directory="/tmp/test",
            callbacks=callbacks,
        )
        counters, _ = process_activities(
            activities=[],
            context=context,
            parent_activity_ids=set(),
        )

        self.assertIn("new_count", counters)
        self.assertIn("updated_count", counters)
        self.assertIn("api_updated_count", counters)
        self.assertIn("skipped_count", counters)
        self.assertEqual(counters["new_count"], 0)
        self.assertEqual(counters["updated_count"], 0)


if __name__ == "__main__":
    unittest.main()
