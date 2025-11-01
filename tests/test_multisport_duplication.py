"""
TDD test to fix multisport activity duplication bug.

Bug: When downloading a multisport activity (e.g., bike + strength),
     the system downloads both child activities AND the parent activity,
     resulting in duplicate sessions in the summary.

Expected: Only child activities should be downloaded for multisport activities.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fitanalyzer.sync import _get_child_activity_ids, download_new_activities


class TestMultisportDuplication(unittest.TestCase):
    """Test that multisport activities don't create duplicates."""

    def test_get_child_activity_ids_extracts_child_ids(self):
        """Test that _get_child_activity_ids correctly extracts child IDs."""
        # Mock activity with child IDs (multisport)
        activity_with_children = {
            "activityId": 12345,
            "childIds": [12346, 12347],
            "activityType": {"typeKey": "multi_sport"},
        }

        child_ids = _get_child_activity_ids(activity_with_children)

        self.assertEqual(child_ids, [12346, 12347])
        self.assertEqual(len(child_ids), 2)

    def test_get_child_activity_ids_returns_empty_for_single_activity(self):
        """Test that _get_child_activity_ids returns empty list for single activities."""
        # Regular activity without children
        single_activity = {
            "activityId": 12345,
            "activityType": {"typeKey": "strength_training"},
        }

        child_ids = _get_child_activity_ids(single_activity)

        self.assertEqual(child_ids, [])

    @patch("fitanalyzer.sync.garth")
    @patch("fitanalyzer.sync.get_existing_activity_ids")
    @patch("fitanalyzer.sync._filter_recent_activities")
    def test_multisport_only_downloads_children_not_parent(
        self, mock_filter, mock_existing, mock_garth
    ):
        """
        FAILING TEST: Multisport parent should be skipped if children are present.

        When a multisport activity has child activities, we should:
        1. Download the child activities (e.g., cycling + strength)
        2. Skip the parent activity (which would duplicate the data)

        Current behavior: Downloads both parent AND children, causing duplicates.
        """
        # Mock: No existing activities
        mock_existing.return_value = {}

        # Mock: API returns multisport parent + its children
        parent_activity = {
            "activityId": 20744294782,
            "activityName": "Bike + Strength",
            "startTimeLocal": "2025-10-20T13:09:48.0",
            "updateDate": "2025-10-20T14:30:00.0",
            "childIds": [20744294788, 20744294802],  # Child activities
        }

        child1_cycling = {
            "activityId": 20744294788,
            "activityName": "Cycling",
            "startTimeLocal": "2025-10-20T13:09:48.0",
            "updateDate": "2025-10-20T13:19:50.0",
        }

        child2_strength = {
            "activityId": 20744294802,
            "activityName": "Strength Training",
            "startTimeLocal": "2025-10-20T13:21:30.0",
            "updateDate": "2025-10-20T14:26:00.0",
        }

        # API returns parent + children (this is what Garmin API does)
        all_activities = [parent_activity, child1_cycling, child2_strength]
        mock_filter.return_value = all_activities

        # Mock connectapi to return activity list
        mock_garth.connectapi.return_value = all_activities

        # Mock successful downloads
        with patch("fitanalyzer.sync._download_single_activity") as mock_download:
            mock_download.return_value = True

            # Run the download
            count, updated_files = download_new_activities(days=7, directory="/tmp/test")

            # ASSERTION: Should only download the 2 children, NOT the parent
            # This test will FAIL until we fix the bug
            self.assertEqual(
                mock_download.call_count,
                2,
                "Should only download 2 child activities, not the parent",
            )

            # Verify the parent (20744294782) was NOT downloaded
            downloaded_ids = [call[0][0] for call in mock_download.call_args_list]
            self.assertNotIn(
                "20744294782",
                downloaded_ids,
                "Parent multisport activity should be skipped",
            )

            # Verify the children WERE downloaded
            self.assertIn("20744294788", downloaded_ids, "Child 1 (cycling) should be downloaded")
            self.assertIn("20744294802", downloaded_ids, "Child 2 (strength) should be downloaded")


if __name__ == "__main__":
    unittest.main()
