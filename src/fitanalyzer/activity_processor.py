"""Activity processing and filtering module.

This module handles activity processing, filtering by date,
identifying multisport parent activities, and orchestrating
the processing loop.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from fitanalyzer.garmin_api import check_and_update_api_data, get_child_activity_ids


@dataclass
class ProcessorCallbacks:
    """Callback functions for activity processing."""

    should_download_fn: Callable[..., Any]
    download_fn: Callable[..., Any]


@dataclass
class ProcessorContext:
    """Context for processing activities."""

    existing_activities: Dict[str, float]
    directory: str
    callbacks: ProcessorCallbacks


def get_existing_activity_ids(directory: str) -> Dict[str, float]:
    """Get dictionary of existing activity IDs with their modification times.

    Args:
        directory: Directory containing FIT files

    Returns:
        Dictionary mapping activity ID strings to modification times (as floats)
    """
    existing_activities: Dict[str, float] = {}
    activity_files = Path(directory).glob("*_ACTIVITY.fit")

    for fit_file in activity_files:
        # Extract activity ID from filename (e.g., "12345678901_ACTIVITY.fit")
        filename = fit_file.stem  # Gets filename without .fit extension
        activity_id_str = filename.replace("_ACTIVITY", "")

        # Only include files with numeric activity IDs
        if activity_id_str.isdigit():
            existing_activities[activity_id_str] = fit_file.stat().st_mtime

    return existing_activities


def _parse_activity_date(activity: Dict[str, Any]) -> datetime:
    """Parse activity date and ensure it's timezone-aware.

    Args:
        activity: Activity dictionary with startTimeLocal field

    Returns:
        Timezone-aware datetime object
    """
    activity_date = datetime.fromisoformat(activity["startTimeLocal"].replace("Z", "+00:00"))
    if activity_date.tzinfo is None:
        activity_date = activity_date.replace(tzinfo=timezone.utc)
    return activity_date


def filter_recent_activities(activities: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    """Filter activities to only include recent ones.

    Args:
        activities: List of activity dictionaries
        days: Number of days to look back

    Returns:
        Filtered list of activities within the date range
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_activities = []

    for activity in activities:
        activity_date = _parse_activity_date(activity)
        if activity_date >= cutoff_date:
            recent_activities.append(activity)

    return recent_activities


def _process_activity(
    activity: Dict[str, Any],
    context: ProcessorContext,
    counters: Dict[str, int],
    updated_files: Optional[List[str]] = None,
) -> None:
    """Process a single activity (download, update, or skip).

    Args:
        activity: Activity dictionary from Garmin API
        context: ProcessorContext with existing_activities, directory, and callbacks
        counters: Counter dictionary to update
        updated_files: Optional list to append updated filenames to
    """
    if updated_files is None:
        updated_files = []

    activity_id, activity_name, activity_date = (
        activity["activityId"],
        activity.get("activityName", "Unknown"),
        activity.get("startTimeLocal", ""),
    )

    should_download, is_update, check_api_anyway = context.callbacks.should_download_fn(
        activity, context.existing_activities
    )

    # Build file path once - used for both download success and API updates
    fit_path = str(Path(context.directory) / f"{activity_id}_ACTIVITY.fit")

    if should_download:
        if context.callbacks.download_fn(
            activity_id, activity_name, activity_date, context.directory
        ):
            updated_files.append(fit_path)
            counters["updated_count" if is_update else "new_count"] += 1
    elif check_api_anyway:
        # Even if we don't download the FIT file, check for API exercise data updates
        if check_and_update_api_data(activity_id, context.directory):
            counters["api_updated_count"] += 1
            updated_files.append(fit_path)
        else:
            counters["skipped_count"] += 1
    else:
        counters["skipped_count"] += 1


def identify_multisport_parents(activities: List[Dict[str, Any]]) -> Set[int]:
    """Identify parent multisport activities that should be skipped.

    Args:
        activities: List of activity dictionaries

    Returns:
        Set of activity IDs that are multisport parents
    """
    parent_activity_ids: Set[int] = set()

    for activity in activities:
        activity_id = activity["activityId"]
        child_ids = get_child_activity_ids(activity)

        if child_ids:
            parent_activity_ids.add(activity_id)

    return parent_activity_ids


def process_activities(
    activities: List[Dict[str, Any]],
    context: ProcessorContext,
    *,
    parent_activity_ids: Set[int],
) -> Tuple[Dict[str, int], List[str]]:
    """Process all activities, skipping multisport parents.

    Args:
        activities: List of activity dictionaries
        context: ProcessorContext with existing_activities, directory, and callbacks
        parent_activity_ids: Set of parent activity IDs to skip

    Returns:
        Tuple of (counters dict, list of updated filenames)
    """
    counters = {
        "new_count": 0,
        "updated_count": 0,
        "api_updated_count": 0,
        "skipped_count": 0,
    }
    updated_files: List[str] = []

    for activity in activities:
        activity_id = activity["activityId"]

        if activity_id in parent_activity_ids:
            counters["skipped_count"] += 1
            continue

        _process_activity(
            activity=activity,
            context=context,
            counters=counters,
            updated_files=updated_files,
        )

    return counters, updated_files
