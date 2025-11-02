"""Garmin Connect API interaction and exercise data management.

This module handles all interactions with the Garmin Connect API for fetching
and managing exercise set data. It provides functions to:
- Fetch exercise sets from the API (with multisport support)
- Compare and update exercise data when changes are detected
- Handle API errors gracefully with appropriate fallbacks
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .garth_utils import GarthHTTPError, garth
from .strength import load_exercise_sets_from_json, save_exercise_sets_to_json

__all__ = [
    "fetch_exercise_sets_from_api",
    "_check_and_update_api_data",
    "_exercise_names_differ",
    "_fetch_exercise_sets_for_activity",
    "_get_child_activity_ids",
]


def _exercise_names_differ(
    existing_sets: List[Dict[str, Any]], fresh_sets: List[Dict[str, Any]]
) -> bool:
    """Check if exercise names differ between two sets.

    Args:
        existing_sets: List of existing exercise set dictionaries
        fresh_sets: List of fresh exercise set dictionaries from API

    Returns:
        True if any exercise names differ, False otherwise
    """
    for ex_set, fr_set in zip(existing_sets, fresh_sets):
        ex_exercises = ex_set.get("exercises", [{}])
        fr_exercises = fr_set.get("exercises", [{}])
        ex_name = ex_exercises[0].get("name") if ex_exercises else None
        fr_name = fr_exercises[0].get("name") if fr_exercises else None
        if ex_name != fr_name:
            return True
    return False


def _get_child_activity_ids(
    activity_details: Dict[str, Any] | List[Any],
) -> List[Any]:
    """Extract child activity IDs from activity details.

    Args:
        activity_details: Activity details from API (can be dict or list)

    Returns:
        List of child activity IDs, or empty list if none
    """
    if isinstance(activity_details, list):
        return []

    # Try direct childIds first (from activity list API)
    if "childIds" in activity_details:
        child_ids = activity_details.get("childIds", [])
        return child_ids if isinstance(child_ids, list) else []

    # Fall back to metadataDTO.childIds (from activity details API)
    metadata = activity_details.get("metadataDTO", {})
    child_ids_meta = metadata.get("childIds", [])
    return child_ids_meta if isinstance(child_ids_meta, list) else []


def _fetch_exercise_sets_for_activity(activity_id: int) -> Optional[Dict[str, Any]]:
    """Fetch exercise sets for a single activity ID.

    Args:
        activity_id: Garmin activity ID

    Returns:
        Dict with activityId and exerciseSets array, or None if not found
    """
    try:
        exercise_sets = garth.connectapi(f"/activity-service/activity/{activity_id}/exerciseSets")
        # Handle case where API might return unexpected types
        if isinstance(exercise_sets, dict) and exercise_sets.get("exerciseSets"):
            return exercise_sets
    except (GarthHTTPError, KeyError, TypeError, RuntimeError):
        pass
    return None


def fetch_exercise_sets_from_api(activity_id: int) -> Optional[Dict[str, Any]]:
    """Fetch exercise sets from Garmin Connect API for an activity.

    Retrieves detailed strength training exercise data from Garmin Connect,
    including manually edited exercise names, set counts, reps, and weight.
    This data is more accurate than FIT file data because it reflects user
    corrections made in the Garmin Connect interface.

    Handles both regular activities and multisport activities by checking
    child activity IDs. For multisport activities (e.g., triathlon), it
    searches child activities first since strength exercises are typically
    in a child segment.

    Args:
        activity_id: Garmin Connect activity ID (numeric identifier).
                     Can be found in the FIT filename or activity URL.

    Returns:
        Dictionary containing exercise sets data with structure:
        {
            "activityId": int,
            "exerciseSets": [
                {
                    "messageIndex": int,
                    "exercises": [
                        {
                            "name": str,  # e.g., "BARBELL_SQUAT"
                            "category": str,
                            "exerciseName": str
                        }
                    ],
                    "setCount": int,
                    "reps": float,
                    "weight": float,
                    ...
                }
            ]
        }
        Returns None if:
        - garth library is not available
        - Activity has no exercise sets (not a strength workout)
        - API returns an error
        - Network request fails

    Raises:
        Does not raise exceptions - errors are caught and logged to stderr.
        Returns None on any error condition.

    Example:
        >>> exercise_data = fetch_exercise_sets_from_api(20753039222)
        >>> if exercise_data:
        ...     num_sets = len(exercise_data['exerciseSets'])
        ...     print(f"Found {num_sets} exercise sets")
        Found 15 exercise sets

    Notes:
        - Requires active Garmin Connect authentication
        - For multisport activities, checks child activities first
        - Exercise names use Garmin's UPPER_SNAKE_CASE format
        - messageIndex links exercises to FIT file set records
        - Weight values in kilograms, reps as floating point
    """
    if garth is None:
        return None  # type: ignore[unreachable]

    try:
        # Get activity details to check for child activities (multisport)
        activity_details = garth.connectapi(f"/activity-service/activity/{activity_id}")
        child_ids = _get_child_activity_ids(activity_details) if activity_details else []

        # Try child activities first (for multisport)
        for child_id in child_ids:
            result = _fetch_exercise_sets_for_activity(child_id)
            if result:
                return result

        # Try the main activity if no children or no child had exercise sets
        return _fetch_exercise_sets_for_activity(activity_id)

    except (GarthHTTPError, KeyError, TypeError, RuntimeError) as e:
        print(f"      ⚠️  Error fetching exercise sets for {activity_id}: {e}")
        return None


def _get_update_reason(
    existing_data: Optional[Dict[str, Any]], fresh_data: Dict[str, Any]
) -> Optional[str]:
    """Determine if API data needs updating and return the reason.

    Args:
        existing_data: Current API data from JSON file, or None if no file exists
        fresh_data: Fresh API data from Garmin Connect

    Returns:
        String describing the reason for update, or None if no update needed
    """
    if not existing_data:
        return "no existing data"

    existing_sets = existing_data.get("exerciseSets", [])
    fresh_sets = fresh_data.get("exerciseSets", [])

    if len(existing_sets) != len(fresh_sets):
        return f"set count changed ({len(existing_sets)} → {len(fresh_sets)})"

    if _exercise_names_differ(existing_sets, fresh_sets):
        return "exercise names changed"

    if existing_sets != fresh_sets:
        return "set values changed (reps/weight/etc)"

    return None


def _check_and_update_api_data(activity_id: int, directory: str) -> bool:
    """Check if API exercise data needs updating and update if necessary.

    Args:
        activity_id: Activity ID to check
        directory: Directory containing FIT files

    Returns:
        True if data was updated, False otherwise
    """
    try:
        filename = Path(directory) / f"{activity_id}_ACTIVITY.fit"
        if not filename.exists():
            return False

        # Fetch fresh API data
        fresh_data = fetch_exercise_sets_from_api(activity_id)
        if not fresh_data:
            return False

        # Load existing API data
        existing_data = load_exercise_sets_from_json(str(filename))

        # Determine if update is needed and get reason
        update_reason = _get_update_reason(existing_data, fresh_data)

        if update_reason:
            save_exercise_sets_to_json(str(filename), fresh_data)
            print(f"      └─ Reason: {update_reason}")
            return True

        return False

    except (OSError, RuntimeError, ValueError) as e:
        print(f"      ⚠️  Error checking API data for {activity_id}: {e}")
        return False
