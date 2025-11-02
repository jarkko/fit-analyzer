"""Activity download module - handles FIT file downloading and management from Garmin Connect."""

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .garmin_api import fetch_exercise_sets_from_api
from .garth_utils import GARTH_AVAILABLE, GarthHTTPError, garth
from .strength import save_exercise_sets_to_json

__all__ = [
    "GARTH_AVAILABLE",
    "GarthHTTPError",
    "should_download_activity",
    "download_single_activity",
    "print_download_summary",
]


def _extract_fit_from_zip(fit_data: bytes) -> Optional[bytes]:
    """Extract FIT file from ZIP if needed.

    Args:
        fit_data: Raw bytes from Garmin download

    Returns:
        FIT file bytes, or None if no FIT file found in ZIP
    """
    # Check if it's a ZIP file
    if fit_data[:2] != b"PK":  # Not a ZIP file
        return fit_data

    # Extract FIT file from ZIP
    with zipfile.ZipFile(io.BytesIO(fit_data)) as zip_file:
        # Get the first .fit file in the archive
        fit_files = [name for name in zip_file.namelist() if name.lower().endswith(".fit")]
        if fit_files:
            return zip_file.read(fit_files[0])

    return None


def should_download_activity(
    activity: Dict[str, Any], existing_activities: Dict[str, float]
) -> Tuple[bool, bool, bool]:
    """Determine if an activity should be downloaded.

    Args:
        activity: Activity dict from Garmin API
        existing_activities: Dict mapping activity_id (str) -> file modification timestamp (float)

    Returns:
        Tuple of (should_download, is_update, check_api_anyway):
        - should_download: True if activity should be downloaded
        - is_update: True if this is an update to existing activity
        - check_api_anyway: True if we should check API for exercise data updates
          even if not downloading
    """
    activity_id = str(activity["activityId"])

    # New activity - always download
    if activity_id not in existing_activities:
        return (True, False, False)

    # Existing activity - check if Garmin has a newer version
    garmin_timestamp = activity.get("updateDate") or activity.get("lastModified")
    if not garmin_timestamp:
        # No timestamp available, don't download but check API
        return (False, False, True)

    # Convert Garmin timestamp (milliseconds) to seconds
    garmin_time = garmin_timestamp / 1000.0
    local_time = existing_activities[activity_id]

    # If Garmin version is newer (with 1 second tolerance), download it
    if garmin_time > local_time + 1:
        return (True, True, False)

    # FIT file is up-to-date, but still check if exercise data was edited
    return (False, False, True)


def download_single_activity(
    activity_id: int,
    activity_name: str,
    activity_date: str,
    directory: str,
) -> bool:
    """Download a single activity FIT file and exercise data.

    Args:
        activity_id: Garmin activity ID
        activity_name: Name of the activity
        activity_date: Date string (YYYY-MM-DD)
        directory: Directory to save files

    Returns:
        True if download successful, False otherwise
    """
    if not GARTH_AVAILABLE:
        raise ImportError("garth library not available")

    try:
        # Download the activity FIT file
        fit_data = garth.download(f"/download-service/files/activity/{activity_id}")

        # Extract FIT file if it's in a ZIP
        fit_bytes = _extract_fit_from_zip(fit_data)
        if fit_bytes is None:
            print(f"   ❌ No .fit file found in download for: {activity_name} [ID: {activity_id}]")
            return False

        # Save FIT file
        fit_filename = Path(directory) / f"{activity_id}_ACTIVITY.fit"
        with open(fit_filename, "wb") as f:
            f.write(fit_bytes)

        print(f"   ✅ Downloaded: {activity_name} ({activity_date}) [ID: {activity_id}]")

        # Try to fetch exercise sets from API
        exercise_data = fetch_exercise_sets_from_api(activity_id)
        if exercise_data:
            save_exercise_sets_to_json(str(fit_filename), exercise_data)

        return True

    except (OSError, RuntimeError, ValueError) as e:
        print(f"   ❌ Error downloading {activity_name}: {e}")
        return False


def print_download_summary(counters: Dict[str, int]) -> None:
    """Print summary of download results.

    Args:
        counters: Dict with keys: new_count, updated_count, api_updated_count, skipped_count
    """
    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"New activities: {counters['new_count']}")
    print(f"Updated activities: {counters['updated_count']}")
    if counters["api_updated_count"] > 0:
        print(f"Exercise data updated: {counters['api_updated_count']}")
    print(f"Skipped (unchanged): {counters['skipped_count']}")
    print("=" * 50)
