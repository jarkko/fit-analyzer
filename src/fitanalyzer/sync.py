#!/usr/bin/env python3
"""
Garmin Connect Auto-Sync Script.

Automatically downloads new activities from Garmin Connect and updates your workout summary.
"""

import argparse
import getpass
import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST, DEFAULT_SYNC_DAYS

__all__ = [
    "authenticate_garmin",
    "download_new_activities",
    "run_analysis",
    "fetch_exercise_sets_from_api",
    "save_exercise_sets_to_json",
    "load_exercise_sets_from_json",
    "main",
]

# Try to import garth at module level
try:
    import garth
    from garth.http import GarthHTTPError

    GARTH_AVAILABLE = True
except ImportError:
    garth = None
    GarthHTTPError = Exception  # Fallback type for type hints
    GARTH_AVAILABLE = False


def check_and_install_garth() -> bool:
    """Check if garth is installed, offer to install if not"""
    if GARTH_AVAILABLE:
        return True

    print("📦 garth library not found.")
    print("\n⚠️  Please install it using one of these methods:")
    print("   1. If using the venv (recommended):")
    print("      source .venv/bin/activate")
    print("      pip install garth")
    print("\n   2. Or run directly with venv Python:")
    print("      .venv/bin/python garmin_sync.py")
    print("\n   3. Or use make command:")
    print("      make install-dev  # installs all dependencies")
    print("")

    response = input("Would you like to try auto-installing now? (y/n): ")
    if response.lower() == "y":
        print("Installing garth...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "garth"])
            print("✅ garth installed successfully!")
            print("Please restart the script to use the newly installed library.")
            return False  # Still need to restart
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("\n💡 If you see 'externally-managed-environment' error:")
            print("   You're using system Python. Please use the venv instead.")
            print("   Run: source .venv/bin/activate")
            print("   Then: pip install garth")
            return False

    print("❌ Cannot proceed without garth library.")
    return False


def authenticate_garmin(
    email: Optional[str] = None,
    password: Optional[str] = None,
    token_store: str = "~/.garth"
) -> bool:
    """Authenticate with Garmin Connect and manage session tokens.

    Handles authentication to Garmin Connect using the garth library, with support
    for session token caching to avoid repeated logins. Attempts to resume an
    existing session first, and only prompts for credentials if needed.

    Args:
        email: Garmin Connect account email. If None, tries GARMIN_EMAIL env var,
               then prompts user for input.
        password: Garmin Connect account password. If None, tries GARMIN_PASSWORD
                  env var, then prompts securely using getpass.
        token_store: Path to store authentication tokens for session persistence.
                     Supports tilde (~) expansion for home directory.
                     Default: "~/.garth"

    Returns:
        bool: True if authentication successful (new or resumed session),
              False if authentication failed.

    Raises:
        ImportError: If garth library is not installed or not available.

    Example:
        >>> # Auto-authenticate using environment variables
        >>> authenticate_garmin()
        ✅ Resumed existing Garmin Connect session
        True

        >>> # Force new authentication with credentials
        >>> authenticate_garmin(email="user@example.com", password="secret")
        🔐 Authenticating with Garmin Connect...
        ✅ Authentication successful! Session saved.
        True

    Notes:
        - Session tokens are saved to avoid repeated MFA prompts
        - For MFA-enabled accounts, consider using app-specific passwords
        - Credentials are never stored, only session tokens
        - Failed authentications provide helpful troubleshooting hints
    """
    if garth is None:
        raise ImportError("garth library not available")

    token_path = Path(token_store).expanduser()

    # Try to resume existing session
    if token_path.exists():
        try:
            garth.resume(str(token_path))
            # Test if session is valid
            _ = garth.client.username
            print("✅ Resumed existing Garmin Connect session")
            return True
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            print(f"⚠️  Saved session expired or invalid: {e}")
            print("   Need to re-authenticate...")

    # Need new authentication
    if not email:
        email = os.getenv("GARMIN_EMAIL")
        if not email:
            email = input("Garmin Connect email: ")

    if not password:
        password = os.getenv("GARMIN_PASSWORD")
        if not password:
            password = getpass.getpass("Garmin Connect password: ")

    try:
        print("🔐 Authenticating with Garmin Connect...")
        garth.login(email, password)

        # Save credentials for next time
        token_path.parent.mkdir(parents=True, exist_ok=True)
        garth.save(str(token_path))
        print("✅ Authentication successful! Session saved.")
        return True

    except (OSError, RuntimeError, ValueError) as e:
        print(f"❌ Authentication failed: {e}")
        if "MFA" in str(e) or "verification" in str(e).lower():
            print("\n💡 If you have MFA enabled, you may need to:")
            print("   1. Generate an app-specific password in your Garmin account")
            print("   2. Or disable MFA temporarily during first setup")
        return False


def get_existing_activity_ids(directory: str = ".") -> Dict[str, float]:
    """Get set of activity IDs and their file modification times.

    Returns:
        Dict mapping activity_id -> file modification timestamp
    """
    existing_activities = {}
    fit_files = Path(directory).glob("*_ACTIVITY.fit")

    for fit_file in fit_files:
        # Extract activity ID from filename (e.g., "20744294782_ACTIVITY.fit" -> "20744294782")
        activity_id = fit_file.stem.replace("_ACTIVITY", "")
        try:
            # Verify it's a numeric ID
            int(activity_id)
            # Get file modification time
            mtime = fit_file.stat().st_mtime
            existing_activities[activity_id] = mtime
        except (ValueError, OSError):
            # Skip files that don't match the pattern or can't be accessed
            continue

    return existing_activities


def _parse_activity_date(activity: Dict[str, Any]) -> datetime:
    """Parse activity date and ensure it's timezone-aware"""
    activity_date_str = activity["startTimeLocal"].replace("Z", "+00:00")
    activity_date = datetime.fromisoformat(activity_date_str)

    # If the parsed date is naive, make it timezone-aware (assume UTC)
    if activity_date.tzinfo is None:
        activity_date = activity_date.replace(tzinfo=timezone.utc)

    return activity_date


def _filter_recent_activities(activities: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    """Filter activities by date range"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_activities = []

    for activity in activities:
        activity_date = _parse_activity_date(activity)
        if activity_date >= cutoff_date:
            recent_activities.append(activity)

    return recent_activities


def _extract_fit_from_zip(fit_data: bytes) -> Optional[bytes]:
    """Extract FIT file from ZIP if needed"""
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


def _should_download_activity(
    activity: Dict[str, Any],
    existing_activities: Dict[str, float]
) -> Tuple[bool, bool, bool]:
    """Check if activity should be downloaded based on update timestamp.

    Args:
        activity: Activity dict from Garmin API
        existing_activities: Dict of activity_id -> local file mtime

    Returns:
        Tuple of (should_download, is_update, check_api_update)
            - should_download: Whether to download FIT file
            - is_update: Whether this is an update to existing activity
            - check_api_update: Whether to check for API exercise data updates
    """
    activity_id = str(activity["activityId"])

    if activity_id not in existing_activities:
        return (True, False, False)

    # Activity exists - check if it was updated on Garmin
    garmin_update_time = activity.get("updateDate") or activity.get("lastModified")

    if not garmin_update_time:
        # No update timestamp from activity list, but we should still
        # check the API exercise data for updates (user may have edited exercises)
        return (False, False, True)

    # Parse Garmin timestamp (milliseconds since epoch)
    garmin_timestamp = garmin_update_time / 1000.0
    local_timestamp = existing_activities[activity_id]

    # Re-download if Garmin version is newer (with 1 second tolerance)
    if garmin_timestamp > local_timestamp + 1:
        return (True, True, False)

    return (False, False, True)


def _exercise_names_differ(existing_sets: List[Dict], fresh_sets: List[Dict]) -> bool:
    """Check if exercise names differ between two sets."""
    for ex_set, fr_set in zip(existing_sets, fresh_sets):
        ex_exercises = ex_set.get('exercises', [{}])
        fr_exercises = fr_set.get('exercises', [{}])
        ex_name = ex_exercises[0].get('name') if ex_exercises else None
        fr_name = fr_exercises[0].get('name') if fr_exercises else None
        if ex_name != fr_name:
            return True
    return False


def _check_and_update_api_data(activity_id: str, directory: str) -> bool:
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

        # Determine if we need to update
        needs_update = False
        if not existing_data:
            needs_update = True
        else:
            existing_sets = existing_data.get('exerciseSets', [])
            fresh_sets = fresh_data.get('exerciseSets', [])

            # Update if lengths differ or exercise names differ
            if len(existing_sets) != len(fresh_sets):
                needs_update = True
            elif _exercise_names_differ(existing_sets, fresh_sets):
                needs_update = True

        if needs_update:
            save_exercise_sets_to_json(str(filename), fresh_data)
            return True

        return False

    except (OSError, RuntimeError, ValueError) as e:
        print(f"      ⚠️  Error checking API data for {activity_id}: {e}")
        return False


def _download_single_activity(
    activity_id: str,
    activity_name: str,
    activity_date: str,
    directory: str
) -> bool:
    """Download a single activity and save to file"""
    try:
        print(f"   ⬇️  Downloading: {activity_name} ({activity_date}) [ID: {activity_id}]")

        # Download FIT file using garth.download
        fit_data = garth.download(f"/download-service/files/activity/{activity_id}")

        # Garmin returns a ZIP file, so we need to extract the FIT file
        fit_data = _extract_fit_from_zip(fit_data)

        if fit_data is None:
            print(f"      ⚠️  No .fit file found in ZIP for activity {activity_id}")
            return False

        # Save to file
        filename = Path(directory) / f"{activity_id}_ACTIVITY.fit"
        with open(filename, "wb") as f:
            f.write(fit_data)

        # Fetch and save exercise sets from API (for strength training)
        exercise_sets = fetch_exercise_sets_from_api(activity_id)
        if exercise_sets:
            save_exercise_sets_to_json(str(filename), exercise_sets)
            num_sets = len(exercise_sets.get('exerciseSets', []))
            print(f"      ✅ Saved exercise data ({num_sets} sets)")

        return True

    except (OSError, RuntimeError, ValueError) as e:
        print(f"      ⚠️  Error downloading activity {activity_id}: {e}")
        return False


def _fetch_exercise_sets_for_activity(activity_id: int) -> Optional[Dict[str, Any]]:
    """Fetch exercise sets for a single activity ID.

    Args:
        activity_id: Garmin activity ID

    Returns:
        Dict with activityId and exerciseSets array, or None if not found
    """
    try:
        exercise_sets = garth.connectapi(
            f'/activity-service/activity/{activity_id}/exerciseSets'
        )
        # Handle case where API might return unexpected types
        if isinstance(exercise_sets, dict) and exercise_sets.get('exerciseSets'):
            return exercise_sets
    except (GarthHTTPError, KeyError, TypeError):
        pass
    return None


def _get_child_activity_ids(activity_details: Dict[str, Any]) -> list:
    """Extract child activity IDs from activity details.

    Args:
        activity_details: Activity details from API

    Returns:
        List of child activity IDs, or empty list if none
    """
    if isinstance(activity_details, list):
        return []
    metadata = activity_details.get('metadataDTO', {})
    return metadata.get('childIds', [])


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
        return None

    try:
        # Get activity details to check for child activities (multisport)
        activity_details = garth.connectapi(f'/activity-service/activity/{activity_id}')
        child_ids = _get_child_activity_ids(activity_details)

        # Try child activities first (for multisport)
        for child_id in child_ids:
            result = _fetch_exercise_sets_for_activity(child_id)
            if result:
                return result

        # Try the main activity if no children or no child had exercise sets
        return _fetch_exercise_sets_for_activity(activity_id)

    except (GarthHTTPError, KeyError, TypeError) as e:
        print(f"      ⚠️  Error fetching exercise sets for {activity_id}: {e}")
        return None


def save_exercise_sets_to_json(fit_file_path: str, exercise_sets: Dict[str, Any]) -> None:
    """Save exercise sets data to JSON file alongside FIT file.

    Args:
        fit_file_path: Path to the FIT file
        exercise_sets: Exercise sets data from API
    """
    fit_path = Path(fit_file_path)
    json_path = fit_path.with_name(f"{fit_path.stem}_exercises.json")

    # Create directory if it doesn't exist
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(exercise_sets, f, indent=2)


def load_exercise_sets_from_json(fit_file_path: str) -> Optional[Dict[str, Any]]:
    """Load exercise sets data from JSON file.

    Args:
        fit_file_path: Path to the FIT file

    Returns:
        Exercise sets data, or None if file doesn't exist
    """
    fit_path = Path(fit_file_path)
    json_path = fit_path.with_name(f"{fit_path.stem}_exercises.json")

    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _process_activity(
    activity: Dict[str, Any],
    existing_activities: Dict[str, float],
    directory: str,
    counters: Dict[str, int]
) -> None:
    """Process a single activity (download, update, or skip).

    Args:
        activity: Activity dict from Garmin API
        existing_activities: Dict of existing activity IDs
        directory: Directory to save files
        counters: Dict with keys: new_count, updated_count, api_updated_count, skipped_count
    """
    activity_id = str(activity["activityId"])
    activity_name = activity.get("activityName", "Unknown")
    activity_date = activity["startTimeLocal"][:10]

    # Check if we need to download this activity
    should_download, is_update, check_api = _should_download_activity(
        activity, existing_activities
    )

    if should_download:
        if is_update:
            print(f"   🔄 Update detected for: {activity_name} [ID: {activity_id}]")

        if _download_single_activity(activity_id, activity_name, activity_date, directory):
            if is_update:
                counters['updated_count'] += 1
            else:
                counters['new_count'] += 1
    elif check_api:
        # FIT file exists and up-to-date, but check if exercise data was updated
        if _check_and_update_api_data(activity_id, directory):
            print(f"   📝 Exercise data updated for: {activity_name} [ID: {activity_id}]")
            counters['api_updated_count'] += 1
        else:
            counters['skipped_count'] += 1
    else:
        counters['skipped_count'] += 1


def download_new_activities(
    days: int = DEFAULT_SYNC_DAYS,
    limit: Optional[int] = None,
    directory: str = ".",
    force: bool = False,
) -> int:
    """Download new and updated activities from Garmin Connect.

    Fetches activities from the specified time range and downloads FIT files
    that are new or have been updated since the last sync. Intelligently skips
    unchanged files to minimize API calls and bandwidth usage.

    The function performs smart synchronization:
    1. Checks existing FIT files and their modification times
    2. Compares with Garmin's updateDate to detect changes
    3. Downloads only new or modified activities
    4. Updates exercise data (strength training sets) when edited in Garmin Connect
    5. Skips files that are already up-to-date

    Args:
        days: Number of days to look back when fetching activities.
              For example, days=30 fetches all activities from the last 30 days.
              Default is DEFAULT_SYNC_DAYS from constants (typically 30).
        limit: Maximum number of activities to download in this sync.
               If None, downloads all activities in the date range (up to API limit).
               Useful for testing or rate limiting.
        directory: Directory path where FIT files will be saved.
                   Files are named as "{activity_id}_ACTIVITY.fit".
                   Exercise data saved as "{activity_id}_ACTIVITY_exercises.json".
                   Default is current directory (".").
        force: If True, re-downloads all activities regardless of modification time.
               Useful for recovery or fixing corrupted files.
               Default is False (smart sync mode).

    Returns:
        int: Total number of activities successfully downloaded (new + updated).
             Does not include API-only updates or skipped activities.

    Raises:
        ImportError: If garth library is not installed or not available.
                     Call check_and_install_garth() before this function.
        GarthHTTPError: If Garmin Connect API returns an error (network issues,
                        authentication expired, rate limiting).
        OSError: If directory cannot be created or files cannot be written.

    Example:
        >>> # Download last 7 days of activities
        >>> count = download_new_activities(days=7, directory="./fit_files")
        📥 Fetching activities from last 7 days...
           Found 5 existing FIT files
        ⬇️  Downloading: Morning Run (2025-10-20) [ID: 12345]
           ✅ Saved exercise data (15 sets)
        📊 Summary: 2 new, 1 updated, 3 skipped (6 total)
        2

        >>> # Force re-download all activities from last month
        >>> count = download_new_activities(days=30, force=True)
        Force mode: will re-download all activities
        ...
        15

    Notes:
        - Requires active Garmin Connect session (call authenticate_garmin() first)
        - Automatically extracts FIT files from ZIP archives
        - Exercise data fetched separately via API (includes user edits)
        - Handles multisport activities by checking child activity IDs
        - Prints detailed progress with emoji indicators for status
        - Creates directory structure automatically if needed
    """
    if garth is None:
        raise ImportError("garth library not available")

    print(f"\n📥 Fetching activities from last {days} days...")

    # Get existing activity IDs and their modification times (unless force mode)
    existing_activities = {} if force else get_existing_activity_ids(directory)
    if force:
        print("   Force mode: will re-download all activities")
    else:
        print(f"   Found {len(existing_activities)} existing FIT files")

    # Fetch activities
    try:
        # Get activities using connectapi
        max_fetch = limit if limit else 100

        # Use the garth client to fetch activities
        activities = garth.connectapi(
            "/activitylist-service/activities/search/activities",
            params={"start": 0, "limit": max_fetch},
        )

        if not activities:
            print("   No activities found")
            return 0

        # Filter by date
        recent_activities = _filter_recent_activities(activities, days)
        print(f"   Found {len(recent_activities)} activities in date range")

        # Download new activities
        counters = {
            'new_count': 0,
            'updated_count': 0,
            'api_updated_count': 0,
            'skipped_count': 0
        }

        for activity in recent_activities:
            _process_activity(activity, existing_activities, directory, counters)

        print("\n✅ Download complete!")
        print(f"   New activities: {counters['new_count']}")
        print(f"   Updated activities: {counters['updated_count']}")
        if counters['api_updated_count'] > 0:
            print(f"   Exercise data updated: {counters['api_updated_count']}")
        print(f"   Skipped (already up-to-date): {counters['skipped_count']}")

        return counters['new_count'] + counters['updated_count'] + counters['api_updated_count']

    except (OSError, RuntimeError, ValueError) as e:
        print(f"❌ Error fetching activities: {e}")
        return 0


def run_analysis(
    ftp: float = DEFAULT_FTP,
    hrrest: int = DEFAULT_HR_REST,
    hrmax: int = DEFAULT_HR_MAX,
    multisport: bool = True,
    directory: str = ".",
) -> None:
    """Run the FIT file analysis using the parser module.

    Args:
        ftp: Functional Threshold Power in watts
        hrrest: Resting heart rate in bpm
        hrmax: Maximum heart rate in bpm
        multisport: Whether to process multisport activities
        directory: Directory containing FIT files
    """
    print("\n📊 Running analysis on all FIT files...")

    try:
        # Import parser module (relative import must be inside function)
        from . import parser  # pylint: disable=import-outside-toplevel

        # Get all FIT files in the directory
        fit_files = list(Path(directory).glob("*_ACTIVITY.fit"))
        if not fit_files:
            print("⚠️  No FIT files found to analyze")
            return False

        # Build arguments list as if calling from command line
        args = [str(f) for f in fit_files]
        args.extend(["--ftp", str(ftp)])
        args.extend(["--hrrest", str(hrrest)])
        args.extend(["--hrmax", str(hrmax)])
        args.append("--dump-sets")  # Always save strength training sets
        if multisport:
            args.append("--multisport")

        # Parse arguments using parser's argument parser
        parsed_args = parser.parse_arguments(args)

        # Run the parser main logic
        result = parser.main_with_args(parsed_args)

        if result == 0:
            print("✅ Analysis complete!")
            return True

        print(f"❌ Analysis failed with error code {result}")
        return False

    except (ImportError, OSError, ValueError) as e:
        print(f"❌ Error running analysis: {e}")
        return False


def main():
    """Main entry point for the sync command-line tool."""
    parser = argparse.ArgumentParser(
        description="Sync activities from Garmin Connect and analyze them"
    )
    parser.add_argument("--email", help="Garmin Connect email (or set GARMIN_EMAIL env var)")
    parser.add_argument(
        "--password", help="Garmin Connect password (or set GARMIN_PASSWORD env var)"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Download activities from last N days (default: 30)"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of activities to fetch (default: 100)"
    )
    parser.add_argument(
        "--directory",
        default="data/samples",
        help="Directory to save FIT files (default: data/samples)",
    )
    parser.add_argument(
        "--ftp", type=float, default=300, help="Functional Threshold Power in watts (default: 300)"
    )
    parser.add_argument("--hrrest", type=int, default=50, help="Resting heart rate (default: 50)")
    parser.add_argument("--hrmax", type=int, default=190, help="Maximum heart rate (default: 190)")
    parser.add_argument(
        "--no-multisport", action="store_true", help="Disable multisport session separation"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Only download, don't run analysis"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only run analysis, don't download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of activities even if they already exist",
    )

    args = parser.parse_args()

    print("🏃 Garmin Connect Auto-Sync")
    print("=" * 50)

    # Ensure directory exists
    directory = Path(args.directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    # Check for garth library
    if not args.analyze_only:
        if not check_and_install_garth():
            return 1

    # Download activities
    new_activities = 0
    if not args.analyze_only:
        # Authenticate
        if not authenticate_garmin(args.email, args.password):
            return 1

        # Download new activities
        new_activities = download_new_activities(
            days=args.days, limit=args.limit, directory=directory, force=args.force
        )

    # Run analysis
    if not args.download_only:
        run_analysis(
            ftp=args.ftp,
            hrrest=args.hrrest,
            hrmax=args.hrmax,
            multisport=not args.no_multisport,
            directory=directory,
        )

    print("\n🎉 Done!")
    if new_activities > 0:
        print(f"   Downloaded {new_activities} new activities")
    print("   Summary saved to: data/workout_summary_from_fit.csv")
    print("   Strength sets saved to: data/strength_training_summary.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
