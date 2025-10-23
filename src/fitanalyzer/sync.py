#!/usr/bin/env python3
"""
Garmin Connect Auto-Sync Script.

Automatically downloads new activities from Garmin Connect and updates your workout summary.
"""

import argparse
import getpass
import io
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST, DEFAULT_SYNC_DAYS

__all__ = [
    "authenticate_garmin",
    "download_new_activities",
    "run_analysis",
    "main",
]

# Try to import garth at module level
try:
    import garth

    GARTH_AVAILABLE = True
except ImportError:
    garth = None
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


def authenticate_garmin(email=None, password=None, token_store="~/.garth"):
    """Authenticate with Garmin Connect"""
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


def get_existing_activity_ids(directory="."):
    """Get set of activity IDs that have already been downloaded"""
    existing_ids = set()
    fit_files = Path(directory).glob("*_ACTIVITY.fit")

    for fit_file in fit_files:
        # Extract activity ID from filename (e.g., "20744294782_ACTIVITY.fit" -> "20744294782")
        activity_id = fit_file.stem.replace("_ACTIVITY", "")
        try:
            # Verify it's a numeric ID
            int(activity_id)
            existing_ids.add(activity_id)
        except ValueError:
            # Skip files that don't match the pattern
            continue

    return existing_ids


def _parse_activity_date(activity):
    """Parse activity date and ensure it's timezone-aware"""
    activity_date_str = activity["startTimeLocal"].replace("Z", "+00:00")
    activity_date = datetime.fromisoformat(activity_date_str)

    # If the parsed date is naive, make it timezone-aware (assume UTC)
    if activity_date.tzinfo is None:
        activity_date = activity_date.replace(tzinfo=timezone.utc)

    return activity_date


def _filter_recent_activities(activities, days):
    """Filter activities by date range"""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_activities = []

    for activity in activities:
        activity_date = _parse_activity_date(activity)
        if activity_date >= cutoff_date:
            recent_activities.append(activity)

    return recent_activities


def _extract_fit_from_zip(fit_data):
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


def _download_single_activity(activity_id, activity_name, activity_date, directory):
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

        return True

    except (OSError, RuntimeError, ValueError) as e:
        print(f"      ⚠️  Error downloading activity {activity_id}: {e}")
        return False


def download_new_activities(
    days: int = DEFAULT_SYNC_DAYS, limit: Optional[int] = None, directory: str = "."
) -> int:
    """Download new activities from Garmin Connect.

    Args:
        days: Number of days of activities to fetch
        limit: Maximum number of activities to download
        directory: Directory to save FIT files to

    Returns:
        Number of activities downloaded

    Raises:
        ImportError: If garth library is not available
    """
    if garth is None:
        raise ImportError("garth library not available")

    print(f"\n📥 Fetching activities from last {days} days...")

    # Get existing activity IDs
    existing_ids = get_existing_activity_ids(directory)
    print(f"   Found {len(existing_ids)} existing FIT files")

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
        new_count = 0
        skipped_count = 0

        for activity in recent_activities:
            activity_id = str(activity["activityId"])
            activity_name = activity.get("activityName", "Unknown")
            activity_date = activity["startTimeLocal"][:10]

            # Skip if already downloaded
            if activity_id in existing_ids:
                skipped_count += 1
                continue

            if _download_single_activity(activity_id, activity_name, activity_date, directory):
                new_count += 1

        print("\n✅ Download complete!")
        print(f"   New activities: {new_count}")
        print(f"   Skipped (already downloaded): {skipped_count}")

        return new_count

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
            days=args.days, limit=args.limit, directory=directory
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
    print("   Summary saved to: workout_summary_from_fit.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
