#!/usr/bin/env python3
"""
Garmin Connect Auto-Sync Script
Automatically downloads new activities from Garmin Connect and updates your workout summary.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# Import garth at module level for easier testing
try:
    import garth
except ImportError:
    garth = None

def check_and_install_garth():
    """Check if garth is installed, offer to install if not"""
    global garth
    if garth is not None:
        return True

    try:
        import garth as g
        garth = g
        return True
    except ImportError:
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
        if response.lower() == 'y':
            print("Installing garth...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "garth"])
                print("✅ garth installed successfully!")
                import garth as g
                garth = g
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print("\n💡 If you see 'externally-managed-environment' error:")
                print("   You're using system Python. Please use the venv instead.")
                print("   Run: source .venv/bin/activate")
                print("   Then: pip install garth")
                return False
        else:
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
            garth.client.username  # Test if session is valid
            print("✅ Resumed existing Garmin Connect session")
            return True
        except Exception as e:
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
            import getpass
            password = getpass.getpass("Garmin Connect password: ")

    try:
        print("🔐 Authenticating with Garmin Connect...")
        garth.login(email, password)

        # Save credentials for next time
        token_path.parent.mkdir(parents=True, exist_ok=True)
        garth.save(str(token_path))
        print("✅ Authentication successful! Session saved.")
        return True

    except Exception as e:
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

def download_new_activities(days=30, limit=None, directory="."):
    """Download new activities from Garmin Connect"""
    if garth is None:
        raise ImportError("garth library not available")

    print(f"\n📥 Fetching activities from last {days} days...")

    # Get existing activity IDs
    existing_ids = get_existing_activity_ids(directory)
    print(f"   Found {len(existing_ids)} existing FIT files")

    # Fetch activities
    try:
        # Get activities using connectapi
        # API endpoint: /activitylist-service/activities/search/activities
        max_fetch = limit if limit else 100

        # Use the garth client to fetch activities
        activities = garth.connectapi(
            f"/activitylist-service/activities/search/activities",
            params={"start": 0, "limit": max_fetch}
        )

        if not activities:
            print("   No activities found")
            return 0

        # Filter by date (use timezone-aware datetime for comparison)
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_activities = []

        for activity in activities:
            activity_date = datetime.fromisoformat(activity["startTimeLocal"].replace("Z", "+00:00"))
            if activity_date >= cutoff_date:
                recent_activities.append(activity)

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

            try:
                print(f"   ⬇️  Downloading: {activity_name} ({activity_date}) [ID: {activity_id}]")

                # Download FIT file using garth.download
                # API endpoint: /download-service/files/activity/{activity_id}
                fit_data = garth.download(f"/download-service/files/activity/{activity_id}")

                # Save to file
                filename = Path(directory) / f"{activity_id}_ACTIVITY.fit"
                with open(filename, "wb") as f:
                    f.write(fit_data)

                new_count += 1

            except Exception as e:
                print(f"      ⚠️  Error downloading activity {activity_id}: {e}")
                continue

        print(f"\n✅ Download complete!")
        print(f"   New activities: {new_count}")
        print(f"   Skipped (already downloaded): {skipped_count}")

        return new_count

    except Exception as e:
        print(f"❌ Error fetching activities: {e}")
        return 0

def run_analysis(ftp=300, hrrest=50, hrmax=190, multisport=True, directory="."):
    """Run the FIT file analysis script"""
    print(f"\n📊 Running analysis on all FIT files...")

    script_path = Path(directory) / "fit_to_summary.py"
    if not script_path.exists():
        print(f"⚠️  Analysis script not found: {script_path}")
        return False

    try:
        cmd = [
            sys.executable,
            str(script_path),
            "--ftp", str(ftp),
            "--hrrest", str(hrrest),
            "--hrmax", str(hrmax),
        ]

        if multisport:
            cmd.append("--multisport")

        # Add all FIT files
        fit_files = list(Path(directory).glob("*_ACTIVITY.fit"))
        if not fit_files:
            print("⚠️  No FIT files found to analyze")
            return False

        cmd.extend([str(f) for f in fit_files])

        result = subprocess.run(cmd, cwd=directory, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Analysis complete!")
            # Print the summary output
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ Analysis failed with error code {result.returncode}")
            if result.stderr:
                print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Sync activities from Garmin Connect and analyze them"
    )
    parser.add_argument("--email", help="Garmin Connect email (or set GARMIN_EMAIL env var)")
    parser.add_argument("--password", help="Garmin Connect password (or set GARMIN_PASSWORD env var)")
    parser.add_argument("--days", type=int, default=30, help="Download activities from last N days (default: 30)")
    parser.add_argument("--limit", type=int, help="Maximum number of activities to fetch (default: 100)")
    parser.add_argument("--directory", default=".", help="Directory to save FIT files (default: current)")
    parser.add_argument("--ftp", type=float, default=300, help="Functional Threshold Power in watts (default: 300)")
    parser.add_argument("--hrrest", type=int, default=50, help="Resting heart rate (default: 50)")
    parser.add_argument("--hrmax", type=int, default=190, help="Maximum heart rate (default: 190)")
    parser.add_argument("--no-multisport", action="store_true", help="Disable multisport session separation")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't run analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Only run analysis, don't download")

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
            days=args.days,
            limit=args.limit,
            directory=directory
        )

    # Run analysis
    if not args.download_only:
        run_analysis(
            ftp=args.ftp,
            hrrest=args.hrrest,
            hrmax=args.hrmax,
            multisport=not args.no_multisport,
            directory=directory
        )

    print("\n🎉 Done!")
    if new_activities > 0:
        print(f"   Downloaded {new_activities} new activities")
    print(f"   Summary saved to: {directory / 'workout_summary_from_fit.csv'}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
