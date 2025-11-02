#!/usr/bin/env python3
"""
Garmin Connect Auto-Sync Script.

Automatically downloads new activities from Garmin Connect and updates your workout summary.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .activity_download import (
    download_single_activity,
    print_download_summary,
    should_download_activity,
)
from .activity_processor import (
    ProcessorCallbacks,
    ProcessorContext,
    filter_recent_activities,
    get_existing_activity_ids,
    identify_multisport_parents,
    process_activities,
)
from .constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST
from .garmin_auth import authenticate_garmin, check_and_install_garth
from .garth_utils import GARTH_AVAILABLE, garth
from .strength import load_exercise_sets_from_json, save_exercise_sets_to_json
from .sync_config import AnalysisParams, SyncConfig, SyncMode

__all__ = [
    "authenticate_garmin",
    "download_new_activities",
    "get_existing_activity_ids",
    "run_analysis",
    "load_exercise_sets_from_json",
    "save_exercise_sets_to_json",
    "sync_activities",
    "SyncConfig",
    "AnalysisParams",
    "SyncMode",
    "main",
]


def download_new_activities(
    days: int = 7, limit: int = 100, directory: str = ".", force: bool = False
) -> Tuple[int, List[str]]:
    """Download new activities from Garmin Connect.

    This function fetches recent activities from Garmin Connect and downloads their
    FIT files along with exercise data. It intelligently:
    - Skips activities that haven't been modified
    - Downloads updated activities when Garmin has newer data
    - Checks for exercise data updates even when FIT file is unchanged
    - Handles multisport activities by downloading child activities instead of parents

    Args:
        days: Number of days to look back (default: 7)
        limit: Maximum number of activities to fetch from API (default: 100)
        directory: Directory to save FIT files (default: current directory)
        force: If True, re-download all activities regardless of modification time

    Returns:
        Tuple of (count, updated_files):
        - count: Total number of new/updated activities
        - updated_files: List of FIT file paths that were downloaded or had API updates

    Raises:
        ImportError: If garth library is not available
    """
    if not GARTH_AVAILABLE:
        raise ImportError("garth library not available")

    print(f"\n🔍 Fetching activities from last {days} days...")

    try:
        # Fetch activities from Garmin Connect
        activities_data = garth.connectapi(
            "/activitylist-service/activities/search/activities", params={"limit": limit}
        )

        # Handle case where API returns None or single activity as dict
        if activities_data is None:
            print("No activities found")
            return (0, [])

        if not isinstance(activities_data, list):
            activities_data = [activities_data]

        # Filter to activities in date range
        recent_activities = filter_recent_activities(activities_data, days)
        print(f"Found {len(recent_activities)} activities in date range")

        if not recent_activities:
            return (0, [])

        # Get existing activity IDs (unless force mode)
        if force:
            print("🔄 Force mode: Re-downloading all activities")
            existing_activities: Dict[str, float] = {}
        else:
            existing_activities = get_existing_activity_ids(directory)

        # Identify parent multisport activities
        parent_ids = identify_multisport_parents(recent_activities)

        # Process activities
        callbacks = ProcessorCallbacks(
            should_download_fn=should_download_activity,
            download_fn=download_single_activity,
        )
        context = ProcessorContext(
            existing_activities=existing_activities,
            directory=directory,
            callbacks=callbacks,
        )
        counters, updated_files = process_activities(
            activities=recent_activities,
            context=context,
            parent_activity_ids=parent_ids,
        )

        print_download_summary(counters)

        return (
            counters["new_count"] + counters["updated_count"] + counters["api_updated_count"],
            updated_files,
        )

    except (OSError, RuntimeError, ValueError) as e:
        print(f"❌ Error fetching activities: {e}")
        return (0, [])


def run_analysis(
    directory: str = ".",
    output_dir: str = "data",
    updated_files: Optional[List[str]] = None,
    **kwargs: Any,
) -> bool:
    """Run the FIT file analysis using the parser module.

    Returns:
        True if analysis completed successfully, False otherwise

    Args:
        directory: Directory containing FIT files
        output_dir: Directory for output CSV files
        updated_files: List of file paths that were updated (downloaded or API changed)
        **kwargs: Additional arguments (ftp, hrrest, hrmax)
    """
    # Extract kwargs with defaults
    ftp = kwargs.get("ftp", DEFAULT_FTP)
    hrrest = kwargs.get("hrrest", DEFAULT_HR_REST)
    hrmax = kwargs.get("hrmax", DEFAULT_HR_MAX)

    print("\n📊 Running analysis on all FIT files...")

    try:
        # Import cli module (relative import must be inside function)
        from . import cli  # pylint: disable=import-outside-toplevel

        # Determine which files to analyze
        # If updated_files is provided (even if empty), use those
        # If updated_files is None, analyze all FIT files in the directory
        if updated_files is not None:
            # Only analyze the files that were actually downloaded/updated
            fit_files = [Path(f) for f in updated_files if Path(f).exists()]
            if not fit_files:
                print("⚠️  No updated files to analyze")
                return True  # Success, just nothing to do
        else:
            # Analyze all files in directory (for analyze_only mode or initial run)
            directory_path = Path(directory)
            if directory_path.is_file() and directory_path.name.endswith("_ACTIVITY.fit"):
                fit_files = [directory_path]
            else:
                fit_files = list(directory_path.glob("*_ACTIVITY.fit"))

        if not fit_files:
            print("⚠️  No FIT files found to analyze")
            return False

        # Build arguments list as if calling from command line
        args = [str(f) for f in fit_files]
        args.extend(["--ftp", str(ftp)])
        args.extend(["--hrrest", str(hrrest)])
        args.extend(["--hrmax", str(hrmax)])
        args.extend(["--output-dir", str(output_dir)])
        args.append("--dump-sets")  # Always save strength training sets

        # Parse arguments using parser's argument parser
        parsed_args = cli.parse_arguments(args)

        # Add updated_files to parsed_args for strength aggregation
        if updated_files:
            parsed_args.updated_files = updated_files
        else:
            parsed_args.updated_files = []

        # Run the parser main logic
        result = cli.main_with_args(parsed_args)

        if result == 0:
            print("✅ Analysis complete!")
            return True

        print(f"❌ Analysis failed with error code {result}")
        return False

    except (ImportError, OSError, ValueError) as e:
        print(f"❌ Error running analysis: {e}")
        return False


def main() -> int:
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
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output CSV files (default: data)",
    )

    args = parser.parse_args()

    print("🏃 Garmin Connect Auto-Sync")
    print("=" * 50)

    # Ensure directory exists (unless it's a single file)
    directory = Path(args.directory).expanduser()
    if not directory.is_file():
        directory.mkdir(parents=True, exist_ok=True)

    # Build configuration from CLI arguments
    analysis_params = AnalysisParams(
        ftp=args.ftp,
        hrrest=args.hrrest,
        hrmax=args.hrmax,
    )
    mode = SyncMode(
        analyze_only=args.analyze_only,
        download_only=args.download_only,
        force=args.force,
    )
    config = SyncConfig(
        directory=str(directory),
        output_dir=args.output_dir,
        days=args.days,
        limit=args.limit,
        analysis=analysis_params,
        mode=mode,
    )

    # Use the high-level sync_activities function
    result = sync_activities(config, email=args.email, password=args.password)

    if not result["success"]:
        print(f"\n❌ Error: {result['error']}")
        return 1

    print("\n🎉 Done!")
    if result["new_activities"] > 0:
        print(f"   Downloaded {result['new_activities']} new activities")
    print(f"   Summary saved to: {result['csv_path']}")
    print(f"   Strength sets saved to: {result['strength_csv_path']}")

    return 0


def sync_activities(config: Optional[SyncConfig] = None, /, **kwargs: Any) -> Dict[str, Any]:
    """High-level function to sync activities from Garmin Connect (incremental by default).

    This is the recommended way to use the sync functionality programmatically.
    Incremental sync is automatic - only new/changed activities are downloaded and analyzed.

    Example:
        >>> from fitanalyzer import sync_activities, SyncConfig
        >>>
        >>> # Simple sync (uses env vars for credentials and defaults)
        >>> result = sync_activities(days=7)
        >>>
        >>> # With explicit credentials
        >>> result = sync_activities(
        ...     email="user@example.com",
        ...     password="secret",
        ...     days=30,
        ...     directory="./activities",
        ...     output_dir="./output"
        ... )
        >>>
        >>> # Advanced: using config object for full control
        >>> from fitanalyzer import AnalysisParams, SyncMode
        >>> config = SyncConfig(
        ...     directory="./activities",
        ...     output_dir="./data",
        ...     days=30,
        ...     analysis=AnalysisParams(ftp=250, hrmax=185),
        ...     mode=SyncMode(force=True)
        ... )
        >>> result = sync_activities(config, email="user@example.com")
        >>>
        >>> print(f"Downloaded {result['new_activities']} new activities")
        >>> print(f"CSV: {result['csv_path']}")

    Args:
        config: Optional SyncConfig object for advanced configuration. If provided,
                keyword arguments override config values.
        **kwargs: Keyword arguments for quick configuration:
            - email: Garmin Connect email (uses GARMIN_EMAIL env var if None)
            - password: Garmin Connect password (uses GARMIN_PASSWORD env var if None)
            - days: Number of days to sync (default: 30)
            - directory: Directory to store FIT files (default: ".")
            - output_dir: Directory for CSV output (default: "data")

    Returns:
        Dict with keys:
            - success (bool): Whether sync completed successfully
            - new_activities (int): Number of new activities downloaded
            - csv_path (str): Path to workout summary CSV
            - strength_csv_path (str): Path to strength training CSV
            - error (str): Error message if success=False

    Raises:
        ImportError: If garth library is not installed
    """
    # Extract authentication from kwargs
    email = kwargs.pop("email", None)
    password = kwargs.pop("password", None)

    # Start with provided config or create default
    if config is None:
        config = SyncConfig()

    # Override config with any explicit keyword arguments
    for key, value in kwargs.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    return _sync_with_config(config, email=email, password=password)


def _sync_with_config(
    config: SyncConfig, email: Optional[str] = None, password: Optional[str] = None
) -> Dict[str, Any]:
    """Internal sync implementation using configuration object."""
    if not check_and_install_garth():
        return {"success": False, "new_activities": 0, "error": "garth library not available"}

    try:
        directory_path = Path(config.directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Download phase
        new_activities, updated_files = _download_phase(config, directory_path, email, password)

        # Analysis phase
        if not config.mode.download_only:
            _analysis_phase(config, directory_path, updated_files)

        return _create_success_result(config.output_dir, new_activities)

    except Exception as e:  # pylint: disable=broad-except
        return {"success": False, "new_activities": 0, "error": str(e)}


def _download_phase(
    config: SyncConfig, directory_path: Path, email: Optional[str], password: Optional[str]
) -> Tuple[int, List[str]]:
    """Handle the download phase of synchronization.

    Returns:
        Tuple of (new_activities_count, updated_files_list)
    """
    if config.mode.analyze_only:
        return 0, []

    if not authenticate_garmin(email, password):
        raise RuntimeError("Authentication failed")

    new_activities, updated_files = download_new_activities(
        days=config.days,
        limit=config.limit if config.limit is not None else 100,
        directory=str(directory_path),
        force=config.mode.force,
    )
    return new_activities, updated_files


def _analysis_phase(config: SyncConfig, directory_path: Path, updated_files: List[str]) -> None:
    """Handle the analysis phase of synchronization."""
    run_analysis(
        directory=str(directory_path),
        output_dir=config.output_dir,
        ftp=config.analysis.ftp,
        hrrest=config.analysis.hrrest,
        hrmax=config.analysis.hrmax,
        updated_files=updated_files,
    )


def _create_success_result(output_dir: str, new_activities: int) -> Dict[str, Any]:
    """Create a success result dictionary."""
    output_path = Path(output_dir)
    return {
        "success": True,
        "new_activities": new_activities,
        "csv_path": str(output_path / "workout_summary_from_fit.csv"),
        "strength_csv_path": str(output_path / "strength_training_summary.csv"),
    }


if __name__ == "__main__":
    sys.exit(main())
