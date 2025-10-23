"""
FIT File Analyzer - A Python library for analyzing Garmin FIT files.

This package provides tools for:
- Parsing FIT files and extracting training metrics
- Syncing activities from Garmin Connect
- Calculating performance metrics (NP, TSS, TRIMP, IF)
"""

from .credentials import create_env_file as setup_credentials
from .parser import (
    np_power,
    process_session_data,
    summarize_fit_original,
    summarize_fit_sessions,
    trimp_from_hr,
)
from .sync import (
    authenticate_garmin,
    download_new_activities,
    get_existing_activity_ids,
    run_analysis,
)

__version__ = "0.1.0"
__author__ = "FIT Analyzer Contributors"

__all__ = [
    "summarize_fit_original",
    "summarize_fit_sessions",
    "process_session_data",
    "np_power",
    "trimp_from_hr",
    "authenticate_garmin",
    "download_new_activities",
    "run_analysis",
    "get_existing_activity_ids",
    "setup_credentials",
]
