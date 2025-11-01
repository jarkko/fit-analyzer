"""
FIT file parser for Garmin activity files.

This module provides low-level functions to parse FIT files and extract raw data.
For high-level workflow and analysis, see workflow.py.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fitparse import FitFile

from fitanalyzer.constants import SPORT_MAPPING, SUB_SPORT_MAPPING

# Apply monkey patch to fix fitparse deprecation warnings (side-effect import)
from . import fitparse_fix  # noqa: F401 pylint: disable=unused-import

__all__ = [
    "extract_sessions_from_fit",
    "extract_records_from_fit",
    "extract_valid_value",
    "get_sport_names",
    "create_record_dict",
]


def create_record_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Create a standardized record dictionary from FIT message data.

    This is the single source of truth for record field extraction,
    ensuring consistency across all record processing.

    Args:
        d: Dictionary of FIT message data (field name -> value)

    Returns:
        Standardized record dictionary with time, hr, power, speed, cadence, distance, altitude
    """
    return {
        "time": d["timestamp"],
        "hr": d.get("heart_rate", np.nan),
        "power": d.get("power", np.nan),
        "speed": d.get("enhanced_speed", d.get("speed", np.nan)),
        "cadence": d.get("cadence", np.nan),
        "distance": d.get("distance", np.nan),
        "altitude": d.get("enhanced_altitude", d.get("altitude", np.nan)),
    }


def extract_sessions_from_fit(ff: FitFile) -> List[Dict[str, Any]]:
    """Extract session info from FIT file.

    Args:
        ff: FitFile object

    Returns:
        List of session dictionaries with keys like 'sport', 'sub_sport',
        'start_time', 'total_timer_time', etc.
    """
    sessions = []
    for m in ff.get_messages("session"):
        d = {d.name: d.value for d in m}
        sessions.append(d)
    return sessions


def extract_records_from_fit(ff: FitFile) -> pd.DataFrame:
    """Extract aerobic data records from FIT file.

    Args:
        ff: FitFile object

    Returns:
        DataFrame with columns: time, hr, power, speed, cadence, distance, altitude
    """
    recs = []
    for m in ff.get_messages("record"):
        d = {d.name: d.value for d in m}
        if "timestamp" in d:
            recs.append(create_record_dict(d))
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values("time")
    return df


def extract_valid_value(value: Any, invalid_value: int = 65534) -> Optional[int]:
    """Extract first valid value from tuple or return single value.

    Args:
        value: Value to extract (int, tuple, or None)
        invalid_value: Value to treat as invalid (default: 65534)

    Returns:
        First valid value, or None if all invalid
    """
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, tuple):
        for v in value:
            if v is not None and v != invalid_value:
                return int(v) if isinstance(v, (int, float)) else None
        return None
    # Check if value is the invalid marker
    if value == invalid_value:
        return None
    return int(value) if isinstance(value, (int, float)) else None


def get_sport_names(sessions: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Get human-readable sport and sub-sport names from sessions.

    Args:
        sessions: List of session dictionaries from extract_sessions_from_fit()

    Returns:
        Tuple of (sport, sub_sport) as human-readable strings
    """
    raw_sport = sessions[0].get("sport", "") if sessions else ""
    raw_subsport = sessions[0].get("sub_sport", "") if sessions else ""

    # Convert numeric sport codes to names
    if isinstance(raw_sport, int):
        session_sport = SPORT_MAPPING.get(raw_sport, str(raw_sport))
    else:
        session_sport = raw_sport

    if isinstance(raw_subsport, int):
        session_subsport = SUB_SPORT_MAPPING.get(raw_subsport, str(raw_subsport))
    else:
        session_subsport = raw_subsport

    return session_sport, session_subsport
