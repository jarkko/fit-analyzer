"""
Configuration classes for FIT file analysis.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for FIT file analysis."""

    ftp: float
    hr_rest: int
    hr_max: int
    tz_name: str


@dataclass(frozen=True)
class SetMetadata:
    """Metadata for strength training set."""

    activity_id: str
    file_name: str
    date: str
    sport: str
    sub_sport: str
