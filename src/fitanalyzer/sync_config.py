"""
Configuration dataclasses for Garmin Connect synchronization.

This module provides dataclasses for configuring activity synchronization,
analysis parameters, and sync modes.
"""

from dataclasses import dataclass
from typing import Optional

from .constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST, DEFAULT_SYNC_DAYS

__all__ = ["AnalysisParams", "SyncMode", "SyncConfig"]


@dataclass
class AnalysisParams:
    """Parameters for activity analysis."""

    ftp: int = DEFAULT_FTP
    hrrest: int = DEFAULT_HR_REST
    hrmax: int = DEFAULT_HR_MAX


@dataclass
class SyncMode:
    """Synchronization mode flags."""

    analyze_only: bool = False
    download_only: bool = False
    force: bool = False


@dataclass
class SyncConfig:
    """Configuration for activity synchronization."""

    # Directories
    directory: str = "."
    output_dir: str = "data"

    # Sync parameters
    days: int = DEFAULT_SYNC_DAYS
    limit: Optional[int] = None

    # Nested configurations
    analysis: AnalysisParams = None  # type: ignore[assignment]
    mode: SyncMode = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize nested params if not provided."""
        if self.analysis is None:
            self.analysis = AnalysisParams()  # type: ignore[unreachable]
        if self.mode is None:
            self.mode = SyncMode()  # type: ignore[unreachable]
