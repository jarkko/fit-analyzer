#!/usr/bin/env .venv/bin/python3
"""
Garmin Connect Auto-Sync Script
Automatically downloads new activities from Garmin Connect and updates your workout summary.
"""

import sys
from pathlib import Path

# Add src directory to path so we can import fitanalyzer
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fitanalyzer.sync import main

if __name__ == "__main__":
    sys.exit(main())
