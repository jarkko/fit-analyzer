#!/usr/bin/env .venv/bin/python3
"""
Command-line script to analyze FIT files and generate workout summary CSV.

Usage:
    ./analyze.py data/samples/*.fit --ftp 300
    ./analyze.py data/samples/*.fit --ftp 300 --multisport

Or with explicit python:
    .venv/bin/python3 analyze.py data/samples/*.fit --ftp 300
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fitanalyzer.parser import main

if __name__ == "__main__":
    main()
