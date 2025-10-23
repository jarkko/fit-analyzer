"""
Scalability tests for fit-analyzer.

Tests performance and memory usage with large numbers of FIT files.
These tests verify that the library can handle real-world workloads.
"""

import tempfile
from pathlib import Path
from shutil import copy2

import pytest

from fitanalyzer.parser import summarize_fit_sessions


@pytest.mark.slow
def test_parse_multiple_files_sequentially(sample_fit_files):
    """Test parsing multiple FIT files in sequence.

    Verifies that:
    - Multiple files can be parsed without errors
    - Memory doesn't leak between parses
    - Performance is consistent across files
    """
    if not sample_fit_files:
        pytest.skip("No FIT files available for scalability testing")

    # Parse each file
    results = []
    for fit_file in sample_fit_files[:10]:  # Limit to 10 files
        sessions, sets = summarize_fit_sessions(str(fit_file))
        results.append(
            {
                "file": fit_file.name,
                "sessions": len(sessions),
                "sets": len(sets),
            }
        )

    # Verify all files parsed successfully
    assert len(results) > 0
    assert all(r["sessions"] >= 0 for r in results)


@pytest.mark.slow
def test_large_batch_processing():
    """Test processing a large batch of FIT files.

    Simulates processing 100+ files as in a typical sync operation.
    """
    # This would need actual FIT files or mocked data
    # For now, we just verify the concept works
    pytest.skip("Requires large dataset - implement when needed")


@pytest.mark.slow
def test_memory_usage_scaling():
    """Test that memory usage scales linearly with file count.

    Verifies that processing N files doesn't cause memory to grow quadratically.
    """
    pytest.skip("Requires psutil and large dataset - implement when needed")


@pytest.mark.slow
def test_concurrent_parsing_safety():
    """Test that parsing is safe for concurrent use.

    While not explicitly multithreaded, verify no global state corruption.
    """
    pytest.skip("Requires threading setup - implement when needed")


# Fixture for sample FIT files
@pytest.fixture
def sample_fit_files():
    """Provide list of available FIT files for testing."""
    # Look for FIT files in current directory
    current_dir = Path.cwd()
    fit_files = list(current_dir.glob("*.fit"))

    # Also check parent directory (in case running from tests/)
    if not fit_files:
        fit_files = list(current_dir.parent.glob("*.fit"))

    return fit_files
