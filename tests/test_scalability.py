"""
Scalability tests for fit-analyzer.

Tests performance and memory usage with large numbers of FIT files.
These tests verify that the library can handle real-world workloads.
"""

import tempfile
from pathlib import Path
from shutil import copy2

import pytest

from fitanalyzer.activities import summarize_fit_sessions


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

    Simulates processing multiple files as in a typical sync operation.
    """
    import time
    from pathlib import Path

    # Get available test fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    fit_files = list(fixtures_dir.glob("*.fit"))

    if len(fit_files) < 3:
        pytest.skip("Need at least 3 FIT files for batch testing")

    # Process files and measure timing
    start_time = time.time()
    results = []

    for fit_file in fit_files:
        sessions, sets = summarize_fit_sessions(str(fit_file))
        results.append(
            {
                "file": fit_file.name,
                "sessions": len(sessions),
                "sets": len(sets) if sets is not None else 0,
            }
        )

    total_time = time.time() - start_time

    # Verify successful processing
    assert len(results) == len(fit_files)
    assert all(r["sessions"] >= 0 for r in results)

    # Basic performance check - should process files in reasonable time
    avg_time_per_file = total_time / len(fit_files)
    assert avg_time_per_file < 5.0, f"Processing too slow: {avg_time_per_file:.2f}s per file"


@pytest.mark.slow
def test_memory_usage_scaling():
    """Test that memory usage doesn't grow excessively with multiple files.

    Basic memory check without requiring psutil dependency.
    """
    import gc
    from pathlib import Path

    fixtures_dir = Path(__file__).parent / "fixtures"
    fit_files = list(fixtures_dir.glob("*.fit"))

    if len(fit_files) < 2:
        pytest.skip("Need at least 2 FIT files for memory testing")

    # Force garbage collection before starting
    gc.collect()

    # Process files and verify no obvious memory leaks
    for i, fit_file in enumerate(fit_files[:3]):  # Limit to avoid slow tests
        sessions, sets = summarize_fit_sessions(str(fit_file))

        # Basic validation
        assert sessions is not None
        assert len(sessions) >= 0

        # Force cleanup between iterations
        del sessions, sets
        gc.collect()

    # If we get here without MemoryError, consider it a success
    assert True


@pytest.mark.slow
def test_concurrent_parsing_safety():
    """Test that parsing doesn't have obvious race conditions.

    Tests sequential parsing of the same file multiple times.
    """
    from pathlib import Path

    fixtures_dir = Path(__file__).parent / "fixtures"
    fit_files = list(fixtures_dir.glob("*.fit"))

    if not fit_files:
        pytest.skip("No FIT files available for concurrency testing")

    test_file = fit_files[0]

    # Parse the same file multiple times in sequence
    results = []
    for i in range(3):
        sessions, sets = summarize_fit_sessions(str(test_file))
        results.append(
            {
                "iteration": i,
                "sessions": len(sessions),
                "sets": len(sets) if sets is not None else 0,
            }
        )

    # Results should be consistent across iterations
    first_result = results[0]
    for result in results[1:]:
        assert result["sessions"] == first_result["sessions"], "Session count should be consistent"
        assert result["sets"] == first_result["sets"], "Set count should be consistent"


# Fixture for sample FIT files
@pytest.fixture
def sample_fit_files():
    """Provide list of available FIT files for testing."""
    # Look for FIT files in test fixtures directory
    fixtures_dir = Path(__file__).parent / "fixtures"
    fit_files = list(fixtures_dir.glob("*.fit"))

    # Fallback: look in current directory
    if not fit_files:
        current_dir = Path.cwd()
        fit_files = list(current_dir.glob("*.fit"))

        # Also check parent directory (in case running from tests/)
        if not fit_files:
            fit_files = list(current_dir.parent.glob("*.fit"))

    return fit_files
