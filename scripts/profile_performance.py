#!/usr/bin/env python3
"""
Performance profiling script for fit-analyzer.

Profiles parsing performance, identifies bottlenecks, and measures memory usage.
Run this script to benchmark fitanalyzer performance and find optimization opportunities.

Usage:
    python scripts/profile_performance.py
    python scripts/profile_performance.py --file path/to/activity.fit
    python scripts/profile_performance.py --directory data/
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import List, Optional

import psutil

# Add parent directory to path to import fitanalyzer
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fitanalyzer.parser import summarize_fit_sessions


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def profile_single_file(fit_file: Path) -> dict:
    """Profile parsing of a single FIT file.

    Args:
        fit_file: Path to FIT file to profile

    Returns:
        Dict with performance metrics: time, memory, file_size
    """
    print(f"\n📊 Profiling: {fit_file.name}")
    print(f"   File size: {fit_file.stat().st_size / 1024:.2f} KB")

    # Measure memory before
    mem_before = get_memory_usage()

    # Profile execution time
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()
    try:
        sessions, sets = summarize_fit_sessions(str(fit_file))
        end_time = time.perf_counter()
        success = True
    except Exception as e:
        end_time = time.perf_counter()
        print(f"   ❌ Error: {e}")
        success = False
        sessions = []
        sets = []

    profiler.disable()

    # Measure memory after
    mem_after = get_memory_usage()

    elapsed = end_time - start_time
    mem_used = mem_after - mem_before

    print(f"   ⏱️  Parse time: {elapsed:.3f}s")
    print(f"   💾 Memory used: {mem_used:.2f} MB")
    print(f"   📈 Sessions: {len(sessions)}, Sets: {len(sets)}")

    return {
        "file": fit_file.name,
        "file_size": fit_file.stat().st_size,
        "time": elapsed,
        "memory": mem_used,
        "success": success,
        "sessions": len(sessions),
        "sets": len(sets),
        "profiler": profiler,
    }


def profile_directory(directory: Path, limit: Optional[int] = None) -> List[dict]:
    """Profile all FIT files in a directory.

    Args:
        directory: Directory containing FIT files
        limit: Optional limit on number of files to profile

    Returns:
        List of performance metrics for each file
    """
    fit_files = sorted(directory.glob("*.fit"))

    if not fit_files:
        print(f"❌ No FIT files found in {directory}")
        return []

    if limit:
        fit_files = fit_files[:limit]

    print(f"\n📁 Found {len(fit_files)} FIT files to profile")

    results = []
    for fit_file in fit_files:
        result = profile_single_file(fit_file)
        results.append(result)

    return results


def print_summary(results: List[dict]) -> None:
    """Print summary statistics of profiling results.

    Args:
        results: List of profiling results from profile_directory
    """
    if not results:
        return

    successful = [r for r in results if r["success"]]
    if not successful:
        print("\n❌ No successful parses to summarize")
        return

    total_time = sum(r["time"] for r in successful)
    avg_time = total_time / len(successful)
    max_time = max(r["time"] for r in successful)
    min_time = min(r["time"] for r in successful)

    total_memory = sum(r["memory"] for r in successful)
    avg_memory = total_memory / len(successful)
    max_memory = max(r["memory"] for r in successful)

    total_size = sum(r["file_size"] for r in successful) / 1024  # KB
    avg_size = total_size / len(successful)

    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Files processed: {len(successful)} / {len(results)}")
    print(f"Total size: {total_size:.2f} KB")
    print(f"Average file size: {avg_size:.2f} KB")
    print()
    print(f"⏱️  TIMING:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average time: {avg_time:.3f}s")
    print(f"   Min time: {min_time:.3f}s")
    print(f"   Max time: {max_time:.3f}s")
    print(f"   Throughput: {len(successful) / total_time:.2f} files/sec")
    print()
    print(f"💾 MEMORY:")
    print(f"   Total memory: {total_memory:.2f} MB")
    print(f"   Average memory: {avg_memory:.2f} MB")
    print(f"   Max memory: {max_memory:.2f} MB")

    # Find slowest files
    print()
    print("🐌 SLOWEST FILES:")
    slowest = sorted(successful, key=lambda r: r["time"], reverse=True)[:5]
    for i, result in enumerate(slowest, 1):
        print(f"   {i}. {result['file']}: {result['time']:.3f}s ({result['file_size'] / 1024:.1f} KB)")


def print_detailed_profile(result: dict, top_n: int = 20) -> None:
    """Print detailed profiling information for a single file.

    Args:
        result: Profiling result dict containing profiler
        top_n: Number of top functions to show
    """
    print("\n" + "=" * 80)
    print(f"🔍 DETAILED PROFILE: {result['file']}")
    print("=" * 80)

    # Get profiling stats
    s = io.StringIO()
    ps = pstats.Stats(result["profiler"], stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(top_n)

    print(s.getvalue())


def main():
    """Main profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile fit-analyzer performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Single FIT file to profile",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory containing FIT files (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to profile",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed profiling for each file",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top functions to show in detailed view (default: 20)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 FIT-ANALYZER PERFORMANCE PROFILER")
    print("=" * 80)

    if args.file:
        # Profile single file
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)

        result = profile_single_file(args.file)
        print_detailed_profile(result, args.top)

    else:
        # Profile directory
        if not args.directory.exists():
            print(f"❌ Directory not found: {args.directory}")
            sys.exit(1)

        results = profile_directory(args.directory, args.limit)

        if results:
            print_summary(results)

            if args.detailed:
                for result in results:
                    if result["success"]:
                        print_detailed_profile(result, args.top)

    print("\n✅ Profiling complete!")


if __name__ == "__main__":
    main()
