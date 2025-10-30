"""
Module reload utilities to prevent Python caching issues during development.
This ensures new code changes take effect immediately without restart.
"""

import sys
import importlib
from typing import List, Optional


def force_reload_fitanalyzer_modules() -> None:
    """
    Force reload all fitanalyzer modules to prevent caching issues.

    This should be called when:
    1. Running tests after code changes
    2. Running analysis after adding new features
    3. Any time you suspect caching issues
    """
    # List of all fitanalyzer modules that might be cached
    fitanalyzer_modules = [
        "fitanalyzer",
        "fitanalyzer.parser",
        "fitanalyzer.sync",
        "fitanalyzer.strength",
        "fitanalyzer.credentials",
        "fitanalyzer.metrics",
        "fitanalyzer.constants",
        "fitanalyzer.exceptions",
        "fitanalyzer.fitparse_fix",
    ]

    # Remove from sys.modules to force fresh import
    modules_removed = []
    for module_name in fitanalyzer_modules:
        if module_name in sys.modules:
            del sys.modules[module_name]
            modules_removed.append(module_name)

    # Force re-import of main modules
    try:
        import fitanalyzer.parser
        import fitanalyzer.sync

        importlib.reload(fitanalyzer.parser)
        importlib.reload(fitanalyzer.sync)
    except ImportError as e:
        print(f"Warning: Could not reload modules: {e}")

    if modules_removed:
        print(f"Reloaded modules: {', '.join(modules_removed)}")


def with_fresh_modules(func):
    """
    Decorator to ensure function runs with fresh module imports.

    Usage:
        @with_fresh_modules
        def test_new_feature():
            # This will run with reloaded modules
            pass
    """

    def wrapper(*args, **kwargs):
        force_reload_fitanalyzer_modules()
        return func(*args, **kwargs)

    return wrapper


def ensure_module_freshness() -> bool:
    """
    Check if modules seem fresh or might be cached.
    Returns True if modules appear fresh, False if suspicious caching detected.
    """
    try:
        # Try to import and check a timestamp or version indicator
        import fitanalyzer.parser

        # Check if the module has our recent changes by looking for new functions
        has_new_features = hasattr(fitanalyzer.parser, "_extract_records_from_fit") and hasattr(
            fitanalyzer.parser, "_calculate_metrics"
        )

        if not has_new_features:
            print("Warning: Modules appear to be cached - missing recent features")
            return False

        return True

    except ImportError:
        print("Warning: Could not import fitanalyzer modules")
        return False


if __name__ == "__main__":
    # Test the reload functionality
    print("Testing module reload functionality...")

    print("Before reload:")
    fresh = ensure_module_freshness()
    print(f"Modules fresh: {fresh}")

    print("\nForcing reload...")
    force_reload_fitanalyzer_modules()

    print("After reload:")
    fresh = ensure_module_freshness()
    print(f"Modules fresh: {fresh}")
