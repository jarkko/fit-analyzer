"""Garth library availability check and common imports.

This module centralizes the garth import pattern to avoid code duplication.
"""

# Try to import garth at module level
try:
    import garth
    from garth.http import GarthHTTPError as _GarthHTTPError  # type: ignore[attr-defined]

    GARTH_AVAILABLE = True
    GarthHTTPError = _GarthHTTPError
except ImportError:
    garth = None  # type: ignore[assignment]
    GarthHTTPError = Exception  # type: ignore[misc, assignment]
    GARTH_AVAILABLE = False

__all__ = ["garth", "GARTH_AVAILABLE", "GarthHTTPError"]
