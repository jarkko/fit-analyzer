"""
Monkey patch for fitparse to fix Python 3.13 deprecation warnings.

The fitparse library uses deprecated datetime.datetime.utcfromtimestamp()
which generates warnings in Python 3.13+. This module patches those methods
to use the modern timezone-aware approach.

This can be removed once fitparse is updated upstream.
See: https://github.com/dtcooper/python-fitparse/issues
"""

import datetime

__all__ = ["is_patched"]

try:
    from fitparse.processors import FitFileDataProcessor

    # Store original methods
    _original_process_type_date_time = FitFileDataProcessor.process_type_date_time
    _original_process_type_local_date_time = FitFileDataProcessor.process_type_local_date_time

    # UTC reference for FIT timestamps (seconds since UTC 00:00 Dec 31 1989)
    UTC_REFERENCE = 631065600

    # 'self' is required for monkey-patched instance methods
    def _patched_process_type_date_time(self, _field_data):  # pylint: disable=unused-argument
        """Fixed version using timezone-aware datetime"""
        value = _field_data.value
        if value is not None and value >= 0x10000000:
            # Use timezone-aware fromtimestamp instead of deprecated utcfromtimestamp
            _field_data.value = datetime.datetime.fromtimestamp(
                UTC_REFERENCE + value, tz=datetime.timezone.utc
            )
            _field_data.units = None

    # 'self' is required for monkey-patched instance methods
    def _patched_process_type_local_date_time(self, _field_data):  # pylint: disable=unused-argument
        """Fixed version using timezone-aware datetime"""
        if _field_data.value is not None:
            # Use timezone-aware fromtimestamp instead of deprecated utcfromtimestamp
            _field_data.value = datetime.datetime.fromtimestamp(
                UTC_REFERENCE + _field_data.value, tz=datetime.timezone.utc
            )
            _field_data.units = None

    # Apply monkey patches
    FitFileDataProcessor.process_type_date_time = _patched_process_type_date_time
    FitFileDataProcessor.process_type_local_date_time = _patched_process_type_local_date_time

    _PATCH_APPLIED = True

except ImportError:
    # fitparse not installed, skip patching
    _PATCH_APPLIED = False


def is_patched():
    """Check if the fitparse patch has been applied"""
    return _PATCH_APPLIED
