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

    # Note: 'self' unused but required to match instance method signature for monkey patching
    def _patched_process_type_date_time(_self, field_data):
        """Fixed version using timezone-aware datetime"""
        value = field_data.value
        if value is not None and value >= 0x10000000:
            # Use timezone-aware fromtimestamp instead of deprecated utcfromtimestamp
            field_data.value = datetime.datetime.fromtimestamp(
                UTC_REFERENCE + value, tz=datetime.timezone.utc
            )
            field_data.units = None

    # Note: 'self' unused but required to match instance method signature for monkey patching
    def _patched_process_type_local_date_time(_self, field_data):
        """Fixed version using timezone-aware datetime"""
        if field_data.value is not None:
            # Use timezone-aware fromtimestamp instead of deprecated utcfromtimestamp
            field_data.value = datetime.datetime.fromtimestamp(
                UTC_REFERENCE + field_data.value, tz=datetime.timezone.utc
            )
            field_data.units = None

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
