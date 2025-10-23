"""
Custom exceptions for fit-analyzer.

Defines specific exception types for better error handling and debugging.
"""


class FitAnalyzerError(Exception):
    """Base exception for all fit-analyzer errors."""


class FitFileError(FitAnalyzerError):
    """Exception raised for FIT file parsing errors."""


class FitFileNotFoundError(FitFileError):
    """Exception raised when a FIT file cannot be found."""


class FitFileCorruptedError(FitFileError):
    """Exception raised when a FIT file is corrupted or invalid."""


class AuthenticationError(FitAnalyzerError):
    """Exception raised for Garmin Connect authentication failures."""


class APIError(FitAnalyzerError):
    """Exception raised for Garmin Connect API errors."""


class ConfigurationError(FitAnalyzerError):
    """Exception raised for configuration errors."""


class ValidationError(FitAnalyzerError):
    """Exception raised for data validation errors."""
