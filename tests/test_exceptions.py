"""Tests for custom exceptions."""

import pytest

from fitanalyzer.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    FitAnalyzerError,
    FitFileCorruptedError,
    FitFileError,
    FitFileNotFoundError,
    ValidationError,
)


def test_base_exception():
    """Test FitAnalyzerError can be raised and caught."""
    with pytest.raises(FitAnalyzerError):
        raise FitAnalyzerError("Base error")


def test_fit_file_error():
    """Test FitFileError can be raised and caught."""
    with pytest.raises(FitFileError):
        raise FitFileError("File error")
    
    # Should also be caught by base exception
    with pytest.raises(FitAnalyzerError):
        raise FitFileError("File error")


def test_fit_file_not_found_error():
    """Test FitFileNotFoundError can be raised and caught."""
    with pytest.raises(FitFileNotFoundError):
        raise FitFileNotFoundError("File not found")
    
    # Should also be caught by parent exceptions
    with pytest.raises(FitFileError):
        raise FitFileNotFoundError("File not found")
    with pytest.raises(FitAnalyzerError):
        raise FitFileNotFoundError("File not found")


def test_fit_file_corrupted_error():
    """Test FitFileCorruptedError can be raised and caught."""
    with pytest.raises(FitFileCorruptedError):
        raise FitFileCorruptedError("File corrupted")
    
    # Should also be caught by parent exceptions
    with pytest.raises(FitFileError):
        raise FitFileCorruptedError("File corrupted")
    with pytest.raises(FitAnalyzerError):
        raise FitFileCorruptedError("File corrupted")


def test_authentication_error():
    """Test AuthenticationError can be raised and caught."""
    with pytest.raises(AuthenticationError):
        raise AuthenticationError("Auth failed")
    
    # Should also be caught by base exception
    with pytest.raises(FitAnalyzerError):
        raise AuthenticationError("Auth failed")


def test_api_error():
    """Test APIError can be raised and caught."""
    with pytest.raises(APIError):
        raise APIError("API call failed")
    
    # Should also be caught by base exception
    with pytest.raises(FitAnalyzerError):
        raise APIError("API call failed")


def test_configuration_error():
    """Test ConfigurationError can be raised and caught."""
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid config")
    
    # Should also be caught by base exception
    with pytest.raises(FitAnalyzerError):
        raise ConfigurationError("Invalid config")


def test_validation_error():
    """Test ValidationError can be raised and caught."""
    with pytest.raises(ValidationError):
        raise ValidationError("Invalid data")
    
    # Should also be caught by base exception
    with pytest.raises(FitAnalyzerError):
        raise ValidationError("Invalid data")


def test_exception_messages():
    """Test that exception messages are preserved."""
    msg = "Custom error message"
    
    try:
        raise FitAnalyzerError(msg)
    except FitAnalyzerError as e:
        assert str(e) == msg
    
    try:
        raise ValidationError(msg)
    except ValidationError as e:
        assert str(e) == msg
