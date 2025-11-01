"""
Contract tests for aggregation.py functions.

These tests document and enforce contracts for aggregation utility functions.
"""

import pandas as pd
import pytest

from fitanalyzer.aggregation import extract_session_metadata


class TestExtractSessionMetadataContract:
    """Contract tests for extract_session_metadata() function.
    
    Contract: Extract sport, sub_sport, and date from first session
    regardless of input type (dict, list, DataFrame).
    
    Parameter matrix:
    - dict input → extract from dict
    - list with sessions → extract from first element
    - empty list → return defaults
    - DataFrame with rows → extract from first row
    - empty DataFrame → return defaults  
    - None/invalid → return defaults
    """

    def test_dict_input_extracts_metadata(self):
        """Contract: When df_sessions is a dict, extract sport info from it."""
        session_dict = {
            "sport": "cycling",
            "sub_sport": "road",
            "date": "2025-10-31"
        }
        
        sport, sub_sport, date = extract_session_metadata(session_dict)
        
        assert sport == "cycling"
        assert sub_sport == "road"
        assert date == "2025-10-31"

    def test_dict_with_missing_keys_returns_defaults(self):
        """Contract: Dict without sport keys returns 'unknown' defaults."""
        session_dict = {"other_key": "value"}
        
        sport, sub_sport, date = extract_session_metadata(session_dict)
        
        assert sport == "unknown"
        assert sub_sport == "unknown"
        assert date is None

    def test_list_with_sessions_extracts_first(self):
        """Contract: When df_sessions is a list, extract from first element."""
        sessions_list = [
            {"sport": "running", "sub_sport": "trail", "date": "2025-10-30"},
            {"sport": "cycling", "sub_sport": "road", "date": "2025-10-31"}
        ]
        
        sport, sub_sport, date = extract_session_metadata(sessions_list)
        
        # Should extract from first session
        assert sport == "running"
        assert sub_sport == "trail"
        assert date == "2025-10-30"

    def test_empty_list_returns_defaults(self):
        """Contract: Empty list returns default values."""
        sessions_list = []
        
        sport, sub_sport, date = extract_session_metadata(sessions_list)
        
        assert sport == "unknown"
        assert sub_sport == "unknown"
        assert date is None

    def test_dataframe_with_rows_extracts_first(self):
        """Contract: When df_sessions is a DataFrame, extract from first row."""
        df_sessions = pd.DataFrame([
            {"sport": "strength_training", "sub_sport": "generic", "date": "2025-10-29"},
            {"sport": "cycling", "sub_sport": "indoor", "date": "2025-10-30"}
        ])
        
        sport, sub_sport, date = extract_session_metadata(df_sessions)
        
        # Should extract from first row
        assert sport == "strength_training"
        assert sub_sport == "generic"
        assert date == "2025-10-29"

    def test_empty_dataframe_returns_defaults(self):
        """Contract: Empty DataFrame returns default values."""
        df_sessions = pd.DataFrame()
        
        sport, sub_sport, date = extract_session_metadata(df_sessions)
        
        assert sport == "unknown"
        assert sub_sport == "unknown"
        assert date is None

    def test_none_input_returns_defaults(self):
        """Contract: None input returns default values."""
        sport, sub_sport, date = extract_session_metadata(None)
        
        assert sport == "unknown"
        assert sub_sport == "unknown"
        assert date is None

    def test_dataframe_partial_columns(self):
        """Contract: DataFrame with partial data uses defaults for missing fields."""
        df_sessions = pd.DataFrame([
            {"sport": "volleyball"}  # Missing sub_sport and date
        ])
        
        sport, sub_sport, date = extract_session_metadata(df_sessions)
        
        assert sport == "volleyball"
        assert sub_sport == "unknown"  # Should use default
        assert date is None
