"""
Strength training and exercise set extraction for FIT files.

This module handles all strength training related functionality including:
- Exercise name resolution from Garmin FIT SDK
- API exercise name merging from Garmin Connect
- Set extraction from FIT files
- Strength training aggregation across multiple workouts
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fitparse import FitFile

from fitanalyzer.constants import EXERCISE_CATEGORY_MAPPING


class _GarminProfileLoader:
    """Singleton loader for Garmin FIT SDK Profile (lazy loading)"""

    _instance = None
    _profile = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_profile(self):
        """Get the Garmin Profile, loading it on first access

        Returns:
            Garmin FIT SDK Profile object, or empty dict if SDK not installed
        """
        if self._profile is None:
            try:
                from garmin_fit_sdk import Profile  # pylint: disable=import-outside-toplevel

                self._profile = Profile
            except ImportError:
                self._profile = {}  # Fallback if SDK not installed
        return self._profile


# Global instance of the loader
_profile_loader = _GarminProfileLoader()


def _get_garmin_profile():
    """Lazy-load Garmin FIT SDK Profile

    Returns:
        Garmin FIT SDK Profile object, or empty dict if not available
    """
    return _profile_loader.get_profile()


def _load_exercise_sets_from_json(fit_file_path: str) -> Optional[Dict[str, Any]]:
    """Load exercise sets data from JSON file (wrapper to avoid circular import).

    Args:
        fit_file_path: Path to the FIT file

    Returns:
        Exercise sets data, or None if file doesn't exist
    """
    # pylint: disable=import-outside-toplevel
    from fitanalyzer.sync import load_exercise_sets_from_json

    return load_exercise_sets_from_json(fit_file_path)


def get_specific_exercise_name(category_id: int, subtype_id: int) -> Optional[str]:
    """Get specific exercise name from category and subtype IDs using Garmin FIT SDK.

    Uses the Garmin FIT SDK Profile to look up detailed exercise names like
    "Barbell Power Clean" instead of just the generic category "Olympic Lift".
    This provides more accurate exercise naming when the Garmin SDK is available.

    Args:
        category_id: Exercise category ID from FIT file (e.g., 18 for Olympic Lift).
                    See EXERCISE_CATEGORY_MAPPING for common categories.
        subtype_id: Exercise subtype ID within the category (e.g., 2 for Barbell Power Clean).
                   Each category has its own set of subtype values.

    Returns:
        Specific exercise name as a string (e.g., "Barbell Power Clean"), or None if:
        - Garmin FIT SDK is not installed
        - Category ID is unknown or invalid
        - Subtype ID is not found in the category
        - Profile data is unavailable

    Example:
        >>> get_specific_exercise_name(18, 2)
        'Barbell Power Clean'
        >>> get_specific_exercise_name(999, 1)  # Unknown category
        None
    """
    profile = _get_garmin_profile()
    if not profile or "types" not in profile:
        return None

    # Get the category name to find the right exercise_name type
    category_types = profile["types"].get("exercise_category", {})
    category_name = category_types.get(str(category_id))

    if not category_name or category_name == "unknown":
        return None

    # Map category name to its exercise_name type
    type_name = f"{category_name}_exercise_name"
    exercise_types = profile["types"].get(type_name, {})

    exercise_name = exercise_types.get(str(subtype_id))
    if exercise_name and exercise_name != "unknown":
        # Convert snake_case to Title Case
        return " ".join(word.capitalize() for word in exercise_name.split("_"))

    return None


def merge_api_exercise_names(
    fit_df: pd.DataFrame, api_data: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Merge exercise names from Garmin API into FIT DataFrame.

    Matches sets by message_index and replaces exercise_name with API data.
    API names are more accurate as they reflect manual corrections in Garmin Connect.

    Args:
        fit_df: DataFrame with FIT file data including message_index and exercise_name
        api_data: Exercise sets data from Garmin API (or None)

    Returns:
        DataFrame with updated exercise names where API data is available

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'message_index': [0, 1, 2],
        ...     'exercise_name': ['Unknown', 'Unknown', 'Unknown']
        ... })
        >>> api_data = {
        ...     'exerciseSets': [
        ...         {'messageIndex': 0, 'exercises': [{'name': 'BARBELL_SQUAT'}]},
        ...         {'messageIndex': 1, 'exercises': [{'name': 'BENCH_PRESS'}]}
        ...     ]
        ... }
        >>> result = merge_api_exercise_names(df, api_data)
        >>> result['exercise_name'].tolist()
        ['Barbell Squat', 'Bench Press', 'Unknown']
    """
    if api_data is None or not api_data.get("exerciseSets"):
        return fit_df

    # Create a copy to avoid modifying original
    result_df = fit_df.copy()

    # Build a mapping from messageIndex to exercise name
    api_exercise_map = {}
    for ex_set in api_data["exerciseSets"]:
        message_index = ex_set.get("messageIndex")
        exercises = ex_set.get("exercises", [])

        if message_index is not None and exercises:
            # Take the first exercise (highest probability)
            exercise_name = exercises[0].get("name", "")
            if exercise_name:
                # Convert from UPPER_SNAKE_CASE to Title Case
                formatted_name = " ".join(word.capitalize() for word in exercise_name.split("_"))
                api_exercise_map[message_index] = formatted_name

    # Update exercise names in DataFrame where we have API data
    for msg_idx, api_name in api_exercise_map.items():
        mask = result_df["message_index"] == msg_idx
        if mask.any():
            result_df.loc[mask, "exercise_name"] = api_name

    return result_df


def _extract_valid_value(val: Any) -> Optional[int]:
    """Extract valid integer from value (handles tuples and None).

    Args:
        val: Value to extract from (can be tuple, int, or None)

    Returns:
        Integer value if valid, None otherwise
    """
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, tuple):
        for v in val:
            if v is not None and v != 65534:  # 65534 is invalid marker in FIT files
                return int(v)
        return None
    return int(val) if val != 65534 else None


def _get_specific_exercise_name_from_subtype(cat_val: int, category_subtype: Any) -> Optional[str]:
    """Try to get specific exercise name from category and subtype.

    Args:
        cat_val: Category ID value
        category_subtype: Subtype value (may be tuple or int)

    Returns:
        Specific exercise name, or None if not found
    """
    sub_val = _extract_valid_value(category_subtype)
    if sub_val is not None:
        specific_name = get_specific_exercise_name(cat_val, sub_val)
        if specific_name:
            return specific_name
    return None


def _convert_category_to_exercise_name(row: pd.Series) -> str:
    """Convert category and category_subtype to exercise name.

    Args:
        row: DataFrame row with 'category' and 'category_subtype' columns

    Returns:
        Human-readable exercise name
    """
    cat_val = _extract_valid_value(row.get("category"))
    if cat_val is None:
        return "Unknown"

    # Try specific exercise name from subtype
    specific_name = _get_specific_exercise_name_from_subtype(cat_val, row.get("category_subtype"))
    if specific_name:
        return specific_name

    # Fall back to category-level name
    return EXERCISE_CATEGORY_MAPPING.get(cat_val, f"Exercise {cat_val}")


def extract_sets_from_fit(ff: FitFile, fit_file_path: Optional[str] = None) -> pd.DataFrame:
    """Extract strength training sets from FIT file and add exercise names.

    Exercise names are extracted using a two-level system:
    1. Category (e.g., 18 = "Olympic Lift", 13 = "Hyperextension")
    2. Category subtype (e.g., category 18, subtype 2 = "Barbell Power Clean")

    If API exercise data is available (from Garmin Connect), those names are
    preferred as they reflect manual corrections. Otherwise, we use the
    category/subtype mapping from the FIT file.

    Args:
        ff: FitFile object to extract sets from
        fit_file_path: Optional path to FIT file (to load API exercise data)

    Returns:
        DataFrame with columns: message_index, category, category_subtype,
        exercise_name, set_type, repetitions, weight, duration, timestamp, etc.
        Returns empty DataFrame if no sets found.

    Example:
        >>> from fitparse import FitFile
        >>> ff = FitFile('workout.fit')
        >>> df = extract_sets_from_fit(ff, 'workout.fit')
        >>> df[['exercise_name', 'repetitions', 'weight']].head()
           exercise_name  repetitions  weight
        0  Barbell Squat           10    80.0
        1  Barbell Squat           10    80.0
        2  Bench Press              8    60.0
    """
    sets = []
    for m in ff.get_messages("set"):
        d = {d.name: d.value for d in m}
        sets.append(d)

    if not sets:
        return pd.DataFrame()

    df = pd.DataFrame(sets)
    df["exercise_name"] = df.apply(_convert_category_to_exercise_name, axis=1)

    # Merge API exercise names if available
    if fit_file_path:
        api_data = _load_exercise_sets_from_json(fit_file_path)
        if api_data:
            df = merge_api_exercise_names(df, api_data)

    return df


def save_strength_sets_csv(fit_file: str, df_sets: pd.DataFrame) -> Optional[str]:
    """Save strength sets to CSV file.

    Args:
        fit_file: Path to the FIT file
        df_sets: DataFrame with strength training sets

    Returns:
        Path to the saved CSV file, or None if no sets to save

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'exercise_name': ['Squat'], 'repetitions': [10]})
        >>> csv_path = save_strength_sets_csv('workout.fit', df)
        >>> csv_path
        'workout_strength_sets.csv'
    """
    if df_sets is None or (isinstance(df_sets, pd.DataFrame) and df_sets.empty):
        return None

    filename = Path(fit_file).with_suffix("").name + "_strength_sets.csv"
    df_sets.to_csv(filename, index=False)
    return filename


__all__ = [
    "get_specific_exercise_name",
    "merge_api_exercise_names",
    "extract_sets_from_fit",
    "save_strength_sets_csv",
]
