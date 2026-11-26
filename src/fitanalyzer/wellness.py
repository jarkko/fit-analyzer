"""Daily wellness sync from Garmin Connect: overnight HRV, resting heart rate and sleep.

Uses the authenticated garth session (see ``authenticate_garmin``). The endpoints are
the ones garth's own data classes use (garth 0.8: ``DailyHRV``, ``DailySummary`` and
``SleepData``). They are called through ``garth.client.connectapi`` directly so that a
missing or partial day yields ``None`` values instead of a validation error.
"""

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .garth_utils import GarthHTTPError, garth

__all__ = ["WELLNESS_FIELDS", "fetch_daily_wellness", "sync_wellness"]

WELLNESS_FIELDS = [
    "date",
    "hrv_last_night_avg",
    "hrv_weekly_avg",
    "hrv_status",
    "resting_hr",
    "sleep_seconds",
    "sleep_score",
]

# Garmin serves at most 28 days of HRV summaries per request (garth DailyHRV page size).
HRV_PAGE_DAYS = 28

_API_ERRORS = (GarthHTTPError, OSError, ValueError)

Row = Dict[str, Any]


def _connectapi(path: str) -> Any:
    """Call Garmin Connect, returning None (with a warning) on API or network errors."""
    try:
        return garth.client.connectapi(path)
    except _API_ERRORS as e:
        print(f"   ⚠️  Garmin wellness request failed ({path}): {e}")
        return None


def _date_range(start: date, end: date) -> List[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _fetch_hrv(start: date, end: date) -> Dict[str, Row]:
    """Fetch HRV summaries for the range, keyed by ISO date."""
    summaries: Dict[str, Row] = {}
    page_start = start
    while page_start <= end:
        page_end = min(page_start + timedelta(days=HRV_PAGE_DAYS - 1), end)
        response = _connectapi(f"/hrv-service/hrv/daily/{page_start}/{page_end}")
        page = response.get("hrvSummaries") if isinstance(response, dict) else None
        for summary in page if isinstance(page, list) else []:
            if isinstance(summary, dict) and summary.get("calendarDate"):
                summaries[summary["calendarDate"]] = summary
        page_start = page_end + timedelta(days=1)
    return summaries


def _fetch_resting_hr(day: date) -> Optional[int]:
    response = _connectapi(f"/usersummary-service/usersummary/daily/?calendarDate={day}")
    return response.get("restingHeartRate") if isinstance(response, dict) else None


def _fetch_sleep(day: date, username: str) -> Row:
    """Return sleep_seconds and sleep_score for the night ending on ``day``."""
    response = _connectapi(
        f"/wellness-service/wellness/dailySleepData/{username}?nonSleepBufferMinutes=60&date={day}"
    )
    dto = response.get("dailySleepDTO") if isinstance(response, dict) else None
    if not isinstance(dto, dict) or not dto.get("id"):
        return {"sleep_seconds": None, "sleep_score": None}
    overall = (dto.get("sleepScores") or {}).get("overall") or {}
    return {"sleep_seconds": dto.get("sleepTimeSeconds"), "sleep_score": overall.get("value")}


def _username() -> Optional[str]:
    """Return the Garmin username needed by the sleep endpoint, or None if unavailable."""
    try:
        return str(garth.client.username)
    except (*_API_ERRORS, AssertionError, KeyError) as e:
        print(f"   ⚠️  Could not look up Garmin username; skipping sleep data: {e}")
        return None


def _build_row(day: date, hrv: Optional[Row], username: Optional[str]) -> Row:
    hrv = hrv or {}
    sleep = _fetch_sleep(day, username) if username else {}
    return {
        "date": day.isoformat(),
        "hrv_last_night_avg": hrv.get("lastNightAvg"),
        "hrv_weekly_avg": hrv.get("weeklyAvg"),
        "hrv_status": hrv.get("status"),
        "resting_hr": _fetch_resting_hr(day),
        "sleep_seconds": sleep.get("sleep_seconds"),
        "sleep_score": sleep.get("sleep_score"),
    }


def fetch_daily_wellness(start: date, end: date) -> List[Row]:
    """Fetch daily wellness metrics from Garmin Connect for ``start``..``end`` (inclusive).

    Requires an authenticated garth session (call ``authenticate_garmin`` first).

    Args:
        start: First calendar day to fetch.
        end: Last calendar day to fetch.

    Returns:
        One dict per day, in date order, with the keys in ``WELLNESS_FIELDS``:
        ``date`` (ISO string), ``hrv_last_night_avg``, ``hrv_weekly_avg`` (ms),
        ``hrv_status`` (e.g. "BALANCED"), ``resting_hr`` (bpm), ``sleep_seconds``
        and ``sleep_score``. A value is None when Garmin has no data for it or the
        request failed; one failing source does not affect the others.

    Raises:
        ImportError: If garth is not installed.
        ValueError: If ``start`` is after ``end``.
    """
    if garth is None:
        raise ImportError("garth library not available")
    if start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")

    hrv_by_day = _fetch_hrv(start, end)
    username = _username()
    return [
        _build_row(day, hrv_by_day.get(day.isoformat()), username)
        for day in _date_range(start, end)
    ]


def _read_csv_rows(path: Path) -> Dict[str, Row]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["date"]: dict(row) for row in csv.DictReader(f) if row.get("date")}


def _merge_row(existing: Optional[Row], new: Row) -> Row:
    """Overlay ``new`` on ``existing``; None in ``new`` keeps the existing value."""
    merged: Row = dict(existing or {})
    for field in WELLNESS_FIELDS:
        if new.get(field) is not None or field not in merged:
            merged[field] = new.get(field)
    return merged


def _write_csv_rows(path: Path, rows: Dict[str, Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WELLNESS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for day in sorted(rows):
            writer.writerow(rows[day])


def sync_wellness(csv_path: Union[str, Path], days: int, end: Optional[date] = None) -> List[Row]:
    """Fetch the last ``days`` days of wellness data and upsert them into a CSV by date.

    Requires an authenticated garth session (call ``authenticate_garmin`` first).
    Rows for other dates are kept. For a date already in the file, fetched values
    replace stored ones, but a missing (None) value never erases a stored value, so
    a failed request does not wipe earlier data.

    Args:
        csv_path: CSV file to create or update (parent directories are created).
        days: Number of days to fetch, ending at ``end`` (inclusive). Must be >= 1.
        end: Last day to fetch. Defaults to today.

    Returns:
        The rows fetched from Garmin Connect, as returned by ``fetch_daily_wellness``.

    Raises:
        ValueError: If ``days`` is less than 1.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    end = end or date.today()
    fetched = fetch_daily_wellness(end - timedelta(days=days - 1), end)

    path = Path(csv_path)
    rows = _read_csv_rows(path)
    for row in fetched:
        rows[row["date"]] = _merge_row(rows.get(row["date"]), row)
    _write_csv_rows(path, rows)
    return fetched
