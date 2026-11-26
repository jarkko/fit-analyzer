"""Daily wellness sync from Garmin Connect: overnight HRV, resting heart rate and sleep.

Uses the authenticated garth session (see ``authenticate_garmin``). The endpoints are
the ones garth's own data classes use (garth 0.8: ``DailyHRV``, ``DailySummary`` and
``SleepData``). They are called through ``garth.client.connectapi`` directly so that a
missing or partial day yields ``None`` values instead of a validation error.
"""

import csv
import logging
import os
import secrets
import shutil
import time
from collections import Counter
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from requests import HTTPError as RequestsHTTPError

from .garth_utils import GarthHTTPError, garth

__all__ = [
    "WELLNESS_FIELDS",
    "WellnessAuthError",
    "WellnessSyncError",
    "fetch_daily_wellness",
    "sync_wellness",
]

logger = logging.getLogger(__name__)

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

# Pause between per-day requests so long backfills don't hit Garmin's rate limit (429).
REQUEST_DELAY_SECONDS = 0.3

# Auth and rate-limit responses mean every later request will fail too: stop at once.
AUTH_HTTP_STATUSES = frozenset({401, 403})
FATAL_HTTP_STATUSES = AUTH_HTTP_STATUSES | {429}

SOURCE_HRV = "HRV"
SOURCE_RESTING_HR = "resting HR"
SOURCE_SLEEP = "sleep"

Row = Dict[str, Any]


class WellnessSyncError(RuntimeError):
    """Wellness data could not be synced (a whole source failed, or the CSV is unusable)."""


class WellnessAuthError(WellnessSyncError):
    """The garth session is not authenticated, or Garmin rejected it.

    Raised when garth has no OAuth token, and when Garmin answers 401 or 403 to an API
    call or to garth's token refresh. Call ``authenticate_garmin()`` again.
    """


class _RequestFailed(Exception):
    """One request failed; it has already been logged."""


def _http_status(error: Exception) -> Optional[int]:
    """Status code of a GarthHTTPError (wraps ``.error``) or a bare requests.HTTPError."""
    response = getattr(getattr(error, "error", error), "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _request_failed(what: str, error: Exception) -> _RequestFailed:
    logger.warning("Garmin wellness request failed (%s): %s", what, error)
    return _RequestFailed(what)


def _http_failure(error: Exception, what: str) -> Exception:
    """Map an HTTP error to the exception to raise: auth, the error itself (429) or a miss."""
    status = _http_status(error)
    if status in AUTH_HTTP_STATUSES:
        # The callers raise this ``from error``, which sets __cause__
        return WellnessAuthError(
            f"Garmin rejected the session (HTTP {status}); call authenticate_garmin() again"
        )
    if status in FATAL_HTTP_STATUSES:
        return error
    return _request_failed(what, error)


def _guarded(call: Callable[[], Any], what: str) -> Any:
    """Run a garth call, turning recoverable errors into ``_RequestFailed``.

    Raises:
        WellnessAuthError: garth has no OAuth token, or Garmin answered 401/403 (to an
            API call or to garth's token refresh).
        GarthHTTPError: on 429, which no retry of another day would fix.
        requests.HTTPError: on 429 while garth refreshes its token.
        _RequestFailed: on any other HTTP, network or JSON error.
    """
    try:
        return call()
    except AssertionError as e:
        if "OAuth" in str(e):
            raise WellnessAuthError(
                f"Garmin session is not authenticated ({e}); call authenticate_garmin() first"
            ) from e
        raise _request_failed(what, e) from e
    except RequestsHTTPError as e:
        # garth refreshes an expired OAuth2 token before its GarthHTTPError wrapper, so
        # sso.exchange's HTTPError arrives bare. It must be checked before OSError.
        failure = _http_failure(e, what)
        if failure is e:
            raise
        raise failure from e
    except (OSError, ValueError, KeyError) as e:
        raise _request_failed(what, e) from e
    except GarthHTTPError as e:
        failure = _http_failure(e, what)
        if failure is e:
            raise
        raise failure from e


def _connectapi(path: str) -> Any:
    return _guarded(partial(garth.client.connectapi, path), path)


def _date_range(start: date, end: date) -> List[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _hrv_page(page_start: date, page_end: date) -> Dict[str, Row]:
    response = _connectapi(f"/hrv-service/hrv/daily/{page_start}/{page_end}")
    page = response.get("hrvSummaries") if isinstance(response, dict) else None
    return {
        summary["calendarDate"]: summary
        for summary in (page if isinstance(page, list) else [])
        if isinstance(summary, dict) and summary.get("calendarDate")
    }


def _fetch_hrv(start: date, end: date) -> Optional[Dict[str, Row]]:
    """Fetch HRV summaries keyed by ISO date, or None if every page request failed."""
    summaries: Dict[str, Row] = {}
    pages = failed = 0
    page_start = start
    while page_start <= end:
        page_end = min(page_start + timedelta(days=HRV_PAGE_DAYS - 1), end)
        pages += 1
        try:
            summaries.update(_hrv_page(page_start, page_end))
        except _RequestFailed:
            failed += 1
        page_start = page_end + timedelta(days=1)
    return None if failed == pages else summaries


def _fetch_resting_hr(day: date) -> Row:
    response = _connectapi(f"/usersummary-service/usersummary/daily/?calendarDate={day}")
    value = response.get("restingHeartRate") if isinstance(response, dict) else None
    return {"resting_hr": value}


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
    """Return the Garmin username needed by the sleep endpoint, or None if the lookup failed."""
    try:
        return str(_guarded(lambda: garth.client.username, "username lookup"))
    except _RequestFailed:
        return None


def _sources_failed(sources: List[str], start: date, end: date) -> WellnessSyncError:
    return WellnessSyncError(
        f"Garmin {' and '.join(sources)} requests failed for every day {start}..{end} "
        "(see the warnings above)"
    )


def _try_source(fetch: Callable[[], Row], source: str, failures: "Counter[str]") -> Row:
    try:
        return fetch()
    except _RequestFailed:
        failures[source] += 1
        return {}


def _fetch_day(day: date, hrv: Row, username: Optional[str], failures: "Counter[str]") -> Row:
    row: Row = {field: None for field in WELLNESS_FIELDS}
    row["date"] = day.isoformat()
    row["hrv_last_night_avg"] = hrv.get("lastNightAvg")
    row["hrv_weekly_avg"] = hrv.get("weeklyAvg")
    row["hrv_status"] = hrv.get("status")
    row.update(_try_source(partial(_fetch_resting_hr, day), SOURCE_RESTING_HR, failures))
    if username is None:
        failures[SOURCE_SLEEP] += 1  # the sleep endpoint needs the username
    else:
        row.update(_try_source(partial(_fetch_sleep, day, username), SOURCE_SLEEP, failures))
    return row


def fetch_daily_wellness(
    start: date,
    end: date,
    request_delay: float = REQUEST_DELAY_SECONDS,
    progress: Optional[List[Row]] = None,
) -> List[Row]:
    """Fetch daily wellness metrics from Garmin Connect for ``start``..``end`` (inclusive).

    Requires an authenticated garth session (call ``authenticate_garmin`` first).

    Args:
        start: First calendar day to fetch.
        end: Last calendar day to fetch.
        request_delay: Seconds to wait between per-day requests (0 disables the wait).
        progress: Optional list that each day's row is appended to as soon as it is
            fetched, so the caller keeps the finished days when this function raises.

    Returns:
        One dict per day, in date order, with the keys in ``WELLNESS_FIELDS``:
        ``date`` (ISO string), ``hrv_last_night_avg``, ``hrv_weekly_avg`` (ms),
        ``hrv_status`` (e.g. "BALANCED"), ``resting_hr`` (bpm), ``sleep_seconds``
        and ``sleep_score``. A value is None when Garmin has no data for it or that
        day's request failed (logged as a warning).

    Raises:
        ImportError: If garth is not installed.
        ValueError: If ``start`` is after ``end``.
        WellnessAuthError: If the garth session is not authenticated, or Garmin answers
            401 or 403 (the original error is the ``__cause__``).
        WellnessSyncError: If one source (HRV, resting HR or sleep) failed for every day.
            Raised after every day was fetched; the rows are in ``progress``.
        GarthHTTPError: On a 429 response from Garmin.
        requests.HTTPError: On a 429 while garth refreshes its OAuth token.
    """
    if garth is None:
        raise ImportError("garth library not available")
    if start > end:
        raise ValueError(f"start ({start}) must not be after end ({end})")

    rows: List[Row] = [] if progress is None else progress
    days = _date_range(start, end)
    failures: "Counter[str]" = Counter()
    hrv_by_day = _fetch_hrv(start, end)
    if hrv_by_day is None:
        failures[SOURCE_HRV] = len(days)
    username = _username()
    for i, day in enumerate(days):
        if i and request_delay > 0:
            time.sleep(request_delay)
        hrv = (hrv_by_day or {}).get(day.isoformat(), {})
        rows.append(_fetch_day(day, hrv, username, failures))
    failed = [s for s in (SOURCE_HRV, SOURCE_RESTING_HR, SOURCE_SLEEP) if failures[s] == len(days)]
    if failed:
        raise _sources_failed(failed, start, end)
    return rows


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _merge_row(existing: Optional[Row], new: Row) -> Row:
    """Overlay ``new`` on ``existing``; a missing (None or "") value keeps the existing one."""
    merged: Row = dict(existing or {})
    for field, value in new.items():
        if not _is_missing(value) or field not in merged:
            merged[field] = value
    return merged


def _read_csv_rows(path: Path) -> Tuple[List[str], Dict[str, Row]]:
    """Read an existing wellness CSV into (header, rows keyed by date).

    Raises:
        WellnessSyncError: If a non-empty file has no ``date`` column or no dated rows,
            so that it is never overwritten by mistake.
    """
    if not path.exists() or not path.read_text(encoding="utf-8-sig").strip():
        return [], {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        records = [{k: v for k, v in record.items() if k is not None} for record in reader]
    if "date" not in header:
        raise WellnessSyncError(f"{path} has no 'date' column; refusing to overwrite it")
    rows: Dict[str, Row] = {}
    for record in records:
        if record.get("date"):
            rows[record["date"]] = _merge_row(rows.get(record["date"]), record)
    if records and not rows:
        raise WellnessSyncError(f"{path} has rows but none with a date; refusing to overwrite it")
    return header, rows


def _create_temp_file(directory: Path, name: str) -> Tuple[int, str]:
    """Create a unique temp file next to the target, mode 0o666 minus the umask.

    Unlike ``tempfile.mkstemp`` (always 0600), the system applies the umask, as for a
    plain ``open(path, "w")``; no process-wide ``os.umask`` toggling is needed.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp_name = str(directory / f".{name}.{secrets.token_hex(8)}.tmp")
        try:
            return os.open(tmp_name, flags, 0o666), tmp_name
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique temp file in {directory}")


def _write_csv_rows(path: Path, header: List[str], rows: Dict[str, Row]) -> None:
    """Write rows sorted by date, atomically (temp file in the same directory + os.replace).

    A symlinked ``path`` is resolved first, so the link's target is updated and the
    link stays. A new file gets the usual umask-based mode; an existing one keeps its mode.
    """
    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = _create_temp_file(path.parent, path.name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for day in sorted(rows):
                writer.writerow(rows[day])
        if path.exists():
            shutil.copymode(path, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sync_wellness(
    csv_path: Union[str, Path],
    days: int,
    end: Optional[date] = None,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> List[Row]:
    """Fetch the last ``days`` days of wellness data and upsert them into a CSV by date.

    Requires an authenticated garth session (call ``authenticate_garmin`` first).

    Merge rules: rows for other dates are kept, and columns other than
    ``WELLNESS_FIELDS`` are kept with their values. For a date already in the file,
    fetched values replace stored ones, but a missing (None) fetched value never
    erases a stored value. This is deliberate: a day that failed or has no data yet
    does not wipe what an earlier sync stored. The flip side is that a value Garmin
    later deletes stays in the CSV until it is edited by hand.

    The file is written atomically. If the fetch fails part way (a source failed for
    every day, or Garmin answered 401/403/429), the days fetched so far are written
    first and the fetch error is raised after that. If that final write itself fails,
    its ``OSError`` is raised instead, with the fetch error as its ``__context__``.

    Args:
        csv_path: CSV file to create or update (parent directories are created).
        days: Number of days to fetch, ending at ``end`` (inclusive). Must be >= 1.
        end: Last day to fetch. Defaults to today.
        request_delay: Seconds to wait between per-day requests.

    Returns:
        The rows fetched from Garmin Connect, as returned by ``fetch_daily_wellness``.

    Raises:
        ValueError: If ``days`` is less than 1.
        ImportError: If garth is not installed.
        WellnessSyncError: If the existing CSV is unusable (see ``_read_csv_rows``)
            or a whole source failed for every day (see ``fetch_daily_wellness``).
        WellnessAuthError: If the garth session is not authenticated or Garmin
            answers 401/403.
        GarthHTTPError: On a 429 response from Garmin.
        requests.HTTPError: On a 429 while garth refreshes its OAuth token.
        OSError: If the CSV cannot be read or written. A failed final write supersedes
            any fetch error above (which stays available as ``__context__``).
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    end = end or date.today()
    path = Path(csv_path)
    header, rows = _read_csv_rows(path)

    fetched: List[Row] = []
    try:
        fetch_daily_wellness(
            end - timedelta(days=days - 1), end, request_delay=request_delay, progress=fetched
        )
    finally:
        if fetched:
            for row in fetched:
                rows[row["date"]] = _merge_row(rows.get(row["date"]), row)
            extra_columns = [
                column for column in dict.fromkeys(header) if column not in WELLNESS_FIELDS
            ]
            _write_csv_rows(path, WELLNESS_FIELDS + extra_columns, rows)
    return fetched
