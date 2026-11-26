"""Contract tests for the wellness module (daily HRV, resting HR and sleep).

All Garmin Connect access is mocked: ``fitanalyzer.wellness.garth`` is replaced
with a Mock whose ``client.connectapi`` routes request paths to canned
responses shaped like the real Garmin Connect API.
"""

import csv
import os
import stat
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import ANY, Mock, patch

from requests import ConnectionError as RequestsConnectionError
from requests import HTTPError, Response

from fitanalyzer.garth_utils import GarthHTTPError
from fitanalyzer.wellness import (
    WELLNESS_FIELDS,
    WellnessAuthError,
    WellnessSyncError,
    fetch_daily_wellness,
    sync_wellness,
)

USERNAME = "test-user"
SLEEP_PATH = f"/wellness-service/wellness/dailySleepData/{USERNAME}?nonSleepBufferMinutes=60&date="
RHR_PATH = "/usersummary-service/usersummary/daily/?calendarDate="
OAUTH_ASSERTION = "OAuth1 token is required for API requests"


def _http_error(status: int = 500) -> Exception:
    """Build a GarthHTTPError the way garth raises it, with an HTTP status code."""
    response = Response()
    response.status_code = status
    return GarthHTTPError(
        msg="Error in request", error=HTTPError(f"{status} Error", response=response)
    )


def _raw_http_error(status: int) -> HTTPError:
    """Build the bare requests.HTTPError garth's token refresh (sso.exchange) raises."""
    response = Response()
    response.status_code = status
    return HTTPError(f"{status} Client Error", response=response)


def _hrv_summary(
    day: str, last_night: Optional[int], weekly: Optional[int], status: str
) -> Dict[str, Any]:
    return {
        "calendarDate": day,
        "weeklyAvg": weekly,
        "lastNightAvg": last_night,
        "lastNight5MinHigh": 70,
        "baseline": {"lowUpper": 36, "balancedLow": 39, "balancedUpper": 49, "markerValue": 0.4},
        "status": status,
        "feedbackPhrase": "HRV_BALANCED_5",
        "createTimeStamp": f"{day}T06:00:00.000",
    }


def _sleep_response(day: str, seconds: Optional[int], score: Optional[int]) -> Dict[str, Any]:
    dto: Dict[str, Any] = {"id": 123, "calendarDate": day, "sleepTimeSeconds": seconds}
    if score is not None:
        dto["sleepScores"] = {"overall": {"value": score, "qualifierKey": "GOOD"}}
    return {"dailySleepDTO": dto, "restingHeartRate": 99}


def _no_sleep_response(day: str) -> Dict[str, Any]:
    """Shape Garmin returns for a night without a recorded sleep."""
    return {"dailySleepDTO": {"id": None, "calendarDate": day, "sleepTimeSeconds": None}}


class FakeGarminApi:
    """Routes garth connectapi paths to canned responses and records calls."""

    def __init__(self) -> None:
        self.hrv: Dict[str, Any] = {}
        self.resting_hr: Dict[str, Any] = {}
        self.sleep: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}
        self.raw: Dict[str, Any] = {}
        self.calls: List[str] = []

    def connectapi(self, path: str) -> Any:
        self.calls.append(path)
        for prefix, error in self.errors.items():
            if path.startswith(prefix):
                raise error
        for prefix, payload in self.raw.items():
            if path.startswith(prefix):
                return payload
        if path.startswith("/hrv-service/hrv/daily/"):
            start, end = path.rsplit("/", 2)[-2:]
            summaries = [v for k, v in sorted(self.hrv.items()) if start <= k <= end]
            return {"hrvSummaries": summaries} if summaries else None
        if path.startswith("/usersummary-service/usersummary/daily/"):
            day = path.split("calendarDate=")[1]
            return self.resting_hr.get(day, {"calendarDate": day, "restingHeartRate": None})
        if path.startswith(f"/wellness-service/wellness/dailySleepData/{USERNAME}"):
            day = path.split("date=")[1]
            return self.sleep.get(day, _no_sleep_response(day))
        raise AssertionError(f"Unexpected path: {path}")


def _mock_garth(api: FakeGarminApi, username: Any = USERNAME) -> Mock:
    mock_garth = Mock()
    mock_garth.client.connectapi.side_effect = api.connectapi
    if callable(username) and not isinstance(username, str):
        type(mock_garth.client).username = property(lambda _self: username())
    else:
        mock_garth.client.username = username
    return mock_garth


def _populated_api() -> FakeGarminApi:
    api = FakeGarminApi()
    api.hrv["2026-11-19"] = _hrv_summary("2026-11-19", 46, 43, "BALANCED")
    api.hrv["2026-11-20"] = _hrv_summary("2026-11-20", 35, 42, "UNBALANCED")
    api.resting_hr["2026-11-19"] = {"calendarDate": "2026-11-19", "restingHeartRate": 51}
    api.resting_hr["2026-11-20"] = {"calendarDate": "2026-11-20", "restingHeartRate": 55}
    api.sleep["2026-11-19"] = _sleep_response("2026-11-19", 25860, 75)
    api.sleep["2026-11-20"] = _sleep_response("2026-11-20", 27000, 83)
    return api


class TestFetchDailyWellness(unittest.TestCase):
    """Contract tests for fetch_daily_wellness."""

    def _fetch(
        self, api: FakeGarminApi, start: date, end: date, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        with patch("fitanalyzer.wellness.garth", _mock_garth(api, **kwargs)):
            return fetch_daily_wellness(start, end, request_delay=0)

    def test_returns_one_row_per_day_with_all_fields(self) -> None:
        rows = self._fetch(_populated_api(), date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual(
            rows,
            [
                {
                    "date": "2026-11-19",
                    "hrv_last_night_avg": 46,
                    "hrv_weekly_avg": 43,
                    "hrv_status": "BALANCED",
                    "resting_hr": 51,
                    "sleep_seconds": 25860,
                    "sleep_score": 75,
                },
                {
                    "date": "2026-11-20",
                    "hrv_last_night_avg": 35,
                    "hrv_weekly_avg": 42,
                    "hrv_status": "UNBALANCED",
                    "resting_hr": 55,
                    "sleep_seconds": 27000,
                    "sleep_score": 83,
                },
            ],
        )

    def test_row_keys_match_wellness_fields(self) -> None:
        rows = self._fetch(_populated_api(), date(2026, 11, 19), date(2026, 11, 19))
        self.assertEqual(list(rows[0].keys()), WELLNESS_FIELDS)

    def test_single_day_range(self) -> None:
        rows = self._fetch(_populated_api(), date(2026, 11, 20), date(2026, 11, 20))
        self.assertEqual([r["date"] for r in rows], ["2026-11-20"])

    def test_start_after_end_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._fetch(_populated_api(), date(2026, 11, 21), date(2026, 11, 20))

    def test_day_without_any_data_yields_row_of_none(self) -> None:
        rows = self._fetch(FakeGarminApi(), date(2026, 11, 22), date(2026, 11, 22))
        self.assertEqual(
            rows,
            [
                {
                    "date": "2026-11-22",
                    "hrv_last_night_avg": None,
                    "hrv_weekly_avg": None,
                    "hrv_status": None,
                    "resting_hr": None,
                    "sleep_seconds": None,
                    "sleep_score": None,
                }
            ],
        )

    def test_missing_hrv_day_inside_range_is_none(self) -> None:
        api = _populated_api()
        del api.hrv["2026-11-20"]
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual(rows[0]["hrv_last_night_avg"], 46)
        self.assertIsNone(rows[1]["hrv_last_night_avg"])
        self.assertIsNone(rows[1]["hrv_status"])
        self.assertEqual(rows[1]["resting_hr"], 55)

    def test_hrv_summary_with_null_values_is_none(self) -> None:
        api = _populated_api()
        api.hrv["2026-11-19"] = _hrv_summary("2026-11-19", None, None, "NONE")
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 19))
        self.assertIsNone(rows[0]["hrv_last_night_avg"])
        self.assertIsNone(rows[0]["hrv_weekly_avg"])
        self.assertEqual(rows[0]["hrv_status"], "NONE")

    def test_sleep_without_scores_keeps_duration(self) -> None:
        api = _populated_api()
        api.sleep["2026-11-19"] = _sleep_response("2026-11-19", 20000, None)
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 19))
        self.assertEqual(rows[0]["sleep_seconds"], 20000)
        self.assertIsNone(rows[0]["sleep_score"])

    def test_resting_hr_comes_from_daily_summary_not_sleep_data(self) -> None:
        # The sleep response carries its own restingHeartRate (99 in the fixture);
        # the daily summary value is the one Garmin Connect shows for the day.
        rows = self._fetch(_populated_api(), date(2026, 11, 19), date(2026, 11, 19))
        self.assertEqual(rows[0]["resting_hr"], 51)

    def test_empty_api_responses_yield_none(self) -> None:
        api = FakeGarminApi()
        api.resting_hr["2026-11-19"] = None
        api.sleep["2026-11-19"] = None
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 19))
        self.assertIsNone(rows[0]["resting_hr"])
        self.assertIsNone(rows[0]["sleep_seconds"])

    def test_hrv_fetched_in_one_request_for_short_range(self) -> None:
        api = _populated_api()
        self._fetch(api, date(2026, 11, 13), date(2026, 11, 26))
        hrv_calls = [c for c in api.calls if c.startswith("/hrv-service/")]
        self.assertEqual(hrv_calls, ["/hrv-service/hrv/daily/2026-11-13/2026-11-26"])

    def test_hrv_range_longer_than_28_days_is_paginated(self) -> None:
        api = _populated_api()
        self._fetch(api, date(2026, 10, 1), date(2026, 11, 26))
        hrv_calls = [c for c in api.calls if c.startswith("/hrv-service/")]
        self.assertEqual(
            hrv_calls,
            [
                "/hrv-service/hrv/daily/2026-10-01/2026-10-28",
                "/hrv-service/hrv/daily/2026-10-29/2026-11-25",
                "/hrv-service/hrv/daily/2026-11-26/2026-11-26",
            ],
        )

    def test_hrv_page_error_keeps_other_pages_and_fields(self) -> None:
        api = _populated_api()
        api.errors["/hrv-service/hrv/daily/2026-10-23/"] = _http_error()
        rows = self._fetch(api, date(2026, 10, 23), date(2026, 11, 20))
        self.assertIsNone(rows[-2]["hrv_last_night_avg"])
        self.assertEqual(rows[-1]["hrv_last_night_avg"], 35)
        self.assertEqual([r["resting_hr"] for r in rows[-2:]], [51, 55])

    def test_resting_hr_error_on_one_day_keeps_other_days_and_fields(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = _http_error()
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["resting_hr"] for r in rows], [None, 55])
        self.assertEqual(rows[0]["hrv_last_night_avg"], 46)
        self.assertEqual(rows[0]["sleep_seconds"], 25860)

    def test_sleep_error_on_one_day_keeps_other_days_and_fields(self) -> None:
        api = _populated_api()
        api.errors[SLEEP_PATH + "2026-11-19"] = _http_error()
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["sleep_seconds"] for r in rows], [None, 27000])
        self.assertIsNone(rows[0]["sleep_score"])
        self.assertEqual(rows[0]["resting_hr"], 51)

    def test_network_error_on_one_day_is_handled(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = RequestsConnectionError("connection reset")
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["resting_hr"] for r in rows], [None, 55])

    def test_invalid_json_on_one_day_is_handled(self) -> None:
        api = _populated_api()
        api.errors[SLEEP_PATH + "2026-11-20"] = ValueError("Expecting value")
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["sleep_seconds"] for r in rows], [25860, None])

    def test_per_day_miss_is_logged_as_warning(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = _http_error()
        with self.assertLogs("fitanalyzer.wellness", level="WARNING") as logs:
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertIn(RHR_PATH + "2026-11-19", "\n".join(logs.output))

    def test_hrv_failing_for_whole_range_raises(self) -> None:
        api = _populated_api()
        api.errors["/hrv-service/"] = _http_error()
        with self.assertRaisesRegex(WellnessSyncError, "HRV"):
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))

    def test_resting_hr_failing_every_day_raises(self) -> None:
        api = _populated_api()
        api.errors["/usersummary-service/"] = _http_error()
        with self.assertRaisesRegex(WellnessSyncError, "resting HR"):
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))

    def test_sleep_failing_every_day_raises(self) -> None:
        api = _populated_api()
        api.errors["/wellness-service/"] = RequestsConnectionError("offline")
        with self.assertRaisesRegex(WellnessSyncError, "sleep"):
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))

    def test_auth_and_rate_limit_errors_are_raised(self) -> None:
        for status in (401, 403, 429):
            with self.subTest(status=status):
                api = _populated_api()
                api.errors[RHR_PATH + "2026-11-20"] = _http_error(status)
                with self.assertRaises(GarthHTTPError) as ctx:
                    self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
                self.assertEqual(ctx.exception.error.response.status_code, status)

    def test_missing_oauth_token_raises_auth_error(self) -> None:
        api = _populated_api()
        api.errors["/hrv-service/"] = AssertionError(OAUTH_ASSERTION)
        with self.assertRaisesRegex(WellnessAuthError, "authenticate_garmin"):
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))

    def test_other_assertion_from_garth_is_a_request_failure(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = AssertionError("Expected dict")
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["resting_hr"] for r in rows], [None, 55])

    def test_waits_between_per_day_requests(self) -> None:
        with patch("fitanalyzer.wellness.garth", _mock_garth(_populated_api())):
            with patch("fitanalyzer.wellness.time.sleep") as sleep:
                fetch_daily_wellness(date(2026, 11, 18), date(2026, 11, 20), request_delay=0.3)
        self.assertEqual([c.args for c in sleep.call_args_list], [(0.3,), (0.3,)])

    def test_default_request_delay_is_between_point_two_and_half_second(self) -> None:
        with patch("fitanalyzer.wellness.garth", _mock_garth(_populated_api())):
            with patch("fitanalyzer.wellness.time.sleep") as sleep:
                fetch_daily_wellness(date(2026, 11, 19), date(2026, 11, 20))
        (delay,) = sleep.call_args.args
        self.assertTrue(0.2 <= delay <= 0.5)

    def test_zero_delay_does_not_sleep(self) -> None:
        with patch("fitanalyzer.wellness.garth", _mock_garth(_populated_api())):
            with patch("fitanalyzer.wellness.time.sleep") as sleep:
                fetch_daily_wellness(date(2026, 11, 19), date(2026, 11, 20), request_delay=0)
        sleep.assert_not_called()

    def test_hrv_from_later_pages_lands_in_rows(self) -> None:
        api = _populated_api()
        api.hrv["2026-10-15"] = _hrv_summary("2026-10-15", 40, 41, "BALANCED")
        api.hrv["2026-11-26"] = _hrv_summary("2026-11-26", 55, 42, "BALANCED")
        rows = {r["date"]: r for r in self._fetch(api, date(2026, 10, 1), date(2026, 11, 26))}
        self.assertEqual(rows["2026-10-15"]["hrv_last_night_avg"], 40)  # page 1
        self.assertEqual(rows["2026-11-19"]["hrv_last_night_avg"], 46)  # page 2
        self.assertEqual(rows["2026-11-20"]["hrv_status"], "UNBALANCED")  # page 2
        self.assertEqual(rows["2026-11-26"]["hrv_last_night_avg"], 55)  # page 3

    def test_unexpected_response_types_are_ignored(self) -> None:
        api = FakeGarminApi()
        api.resting_hr["2026-11-19"] = ["not", "a", "dict"]
        api.sleep["2026-11-19"] = {"dailySleepDTO": "garbage"}
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 19))
        self.assertIsNone(rows[0]["resting_hr"])
        self.assertIsNone(rows[0]["sleep_seconds"])

    def test_unexpected_hrv_payload_is_ignored(self) -> None:
        api = FakeGarminApi()
        api.raw["/hrv-service/"] = {"hrvSummaries": "garbage"}
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 19))
        self.assertIsNone(rows[0]["hrv_status"])

    def test_username_lookup_failure_raises_sleep_error(self) -> None:
        def failing_username() -> str:
            raise _http_error()

        with self.assertRaisesRegex(WellnessSyncError, "sleep"):
            self._fetch(
                _populated_api(), date(2026, 11, 19), date(2026, 11, 19), username=failing_username
            )

    def test_username_lookup_without_oauth_token_raises_auth_error(self) -> None:
        def unauthenticated_username() -> str:
            raise AssertionError(OAUTH_ASSERTION)

        with self.assertRaises(WellnessAuthError):
            self._fetch(
                _populated_api(),
                date(2026, 11, 19),
                date(2026, 11, 19),
                username=unauthenticated_username,
            )

    def test_raw_http_auth_error_from_token_refresh_raises_auth_error(self) -> None:
        # garth refreshes an expired OAuth2 token outside its GarthHTTPError wrapper,
        # so sso.exchange's requests.HTTPError (an OSError) reaches us unwrapped.
        for status in (401, 403):
            with self.subTest(status=status):
                api = _populated_api()
                api.errors["/hrv-service/"] = _raw_http_error(status)
                with self.assertRaisesRegex(WellnessAuthError, "authenticate_garmin"):
                    self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))

    def test_raw_http_rate_limit_from_token_refresh_is_raised(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = _raw_http_error(429)
        with self.assertRaises(HTTPError) as ctx:
            self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual(ctx.exception.response.status_code, 429)

    def test_raw_http_server_error_is_a_per_day_failure(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = _raw_http_error(503)
        rows = self._fetch(api, date(2026, 11, 19), date(2026, 11, 20))
        self.assertEqual([r["resting_hr"] for r in rows], [None, 55])

    def test_progress_gets_each_finished_day(self) -> None:
        progress: List[Dict[str, Any]] = []
        with patch("fitanalyzer.wellness.garth", _mock_garth(_populated_api())):
            rows = fetch_daily_wellness(
                date(2026, 11, 19), date(2026, 11, 20), request_delay=0, progress=progress
            )
        self.assertEqual(progress, rows)

    def test_progress_keeps_rows_when_a_source_fails_for_every_day(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-19"] = _http_error()
        progress: List[Dict[str, Any]] = []
        with patch("fitanalyzer.wellness.garth", _mock_garth(api)):
            with self.assertRaisesRegex(WellnessSyncError, "resting HR"):
                fetch_daily_wellness(
                    date(2026, 11, 19), date(2026, 11, 19), request_delay=0, progress=progress
                )
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["hrv_last_night_avg"], 46)
        self.assertEqual(progress[0]["sleep_score"], 75)
        self.assertIsNone(progress[0]["resting_hr"])

    def test_hrv_failing_for_whole_range_still_fetches_other_sources(self) -> None:
        api = _populated_api()
        api.errors["/hrv-service/"] = _http_error()
        progress: List[Dict[str, Any]] = []
        with patch("fitanalyzer.wellness.garth", _mock_garth(api)):
            with self.assertRaisesRegex(WellnessSyncError, "HRV"):
                fetch_daily_wellness(
                    date(2026, 11, 19), date(2026, 11, 20), request_delay=0, progress=progress
                )
        self.assertEqual([r["resting_hr"] for r in progress], [51, 55])
        self.assertEqual([r["sleep_score"] for r in progress], [75, 83])

    def test_progress_keeps_days_before_a_rate_limit(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-20"] = _http_error(429)
        progress: List[Dict[str, Any]] = []
        with patch("fitanalyzer.wellness.garth", _mock_garth(api)):
            with self.assertRaises(GarthHTTPError):
                fetch_daily_wellness(
                    date(2026, 11, 19), date(2026, 11, 20), request_delay=0, progress=progress
                )
        self.assertEqual([r["date"] for r in progress], ["2026-11-19"])

    def test_raises_import_error_without_garth(self) -> None:
        with patch("fitanalyzer.wellness.garth", None):
            with self.assertRaises(ImportError):
                fetch_daily_wellness(date(2026, 11, 19), date(2026, 11, 19))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class TestSyncWellness(unittest.TestCase):
    """Contract tests for sync_wellness (CSV upsert by date)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self._tmp.name) / "nested" / "wellness_daily.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _sync(
        self, rows: List[Dict[str, Any]], days: int = 2, end: date = date(2026, 11, 20)
    ) -> Any:
        with patch("fitanalyzer.wellness.fetch_daily_wellness", return_value=rows) as fetch:
            result = sync_wellness(self.csv_path, days, end=end)
        return fetch, result

    @staticmethod
    def _row(day: str, **values: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {field: None for field in WELLNESS_FIELDS}
        row["date"] = day
        row.update(values)
        return row

    def test_creates_csv_with_header_and_rows(self) -> None:
        rows = [self._row("2026-11-19", resting_hr=51, hrv_status="BALANCED")]
        self._sync(rows)
        with self.csv_path.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        self.assertEqual(header, WELLNESS_FIELDS)
        written = _read_csv(self.csv_path)
        self.assertEqual(written[0]["resting_hr"], "51")
        self.assertEqual(written[0]["hrv_status"], "BALANCED")
        self.assertEqual(written[0]["sleep_score"], "")

    def test_fetch_window_covers_days_ending_at_end(self) -> None:
        fetch, _ = self._sync([], days=14, end=date(2026, 11, 26))
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args, (date(2026, 11, 13), date(2026, 11, 26)))

    def test_end_defaults_to_today(self) -> None:
        with patch("fitanalyzer.wellness.fetch_daily_wellness", return_value=[]) as fetch:
            sync_wellness(self.csv_path, 1)
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args, (date.today(), date.today()))

    def test_accepts_string_path(self) -> None:
        with patch(
            "fitanalyzer.wellness.fetch_daily_wellness", return_value=[self._row("2026-11-19")]
        ):
            sync_wellness(str(self.csv_path), 1, end=date(2026, 11, 19))
        self.assertTrue(self.csv_path.exists())

    def test_returns_fetched_rows(self) -> None:
        rows = [self._row("2026-11-19", resting_hr=51)]
        _, result = self._sync(rows)
        self.assertEqual(result, rows)

    def test_days_below_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            sync_wellness(self.csv_path, 0)

    def test_upsert_replaces_existing_date_and_keeps_others(self) -> None:
        self._sync([self._row("2026-11-18", resting_hr=50), self._row("2026-11-19", resting_hr=60)])
        self._sync([self._row("2026-11-19", resting_hr=51), self._row("2026-11-20", resting_hr=55)])
        written = _read_csv(self.csv_path)
        self.assertEqual(
            [(r["date"], r["resting_hr"]) for r in written],
            [("2026-11-18", "50"), ("2026-11-19", "51"), ("2026-11-20", "55")],
        )

    def test_missing_new_values_do_not_erase_existing_values(self) -> None:
        self._sync([self._row("2026-11-19", resting_hr=51, sleep_score=75)])
        self._sync([self._row("2026-11-19", resting_hr=52)])
        written = _read_csv(self.csv_path)
        self.assertEqual(written[0]["resting_hr"], "52")
        self.assertEqual(written[0]["sleep_score"], "75")

    def test_rows_are_sorted_by_date(self) -> None:
        self._sync([self._row("2026-11-20"), self._row("2026-11-18")])
        self._sync([self._row("2026-11-19")])
        self.assertEqual(
            [r["date"] for r in _read_csv(self.csv_path)],
            ["2026-11-18", "2026-11-19", "2026-11-20"],
        )

    def test_existing_file_with_only_header_is_handled(self) -> None:
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.write_text(",".join(WELLNESS_FIELDS) + "\n", encoding="utf-8")
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(len(_read_csv(self.csv_path)), 1)

    def _write_existing(self, text: str, encoding: str = "utf-8") -> None:
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.write_text(text, encoding=encoding)

    def test_existing_rows_without_date_are_dropped(self) -> None:
        header = ",".join(WELLNESS_FIELDS)
        self._write_existing(f"{header}\n2026-11-18,,,,50,,\n,,,,51,,\n")
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(
            [r["date"] for r in _read_csv(self.csv_path)], ["2026-11-18", "2026-11-19"]
        )

    def test_existing_file_with_bom_keeps_its_rows(self) -> None:
        header = ",".join(WELLNESS_FIELDS)
        self._write_existing(f"{header}\n2026-11-18,,,,50,,\n", encoding="utf-8-sig")
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(
            [(r["date"], r["resting_hr"]) for r in _read_csv(self.csv_path)],
            [("2026-11-18", "50"), ("2026-11-19", "51")],
        )

    def test_extra_columns_and_their_values_are_preserved(self) -> None:
        header = ",".join(WELLNESS_FIELDS + ["notes"])
        self._write_existing(f"{header}\n2026-11-19,,,,50,,,felt rough\n")
        self._sync([self._row("2026-11-19", resting_hr=51), self._row("2026-11-20")])
        with self.csv_path.open(encoding="utf-8") as f:
            self.assertEqual(f.readline().strip().split(","), WELLNESS_FIELDS + ["notes"])
        written = _read_csv(self.csv_path)
        self.assertEqual(written[0]["notes"], "felt rough")
        self.assertEqual(written[0]["resting_hr"], "51")
        self.assertEqual(written[1]["notes"], "")

    def test_duplicate_existing_dates_collapse_into_one_row(self) -> None:
        header = ",".join(WELLNESS_FIELDS)
        self._write_existing(f"{header}\n2026-11-18,40,,,50,,\n2026-11-18,,,,,,80\n")
        self._sync([self._row("2026-11-19")])
        written = _read_csv(self.csv_path)
        self.assertEqual([r["date"] for r in written], ["2026-11-18", "2026-11-19"])
        self.assertEqual(
            (written[0]["hrv_last_night_avg"], written[0]["resting_hr"], written[0]["sleep_score"]),
            ("40", "50", "80"),
        )

    def test_non_empty_file_without_date_column_raises_and_is_untouched(self) -> None:
        original = "this is not\nthe wellness csv\n"
        self._write_existing(original)
        with self.assertRaises(WellnessSyncError):
            self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(self.csv_path.read_text(encoding="utf-8"), original)

    def test_non_empty_file_with_no_dated_rows_raises_and_is_untouched(self) -> None:
        original = ",".join(WELLNESS_FIELDS) + "\n,,,,51,,\n"
        self._write_existing(original)
        with self.assertRaises(WellnessSyncError):
            self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(self.csv_path.read_text(encoding="utf-8"), original)

    def test_empty_existing_file_is_replaced(self) -> None:
        self._write_existing("")
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual([r["date"] for r in _read_csv(self.csv_path)], ["2026-11-19"])

    def test_write_is_atomic_via_os_replace(self) -> None:
        self._sync([self._row("2026-11-18", resting_hr=50)])
        with patch("fitanalyzer.wellness.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual([r["date"] for r in _read_csv(self.csv_path)], ["2026-11-18"])
        self.assertEqual(os.listdir(self.csv_path.parent), [self.csv_path.name])

    def test_fetch_error_leaves_csv_untouched(self) -> None:
        self._sync([self._row("2026-11-18", resting_hr=50)])
        before = self.csv_path.read_text(encoding="utf-8")
        with patch(
            "fitanalyzer.wellness.fetch_daily_wellness",
            side_effect=WellnessSyncError("resting HR failed"),
        ):
            with self.assertRaises(WellnessSyncError):
                sync_wellness(self.csv_path, 2, end=date(2026, 11, 20))
        self.assertEqual(self.csv_path.read_text(encoding="utf-8"), before)

    def test_new_csv_gets_default_permissions_from_umask(self) -> None:
        old_umask = os.umask(0o022)
        try:
            self._sync([self._row("2026-11-19", resting_hr=51)])
        finally:
            os.umask(old_umask)
        self.assertEqual(stat.S_IMODE(self.csv_path.stat().st_mode), 0o644)

    def test_existing_csv_keeps_its_permissions(self) -> None:
        self._sync([self._row("2026-11-18", resting_hr=50)])
        os.chmod(self.csv_path, 0o640)
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertEqual(stat.S_IMODE(self.csv_path.stat().st_mode), 0o640)

    def test_symlinked_csv_is_updated_through_the_link(self) -> None:
        real = Path(self._tmp.name) / "real" / "wellness.csv"
        real.parent.mkdir()
        header = ",".join(WELLNESS_FIELDS)
        real.write_text(f"{header}\n2026-11-18,,,,50,,\n", encoding="utf-8")
        self.csv_path.parent.mkdir(parents=True)
        self.csv_path.symlink_to(real)
        self._sync([self._row("2026-11-19", resting_hr=51)])
        self.assertTrue(self.csv_path.is_symlink())
        self.assertEqual(
            [(r["date"], r["resting_hr"]) for r in _read_csv(real)],
            [("2026-11-18", "50"), ("2026-11-19", "51")],
        )
        self.assertEqual(os.listdir(real.parent), [real.name])

    def test_duplicate_extra_column_names_are_written_once(self) -> None:
        header = ",".join(WELLNESS_FIELDS + ["note", "note"])
        self._write_existing(f"{header}\n2026-11-18,,,,50,,,a,b\n")
        self._sync([self._row("2026-11-19", resting_hr=51)])
        with self.csv_path.open(encoding="utf-8") as f:
            written_header = f.readline().strip().split(",")
        self.assertEqual(written_header, WELLNESS_FIELDS + ["note"])
        self.assertEqual(_read_csv(self.csv_path)[0]["note"], "b")

    def test_rows_fetched_before_a_failure_are_written_then_error_is_raised(self) -> None:
        self._sync([self._row("2026-11-18", resting_hr=50)])

        def fail_after_one_day(*_args: Any, progress: List[Dict[str, Any]], **_kw: Any) -> None:
            progress.append(self._row("2026-11-19", resting_hr=51))
            raise WellnessSyncError("sleep failed")

        with patch("fitanalyzer.wellness.fetch_daily_wellness", side_effect=fail_after_one_day):
            with self.assertRaises(WellnessSyncError):
                sync_wellness(self.csv_path, 2, end=date(2026, 11, 20))
        self.assertEqual(
            [(r["date"], r["resting_hr"]) for r in _read_csv(self.csv_path)],
            [("2026-11-18", "50"), ("2026-11-19", "51")],
        )

    def test_request_delay_is_passed_to_fetch(self) -> None:
        with patch("fitanalyzer.wellness.fetch_daily_wellness", return_value=[]) as fetch:
            sync_wellness(self.csv_path, 2, end=date(2026, 11, 20), request_delay=0.4)
        fetch.assert_called_once_with(
            date(2026, 11, 19), date(2026, 11, 20), request_delay=0.4, progress=ANY
        )


class TestSyncWellnessEndToEnd(unittest.TestCase):
    """fetch + CSV upsert together, through the fake Garmin router."""

    def test_sync_writes_fetched_values_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "wellness_daily.csv"
            with patch("fitanalyzer.wellness.garth", _mock_garth(_populated_api())):
                sync_wellness(csv_path, 2, end=date(2026, 11, 20), request_delay=0)
            written = _read_csv(csv_path)
        self.assertEqual(
            written,
            [
                {
                    "date": "2026-11-19",
                    "hrv_last_night_avg": "46",
                    "hrv_weekly_avg": "43",
                    "hrv_status": "BALANCED",
                    "resting_hr": "51",
                    "sleep_seconds": "25860",
                    "sleep_score": "75",
                },
                {
                    "date": "2026-11-20",
                    "hrv_last_night_avg": "35",
                    "hrv_weekly_avg": "42",
                    "hrv_status": "UNBALANCED",
                    "resting_hr": "55",
                    "sleep_seconds": "27000",
                    "sleep_score": "83",
                },
            ],
        )

    def _sync_through_api(self, api: FakeGarminApi, csv_path: Path, days: int) -> None:
        with patch("fitanalyzer.wellness.garth", _mock_garth(api)):
            sync_wellness(csv_path, days, end=date(2026, 11, 20), request_delay=0)

    def test_one_day_with_one_failed_request_writes_the_rest(self) -> None:
        api = _populated_api()
        api.errors[RHR_PATH + "2026-11-20"] = _http_error()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "wellness_daily.csv"
            with self.assertRaisesRegex(WellnessSyncError, "resting HR"):
                self._sync_through_api(api, csv_path, 1)
            written = _read_csv(csv_path)
        self.assertEqual(
            [
                (r["date"], r["hrv_last_night_avg"], r["resting_hr"], r["sleep_score"])
                for r in written
            ],
            [("2026-11-20", "35", "", "83")],
        )

    def test_rate_limit_late_in_backfill_keeps_earlier_days(self) -> None:
        api = _populated_api()
        api.errors[SLEEP_PATH + "2026-11-20"] = _http_error(429)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "wellness_daily.csv"
            with self.assertRaises(GarthHTTPError):
                self._sync_through_api(api, csv_path, 2)
            written = _read_csv(csv_path)
        self.assertEqual([(r["date"], r["resting_hr"]) for r in written], [("2026-11-19", "51")])

    def test_auth_failure_before_any_day_leaves_no_file(self) -> None:
        api = _populated_api()
        api.errors["/hrv-service/"] = AssertionError(OAUTH_ASSERTION)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "wellness_daily.csv"
            with self.assertRaises(WellnessAuthError):
                self._sync_through_api(api, csv_path, 2)
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
