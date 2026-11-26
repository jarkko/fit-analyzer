"""Tests for the local-time helpers in fitanalyzer.formatting."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dateutil import tz

from fitanalyzer.formatting import local_zone, to_local_time


def test_local_zone_uses_zoneinfo_for_iana_names() -> None:
    assert local_zone("Europe/Helsinki") == ZoneInfo("Europe/Helsinki")


def test_local_zone_falls_back_to_dateutil_for_names_zoneinfo_rejects() -> None:
    # zoneinfo only takes IANA keys; dateutil also understands POSIX TZ strings
    zone = local_zone("EST5EDT,M3.2.0,M11.1.0")
    assert zone is not None
    assert not isinstance(zone, ZoneInfo)
    assert datetime(2025, 1, 1, 17, tzinfo=timezone.utc).astimezone(zone).hour == 12


def test_local_zone_returns_none_for_unknown_name() -> None:
    assert local_zone("Not/A_Zone") is None


def test_to_local_time_converts_pandas_timestamp() -> None:
    import pandas as pd

    stamp = pd.Timestamp("2025-10-25 15:26:56", tz="UTC")
    local = to_local_time(stamp, "Europe/Helsinki")
    assert local.strftime("%Y-%m-%d %H:%M:%S") == "2025-10-25 18:26:56"


def test_to_local_time_converts_stdlib_datetime_across_midnight() -> None:
    moment = datetime(2025, 10, 25, 15, 26, 56, tzinfo=tz.UTC)
    assert to_local_time(moment, "Pacific/Kiritimati").date().isoformat() == "2025-10-26"
