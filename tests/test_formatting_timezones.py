"""Tests for the local-time helpers in fitanalyzer.formatting.

Every conversion runs for both stdlib datetimes (what fitparse returns) and pandas
Timestamps (what the record DataFrames hold), since pandas 2.x and 3.x convert
Timestamps differently from the stdlib.
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from fitanalyzer.formatting import local_zone, to_local_time

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
SETUP_PY = Path(__file__).parent.parent / "setup.py"


def _as_datetime(year: int, month: int, day: int, hour: int, minute: int, tz: Any) -> Any:
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def _as_timestamp(year: int, month: int, day: int, hour: int, minute: int, tz: Any) -> Any:
    return pd.Timestamp(datetime(year, month, day, hour, minute, tzinfo=tz))


MAKERS: list[Callable[..., Any]] = [_as_datetime, _as_timestamp]
MAKER_IDS = ["datetime", "Timestamp"]
both_types = pytest.mark.parametrize("make", MAKERS, ids=MAKER_IDS)


@pytest.fixture
def machine_in_kiritimati() -> Iterator[None]:
    """Set the process timezone far from UTC, so a naive-as-local bug shows up."""
    if not hasattr(time, "tzset"):  # pragma: no cover - Windows
        pytest.skip("time.tzset is not available")
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Pacific/Kiritimati"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_local_zone_returns_zoneinfo_for_iana_names() -> None:
    assert local_zone("Europe/Helsinki") == ZoneInfo("Europe/Helsinki")
    assert local_zone("UTC") == ZoneInfo("UTC")


@pytest.mark.parametrize(
    "name",
    ["Europe/Helsenki", "Not/A_Zone", "EST5EDT,M3.2.0,M11.1.0", "", "../etc/passwd"],
)
def test_local_zone_rejects_names_zoneinfo_does_not_know(name: str) -> None:
    with pytest.raises(ValueError, match="IANA"):
        local_zone(name)


@both_types
@pytest.mark.parametrize("name", ["Europe/Helsenki", "EST5EDT,M3.2.0,M11.1.0"])
def test_to_local_time_rejects_unknown_zone_instead_of_returning_utc(
    make: Callable[..., Any], name: str
) -> None:
    with pytest.raises(ValueError):
        to_local_time(make(2025, 10, 25, 15, 26, timezone.utc), name)


@both_types
def test_converts_utc_to_configured_zone(make: Callable[..., Any]) -> None:
    local = to_local_time(make(2025, 10, 25, 15, 26, timezone.utc), "Europe/Helsinki")
    assert local.strftime("%Y-%m-%d %H:%M") == "2025-10-25 18:26"  # EEST, UTC+3
    assert local.utcoffset() == timedelta(hours=3)
    assert isinstance(local, type(make(2025, 1, 1, 0, 0, timezone.utc)))


@both_types
def test_conversion_crosses_midnight(make: Callable[..., Any]) -> None:
    local = to_local_time(make(2025, 10, 25, 15, 26, timezone.utc), "Pacific/Kiritimati")
    assert local.strftime("%Y-%m-%d %H:%M") == "2025-10-26 05:26"  # UTC+14


@both_types
def test_conversion_follows_dst(make: Callable[..., Any]) -> None:
    # Helsinki switches to summer time at 01:00 UTC on 2025-03-30
    before = to_local_time(make(2025, 3, 30, 0, 30, timezone.utc), "Europe/Helsinki")
    after = to_local_time(make(2025, 3, 30, 1, 30, timezone.utc), "Europe/Helsinki")
    assert before.strftime("%H:%M") == "02:30"
    assert after.strftime("%H:%M") == "04:30"


@both_types
def test_aware_input_in_another_zone_keeps_its_instant(make: Callable[..., Any]) -> None:
    eastern = make(2025, 1, 1, 12, 0, ZoneInfo("America/New_York"))  # 17:00 UTC
    local = to_local_time(eastern, "Europe/Helsinki")
    assert local.strftime("%Y-%m-%d %H:%M") == "2025-01-01 19:00"
    assert local == eastern


@both_types
def test_naive_input_is_treated_as_utc_whatever_the_machine_zone(
    make: Callable[..., Any], machine_in_kiritimati: None
) -> None:
    naive = make(2025, 10, 25, 15, 26, None)
    local = to_local_time(naive, "Europe/Helsinki")
    assert local.strftime("%Y-%m-%d %H:%M") == "2025-10-25 18:26"


def test_tzdata_is_a_dependency() -> None:
    # zoneinfo falls back to the tzdata package where the OS has no tz database
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    deps = re.search(r"^dependencies = \[(.*?)\]", pyproject, re.M | re.S)
    assert deps is not None
    assert re.search(r'"tzdata\b', deps.group(1))
    install_requires = re.search(r"install_requires=\[(.*?)\]", SETUP_PY.read_text("utf-8"), re.S)
    assert install_requires is not None
    assert re.search(r'"tzdata\b', install_requires.group(1))
