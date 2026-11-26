"""--tz is validated while the arguments are parsed, before any FIT file is touched."""

import sys
from unittest import mock

import pytest

from fitanalyzer import cli
from fitanalyzer.cli import parse_arguments


@pytest.mark.parametrize("name", ["Europe/Helsinki", "UTC", "Etc/GMT-3", "America/New_York"])
def test_accepts_iana_zone_names(name: str) -> None:
    assert parse_arguments(["a.fit", "--tz", name]).tz == name


def test_default_zone_is_valid() -> None:
    assert parse_arguments(["a.fit"]).tz == "Europe/Helsinki"


@pytest.mark.parametrize(
    "name", ["Europe/Helsenki", "EST5EDT,M3.2.0,M11.1.0", "UTC+3", "", "../etc/passwd"]
)
def test_rejects_unknown_zone_with_usage_error(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_arguments(["a.fit", "--tz", name])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--tz" in err
    assert "IANA" in err


def test_main_exits_before_any_work_on_unknown_zone() -> None:
    with (
        mock.patch.object(sys, "argv", ["fitanalyzer", "a.fit", "--tz", "Europe/Helsenki"]),
        mock.patch.object(cli, "main_with_args") as work,
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    work.assert_not_called()
