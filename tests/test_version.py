"""The package version reported at runtime matches pyproject.toml."""

import re
from pathlib import Path

import fitanalyzer

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def test_dunder_version_matches_pyproject() -> None:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert match is not None
    assert fitanalyzer.__version__ == match.group(1)
