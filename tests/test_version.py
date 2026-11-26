"""The package version reported at runtime matches pyproject.toml."""

import ast
import re
import sys
from pathlib import Path

import fitanalyzer

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"


def test_dunder_version_matches_pyproject() -> None:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert match is not None
    assert fitanalyzer.__version__ == match.group(1)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~ ;\[]", req, maxsplit=1)[0].lower() for req in requirements}


def test_setup_py_install_requires_matches_pyproject_dependencies() -> None:
    if sys.version_info >= (3, 11):
        import tomllib

        pyproject_deps = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
            "dependencies"
        ]
    else:  # pragma: no cover - tomllib is 3.11+
        import pytest

        pytest.skip("tomllib needs Python 3.11")
    setup_call = next(
        node
        for node in ast.walk(ast.parse(SETUP_PY.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    install_requires = next(kw.value for kw in setup_call.keywords if kw.arg == "install_requires")
    assert _requirement_names(ast.literal_eval(install_requires)) == _requirement_names(
        pyproject_deps
    )
