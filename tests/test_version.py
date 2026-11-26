"""The package version reported at runtime matches pyproject.toml."""

import ast
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import fitanalyzer

ROOT = Path(__file__).parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SETUP_PY = ROOT / "setup.py"


def test_dunder_version_matches_pyproject() -> None:
    match = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert match is not None
    assert fitanalyzer.__version__ == match.group(1)


def _requirement_specs(requirements: list[str]) -> dict[str, tuple[str, str, str]]:
    """Map each normalized package name to its (extras, version specifier, marker)."""
    specs = {}
    for line in requirements:
        req = Requirement(line)
        specs[canonicalize_name(req.name)] = (
            ",".join(sorted(req.extras)),
            str(req.specifier),
            str(req.marker or ""),
        )
    return specs


def test_requirement_specs_tell_version_specifiers_apart() -> None:
    assert _requirement_specs(["pandas>=2.0.0"]) != _requirement_specs(["pandas>=2.1.0"])
    assert _requirement_specs(["tzdata"]) != _requirement_specs(["tzdata>=2024.1"])
    assert _requirement_specs(["Python_Dateutil >= 2.8.0"]) == _requirement_specs(
        ["python-dateutil>=2.8.0"]
    )


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
    # Names and version specifiers: a bump in one file only must fail here
    assert _requirement_specs(ast.literal_eval(install_requires)) == _requirement_specs(
        pyproject_deps
    )


def test_packaging_is_a_declared_dev_dependency() -> None:
    # This module imports packaging directly; don't rely on pytest pulling it in
    dev_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert re.search(r"^packaging\b", dev_requirements, re.M)
    dev_extras = re.search(r"^dev = \[(.*?)\]", PYPROJECT.read_text(encoding="utf-8"), re.M | re.S)
    assert dev_extras is not None
    assert re.search(r'"packaging\b', dev_extras.group(1))
