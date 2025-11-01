#!/usr/bin/env python3
"""Check that version is consistent across setup.py and pyproject.toml."""

import sys
import tomllib
from pathlib import Path


def get_pyproject_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def get_setup_version():
    """Get version from setup.py by parsing it."""
    # Since setup.py now reads from pyproject.toml, just verify it does so
    setup_path = Path("setup.py")
    content = setup_path.read_text()

    # Check that setup.py uses pyproject.toml as source
    if "pyproject.toml" in content and 'version = pyproject["project"]["version"]' in content:
        # Setup.py correctly reads from pyproject.toml
        return get_pyproject_version()

    # Fallback: try to extract hardcoded version
    import re

    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)

    return None


def main():
    """Check version consistency."""
    try:
        pyproject_version = get_pyproject_version()
        setup_version = get_setup_version()

        if pyproject_version != setup_version:
            print(f"❌ Version mismatch detected!")
            print(f"   pyproject.toml: {pyproject_version}")
            print(f"   setup.py:       {setup_version}")
            print(f"\n   Both files must have the same version.")
            return 1

        print(f"✅ Version consistent: {pyproject_version}")
        return 0

    except Exception as e:
        print(f"❌ Error checking versions: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
