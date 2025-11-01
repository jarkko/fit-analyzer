#!/usr/bin/env python3
"""Bump version in pyproject.toml (single source of truth)."""

import argparse
import re
import sys
import tomllib
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def bump_version(current: str, bump_type: str) -> str:
    """Bump version according to semver."""
    major, minor, patch = map(int, current.split("."))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        return bump_type  # Explicit version provided


def update_pyproject_toml(new_version: str):
    """Update version in pyproject.toml."""
    path = Path("pyproject.toml")
    content = path.read_text()

    # Replace version line
    new_content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)

    path.write_text(new_content)
    print(f"✅ Updated pyproject.toml to {new_version}")


def main():
    """Bump version."""
    parser = argparse.ArgumentParser(description="Bump project version")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        nargs="?",
        help="Type of version bump (or provide explicit version)",
    )
    parser.add_argument("--version", help="Explicit version number (e.g., 1.2.3)")

    args = parser.parse_args()

    if not args.bump_type and not args.version:
        # Just show current version
        current = get_current_version()
        print(f"Current version: {current}")
        return 0

    try:
        current_version = get_current_version()
        print(f"Current version: {current_version}")

        if args.version:
            new_version = args.version
        else:
            new_version = bump_version(current_version, args.bump_type)

        print(f"New version: {new_version}")

        # Confirm
        response = input("Update version? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return 1

        update_pyproject_toml(new_version)

        print(f"\n✅ Version bumped: {current_version} → {new_version}")
        print(f"\nNext steps:")
        print(
            f"  1. Commit: git add pyproject.toml && git commit -m 'chore: bump version to {new_version}'"
        )
        print(f"  2. Tag: git tag -a v{new_version} -m 'Release v{new_version}'")
        print(f"  3. Push: git push && git push origin v{new_version}")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
