#!/usr/bin/env python3
"""
Helper script to create a .env file for Garmin credentials.
"""

import getpass
from pathlib import Path

from .constants import DEFAULT_FTP, DEFAULT_HR_MAX, DEFAULT_HR_REST

__all__ = ["create_env_file"]


def create_env_file() -> None:
    """Create .env file with Garmin credentials and training parameters."""
    print("🔐 Garmin Connect Credentials Setup")
    print("=" * 50)
    print("\nThis will create a .env file to store your credentials securely.")
    print("⚠️  Make sure .env is in your .gitignore!\n")

    env_path = Path(".env")

    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    email = input("Garmin Connect Email: ")
    password = getpass.getpass("Garmin Connect Password: ")

    ftp = input(f"Your FTP (Functional Threshold Power) in watts [default: {DEFAULT_FTP}]: ")
    ftp = ftp.strip() or str(DEFAULT_FTP)

    hr_rest = input(f"Your resting heart rate [default: {DEFAULT_HR_REST}]: ")
    hr_rest = hr_rest.strip() or str(DEFAULT_HR_REST)

    hr_max = input(f"Your maximum heart rate [default: {DEFAULT_HR_MAX}]: ")
    hr_max = hr_max.strip() or str(DEFAULT_HR_MAX)

    env_content = f"""# Garmin Connect Credentials
# DO NOT COMMIT THIS FILE TO GIT!
GARMIN_EMAIL={email}
GARMIN_PASSWORD={password}

# Training Parameters
FTP={ftp}
HR_REST={hr_rest}
HR_MAX={hr_max}
"""

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)

    # Set restrictive permissions
    env_path.chmod(0o600)

    print("\n✅ .env file created successfully!")
    print(f"   Location: {env_path.absolute()}")
    print("\n📝 Add this to your .gitignore:")
    print("   .env")

    # Check/create .gitignore
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gitignore_content = f.read()

        if ".env" not in gitignore_content:
            response = input("\n.env not found in .gitignore. Add it now? (y/n): ")
            if response.lower() == "y":
                with open(gitignore_path, "a", encoding="utf-8") as f:
                    f.write("\n# Environment variables\n.env\n")
                print("✅ Added .env to .gitignore")
    else:
        response = input("\nNo .gitignore found. Create one? (y/n): ")
        if response.lower() == "y":
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write("# Environment variables\n.env\n")
            print("✅ Created .gitignore with .env entry")

    print("\n💡 Usage:")
    print("   python garmin_sync.py")
    print("\n   The script will automatically load credentials from .env")


if __name__ == "__main__":
    create_env_file()
