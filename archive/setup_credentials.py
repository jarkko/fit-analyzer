#!/usr/bin/env python3
"""
Helper script to create a .env file for Garmin credentials
"""

import getpass
from pathlib import Path

def create_env_file():
    print("🔐 Garmin Connect Credentials Setup")
    print("=" * 50)
    print("\nThis will create a .env file to store your credentials securely.")
    print("⚠️  Make sure .env is in your .gitignore!\n")

    env_path = Path(".env")

    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    email = input("Garmin Connect Email: ")
    password = getpass.getpass("Garmin Connect Password: ")

    ftp = input("Your FTP (Functional Threshold Power) in watts [default: 300]: ")
    ftp = ftp.strip() or "300"

    hr_rest = input("Your resting heart rate [default: 50]: ")
    hr_rest = hr_rest.strip() or "50"

    hr_max = input("Your maximum heart rate [default: 190]: ")
    hr_max = hr_max.strip() or "190"

    env_content = f"""# Garmin Connect Credentials
# DO NOT COMMIT THIS FILE TO GIT!
GARMIN_EMAIL={email}
GARMIN_PASSWORD={password}

# Training Parameters
FTP={ftp}
HR_REST={hr_rest}
HR_MAX={hr_max}
"""

    with open(env_path, 'w') as f:
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
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()

        if '.env' not in gitignore_content:
            response = input("\n.env not found in .gitignore. Add it now? (y/n): ")
            if response.lower() == 'y':
                with open(gitignore_path, 'a') as f:
                    f.write("\n# Environment variables\n.env\n")
                print("✅ Added .env to .gitignore")
    else:
        response = input("\nNo .gitignore found. Create one? (y/n): ")
        if response.lower() == 'y':
            with open(gitignore_path, 'w') as f:
                f.write("# Environment variables\n.env\n")
            print("✅ Created .gitignore with .env entry")

    print("\n💡 Usage:")
    print("   python garmin_sync.py")
    print("\n   The script will automatically load credentials from .env")

if __name__ == "__main__":
    create_env_file()
