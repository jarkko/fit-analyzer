"""
Garmin Connect authentication module.

Handles authentication to Garmin Connect using the garth library,
with support for session token caching to avoid repeated logins.
"""

import getpass
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .garth_utils import GARTH_AVAILABLE, garth

__all__ = ["check_and_install_garth", "authenticate_garmin"]


def check_and_install_garth() -> bool:
    """Check if garth is installed, offer to install if not.

    Returns:
        True if garth is available, False otherwise.
    """
    if GARTH_AVAILABLE:
        return True

    print("📦 garth library not found.")
    print("\n⚠️  Please install it using one of these methods:")
    print("   1. If using the venv (recommended):")
    print("      source .venv/bin/activate")
    print("      pip install garth")
    print("\n   2. Or run directly with venv Python:")
    print("      .venv/bin/python garmin_sync.py")
    print("\n   3. Or use make command:")
    print("      make install-dev  # installs all dependencies")
    print("")

    response = input("Would you like to try auto-installing now? (y/n): ")
    if response.lower() == "y":
        print("Installing garth...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "garth"])
            print("✅ garth installed successfully!")
            print("Please restart the script to use the newly installed library.")
            return False  # Still need to restart
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("\n💡 If you see 'externally-managed-environment' error:")
            print("   You're using system Python. Please use the venv instead.")
            print("   Run: source .venv/bin/activate")
            print("   Then: pip install garth")
            return False

    print("❌ Cannot proceed without garth library.")
    return False


def _try_resume_session(token_path: Path) -> bool:
    """Try to resume an existing Garmin session.

    Args:
        token_path: Path to the stored session token.

    Returns:
        True if session resumed successfully, False otherwise.
    """
    if not token_path.exists():
        return False

    try:
        garth.resume(str(token_path))
        # Test if session is valid
        _ = garth.client.username
        print("✅ Resumed existing Garmin Connect session")
        return True
    except (OSError, RuntimeError, ValueError, AttributeError) as e:
        print(f"⚠️  Saved session expired or invalid: {e}")
        print("   Need to re-authenticate...")
        return False


def _get_credential(value: Optional[str], env_var: str, prompt: str, secure: bool = False) -> str:
    """Get a credential from value, environment variable, or user input.

    Args:
        value: Explicit value provided.
        env_var: Environment variable name to check.
        prompt: User prompt if value not found.
        secure: If True, use getpass for secure input.

    Returns:
        The credential value.
    """
    if value:
        return value

    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    if secure:
        return getpass.getpass(prompt)
    return input(prompt)


def _handle_auth_error(error: Exception) -> None:
    """Handle authentication errors with helpful messages.

    Args:
        error: The exception that occurred during authentication.
    """
    print(f"❌ Authentication failed: {error}")
    error_str = str(error).lower()
    if "mfa" in error_str or "verification" in error_str:
        print("\n💡 If you have MFA enabled, you may need to:")
        print("   1. Generate an app-specific password in your Garmin account")
        print("   2. Or disable MFA temporarily during first setup")


def _perform_login(email: str, password: str, token_path: Path) -> bool:
    """Perform Garmin login and save session.

    Args:
        email: Garmin account email.
        password: Garmin account password.
        token_path: Path to save session token.

    Returns:
        True if login successful, False otherwise.
    """
    try:
        print("🔐 Authenticating with Garmin Connect...")
        garth.login(email, password)  # type: ignore[no-untyped-call]

        # Save credentials for next time
        token_path.parent.mkdir(parents=True, exist_ok=True)
        garth.save(str(token_path))
        print("✅ Authentication successful! Session saved.")
        return True
    except (OSError, RuntimeError, ValueError) as e:
        _handle_auth_error(e)
        return False


def authenticate_garmin(
    email: Optional[str] = None, password: Optional[str] = None, token_store: str = "~/.garth"
) -> bool:
    """Authenticate with Garmin Connect and manage session tokens.

    Handles authentication to Garmin Connect using the garth library, with support
    for session token caching to avoid repeated logins. Attempts to resume an
    existing session first, and only prompts for credentials if needed.

    Args:
        email: Garmin Connect account email. If None, tries GARMIN_EMAIL env var,
               then prompts user for input.
        password: Garmin Connect account password. If None, tries GARMIN_PASSWORD
                  env var, then prompts securely using getpass.
        token_store: Path to store authentication tokens for session persistence.
                     Supports tilde (~) expansion for home directory.
                     Default: "~/.garth"

    Returns:
        bool: True if authentication successful (new or resumed session),
              False if authentication failed.

    Raises:
        ImportError: If garth library is not installed or not available.

    Example:
        >>> # Auto-authenticate using environment variables
        >>> authenticate_garmin()
        ✅ Resumed existing Garmin Connect session
        True

        >>> # Force new authentication with credentials
        >>> authenticate_garmin(email="user@example.com", password="secret")
        🔐 Authenticating with Garmin Connect...
        ✅ Authentication successful! Session saved.
        True

    Notes:
        - Session tokens are saved to avoid repeated MFA prompts
        - For MFA-enabled accounts, consider using app-specific passwords
        - Credentials are never stored, only session tokens
        - Failed authentications provide helpful troubleshooting hints
    """
    if garth is None:
        raise ImportError("garth library not available")

    token_path = Path(token_store).expanduser()

    # Try to resume existing session
    if _try_resume_session(token_path):
        return True

    # Get credentials
    email = _get_credential(email, "GARMIN_EMAIL", "Garmin Connect email: ")
    password = _get_credential(
        password, "GARMIN_PASSWORD", "Garmin Connect password: ", secure=True
    )

    # Perform login
    return _perform_login(email, password, token_path)
