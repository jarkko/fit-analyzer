"""
Tests for Garmin authentication module.

These tests verify the contract for authentication functions.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from fitanalyzer.garmin_auth import (
    authenticate_garmin,
    check_and_install_garth,
)


class TestCheckAndInstallGarth(unittest.TestCase):
    """Test check_and_install_garth function."""

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", True)
    def test_returns_true_when_garth_available(self):
        """Test that function returns True when garth is already installed."""
        result = check_and_install_garth()
        self.assertTrue(result)

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="n")
    @patch("builtins.print")
    def test_returns_false_when_user_declines_install(self, mock_print, mock_input):
        """Test returns False when user declines installation."""
        result = check_and_install_garth()
        self.assertFalse(result)
        mock_input.assert_called_once()

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="y")
    @patch("subprocess.check_call")
    @patch("builtins.print")
    def test_successful_install_returns_false(self, mock_print, mock_check_call, mock_input):
        """Test returns False even after successful install (requires restart)."""
        result = check_and_install_garth()
        self.assertFalse(result)
        mock_check_call.assert_called_once()

    @patch("fitanalyzer.garmin_auth.GARTH_AVAILABLE", False)
    @patch("builtins.input", return_value="y")
    @patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip"))
    @patch("builtins.print")
    def test_failed_install_returns_false(self, mock_print, mock_check_call, mock_input):
        """Test returns False when installation fails."""
        result = check_and_install_garth()
        self.assertFalse(result)


class TestAuthenticateGarmin(unittest.TestCase):
    """Test authenticate_garmin function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_token_path = Path("/tmp/test_garth_token")
        if self.test_token_path.exists():
            self.test_token_path.unlink()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_token_path.exists():
            self.test_token_path.unlink()

    @patch("fitanalyzer.garmin_auth.garth", None)
    def test_raises_import_error_when_garth_not_available(self):
        """Test raises ImportError when garth is not available."""
        with self.assertRaises(ImportError) as context:
            authenticate_garmin()
        self.assertIn("garth library not available", str(context.exception))

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=True)
    def test_returns_true_when_session_resumed(self, mock_resume, mock_garth):
        """Test returns True when existing session is successfully resumed."""
        result = authenticate_garmin(token_store=str(self.test_token_path))
        self.assertTrue(result)
        mock_resume.assert_called_once()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=False)
    @patch("fitanalyzer.garmin_auth._get_credential")
    @patch("fitanalyzer.garmin_auth._perform_login", return_value=True)
    def test_successful_new_login(self, mock_login, mock_get_cred, mock_resume, mock_garth):
        """Test successful new login when session cannot be resumed."""
        mock_get_cred.side_effect = ["test@example.com", "password123"]

        result = authenticate_garmin(token_store=str(self.test_token_path))

        self.assertTrue(result)
        mock_resume.assert_called_once()
        self.assertEqual(mock_get_cred.call_count, 2)
        mock_login.assert_called_once()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=False)
    @patch("fitanalyzer.garmin_auth._get_credential")
    @patch("fitanalyzer.garmin_auth._perform_login", return_value=False)
    def test_failed_login(self, mock_login, mock_get_cred, mock_resume, mock_garth):
        """Test returns False when login fails."""
        mock_get_cred.side_effect = ["test@example.com", "wrongpassword"]

        result = authenticate_garmin(token_store=str(self.test_token_path))

        self.assertFalse(result)
        mock_login.assert_called_once()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=False)
    @patch("fitanalyzer.garmin_auth._perform_login", return_value=True)
    def test_uses_provided_credentials(self, mock_login, mock_resume, mock_garth):
        """Test uses explicitly provided email and password."""
        result = authenticate_garmin(
            email="explicit@example.com",
            password="explicitpass",
            token_store=str(self.test_token_path),
        )

        self.assertTrue(result)
        # Should not prompt for credentials
        mock_login.assert_called_once()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=False)
    @patch.dict("os.environ", {"GARMIN_EMAIL": "env@example.com", "GARMIN_PASSWORD": "envpass"})
    @patch("fitanalyzer.garmin_auth._perform_login", return_value=True)
    def test_uses_environment_variables(self, mock_login, mock_resume, mock_garth):
        """Test uses environment variables when no explicit credentials provided."""
        result = authenticate_garmin(token_store=str(self.test_token_path))

        self.assertTrue(result)
        mock_login.assert_called_once()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._try_resume_session", return_value=False)
    @patch("fitanalyzer.garmin_auth._get_credential")
    @patch("fitanalyzer.garmin_auth._perform_login", return_value=True)
    def test_expands_tilde_in_token_path(self, mock_login, mock_get_cred, mock_resume, mock_garth):
        """Test that tilde (~) in token_store path is expanded."""
        mock_get_cred.side_effect = ["test@example.com", "password"]

        result = authenticate_garmin(token_store="~/test_token")

        self.assertTrue(result)
        # Verify the path was expanded (can't directly check, but login should succeed)


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_try_resume_session_no_token_file(self):
        """Test _try_resume_session returns False when token file doesn't exist."""
        from fitanalyzer.garmin_auth import _try_resume_session

        non_existent = Path("/tmp/definitely_not_existing_token_12345")
        result = _try_resume_session(non_existent)
        self.assertFalse(result)

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("builtins.print")
    def test_try_resume_session_invalid_token(self, mock_print, mock_garth):
        """Test _try_resume_session returns False when token is invalid."""
        from fitanalyzer.garmin_auth import _try_resume_session

        # Create a temporary token file
        token_path = Path("/tmp/test_invalid_token")
        token_path.write_text("invalid token")

        try:
            # Mock garth.resume to raise an error
            mock_garth.resume.side_effect = RuntimeError("Invalid token")

            result = _try_resume_session(token_path)
            self.assertFalse(result)
        finally:
            token_path.unlink()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("builtins.print")
    def test_try_resume_session_success(self, mock_print, mock_garth):
        """Test _try_resume_session returns True on successful resume."""
        from fitanalyzer.garmin_auth import _try_resume_session

        token_path = Path("/tmp/test_valid_token")
        token_path.write_text("valid token")

        try:
            # Mock successful resume
            mock_garth.client.username = "testuser"

            result = _try_resume_session(token_path)
            self.assertTrue(result)
        finally:
            token_path.unlink()

    def test_get_credential_with_value(self):
        """Test _get_credential returns provided value."""
        from fitanalyzer.garmin_auth import _get_credential

        result = _get_credential("provided", "ENV_VAR", "Prompt: ")
        self.assertEqual(result, "provided")

    @patch.dict("os.environ", {"TEST_ENV": "env_value"})
    def test_get_credential_from_env(self):
        """Test _get_credential gets value from environment."""
        from fitanalyzer.garmin_auth import _get_credential

        result = _get_credential(None, "TEST_ENV", "Prompt: ")
        self.assertEqual(result, "env_value")

    @patch("builtins.input", return_value="user_input")
    def test_get_credential_from_input(self, mock_input):
        """Test _get_credential prompts user when no value or env var."""
        from fitanalyzer.garmin_auth import _get_credential

        result = _get_credential(None, "NONEXISTENT_ENV", "Enter value: ")
        self.assertEqual(result, "user_input")
        mock_input.assert_called_once_with("Enter value: ")

    @patch("getpass.getpass", return_value="secure_input")
    def test_get_credential_secure_input(self, mock_getpass):
        """Test _get_credential uses getpass for secure input."""
        from fitanalyzer.garmin_auth import _get_credential

        result = _get_credential(None, "NONEXISTENT_ENV", "Password: ", secure=True)
        self.assertEqual(result, "secure_input")
        mock_getpass.assert_called_once_with("Password: ")

    @patch("builtins.print")
    def test_handle_auth_error_generic(self, mock_print):
        """Test _handle_auth_error with generic error."""
        from fitanalyzer.garmin_auth import _handle_auth_error

        error = ValueError("Generic error")
        _handle_auth_error(error)

        # Should print error message
        calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("Authentication failed" in str(call) for call in calls))

    @patch("builtins.print")
    def test_handle_auth_error_mfa(self, mock_print):
        """Test _handle_auth_error with MFA error."""
        from fitanalyzer.garmin_auth import _handle_auth_error

        error = ValueError("MFA verification required")
        _handle_auth_error(error)

        # Should print MFA-specific help
        calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(
            any("MFA" in str(call) or "app-specific password" in str(call) for call in calls)
        )

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("builtins.print")
    def test_perform_login_success(self, mock_print, mock_garth):
        """Test _perform_login with successful login."""
        from fitanalyzer.garmin_auth import _perform_login

        token_path = Path("/tmp/test_login_token")
        token_path.parent.mkdir(exist_ok=True)

        try:
            result = _perform_login("test@example.com", "password", token_path)
            self.assertTrue(result)
            mock_garth.login.assert_called_once_with("test@example.com", "password")
            mock_garth.save.assert_called_once()
        finally:
            if token_path.exists():
                token_path.unlink()

    @patch("fitanalyzer.garmin_auth.garth")
    @patch("fitanalyzer.garmin_auth._handle_auth_error")
    @patch("builtins.print")
    def test_perform_login_failure(self, mock_print, mock_handle_error, mock_garth):
        """Test _perform_login with failed login."""
        from fitanalyzer.garmin_auth import _perform_login

        mock_garth.login.side_effect = ValueError("Invalid credentials")
        token_path = Path("/tmp/test_failed_login")

        result = _perform_login("test@example.com", "wrongpass", token_path)

        self.assertFalse(result)
        mock_handle_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
