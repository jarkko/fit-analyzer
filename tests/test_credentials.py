"""Tests for credentials module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from fitanalyzer.credentials import create_env_file


class TestCreateEnvFile:
    """Tests for create_env_file function."""

    def test_create_env_file_new(self, tmp_path, monkeypatch):
        """Test creating new .env file."""
        monkeypatch.chdir(tmp_path)

        inputs = [
            "test@example.com",  # email
            "250",  # FTP
            "55",  # hr_rest
            "185",  # hr_max
            "n",  # Don't create .gitignore (not exist yet)
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass123"):
                create_env_file()

        env_path = tmp_path / ".env"
        assert env_path.exists()

        content = env_path.read_text()
        assert "GARMIN_EMAIL=test@example.com" in content
        assert "GARMIN_PASSWORD=testpass123" in content
        assert "FTP=250" in content
        assert "HR_REST=55" in content
        assert "HR_MAX=185" in content

        # Check permissions (Unix only)
        import sys

        if sys.platform != "win32":
            assert oct(env_path.stat().st_mode)[-3:] == "600"

    def test_create_env_file_with_defaults(self, tmp_path, monkeypatch):
        """Test creating .env file with default values."""
        monkeypatch.chdir(tmp_path)

        inputs = [
            "test@example.com",  # email
            "",  # FTP (use default)
            "",  # hr_rest (use default)
            "",  # hr_max (use default)
            "n",  # Don't create .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass123"):
                create_env_file()

        env_path = tmp_path / ".env"
        content = env_path.read_text()
        assert "FTP=300" in content  # DEFAULT_FTP
        assert "HR_REST=50" in content  # DEFAULT_HR_REST
        assert "HR_MAX=190" in content  # DEFAULT_HR_MAX

    def test_create_env_file_overwrite_yes(self, tmp_path, monkeypatch):
        """Test overwriting existing .env file."""
        monkeypatch.chdir(tmp_path)

        # Create existing .env
        env_path = tmp_path / ".env"
        env_path.write_text("OLD_CONTENT=true")

        inputs = [
            "y",  # Overwrite confirmation
            "new@example.com",
            "250",
            "55",
            "185",
            "n",  # Don't create .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="newpass"):
                create_env_file()

        content = env_path.read_text()
        assert "OLD_CONTENT" not in content
        assert "GARMIN_EMAIL=new@example.com" in content

    def test_create_env_file_overwrite_no(self, tmp_path, monkeypatch):
        """Test canceling overwrite of existing .env file."""
        monkeypatch.chdir(tmp_path)

        # Create existing .env
        env_path = tmp_path / ".env"
        original_content = "OLD_CONTENT=true"
        env_path.write_text(original_content)

        with patch("builtins.input", return_value="n"):
            create_env_file()

        # Content should remain unchanged
        assert env_path.read_text() == original_content

    def test_create_env_file_with_existing_gitignore_containing_env(self, tmp_path, monkeypatch):
        """Test when .gitignore already contains .env."""
        monkeypatch.chdir(tmp_path)

        # Create .gitignore with .env already in it
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text("*.pyc\n.env\n")

        inputs = [
            "test@example.com",
            "250",
            "55",
            "185",
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass"):
                create_env_file()

        # .gitignore should not be modified
        gitignore_content = gitignore_path.read_text()
        assert gitignore_content.count(".env") == 1

    def test_create_env_file_with_existing_gitignore_missing_env_add_yes(
        self, tmp_path, monkeypatch
    ):
        """Test adding .env to existing .gitignore when user says yes."""
        monkeypatch.chdir(tmp_path)

        # Create .gitignore without .env
        gitignore_path = tmp_path / ".gitignore"
        gitignore_path.write_text("*.pyc\n")

        inputs = [
            "test@example.com",
            "250",
            "55",
            "185",
            "y",  # Add to .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass"):
                create_env_file()

        gitignore_content = gitignore_path.read_text()
        assert ".env" in gitignore_content

    def test_create_env_file_with_existing_gitignore_missing_env_add_no(
        self, tmp_path, monkeypatch
    ):
        """Test not adding .env to existing .gitignore when user says no."""
        monkeypatch.chdir(tmp_path)

        # Create .gitignore without .env
        gitignore_path = tmp_path / ".gitignore"
        original_content = "*.pyc\n"
        gitignore_path.write_text(original_content)

        inputs = [
            "test@example.com",
            "250",
            "55",
            "185",
            "n",  # Don't add to .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass"):
                create_env_file()

        # .gitignore should remain unchanged
        assert gitignore_path.read_text() == original_content

    def test_create_env_file_no_gitignore_create_yes(self, tmp_path, monkeypatch):
        """Test creating new .gitignore when it doesn't exist."""
        monkeypatch.chdir(tmp_path)

        inputs = [
            "test@example.com",
            "250",
            "55",
            "185",
            "y",  # Create .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass"):
                create_env_file()

        gitignore_path = tmp_path / ".gitignore"
        assert gitignore_path.exists()
        assert ".env" in gitignore_path.read_text()

    def test_create_env_file_no_gitignore_create_no(self, tmp_path, monkeypatch):
        """Test not creating .gitignore when it doesn't exist and user says no."""
        monkeypatch.chdir(tmp_path)

        inputs = [
            "test@example.com",
            "250",
            "55",
            "185",
            "n",  # Don't create .gitignore
        ]

        with patch("builtins.input", side_effect=inputs):
            with patch("getpass.getpass", return_value="testpass"):
                create_env_file()

        gitignore_path = tmp_path / ".gitignore"
        assert not gitignore_path.exists()
