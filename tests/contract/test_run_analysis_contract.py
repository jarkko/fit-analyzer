"""Contract tests for run_analysis() function.

Function Signature:
    run_analysis(
        directory: str = ".",
        output_dir: str = "data",
        updated_files: Optional[List[str]] = None,
        **kwargs
    ) -> bool

Parameter Contracts:
    updated_files:
        - None: Analyze all FIT files in directory (default behavior)
        - []: Skip analysis (no files to process, early return)
        - [...files]: Analyze only specified files
        
    directory:
        - Valid directory: Glob for FIT files
        - Single file path: Analyze that file only
        - Non-existent: Should handle gracefully
        
Return Value:
    - True: Analysis completed successfully
    - False: Analysis failed or no files found
    
Side Effects:
    - Creates CSV files in output_dir
    - Calls CLI module with parsed arguments
    - May print status messages
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_cli():
    """Mock the CLI module to avoid actual file processing."""
    # Mock the import of cli module inside run_analysis
    with patch('fitanalyzer.cli') as mock:
        mock.parse_arguments.return_value = MagicMock(
            ftp=300,
            hrrest=60,
            hrmax=190,
            multisport=True,
            dump_sets=True,
        )
        mock.main_with_args.return_value = 0
        yield mock


class TestRunAnalysisUpdatedFilesContract:
    """Test the updated_files parameter contract."""
    
    def test_updated_files_none_analyzes_all_files(self, tmp_path, mock_cli):
        """When updated_files=None, should analyze all FIT files in directory."""
        from fitanalyzer.sync import run_analysis
        
        # Create test directory with FIT files
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "activity1_ACTIVITY.fit").write_bytes(b"fake")
        (fit_dir / "activity2_ACTIVITY.fit").write_bytes(b"fake")
        
        result = run_analysis(
            directory=str(fit_dir),
            updated_files=None  # Explicit None: analyze all
        )
        
        assert result is True
        # Verify CLI was called with all files
        args = mock_cli.parse_arguments.call_args[0][0]
        fit_files = [arg for arg in args if arg.endswith("_ACTIVITY.fit")]
        assert len(fit_files) == 2
    
    def test_updated_files_empty_list_skips_analysis(self, tmp_path, mock_cli):
        """When updated_files=[], should skip analysis entirely."""
        from fitanalyzer.sync import run_analysis
        
        # Create test directory with FIT files
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "activity_ACTIVITY.fit").write_bytes(b"fake")
        
        result = run_analysis(
            directory=str(fit_dir),
            updated_files=[]  # Empty list: skip analysis
        )
        
        assert result is True
        # Verify CLI was NOT called
        mock_cli.parse_arguments.assert_not_called()
        mock_cli.main_with_args.assert_not_called()
    
    def test_updated_files_single_file_analyzes_only_that(self, tmp_path, mock_cli):
        """When updated_files has one file, should analyze only that file."""
        from fitanalyzer.sync import run_analysis
        
        # Create test directory with multiple FIT files
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        file1 = fit_dir / "activity1_ACTIVITY.fit"
        file2 = fit_dir / "activity2_ACTIVITY.fit"
        file1.write_bytes(b"fake")
        file2.write_bytes(b"fake")
        
        result = run_analysis(
            directory=str(fit_dir),
            updated_files=[str(file1)]  # Only analyze file1
        )
        
        assert result is True
        # Verify CLI was called with only file1
        args = mock_cli.parse_arguments.call_args[0][0]
        fit_files = [arg for arg in args if arg.endswith("_ACTIVITY.fit")]
        assert len(fit_files) == 1
        assert str(file1) in fit_files
        assert str(file2) not in fit_files
    
    def test_updated_files_multiple_files_analyzes_all_specified(self, tmp_path, mock_cli):
        """When updated_files has multiple files, should analyze all of them."""
        from fitanalyzer.sync import run_analysis
        
        # Create test directory with FIT files
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        file1 = fit_dir / "activity1_ACTIVITY.fit"
        file2 = fit_dir / "activity2_ACTIVITY.fit"
        file3 = fit_dir / "activity3_ACTIVITY.fit"
        file1.write_bytes(b"fake")
        file2.write_bytes(b"fake")
        file3.write_bytes(b"fake")
        
        result = run_analysis(
            directory=str(fit_dir),
            updated_files=[str(file1), str(file2)]  # Two files
        )
        
        assert result is True
        # Verify CLI was called with both files
        args = mock_cli.parse_arguments.call_args[0][0]
        fit_files = [arg for arg in args if arg.endswith("_ACTIVITY.fit")]
        assert len(fit_files) == 2
        assert str(file1) in fit_files
        assert str(file2) in fit_files
        assert str(file3) not in fit_files
    
    def test_updated_files_nonexistent_files_skips_gracefully(self, tmp_path, mock_cli):
        """When updated_files contains non-existent files, should skip them."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        
        result = run_analysis(
            directory=str(fit_dir),
            updated_files=[str(fit_dir / "missing.fit")]
        )
        
        # Should return True (success, just nothing to do)
        assert result is True
        # CLI should not be called since no files exist
        mock_cli.parse_arguments.assert_not_called()


class TestRunAnalysisDirectoryContract:
    """Test the directory parameter contract."""
    
    def test_directory_valid_path_globs_for_files(self, tmp_path, mock_cli):
        """When directory is valid path, should glob for FIT files."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "activities"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        result = run_analysis(directory=str(fit_dir))
        
        assert result is True
        mock_cli.parse_arguments.assert_called_once()
    
    def test_directory_single_file_analyzes_that_file(self, tmp_path, mock_cli):
        """When directory is a single FIT file, should analyze only that."""
        from fitanalyzer.sync import run_analysis
        
        fit_file = tmp_path / "single_ACTIVITY.fit"
        fit_file.write_bytes(b"fake")
        
        result = run_analysis(
            directory=str(fit_file),
            updated_files=None  # Should use the single file
        )
        
        assert result is True
        args = mock_cli.parse_arguments.call_args[0][0]
        fit_files = [arg for arg in args if arg.endswith("_ACTIVITY.fit")]
        assert len(fit_files) == 1
        assert str(fit_file) in fit_files
    
    def test_directory_no_fit_files_returns_false(self, tmp_path, mock_cli):
        """When directory has no FIT files, should return False."""
        from fitanalyzer.sync import run_analysis
        
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = run_analysis(
            directory=str(empty_dir),
            updated_files=None  # Try to analyze all
        )
        
        assert result is False
        mock_cli.parse_arguments.assert_not_called()


class TestRunAnalysisReturnValueContract:
    """Test the return value contract."""
    
    def test_returns_true_on_success(self, tmp_path, mock_cli):
        """Should return True when analysis completes successfully."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        mock_cli.main_with_args.return_value = 0  # Success
        
        result = run_analysis(directory=str(fit_dir))
        
        assert result is True
    
    def test_returns_false_on_cli_failure(self, tmp_path, mock_cli):
        """Should return False when CLI returns non-zero exit code."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        mock_cli.main_with_args.return_value = 1  # Failure
        
        result = run_analysis(directory=str(fit_dir))
        
        assert result is False
    
    def test_returns_false_on_known_exception(self, tmp_path, mock_cli):
        """Should return False and handle known exceptions gracefully."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        # Test with OSError (a known exception type that's caught)
        mock_cli.main_with_args.side_effect = OSError("File error")
        
        result = run_analysis(directory=str(fit_dir))
        
        assert result is False
    
    def test_unknown_exception_propagates(self, tmp_path, mock_cli):
        """Unknown exceptions should propagate (not caught by run_analysis)."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        # RuntimeError is not in the caught exception list
        mock_cli.main_with_args.side_effect = RuntimeError("Unexpected error")
        
        # Should raise (not return False)
        with pytest.raises(RuntimeError, match="Unexpected error"):
            run_analysis(directory=str(fit_dir))


class TestRunAnalysisKwargsContract:
    """Test the **kwargs parameter contract (ftp, hrrest, hrmax, multisport)."""
    
    def test_kwargs_passed_to_cli_arguments(self, tmp_path, mock_cli):
        """Should pass kwargs as CLI arguments."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        run_analysis(
            directory=str(fit_dir),
            ftp=250,
            hrrest=55,
            hrmax=185,
            multisport=False
        )
        
        args = mock_cli.parse_arguments.call_args[0][0]
        assert "--ftp" in args
        assert "250" in args
        assert "--hrrest" in args
        assert "55" in args
        assert "--hrmax" in args
        assert "185" in args
        # multisport=False means --multisport flag should NOT be present
        assert "--multisport" not in args
    
    def test_kwargs_default_values(self, tmp_path, mock_cli):
        """Should use default values when kwargs not provided."""
        from fitanalyzer.sync import run_analysis
        
        fit_dir = tmp_path / "fits"
        fit_dir.mkdir()
        (fit_dir / "test_ACTIVITY.fit").write_bytes(b"fake")
        
        run_analysis(directory=str(fit_dir))
        
        args = mock_cli.parse_arguments.call_args[0][0]
        # Should have default FTP (300), hrrest (50), hrmax (190)
        assert "--ftp" in args
        assert "300" in args
        assert "--hrrest" in args
        assert "50" in args
        assert "--hrmax" in args
        assert "190" in args
        # multisport default is True
        assert "--multisport" in args
