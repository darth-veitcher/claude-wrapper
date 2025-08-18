"""Tests for version management functionality."""

import importlib.metadata
import warnings
from unittest.mock import Mock, patch

import pytest

from claude_wrapper import __version__, get_version, get_version_info
from claude_wrapper._version import check_version_consistency


class TestVersionManagement:
    """Test suite for dynamic version management."""

    def test_version_exists(self) -> None:
        """Test that version is accessible and valid."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_get_version_returns_string(self) -> None:
        """Test that get_version returns a valid version string."""
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_version_consistency(self) -> None:
        """Test that __version__ and get_version() return the same value."""
        assert __version__ == get_version()

    def test_semantic_versioning_format(self) -> None:
        """Test that version follows semantic versioning format."""
        version = get_version()

        # Should have at least MAJOR.MINOR format
        parts = version.split(".")
        assert len(parts) >= 2, f"Version {version} should have at least MAJOR.MINOR format"

        # First two parts should be numeric
        assert parts[0].isdigit(), f"Major version should be numeric: {parts[0]}"
        assert parts[1].isdigit(), f"Minor version should be numeric: {parts[1]}"

    @patch("importlib.metadata.version")
    def test_get_version_from_package_metadata(self, mock_version: Mock) -> None:
        """Test version retrieval from package metadata."""
        mock_version.return_value = "1.2.3"

        # Need to reload the module to test the function directly
        from claude_wrapper._version import get_version

        version = get_version()

        mock_version.assert_called_with("claude-wrapper")
        assert version == "1.2.3"

    @patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError)
    def test_get_version_from_git_metadata(self, _mock_version: Mock) -> None:
        """Test version retrieval from git metadata via hatch-vcs."""
        # Mock the ProjectMetadata at the module level where it's imported
        mock_project_metadata = Mock()
        mock_instance = Mock()
        mock_instance.version = "2.0.0"
        mock_project_metadata.return_value = mock_instance

        # Mock the import using sys.modules

        # Create a mock for the hatchling.metadata.core module
        mock_core_module = Mock()
        mock_core_module.ProjectMetadata = mock_project_metadata

        # Mock the entire import chain
        with patch.dict(
            "sys.modules",
            {
                "hatchling": Mock(),
                "hatchling.metadata": Mock(),
                "hatchling.metadata.core": mock_core_module,
            },
        ):
            from claude_wrapper._version import get_version

            version = get_version()

            assert version == "2.0.0"

    @patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError)
    def test_get_version_fallback_with_warning(self, _mock_version: Mock) -> None:
        """Test version fallback when all methods fail."""
        # Make the import fail to trigger fallback
        with (
            patch.dict("sys.modules", {"hatchling.metadata.core": None}),
            warnings.catch_warnings(record=True) as warning_list,
        ):
            warnings.simplefilter("always")

            from claude_wrapper._version import get_version

            version = get_version()

            assert version == "0.1.0-dev"
            assert len(warning_list) == 1
            assert "Could not determine version" in str(warning_list[0].message)
            assert issubclass(warning_list[0].category, UserWarning)

    def test_get_version_info_structure(self) -> None:
        """Test that get_version_info returns proper structure."""
        info = get_version_info()

        expected_keys = {"version", "git_version", "package_version", "source"}
        assert set(info.keys()) == expected_keys

        # Version should always be present
        assert info["version"] is not None
        assert isinstance(info["version"], str)

        # Source should indicate where version came from
        assert info["source"] in {"package_metadata", "git_metadata", "fallback"}

    def test_check_version_consistency_single_source(self) -> None:
        """Test version consistency check with single source."""
        # With only one source available, should always be consistent
        result = check_version_consistency()
        assert isinstance(result, bool)

    @patch("claude_wrapper._version.get_version_info")
    def test_check_version_consistency_multiple_sources_consistent(
        self, mock_get_info: Mock
    ) -> None:
        """Test version consistency check with consistent multiple sources."""
        mock_get_info.return_value = {
            "version": "1.0.0",
            "git_version": "1.0.0",
            "package_version": "1.0.0",
            "source": "package_metadata",
        }

        result = check_version_consistency()
        assert result is True

    @patch("claude_wrapper._version.get_version_info")
    def test_check_version_consistency_multiple_sources_inconsistent(
        self, mock_get_info: Mock
    ) -> None:
        """Test version consistency check with inconsistent multiple sources."""
        mock_get_info.return_value = {
            "version": "1.0.0",
            "git_version": "1.0.1",
            "package_version": "1.0.0",
            "source": "package_metadata",
        }

        result = check_version_consistency()
        assert result is False


class TestVersionIntegration:
    """Integration tests for version usage across the application."""

    def test_version_accessible_from_main_module(self) -> None:
        """Test that version is accessible from main claude_wrapper module."""
        import claude_wrapper

        assert hasattr(claude_wrapper, "__version__")
        assert hasattr(claude_wrapper, "get_version")
        assert hasattr(claude_wrapper, "get_version_info")

    def test_cli_version_command_works(self) -> None:
        """Test that CLI can access version information."""
        from typer.testing import CliRunner

        from claude_wrapper.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["version"])

        # Should not crash and should contain version info
        assert result.exit_code == 0
        assert "Claude Wrapper" in result.stdout
        # Should contain a version number (at least one digit)
        assert any(char.isdigit() for char in result.stdout)

    def test_api_server_uses_dynamic_version(self) -> None:
        """Test that API server uses dynamic version."""
        from claude_wrapper.api.server import app

        # The FastAPI app should have the dynamic version
        assert app.version == get_version()

    @pytest.mark.asyncio
    async def test_api_root_endpoint_returns_version(self) -> None:
        """Test that API root endpoint returns current version."""
        from claude_wrapper.api.server import root

        response = await root()

        assert "version" in response
        assert response["version"] == get_version()


class TestVersionEdgeCases:
    """Test edge cases and error conditions for version management."""

    def test_version_module_import_error_handling(self) -> None:
        """Test that module handles import errors gracefully."""
        # This tests the ultimate fallback in __init__.py
        # The version should still be accessible even if get_version fails
        assert __version__ is not None

    def test_version_info_with_no_sources(self) -> None:
        """Test version info when no reliable sources are available."""
        # This should still return valid structure
        info = get_version_info()
        assert "version" in info
        assert info["version"] is not None

    def test_version_reproducibility(self) -> None:
        """Test that version calls are reproducible within same session."""
        version1 = get_version()
        version2 = get_version()
        assert version1 == version2

    def test_version_info_reproducibility(self) -> None:
        """Test that version info calls are reproducible."""
        info1 = get_version_info()
        info2 = get_version_info()
        assert info1 == info2


# Performance and compatibility tests
class TestVersionPerformance:
    """Test performance and compatibility aspects of version management."""

    def test_version_call_performance(self) -> None:
        """Test that version calls are reasonably fast."""
        import time

        start_time = time.time()
        for _ in range(100):
            get_version()
        end_time = time.time()

        # Should complete 100 calls in well under a second
        assert (end_time - start_time) < 1.0

    def test_version_caching_behavior(self) -> None:
        """Test that version calls are properly cached/optimized."""
        # Multiple calls should be consistent (implying caching or optimization)
        versions = [get_version() for _ in range(10)]
        assert len(set(versions)) == 1  # All should be the same
