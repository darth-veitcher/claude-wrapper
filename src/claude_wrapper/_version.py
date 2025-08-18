"""Version management for Claude Wrapper.

This module provides centralized version access using modern Python packaging
standards with dynamic versioning from git tags via setuptools-scm/hatch-vcs.

The version is automatically determined from:
1. Git tags (when installed from source with git metadata)
2. Package metadata (when installed from PyPI/wheel)
3. Fallback static version (development/testing)

Examples:
    >>> from claude_wrapper._version import get_version, __version__
    >>> # Get version from function
    >>> version = get_version()
    >>> # Use module-level constant
    >>> v = __version__
"""

from __future__ import annotations

import importlib.metadata
import warnings


def get_version() -> str:
    """Get the current version of Claude Wrapper.

    This function attempts to retrieve the version in the following order:
    1. From package metadata (installed package)
    2. From git tags via hatch-vcs (development with git)
    3. Fallback to a default version (edge cases)

    Returns:
        str: The version string following semantic versioning (MAJOR.MINOR.PATCH)

    Examples:
        >>> version = get_version()
        >>> assert version.count('.') >= 1  # semantic versioning
    """
    try:
        # Try to get version from installed package metadata
        # This works when the package is installed via pip/uv
        return importlib.metadata.version("claude-wrapper")
    except importlib.metadata.PackageNotFoundError:
        # Package not installed, try development approaches
        pass

    try:
        # Try to get version from hatch-vcs during development
        # This requires git metadata and hatch-vcs to be available
        from pathlib import Path

        from hatchling.metadata.core import ProjectMetadata

        project_root = Path(__file__).parent.parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        if pyproject_toml.exists():
            metadata = ProjectMetadata(
                str(project_root),
                None,
                {"version": {"source": "vcs"}},
            )
            return metadata.version
    except (ImportError, Exception):
        # hatch-vcs not available or git metadata missing
        pass

    # Development fallback - warn about static version usage
    fallback_version = "0.1.0-dev"
    warnings.warn(
        f"Could not determine version from git or package metadata. "
        f"Using fallback version: {fallback_version}. "
        f"For accurate versioning, ensure git tags are present or install from PyPI.",
        UserWarning,
        stacklevel=2,
    )
    return fallback_version


def get_version_info() -> dict[str, str | None]:
    """Get comprehensive version information.

    Returns:
        dict[str, str | None]: Version information including:
            - version: The current version string
            - git_version: Version from git metadata (if available)
            - package_version: Version from package metadata (if available)
            - source: The source used for version determination

    Examples:
        >>> info = get_version_info()
        >>> assert info['version'] is not None
        >>> assert info['source'] in ['package_metadata', 'git_metadata', 'fallback']
    """
    result: dict[str, str | None] = {
        "version": None,
        "git_version": None,
        "package_version": None,
        "source": None,
    }

    # Try package metadata first
    try:
        package_version = importlib.metadata.version("claude-wrapper")
        result["package_version"] = package_version
        result["version"] = package_version
        result["source"] = "package_metadata"
        return result
    except importlib.metadata.PackageNotFoundError:
        pass

    # Try git/hatch-vcs
    try:
        from pathlib import Path

        from hatchling.metadata.core import ProjectMetadata

        project_root = Path(__file__).parent.parent.parent
        pyproject_toml = project_root / "pyproject.toml"

        if pyproject_toml.exists():
            metadata = ProjectMetadata(
                str(project_root),
                None,
                {"version": {"source": "vcs"}},
            )
            git_version = metadata.version
            result["git_version"] = git_version
            result["version"] = git_version
            result["source"] = "git_metadata"
            return result
    except (ImportError, Exception):
        pass

    # Fallback
    fallback_version = "0.1.0-dev"
    result["version"] = fallback_version
    result["source"] = "fallback"
    return result


# Module-level version for backward compatibility
try:
    __version__ = get_version()
except Exception:
    # Ultimate fallback - should rarely happen
    __version__ = "0.1.0-dev"


# Expose version checking utility
def check_version_consistency() -> bool:
    """Check if version is consistent across all sources.

    This is primarily used for testing and development to ensure
    version information is properly synchronized.

    Returns:
        bool: True if all available version sources are consistent

    Examples:
        >>> result = check_version_consistency()
        >>> assert isinstance(result, bool)
    """
    info = get_version_info()
    versions = [v for v in [info["package_version"], info["git_version"]] if v]

    if len(versions) <= 1:
        return True  # Only one or no version sources available

    # Check if all versions are the same
    return len(set(versions)) == 1
