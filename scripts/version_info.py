#!/usr/bin/env python3
"""Version information utility script for Claude Wrapper.

This script provides comprehensive version information for debugging,
development, and release management. Optimized for uv workflow.

Usage:
    uv run python scripts/version_info.py
    uv run python scripts/version_info.py --json
    uv run python scripts/version_info.py --check
    uv run python scripts/version_info.py --validate-build
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_wrapper import get_version, get_version_info
from claude_wrapper._version import check_version_consistency


def main() -> None:
    """Main entry point for version information utility."""
    parser = argparse.ArgumentParser(
        description="Claude Wrapper version information utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/version_info.py                 # Human-readable output
    python scripts/version_info.py --json         # JSON output
    python scripts/version_info.py --check        # Check consistency only
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output version information as JSON",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check version consistency and exit with status code",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include additional diagnostic information",
    )

    parser.add_argument(
        "--validate-build",
        action="store_true",
        help="Validate that the package builds correctly and version is accessible",
    )

    args = parser.parse_args()

    if args.validate_build:
        # Validate build mode
        print("🔍 Validating dynamic versioning and build process...")
        exit_code = validate_build_process()
        sys.exit(exit_code)

    if args.check:
        # Check consistency mode
        consistent = check_version_consistency()
        if consistent:
            print("✓ Version information is consistent")
            sys.exit(0)
        else:
            print("✗ Version information is inconsistent")
            if not args.json:
                info = get_version_info()
                print("\nVersion sources:")
                for key, value in info.items():
                    if value:
                        print(f"  {key}: {value}")
            sys.exit(1)

    # Get comprehensive version information
    version = get_version()
    info = get_version_info()
    consistent = check_version_consistency()

    if args.json:
        # JSON output mode
        output = {
            "version": version,
            "info": info,
            "consistent": consistent,
        }

        if args.verbose:
            output["diagnostic"] = get_diagnostic_info()

        print(json.dumps(output, indent=2))
    else:
        # Human-readable output mode
        print(f"Claude Wrapper Version Information")
        print("=" * 40)
        print(f"Current Version: {version}")
        print(f"Version Source:  {info['source']}")
        print(f"Consistency:     {'✓ Consistent' if consistent else '✗ Inconsistent'}")

        if info['package_version']:
            print(f"Package Version: {info['package_version']}")

        if info['git_version']:
            print(f"Git Version:     {info['git_version']}")

        if args.verbose:
            print("\nDiagnostic Information:")
            print("-" * 25)
            diagnostic = get_diagnostic_info()
            for key, value in diagnostic.items():
                print(f"{key.replace('_', ' ').title()}: {value}")


def validate_build_process() -> int:
    """Validate that the dynamic versioning and build process works correctly.

    This function performs comprehensive validation of:
    1. Version can be retrieved from source
    2. Package can be built with uv
    3. Built package contains correct version metadata
    4. Version is accessible after installation

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    import subprocess
    import tempfile
    import shutil
    from pathlib import Path

    print("1. Testing version retrieval from source...")
    try:
        version = get_version()
        print(f"   ✓ Version retrieved: {version}")
    except Exception as e:
        print(f"   ✗ Failed to get version: {e}")
        return 1

    print("2. Testing build process with uv...")
    try:
        # Clean any previous build artifacts
        build_dir = Path("dist")
        if build_dir.exists():
            shutil.rmtree(build_dir)

        result = subprocess.run(
            ["uv", "build"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"   ✗ Build failed: {result.stderr}")
            return 1
        print("   ✓ Build successful")
    except subprocess.TimeoutExpired:
        print("   ✗ Build timed out")
        return 1
    except FileNotFoundError:
        print("   ✗ uv command not found")
        return 1
    except Exception as e:
        print(f"   ✗ Build failed with exception: {e}")
        return 1

    print("3. Validating built package metadata...")
    try:
        # Check that wheel and sdist were created
        dist_files = list(Path("dist").glob("*"))
        if not dist_files:
            print("   ✗ No build artifacts found in dist/")
            return 1

        wheel_files = [f for f in dist_files if f.suffix == ".whl"]
        sdist_files = [f for f in dist_files if f.suffix == ".gz"]

        if not wheel_files:
            print("   ✗ No wheel file found")
            return 1
        if not sdist_files:
            print("   ✗ No source distribution found")
            return 1

        print(f"   ✓ Built {len(wheel_files)} wheel(s) and {len(sdist_files)} sdist(s)")

        # Verify version in filename
        wheel_file = wheel_files[0]
        if version not in str(wheel_file):
            print(f"   ⚠️  Warning: Version {version} not found in wheel filename {wheel_file}")
        else:
            print(f"   ✓ Version {version} correctly embedded in wheel filename")

    except Exception as e:
        print(f"   ✗ Failed to validate build artifacts: {e}")
        return 1

    print("4. Testing installation from built package...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test environment
            venv_path = Path(temp_dir) / "test_env"

            # Create virtual environment
            subprocess.run(
                ["python", "-m", "venv", str(venv_path)],
                check=True,
                capture_output=True,
            )

            # Install the built wheel
            pip_path = venv_path / "bin" / "pip"
            if not pip_path.exists():  # Windows
                pip_path = venv_path / "Scripts" / "pip.exe"

            subprocess.run(
                [str(pip_path), "install", str(wheel_file)],
                check=True,
                capture_output=True,
            )

            # Test import and version access
            python_path = venv_path / "bin" / "python"
            if not python_path.exists():  # Windows
                python_path = venv_path / "Scripts" / "python.exe"

            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "import claude_wrapper; print(claude_wrapper.__version__)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            installed_version = result.stdout.strip()
            if installed_version == version:
                print(f"   ✓ Installed version matches: {installed_version}")
            else:
                print(f"   ✗ Version mismatch: built={version}, installed={installed_version}")
                return 1

    except Exception as e:
        print(f"   ✗ Installation test failed: {e}")
        return 1

    print("\n🎉 All validation checks passed!")
    print(f"Dynamic versioning is working correctly with version: {version}")
    return 0


def get_diagnostic_info() -> dict[str, str | bool]:
    """Get diagnostic information for troubleshooting.

    Returns:
        dict[str, str | bool]: Diagnostic information including:
            - git_available: Whether git is available
            - hatch_vcs_available: Whether hatch-vcs is importable
            - package_installed: Whether package is installed via pip
            - pyproject_exists: Whether pyproject.toml exists
    """
    import subprocess
    from pathlib import Path

    diagnostic = {}

    # Check if git is available and we're in a git repo
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        diagnostic["git_available"] = result.returncode == 0
        diagnostic["git_repo"] = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        diagnostic["git_available"] = False
        diagnostic["git_repo"] = False

    # Check if hatch-vcs is available
    try:
        import hatch_vcs  # noqa: F401
        diagnostic["hatch_vcs_available"] = True
    except ImportError:
        diagnostic["hatch_vcs_available"] = False

    # Check if hatchling metadata core is available
    try:
        from hatchling.metadata.core import ProjectMetadata  # noqa: F401
        diagnostic["hatchling_available"] = True
    except ImportError:
        diagnostic["hatchling_available"] = False

    # Check if package is installed
    try:
        import importlib.metadata
        importlib.metadata.version("claude-wrapper")
        diagnostic["package_installed"] = True
    except importlib.metadata.PackageNotFoundError:
        diagnostic["package_installed"] = False

    # Check if pyproject.toml exists
    project_root = Path(__file__).parent.parent
    diagnostic["pyproject_exists"] = (project_root / "pyproject.toml").exists()

    # Check for git tags
    if diagnostic["git_available"]:
        try:
            result = subprocess.run(
                ["git", "tag", "--list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
            diagnostic["git_tags_count"] = len(tags)
            diagnostic["has_git_tags"] = len(tags) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            diagnostic["git_tags_count"] = 0
            diagnostic["has_git_tags"] = False
    else:
        diagnostic["git_tags_count"] = 0
        diagnostic["has_git_tags"] = False

    return diagnostic


if __name__ == "__main__":
    main()
