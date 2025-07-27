#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

# Try to use setuptools
try:
    from setuptools import Command, find_packages, setup
except ImportError:
    print("setuptools not found. Please install setuptools.")
    sys.exit(1)


def read_file(filename):
    """Read file contents."""
    here = Path(__file__).parent
    return (here / filename).read_text(encoding="utf-8").strip()


def get_version():
    """Get version from package __init__.py file."""
    try:
        # Try to read version from src/auto_thievia/__init__.py
        init_file = Path(__file__).parent / "src" / "auto_thievia" / "__init__.py"
        if init_file.exists():
            with open(init_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        # Extract version from __version__ = "x.y.z"
                        return line.split("=")[1].strip().strip('"').strip("'")

        # Fallback version if file doesn't exist
        return "0.1.0"
    except (FileNotFoundError, ValueError, IndexError):
        return "0.1.0"


def get_long_description():
    """Get long description from README."""
    try:
        return read_file("README.md")
    except FileNotFoundError:
        return "A Python package using GIS to determine auto theft patterns"


def get_requirements():
    """Get requirements list."""
    # Core dependencies
    requirements = [
        "geopandas>=0.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "folium>=0.14.0",
        "shapely>=2.0.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "contextily>=1.4.0",
        "osmnx>=1.6.0",
    ]

    return requirements


def get_extras_require():
    """Get optional dependencies."""
    return {
        "dev": [
            "setuptools",
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "mypy",
            "pre-commit",
            "autoflake",
            "build",
            "bumpver",
            "wheel",
            "pdoc",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "pdoc",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-mock",
        ],
        "gis": [
            "rasterio>=1.3.0",
            "pyproj>=3.6.0",
            "rtree>=1.0.0",
            "geopy>=2.3.0",
        ],
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
        ],
    }


# Constants for code formatting commands
SOURCE_DIRS = ["src/", "tests/", "setup.py"]


class FormatCommand(Command):
    """Custom command to format code using black."""

    description = "Format code using black"
    user_options = []

    def initialize_options(self):
        """Initialize command options - required by setuptools Command interface."""
        # No options to initialize for this command

    def finalize_options(self):
        """Finalize command options - required by setuptools Command interface."""
        # No options to finalize for this command

    def run(self):
        """Run black formatter on source code."""
        print("Formatting code with black...")
        try:
            subprocess.run(
                [sys.executable, "-m", "black"] + SOURCE_DIRS,
                check=True,
                cwd=os.getcwd(),
            )
            print("✓ Code formatting completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Black formatting failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("✗ Black not found. Install with: pip install black")
            sys.exit(1)


class RemoveUnusedImportsCommand(Command):
    """Custom command to remove unused imports using autoflake."""

    description = "Remove unused imports using autoflake"
    user_options = []

    def initialize_options(self):
        """Initialize command options - required by setuptools Command interface."""
        # No options to initialize for this command

    def finalize_options(self):
        """Finalize command options - required by setuptools Command interface."""
        # No options to finalize for this command

    def run(self):
        """Run autoflake to remove unused imports."""
        print("Removing unused imports with autoflake...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autoflake",
                    "--remove-all-unused-imports",
                    "--remove-unused-variables",
                    "--remove-duplicate-keys",
                    "--expand-star-imports",
                    "--exclude",
                    "__init__.py",
                    "--in-place",
                    "--recursive",
                ]
                + SOURCE_DIRS,
                check=True,
                cwd=os.getcwd(),
            )
            print("✓ Unused imports removed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Autoflake failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("✗ Autoflake not found. Install with: pip install autoflake")
            sys.exit(1)


class SortImportsCommand(Command):
    """Custom command to sort imports using isort."""

    description = "Sort imports using isort"
    user_options = []

    def initialize_options(self):
        """Initialize command options - required by setuptools Command interface."""
        # No options to initialize for this command

    def finalize_options(self):
        """Finalize command options - required by setuptools Command interface."""
        # No options to finalize for this command

    def run(self):
        """Run isort to sort imports."""
        print("Sorting imports with isort...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "isort",
                    "--profile",
                    "black",
                    "--multi-line",
                    "3",
                    "--line-length",
                    "88",
                ]
                + SOURCE_DIRS,
                check=True,
                cwd=os.getcwd(),
            )
            print("✓ Imports sorted successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Isort failed: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("✗ Isort not found. Install with: pip install isort")
            sys.exit(1)


class CleanupCodeCommand(Command):
    """Custom command to run all code cleanup tasks."""

    description = (
        "Run all code cleanup tasks (remove unused imports, sort imports, format)"
    )
    user_options = []

    def initialize_options(self):
        """Initialize command options - required by setuptools Command interface."""
        # No options to initialize for this command

    def finalize_options(self):
        """Finalize command options - required by setuptools Command interface."""
        # No options to finalize for this command

    def run(self):
        """Run all code cleanup commands in sequence."""
        print("Running complete code cleanup...")

        # Remove unused imports first
        remove_cmd = RemoveUnusedImportsCommand(self.distribution)
        remove_cmd.ensure_finalized()
        remove_cmd.run()

        # Sort imports second
        sort_cmd = SortImportsCommand(self.distribution)
        sort_cmd.ensure_finalized()
        sort_cmd.run()

        # Format code last
        format_cmd = FormatCommand(self.distribution)
        format_cmd.ensure_finalized()
        format_cmd.run()

        print("✓ All code cleanup tasks completed!")


# Custom command classes
COMMAND_CLASSES = {
    "format": FormatCommand,
    "remove_unused": RemoveUnusedImportsCommand,
    "sort_imports": SortImportsCommand,
    "cleanup": CleanupCodeCommand,
}


# Check if pyproject.toml exists and use it if available
pyproject_toml = Path(__file__).parent / "pyproject.toml"
setup_cfg = Path(__file__).parent / "setup.cfg"

if pyproject_toml.exists():
    # If pyproject.toml exists, let setuptools handle it
    # This setup.py serves as a fallback and entry point
    print("Found pyproject.toml - using modern configuration")
    setup(cmdclass=COMMAND_CLASSES)
elif setup_cfg.exists():
    # If setup.cfg exists, use it
    print("Found setup.cfg - using legacy configuration")
    setup(cmdclass=COMMAND_CLASSES)
else:
    # Fallback: define everything in setup.py
    print("No pyproject.toml or setup.cfg found - using setup.py configuration")

    setup(
        name="auto_thievia",
        version=get_version(),
        author="James G Willmore",
        author_email="willmorejg@gmail.com",
        maintainer="James G Willmore",
        maintainer_email="willmorejg@gmail.com",
        description="A Python package using GIS to determine auto theft patterns",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/willmorejg/auto_thievia",
        project_urls={
            "Bug Tracker": "https://github.com/willmorejg/auto_thievia/issues",
            "Documentation": "https://auto_thievia.readthedocs.io/",
            "Source Code": "https://github.com/willmorejg/auto_thievia",
            "Changelog": "https://github.com/willmorejg/auto_thievia/blob/main/CHANGELOG.md",
        },
        license="Apache-2.0",
        license_files=["LICENSE"],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Text Processing",
            "Topic :: Utilities",
        ],
        keywords=["pdf", "parse", "text", "extraction", "document"],
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.8",
        install_requires=get_requirements(),
        extras_require=get_extras_require(),
        include_package_data=True,
        package_data={
            "auto_thievia": ["data/*.json", "templates/*.html"],
        },
        entry_points={
            "console_scripts": [
                "auto_thievia=auto_thievia.cli:main",
            ],
        },
        cmdclass=COMMAND_CLASSES,
        zip_safe=False,
    )
