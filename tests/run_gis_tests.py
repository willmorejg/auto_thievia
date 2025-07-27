#!/usr/bin/env python3
"""
Test runner for GIS Mapper tests.

This script runs the GIS mapper tests with appropriate configuration.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run GIS mapper tests."""
    print("🗺️  Auto Thievia GIS Mapper - Test Suite")
    print("=" * 45)

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Test commands to run
    test_commands = [
        {
            "name": "GIS Mapper Unit Tests",
            "cmd": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_gis_mapper.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "GIS Mapper Tests with Coverage",
            "cmd": [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_gis_mapper.py",
                "--cov=src/auto_thievia/gis_mapper",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "-v",
            ],
        },
    ]

    success_count = 0

    for test_config in test_commands:
        print(f"\n📊 Running {test_config['name']}...")
        print("-" * 40)

        try:
            subprocess.run(test_config["cmd"], cwd=project_root, check=True)
            print(f"✅ {test_config['name']} passed!")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"❌ {test_config['name']} failed with exit code {e.returncode}")
        except FileNotFoundError:
            print("❌ pytest not found. Install with: pip install pytest pytest-cov")
            return 1

    print(f"\n📈 Test Summary: {success_count}/{len(test_commands)} test suites passed")

    if success_count == len(test_commands):
        print("🎉 All GIS mapper tests passed successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
