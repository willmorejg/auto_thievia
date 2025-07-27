"""
Pytest configuration and shared fixtures for auto_thievia tests.
"""

import sys
from pathlib import Path

import pytest

# Add src directory to path so we can import auto_thievia modules
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.name or "full_conversion_workflow" in item.name:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def test_data_directory():
    """Provide the test data directory path."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def example_html():
    """Provide the example HTML file path."""
    test_data_dir = Path(__file__).parent / "test_data"
    html_file = test_data_dir / "example.html"
    if not html_file.exists():
        pytest.skip(f"Test HTML file {html_file} not found")
    return html_file
