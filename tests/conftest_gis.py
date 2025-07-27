"""
Additional pytest configuration for GIS mapper tests.

This module provides additional fixtures and configuration specific to
the GISMapper testing suite.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from auto_thievia.gis_mapper import WGS84_CRS


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_output_dir(test_data_dir):
    """Ensure the test output directory exists."""
    output_dir = test_data_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def sample_theft_data():
    """Sample auto theft data for testing."""
    return {
        "coordinates": [
            (-74.1723, 40.7357),  # Newark, NJ
            (-74.2107, 40.6640),  # Elizabeth, NJ
            (-74.2632, 40.6976),  # Union, NJ
            (-74.1547, 40.7282),  # East Newark, NJ
            (-74.2090, 40.6581),  # Hillside, NJ
        ],
        "attributes": {
            "incident_id": ["NWK001", "ELZ001", "UNI001", "ENK001", "HIL001"],
            "date": [
                "2024-01-15",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
            ],
            "vehicle_type": ["sedan", "suv", "truck", "sedan", "coupe"],
            "value_stolen": [25000, 45000, 35000, 20000, 30000],
            "recovered": [False, True, False, False, True],
        },
    }


@pytest.fixture(scope="session")
def sample_city_boundaries():
    """Sample city boundary polygons for testing."""
    # Create simplified NJ city boundaries
    newark_boundary = Polygon(
        [
            (-74.19, 40.70),
            (-74.19, 40.76),
            (-74.15, 40.76),
            (-74.15, 40.70),
            (-74.19, 40.70),
        ]
    )

    elizabeth_boundary = Polygon(
        [
            (-74.23, 40.64),
            (-74.23, 40.69),
            (-74.19, 40.69),
            (-74.19, 40.64),
            (-74.23, 40.64),
        ]
    )

    union_boundary = Polygon(
        [
            (-74.28, 40.68),
            (-74.28, 40.72),
            (-74.24, 40.72),
            (-74.24, 40.68),
            (-74.28, 40.68),
        ]
    )

    return gpd.GeoDataFrame(
        {
            "city_name": ["Newark", "Elizabeth", "Union"],
            "state": ["NJ", "NJ", "NJ"],
            "population": [311549, 137298, 59819],
            "area_sq_km": [67.04, 35.37, 23.85],
        },
        geometry=[newark_boundary, elizabeth_boundary, union_boundary],
        crs=WGS84_CRS,
    )


@pytest.fixture
def temp_shapefile(sample_city_boundaries):
    """Create a temporary shapefile for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        shapefile_path = Path(temp_dir) / "test_cities.shp"
        sample_city_boundaries.to_file(shapefile_path)
        yield shapefile_path


@pytest.fixture
def cleanup_output_files(test_output_dir):
    """Fixture to clean up test output files after tests."""
    # Setup: yield control to the test
    yield

    # Teardown: clean up test files (but keep the directory)
    test_files = [
        "test_output.shp",
        "test_output.shx",
        "test_output.dbf",
        "test_output.prj",
        "test_output.geojson",
        "integration_test_output.shp",
        "integration_test_output.shx",
        "integration_test_output.dbf",
        "integration_test_output.prj",
    ]

    for filename in test_files:
        file_path = test_output_dir / filename
        if file_path.exists():
            file_path.unlink()


@pytest.fixture(autouse=True)
def suppress_matplotlib_warnings():
    """Suppress matplotlib backend warnings during testing."""
    import warnings

    import matplotlib

    # Use Agg backend for testing (non-interactive)
    matplotlib.use("Agg")

    # Suppress specific matplotlib warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@pytest.fixture
def mock_osm_data():
    """Mock OSM data structure for testing."""
    return {
        "roads": gpd.GeoDataFrame(
            {
                "highway": ["primary", "secondary", "residential"],
                "name": ["McCarter Hwy", "Elizabeth Ave", "Morris Ave"],
            },
            geometry=[
                Point(-74.17, 40.74),  # Newark
                Point(-74.21, 40.66),  # Elizabeth
                Point(-74.26, 40.70),  # Union
            ],
            crs=WGS84_CRS,
        ),
        "amenities": gpd.GeoDataFrame(
            {
                "amenity": ["gas_station", "parking", "repair_shop"],
                "name": ["Shell Station", "City Parking", "Auto Repair"],
            },
            geometry=[
                Point(-74.175, 40.735),
                Point(-74.215, 40.665),
                Point(-74.265, 40.695),
            ],
            crs=WGS84_CRS,
        ),
    }
