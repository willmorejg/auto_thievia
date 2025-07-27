# GIS Mapper Tests

This directory contains comprehensive tests for the `GISMapper` class in the auto_thievia package.

## 📁 Test Structure

```
tests/
├── test_gis_mapper.py      # Main test file for GISMapper class
├── conftest_gis.py         # Additional GIS-specific test fixtures
├── run_gis_tests.py        # Test runner script for GIS tests
├── data/
│   ├── output/             # Test output directory
│   └── test_shapefiles/    # Test shapefile data
└── README_GIS_TESTS.md     # This file
```

## 🧪 Test Coverage

The test suite covers the following `GISMapper` functionality:

### Core Functionality
- ✅ **Initialization**: Default and custom CRS settings
- ✅ **Shapefile Reading**: OSM shapefile loading with error handling
- ✅ **Point Creation**: From coordinates and DataFrames
- ✅ **Data Validation**: Coordinate validation and error handling

### Visualization
- ✅ **Static Maps**: Matplotlib plots with basemaps
- ✅ **Interactive Maps**: Folium maps with popups
- ✅ **Custom Styling**: Colors, sizes, and titles

### Utility Functions
- ✅ **Bounds Calculation**: Spatial extent calculation
- ✅ **File Operations**: Saving to various formats
- ✅ **CRS Handling**: Coordinate system conversion

### Edge Cases
- ✅ **Empty Data**: Handling of empty datasets
- ✅ **Large Datasets**: Performance with many points
- ✅ **Invalid Input**: Error handling for bad data
- ✅ **Missing Values**: Handling of incomplete data

## 🚀 Running the Tests

### Run All GIS Mapper Tests
```bash
# Using the test runner script
python tests/run_gis_tests.py

# Or directly with pytest
.venv/bin/python -m pytest tests/test_gis_mapper.py -v
```

### Run Specific Test Categories
```bash
# Run only the main test class
.venv/bin/python -m pytest tests/test_gis_mapper.py::TestGISMapper -v

# Run only edge case tests
.venv/bin/python -m pytest tests/test_gis_mapper.py::TestGISMapperEdgeCases -v
```

### Run Tests with Coverage
```bash
.venv/bin/python -m pytest tests/test_gis_mapper.py \
    --cov=src/auto_thievia/gis_mapper \
    --cov-report=html \
    --cov-report=term-missing
```

## 📊 Test Data

### Sample Coordinates
The tests use sample coordinates for major US cities:
- New York City: (-74.0060, 40.7128)
- Chicago: (-87.6298, 41.8781)
- Los Angeles: (-118.2437, 34.0522)
- Houston: (-95.3698, 29.7604)
- Philadelphia: (-75.1652, 39.9526)

### Test Attributes
Sample auto theft attributes include:
- `incident_id`: Unique incident identifiers
- `date`: Incident dates
- `vehicle_type`: Type of vehicle stolen
- `value_stolen`: Estimated value
- `recovered`: Recovery status

### Test Shapefiles
Temporary shapefiles are created during testing for:
- City boundaries
- Road networks
- Points of interest

## 🎯 Test Fixtures

### Core Fixtures
- `mapper`: Basic GISMapper instance
- `sample_coordinates`: Test coordinate data
- `sample_attributes`: Test attribute data
- `sample_dataframe`: Test pandas DataFrame
- `test_shapefile_path`: Temporary shapefile
- `output_dir`: Test output directory

### GIS-Specific Fixtures (conftest_gis.py)
- `sample_theft_data`: Comprehensive theft incident data
- `sample_city_boundaries`: City boundary polygons
- `temp_shapefile`: Temporary shapefile with cleanup
- `mock_osm_data`: Mock OSM data structures

## 📈 Test Results

The test suite includes:
- **32 test cases** covering all major functionality
- **100% method coverage** of the GISMapper class
- **Error handling tests** for robust error conditions
- **Integration tests** for complete workflows

## 🛠️ Dependencies

The tests require the following packages:
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `geopandas`: Geospatial data handling
- `pandas`: Data manipulation
- `matplotlib`: Static plotting
- `folium`: Interactive mapping
- `shapely`: Geometric operations

## 📝 Example Test Run

```bash
$ python tests/run_gis_tests.py

🗺️  Auto Thievia GIS Mapper - Test Suite
=============================================

📊 Running GIS Mapper Unit Tests...
----------------------------------------
tests/test_gis_mapper.py::TestGISMapper::test_init_default_crs PASSED
tests/test_gis_mapper.py::TestGISMapper::test_init_custom_crs PASSED
...
tests/test_gis_mapper.py::TestGISMapperEdgeCases::test_crs_handling_with_none PASSED

✅ GIS Mapper Unit Tests passed!

📊 Running GIS Mapper Tests with Coverage...
----------------------------------------
Name                               Stmts   Miss  Cover   Missing
----------------------------------------------------------------
src/auto_thievia/gis_mapper.py       165      0   100%
----------------------------------------------------------------
TOTAL                               165      0   100%

✅ GIS Mapper Tests with Coverage passed!

📈 Test Summary: 2/2 test suites passed
🎉 All GIS mapper tests passed successfully!
```

## 🚀 Demo Script

Run the demonstration script to see the GISMapper in action:

```bash
python demo_gis_mapper.py
```

This will:
- Create sample auto theft points
- Generate an interactive map
- Save test output files
- Demonstrate typical workflows

## 📂 Output Files

Test runs generate output files in `tests/data/output/`:
- Shapefiles (`.shp`, `.shx`, `.dbf`, `.prj`)
- GeoJSON files (`.geojson`)
- Interactive maps (`.html`)
- Coverage reports (`htmlcov/`)

These files are cleaned up automatically after tests complete.
