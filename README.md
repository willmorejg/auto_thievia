<!--
 Copyright 2025 James G Willmore
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Auto Thievia 🚗🗺️

A Python package using GIS to determine auto theft patterns and identify potential chop shop locations.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

Auto Thievia is a specialized GIS analysis tool designed to help law enforcement and security professionals analyze auto theft patterns in urban areas. The package provides sophisticated mapping capabilities, spatial analysis, and predictive modeling to identify high-risk areas and potential illegal operations.

### Key Features

- 🗺️ **GIS Mapping**: Interactive and static map generation using Folium and Matplotlib
- 📍 **Spatial Analysis**: Point pattern analysis for theft incident clustering
- 🔍 **Pattern Recognition**: Identify theft hotspots and temporal patterns
- 📊 **Data Visualization**: Comprehensive plotting and visualization tools
- 💾 **Multiple Formats**: Support for Shapefiles, GeoJSON, and other GIS formats
- 🌐 **Web Integration**: Interactive maps with popup information

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/willmorejg/auto_thievia.git
cd auto_thievia

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[gis,ml,dev]"
```

### Basic Usage

```python
from auto_thievia import GISMapper
import pandas as pd

# Initialize the mapper
mapper = GISMapper()

# Create theft incident points
theft_coordinates = [
    (-74.1723, 40.7357),  # Newark, NJ
    (-74.2107, 40.6640),  # Elizabeth, NJ
    (-74.2632, 40.6976),  # Union, NJ
]

attributes = {
    "incident_id": ["NWK001", "ELZ001", "UNI001"],
    "vehicle_type": ["sedan", "suv", "truck"],
    "value_stolen": [25000, 45000, 35000]
}

# Create GeoDataFrame
theft_gdf = mapper.create_points_from_coordinates(theft_coordinates, attributes)

# Generate interactive map
interactive_map = mapper.create_interactive_map(
    popup_columns=["incident_id", "vehicle_type", "value_stolen"]
)
interactive_map.save("theft_analysis.html")

# Create static visualization
fig, ax = mapper.plot_static_map(
    title="Auto Theft Incidents - Newark Area",
    add_basemap=True
)
```

## 📁 Project Structure

```
auto_thievia/
├── src/
│   └── auto_thievia/
│       ├── __init__.py          # Package initialization
│       └── gis_mapper.py        # Core GIS mapping functionality
├── tests/
│   ├── test_gis_mapper.py       # Comprehensive test suite
│   ├── conftest_gis.py          # GIS-specific test fixtures
│   ├── run_gis_tests.py         # Test runner script
│   └── data/
│       └── output/              # Test output files
├── data/
│   └── shapefiles/              # Source shapefiles
├── demo_gis_mapper.py           # Demonstration script
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## 🗺️ GIS Mapper Features

### Core Functionality

The `GISMapper` class provides comprehensive GIS operations:

#### Data Input
- **Shapefile Reading**: Load OSM and other standard shapefiles
- **Coordinate Creation**: Convert lat/lon pairs to GeoDataFrames
- **DataFrame Integration**: Work with existing pandas DataFrames
- **CRS Management**: Automatic coordinate system handling

#### Visualization
- **Static Maps**: Matplotlib-based maps with contextual basemaps
- **Interactive Maps**: Folium maps with markers and popups
- **Custom Styling**: Configurable colors, sizes, and symbols
- **Basemap Integration**: Contextily integration for satellite/street maps

#### Data Export
- **Multiple Formats**: Shapefile, GeoJSON, and other OGR formats
- **Metadata Preservation**: Maintain attribute data and CRS information

### Example Workflows

#### Load and Analyze Existing Data
```python
# Load base map
base_map = mapper.read_osm_shapefile("city_boundaries.shp")

# Load theft data from CSV
df = pd.read_csv("theft_incidents.csv")
theft_points = mapper.create_points_from_dataframe(df)

# Generate analysis
bounds = mapper.get_bounds()
print(f"Analysis area: {bounds}")
```

#### Create Visualizations
```python
# Static map with custom styling
fig, ax = mapper.plot_static_map(
    point_color="red",
    point_size=100,
    figsize=(15, 10),
    title="High-Value Vehicle Thefts"
)

# Interactive map with detailed popups
map_obj = mapper.create_interactive_map(
    popup_columns=["date", "vehicle_type", "value_stolen"],
    zoom_start=12
)
```

## 🔧 Development

### Requirements

**Core Dependencies:**
- `geopandas>=0.14.0` - Geospatial data manipulation
- `pandas>=2.0.0` - Data analysis
- `matplotlib>=3.7.0` - Static plotting
- `folium>=0.14.0` - Interactive maps
- `shapely>=2.0.0` - Geometric operations

**Optional Dependencies:**
- `rasterio>=1.3.0` - Raster data support
- `contextily>=1.4.0` - Basemap integration
- `osmnx>=1.6.0` - OpenStreetMap integration

### Testing

The project includes a comprehensive test suite with 32+ test cases:

```bash
# Run all tests
python -m pytest tests/test_gis_mapper.py -v

# Run with coverage
python -m pytest tests/test_gis_mapper.py --cov=src/auto_thievia --cov-report=html

# Run GIS-specific test runner
python tests/run_gis_tests.py

# Run demonstration
python demo_gis_mapper.py
```

### Code Quality

```bash
# Format code
python -m black src/ tests/

# Sort imports
python -m isort src/ tests/

# Lint code
python -m flake8 src/ tests/

# Type checking
python -m mypy src/
```

## 📊 Use Cases

### Law Enforcement
- **Hotspot Analysis**: Identify areas with high theft concentrations
- **Pattern Recognition**: Temporal and spatial theft patterns
- **Resource Allocation**: Optimize patrol routes and coverage
- **Investigation Support**: Visual analysis for case development

### Insurance Companies
- **Risk Assessment**: Geographic risk modeling
- **Premium Calculation**: Location-based risk factors
- **Claims Analysis**: Spatial clustering of claims
- **Fraud Detection**: Unusual geographic patterns

### Urban Planning
- **Security Infrastructure**: Optimal placement of cameras and lighting
- **Parking Design**: Secure parking facility planning
- **Community Safety**: Public safety improvement initiatives

## 🌟 Newark Area Focus

This implementation specifically focuses on the Newark, New Jersey metropolitan area, including:

- **Newark**: Primary urban center with high incident volume
- **Elizabeth**: Major transportation hub analysis
- **Union**: Suburban pattern analysis
- **East Newark**: Border area security
- **Hillside**: Residential area patterns

The choice of Newark area provides:
- Dense urban environment for analysis
- Varied socioeconomic areas
- Major transportation corridors
- Mix of residential and commercial zones

## 📈 Future Enhancements

- **Machine Learning**: Predictive modeling for chop shop locations
- **Temporal Analysis**: Time-series pattern recognition
- **Network Analysis**: Transportation route optimization
- **Real-time Integration**: Live data feeds and alerts
- **Mobile App**: Field data collection capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**James G Willmore**
- Email: willmorejg@gmail.com
- GitHub: [@willmorejg](https://github.com/willmorejg)

## 🙏 Acknowledgments

- OpenStreetMap contributors for geographic data
- GeoPandas and Folium communities
- Law enforcement professionals who provided domain expertise

---

*Auto Thievia - Mapping crime patterns for safer communities* 🚗🔒