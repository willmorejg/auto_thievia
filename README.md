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

A comprehensive Python package using GIS to analyze auto theft patterns, predict criminal activities, and identify potential chop shop locations. Features a modern web interface with REST API endpoints and machine learning capabilities.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

Auto Thievia is a specialized GIS analysis tool designed to help law enforcement and security professionals analyze auto theft patterns in urban areas. The package provides sophisticated mapping capabilities, spatial analysis, predictive modeling, and a modern web interface for comprehensive investigation support.

### Key Features

- 🗺️ **GIS Mapping**: Interactive and static map generation using Folium and Matplotlib
- 📍 **Spatial Analysis**: Point pattern analysis for theft incident clustering
- 🔍 **Pattern Recognition**: Identify theft hotspots and temporal patterns
- 🤖 **Machine Learning**: Self-supervised learning for predicting criminal activities
- 📊 **Data Visualization**: Comprehensive plotting and visualization tools
- 💾 **Multiple Formats**: Support for Shapefiles, GeoJSON, and other GIS formats
- 🌐 **Web Interface**: FastAPI REST endpoints with NuxtJS frontend
- 🔬 **Advanced Analytics**: DBSCAN clustering, anomaly detection, and risk assessment

## 🏗️ Architecture

### Backend Components
- **FastAPI REST API** (`src/auto_thievia/api.py`) - Interactive map generation and analysis endpoints
- **GIS Mapper** (`src/auto_thievia/gis_mapper.py`) - Core mapping and visualization
- **Machine Learning** (`src/auto_thievia/learning_models.py`) - Predictive modeling and analytics
- **Data Management** - Theft, suspect, and recovery data processors
- **Database Integration** - DuckDB for high-performance spatial queries

### Frontend Components (Optional)
- **Vue 3 + NuxtJS** framework for modern web interface
- **Tailwind CSS** for responsive design
- **Interactive Dashboard** with real-time API integration
- **Map Gallery** and generation controls

### Development Tools
- **Docker Configuration** for deployment
- **VS Code Tasks** for development workflow
- **Comprehensive Documentation** and examples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/willmorejg/auto_thievia.git
cd auto_thievia

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Or install with specific dependency groups
pip install -e ".[dev,test,gis,ml,web]"
```

### Start the API Server

```bash
# Start FastAPI server
python api_server.py
```

The API will be available at:
- **Main API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Optional: Start the Frontend

```bash
# Install Node.js dependencies
cd frontend
npm install

# Start development server
npm run dev  # Runs on http://localhost:3000
```

## 📡 REST API Endpoints

### Map Generation
- `POST /maps/create` - Create custom interactive map from coordinates
- `GET /maps/theft` - Generate theft analysis map with hotspots
- `GET /maps/suspects` - Generate suspect risk analysis map  
- `GET /maps/recovery` - Generate vehicle recovery analysis map
- `GET /maps/view/{map_id}` - View generated map in browser
- `GET /maps/list` - List all available maps

### Analysis Endpoints
- `GET /analysis/theft` - Perform theft pattern analysis with filtering
- `GET /analysis/suspects` - Analyze suspect risk and activity patterns

### Utility Endpoints
- `GET /health` - System health check and service status
- `GET /` - API documentation homepage

## 🗺️ Map Types and Analysis

### Theft Analysis Maps
- **Hotspot Detection**: DBSCAN clustering algorithms for theft concentration areas
- **Temporal Analysis**: Time-based pattern recognition and visualization
- **Vehicle Filtering**: Type, make, value, and theft method categorization
- **Interactive Features**: Detailed popups with incident information
- **Risk Prediction**: Machine learning-based risk area identification

### Suspect Analysis Maps  
- **Risk Assessment**: Color-coded visualization (Low/Medium/High/Critical)
- **Location Mapping**: Address and arrest location visualization
- **Activity Patterns**: Temporal and spatial behavior analysis
- **Filtering Options**: Risk levels, time periods, and activity types
- **Predictive Modeling**: Suspect activity prediction using clustering

### Recovery Analysis Maps
- **Recovery Locations**: Vehicle recovery point mapping
- **Condition Analysis**: Vehicle condition scoring and visualization  
- **Criminal Correlation**: Relationship analysis with known criminal locations
- **Time-based Filtering**: Historical recovery pattern analysis
- **Chop Shop Detection**: Proximity analysis for illegal operations

## 🤖 Machine Learning Capabilities

### Self-Supervised Learning Models

#### Theft Location Prediction
- **DBSCAN Clustering**: Automatic hotspot identification
- **K-means Analysis**: General spatial pattern discovery
- **Risk Grid Prediction**: Geographic area risk assessment
- **Feature Engineering**: Spatial, temporal, and vehicle characteristics

#### Suspect Activity Prediction
- **Activity Clustering**: Behavioral pattern identification
- **Area Prediction**: Potential suspect activity zones
- **Risk Scoring**: Proximity-based criminal activity assessment
- **Demographic Integration**: Age, history, and association analysis

#### Criminal Activity Detection
- **Anomaly Detection**: Isolation Forest for unusual patterns
- **Random Forest Models**: Criminal activity likelihood prediction
- **Proximity Analysis**: Distance-based risk calculation
- **Self-Supervised Labels**: Automatic training data generation

### Model Features
- **Automated Training**: Hyperparameter optimization and model selection
- **Persistence**: Save and load trained models for reuse
- **Real-time Prediction**: Live risk assessment for new data
- **Visualization Integration**: Seamless GIS mapper integration

## 🧪 Testing Framework

### Comprehensive Test Suite

#### Python Unit Tests
```bash
# Run all Python tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/auto_thievia --cov-report=html

# Run specific test modules
python -m pytest tests/test_gis_mapper.py -v
python -m pytest tests/test_learning_models.py -v
```

### Test Coverage Areas
- ✅ **Core GIS Functionality**: Mapping, visualization, spatial analysis
- ✅ **Machine Learning Models**: Training, prediction, persistence
- ✅ **Data Processing**: Theft, suspect, and recovery data handling
- ✅ **Integration Tests**: End-to-end workflow validation
- ✅ **Performance Tests**: Large dataset handling and concurrency

## 🎯 Usage Examples

### Generate Theft Analysis Map
```python
from auto_thievia import GISMapper, TheftData, LearningClass

# Initialize components
mapper = GISMapper()
theft_data = TheftData()
ml_models = LearningClass()

# Generate sample data
data = theft_data.generate_sample_data(center_lat=40.7357, center_lon=-74.1723)

# Perform clustering analysis
clusters = ml_models.theft_hotspot_detection(data)

# Create interactive map
map_obj = mapper.create_interactive_map(
    coordinates=data[['longitude', 'latitude']].values,
    popup_columns=['incident_id', 'vehicle_type', 'date'],
    title="NYC Auto Theft Hotspots"
)
```

### REST API Usage
```bash
# Generate theft analysis map
curl "http://localhost:8000/maps/theft?center_lat=40.7357&center_lon=-74.1723&zoom_start=12"

# Create custom map with coordinates
curl -X POST "http://localhost:8000/maps/create" \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": [[-74.0060, 40.7128], [-73.9857, 40.7484]],
    "title": "Custom Investigation Map"
  }'

# Perform theft pattern analysis
curl "http://localhost:8000/analysis/theft?start_date=2023-01-01&end_date=2023-12-31"
```

### Machine Learning Predictions
```python
# Train and use criminal activity prediction model
ml_models = LearningClass()

# Train model on recovery data
recovery_data = RecoveryData().generate_sample_data()
model = ml_models.criminal_activity_prediction(recovery_data)

# Predict criminal activity for new locations
new_locations = [[40.7128, -74.0060], [40.7589, -73.9851]]
predictions = ml_models.predict_criminal_activity_areas(new_locations)
```

## 🛠️ Development Setup

### VS Code Configuration

The project includes comprehensive VS Code configuration for optimal development:

- **Python Environment**: Automatic virtual environment detection
- **Test Discovery**: Integrated pytest test running
- **Code Formatting**: Black, isort, and flake8 integration
- **Debugging**: Pre-configured launch configurations
- **Tasks**: Common development tasks (test, format, lint, type-check)

### Development Tasks
```bash
# Format code
python -m black src/ tests/

# Sort imports
python -m isort src/ tests/

# Lint code
python -m flake8 src/ tests/

# Type checking
python -m mypy src/

# Clean build artifacts
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```

### Docker Deployment
```bash
# Build container
docker build -t auto-thievia .

# Run with docker-compose
docker-compose up -d
```

## 📊 Dependencies and Requirements

### Core Dependencies
- **GIS**: geopandas, folium, shapely, contextily, pyproj
- **Data**: pandas, numpy, duckdb, geopy
- **ML**: scikit-learn, joblib, networkx
- **Web**: fastapi, uvicorn, pydantic, requests
- **Visualization**: matplotlib, plotly, seaborn, pillow

### Optional Dependencies
- **Advanced GIS**: rasterio, osmnx, rtree
- **ML Extensions**: xgboost, lightgbm, catboost
- **Development**: pytest, black, mypy, sphinx
- **Testing**: coverage, pytest-mock

## 📁 Project Structure

```
auto_thievia/
├── src/auto_thievia/           # Core package
│   ├── __init__.py
│   ├── api.py                  # FastAPI REST endpoints
│   ├── gis_mapper.py          # GIS mapping and visualization
│   ├── learning_models.py     # Machine learning models
│   ├── theft_data.py          # Theft data processing
│   ├── suspect_data.py        # Suspect data management
│   ├── recovery_data.py       # Recovery data analysis
│   └── duckdb_persistence.py  # Database integration
├── tests/                      # Test suites
│   ├── test_*.py              # Python unit tests
│   └── data/                  # Test data and outputs
├── frontend/                   # NuxtJS web interface (optional)
├── data/                      # Sample data and outputs
├── .vscode/                   # VS Code configuration
├── api_server.py              # API server launcher
├── docker-compose.yml         # Docker deployment
└── pyproject.toml            # Project configuration
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev,test]"`)
4. Run tests to ensure everything works (`python -m pytest tests/`)
5. Make your changes and add tests
6. Run the test suite (`python -m pytest`)
7. Format and lint your code (`python -m black src/ tests/`)
8. Commit your changes (`git commit -m 'Add amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` directory for detailed guides
- **API Documentation**: Visit http://localhost:8000/docs when server is running
- **Issues**: Report bugs and feature requests on GitHub
- **Tests**: Run `python -m pytest` to validate installation

## 🚀 Roadmap

- [ ] Enhanced machine learning models with deep learning integration
- [ ] Real-time data streaming and live map updates
- [ ] Mobile application for field investigation support
- [ ] Advanced geofencing and alert systems
- [ ] Integration with law enforcement databases
- [ ] Predictive policing optimization algorithms

---

**Auto Thievia** - Empowering law enforcement with advanced GIS analytics and machine learning for auto theft investigation and prevention.
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

## � Data Management

Auto Thievia provides specialized classes for managing different types of crime data:

### TheftData - Auto Theft Incident Management

The `TheftData` class handles auto theft incident data with comprehensive CSV import/export and database persistence.

```python
from auto_thievia import TheftData, TheftDataPersistence

# Initialize theft data manager
theft_manager = TheftData()

# Create sample data or import from CSV
theft_data = theft_manager.import_from_csv("theft_incidents.csv")

# Database persistence with DuckDB
db_manager = TheftDataPersistence("theft_database.db")
db_manager.save_theft_data(theft_data, "incidents")

# Query by location
nearby_thefts = db_manager.query_by_distance(40.7357, -74.1724, radius_km=2.0)
```

### SuspectData - Suspect Information Management

The `SuspectData` class manages suspect information including personal details, addresses, and arrest history.

```python
from auto_thievia import SuspectData, SuspectDataPersistence

# Initialize suspect data manager
suspect_manager = SuspectData()

# Import suspect data from CSV
suspect_data = suspect_manager.import_from_csv("suspect_database.csv")

# Analyze suspect patterns
high_risk = suspect_manager.get_high_risk_suspects()
repeat_offenders = suspect_manager.get_repeat_offenders()

# Geographic analysis
nearby_suspects = suspect_manager.get_suspects_by_distance(
    center_lat=40.7357, 
    center_lon=-74.1724, 
    radius_km=1.0,
    location_type='address'  # or 'arrest'
)

# Create arrest location visualization
arrest_points = suspect_manager.create_arrest_points_gdf()

# Database persistence with DuckDB
with SuspectDataPersistence("suspect_database.db") as db:
    # Save data to database
    db.save_suspect_data(suspect_data)
    
    # Query suspects by various criteria
    high_risk_db = db.query_suspects_by_criteria(risk_levels=['High', 'Critical'])
    young_males = db.query_suspects_by_criteria(min_age=18, max_age=30, gender='Male')
    
    # Geographic queries
    nearby_addresses = db.query_suspects_by_address_distance(40.7357, -74.1724, 2.0)
    nearby_arrests = db.query_suspects_by_arrest_distance(40.7357, -74.1724, 2.0)
    
    # Get database statistics
    stats = db.get_suspect_statistics()
```

### Shared DuckDB Persistence

The `DuckDbPersistence` class provides a generic database layer that can be used by any data type:

```python
from auto_thievia import DuckDbPersistence

# Create a generic database manager
with DuckDbPersistence("shared_database.db") as db:
    # Create custom tables
    schema = {"id": "VARCHAR PRIMARY KEY", "name": "VARCHAR", "value": "DOUBLE"}
    db.create_table("custom_data", schema)
    
    # Execute custom queries
    results = db.execute_query("SELECT COUNT(*) FROM custom_data")
    
    # Get table information
    table_info = db.get_table_info("custom_data")
```

### Required CSV Columns

**TheftData CSV Requirements:**
- `theft_lat`, `theft_lon` - Theft location coordinates
- `make`, `model`, `year` - Vehicle information
- `theft_date` - When the theft occurred

**SuspectData CSV Requirements:**
- `suspect_name`, `suspect_address` - Personal information
- `address_lat`, `address_lon` - Suspect address coordinates
- `last_arrest_date`, `arrest_location` - Arrest information
- `arrest_lat`, `arrest_lon` - Arrest location coordinates

## �📁 Project Structure

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