"""
Tests for the GISMapper class.

This module contains comprehensive tests for the GISMapper functionality,
including shapefile reading, point creation, and visualization methods.
"""

import warnings
from pathlib import Path
from unittest.mock import patch

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from auto_thievia.gis_mapper import WEB_MERCATOR_CRS, WGS84_CRS, GISMapper


class TestGISMapper:
    """Test cases for the GISMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a GISMapper instance for testing."""
        return GISMapper()

    @pytest.fixture
    def sample_coordinates(self):
        """Sample coordinates for testing."""
        return [
            (-74.1723, 40.7357),  # Newark, NJ
            (-74.2107, 40.6640),  # Elizabeth, NJ
            (-74.2632, 40.6976),  # Union, NJ
        ]

    @pytest.fixture
    def sample_attributes(self):
        """Sample attributes for testing."""
        return {
            "incident_id": [1, 2, 3],
            "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "vehicle_type": ["sedan", "suv", "truck"],
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "longitude": [-74.1723, -74.2107, -74.2632],
                "latitude": [40.7357, 40.6640, 40.6976],
                "incident_id": [1, 2, 3],
                "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "vehicle_type": ["sedan", "suv", "truck"],
            }
        )

    @pytest.fixture
    def test_shapefile_path(self, tmp_path):
        """Create a test shapefile for testing."""
        # Create a simple test GeoDataFrame with NJ area boundaries
        geometry = [
            Polygon(
                [(-74.2, 40.7), (-74.2, 40.8), (-74.1, 40.8), (-74.1, 40.7)]
            ),  # Newark area
            Polygon(
                [(-74.3, 40.6), (-74.3, 40.7), (-74.2, 40.7), (-74.2, 40.6)]
            ),  # Elizabeth/Union area
        ]
        gdf = gpd.GeoDataFrame(
            {
                "name": ["Newark Area", "Elizabeth-Union Area"],
                "population": [311549, 150000],
            },
            geometry=geometry,
            crs=WGS84_CRS,
        )

        shapefile_path = tmp_path / "test_boundary.shp"
        gdf.to_file(shapefile_path)
        return shapefile_path

    @pytest.fixture
    def output_dir(self):
        """Get the test output directory."""
        output_path = Path(__file__).parent / "data" / "output"
        output_path.mkdir(exist_ok=True)
        return output_path

    def test_init_default_crs(self):
        """Test GISMapper initialization with default CRS."""
        mapper = GISMapper()
        assert mapper.crs == WGS84_CRS
        assert mapper.base_map is None
        assert mapper.theft_points is None
        assert mapper._osm_data == {}

    def test_init_custom_crs(self):
        """Test GISMapper initialization with custom CRS."""
        custom_crs = WEB_MERCATOR_CRS
        mapper = GISMapper(crs=custom_crs)
        assert mapper.crs == custom_crs

    def test_read_osm_shapefile_success(self, mapper, test_shapefile_path):
        """Test successful reading of a shapefile."""
        gdf = mapper.read_osm_shapefile(test_shapefile_path)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert "name" in gdf.columns
        assert "population" in gdf.columns
        assert gdf.crs == mapper.crs
        assert mapper.base_map is not None

    def test_read_osm_shapefile_file_not_found(self, mapper):
        """Test reading a non-existent shapefile."""
        with pytest.raises(FileNotFoundError):
            mapper.read_osm_shapefile("nonexistent_file.shp")

    def test_read_osm_shapefile_crs_conversion(self, mapper, test_shapefile_path):
        """Test CRS conversion when reading shapefile."""
        # Set mapper to use Web Mercator
        mapper.crs = WEB_MERCATOR_CRS
        gdf = mapper.read_osm_shapefile(test_shapefile_path)

        assert gdf.crs == WEB_MERCATOR_CRS

    def test_create_points_from_coordinates_success(
        self, mapper, sample_coordinates, sample_attributes
    ):
        """Test successful creation of points from coordinates."""
        gdf = mapper.create_points_from_coordinates(
            sample_coordinates, sample_attributes
        )

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert gdf.crs == mapper.crs
        assert all(isinstance(geom, Point) for geom in gdf.geometry)
        assert "incident_id" in gdf.columns
        assert mapper.theft_points is not None

    def test_create_points_from_coordinates_no_attributes(
        self, mapper, sample_coordinates
    ):
        """Test creation of points without attributes."""
        gdf = mapper.create_points_from_coordinates(sample_coordinates)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert len(gdf.columns) == 1  # Only geometry column

    def test_create_points_from_coordinates_empty_list(self, mapper):
        """Test creation of points from empty coordinate list."""
        with pytest.raises(ValueError, match="No coordinates provided"):
            mapper.create_points_from_coordinates([])

    def test_create_points_from_coordinates_invalid_coordinates(self, mapper):
        """Test creation of points with invalid coordinates."""
        invalid_coords = [(200, 40), (-74, 100)]  # Invalid longitude and latitude

        with pytest.raises(ValueError, match="Invalid coordinates"):
            mapper.create_points_from_coordinates(invalid_coords)

    def test_create_points_from_coordinates_mismatched_attributes(
        self, mapper, sample_coordinates
    ):
        """Test creation of points with mismatched attribute lengths."""
        mismatched_attributes = {
            "incident_id": [1, 2]
        }  # Only 2 values for 3 coordinates

        with pytest.raises(ValueError, match="Attribute 'incident_id' has 2 values"):
            mapper.create_points_from_coordinates(
                sample_coordinates, mismatched_attributes
            )

    def test_create_points_from_dataframe_success(self, mapper, sample_dataframe):
        """Test successful creation of points from DataFrame."""
        gdf = mapper.create_points_from_dataframe(sample_dataframe)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert gdf.crs == mapper.crs
        assert "incident_id" in gdf.columns
        assert "longitude" not in gdf.columns  # Should be removed
        assert "latitude" not in gdf.columns  # Should be removed

    def test_create_points_from_dataframe_custom_columns(self, mapper):
        """Test creation of points with custom column names."""
        df = pd.DataFrame(
            {
                "lon": [-74.1723, -74.2107],  # Newark, Elizabeth
                "lat": [40.7357, 40.6640],
                "id": [1, 2],
            }
        )

        gdf = mapper.create_points_from_dataframe(df, lon_col="lon", lat_col="lat")

        assert len(gdf) == 2
        assert "id" in gdf.columns
        assert "lon" not in gdf.columns
        assert "lat" not in gdf.columns

    def test_create_points_from_dataframe_missing_columns(
        self, mapper, sample_dataframe
    ):
        """Test creation of points with missing coordinate columns."""
        with pytest.raises(
            KeyError, match="Columns 'invalid_lon' and/or 'invalid_lat' not found"
        ):
            mapper.create_points_from_dataframe(
                sample_dataframe, "invalid_lon", "invalid_lat"
            )

    def test_create_points_from_dataframe_missing_values(self, mapper):
        """Test handling of missing coordinate values."""
        df = pd.DataFrame(
            {
                "longitude": [-74.1723, None, -74.2632],  # Newark, missing, Union
                "latitude": [40.7357, 40.6640, None],  # Valid, Elizabeth, missing
                "id": [1, 2, 3],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gdf = mapper.create_points_from_dataframe(df)

            assert len(w) == 1
            assert "missing coordinate values" in str(w[0].message)
            assert len(gdf) == 1  # Only one valid coordinate pair

    def test_plot_static_map_with_points(self, mapper, sample_coordinates):
        """Test static map plotting with points."""
        mapper.create_points_from_coordinates(sample_coordinates)

        fig, ax = mapper.plot_static_map(
            add_basemap=False
        )  # Disable basemap for testing

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)  # Clean up

    def test_plot_static_map_with_title(self, mapper, sample_coordinates):
        """Test static map plotting with custom title."""
        mapper.create_points_from_coordinates(sample_coordinates)

        custom_title = "Test Auto Theft Map - Newark Area"
        fig, ax = mapper.plot_static_map(title=custom_title, add_basemap=False)

        assert ax.get_title() == custom_title
        plt.close(fig)

    def test_plot_static_map_custom_styling(self, mapper, sample_coordinates):
        """Test static map plotting with custom styling."""
        mapper.create_points_from_coordinates(sample_coordinates)

        fig, _ = mapper.plot_static_map(
            point_color="blue", point_size=100, figsize=(10, 6), add_basemap=False
        )

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)

    @patch("contextily.add_basemap")
    def test_plot_static_map_basemap_error(
        self, mock_add_basemap, mapper, sample_coordinates
    ):
        """Test handling of basemap errors."""
        mock_add_basemap.side_effect = Exception("Basemap error")
        mapper.create_points_from_coordinates(sample_coordinates)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, _ = mapper.plot_static_map(add_basemap=True)

            # Check that our specific basemap warning was issued
            basemap_warnings = [
                warning
                for warning in w
                if "Could not add basemap" in str(warning.message)
            ]
            assert len(basemap_warnings) >= 1
            assert "Could not add basemap" in str(basemap_warnings[0].message)

        plt.close(fig)

    def test_create_interactive_map_with_points(
        self, mapper, sample_coordinates, sample_attributes
    ):
        """Test interactive map creation with points."""
        mapper.create_points_from_coordinates(sample_coordinates, sample_attributes)

        map_obj = mapper.create_interactive_map(popup_columns=["incident_id", "date"])

        assert isinstance(map_obj, folium.Map)
        # Check that markers were added (this is a bit tricky to test directly)
        map_html = map_obj._repr_html_()
        assert "CircleMarker" in map_html or "Marker" in map_html

    def test_create_interactive_map_custom_center(self, mapper, sample_coordinates):
        """Test interactive map with custom center coordinates."""
        mapper.create_points_from_coordinates(sample_coordinates)

        center_lat, center_lon = 40.7, -74.2  # Center of Newark area
        map_obj = mapper.create_interactive_map(
            center_lat=center_lat, center_lon=center_lon, zoom_start=12
        )

        assert isinstance(map_obj, folium.Map)
        # Folium's location is [lat, lon]
        assert map_obj.location == [center_lat, center_lon]

    def test_create_interactive_map_no_points(self, mapper):
        """Test interactive map creation without points."""
        map_obj = mapper.create_interactive_map()

        assert isinstance(map_obj, folium.Map)
        # Should use default center (US geographic center)
        assert map_obj.location == [39.8283, -98.5795]

    def test_get_bounds_with_points(self, mapper, sample_coordinates):
        """Test getting bounds from points."""
        mapper.create_points_from_coordinates(sample_coordinates)
        bounds = mapper.get_bounds()

        assert len(bounds) == 4
        minx, miny, maxx, maxy = bounds
        assert minx <= maxx
        assert miny <= maxy

    def test_get_bounds_with_custom_gdf(self, mapper, sample_coordinates):
        """Test getting bounds from custom GeoDataFrame."""
        gdf = mapper.create_points_from_coordinates(sample_coordinates)
        bounds = mapper.get_bounds(gdf)

        assert len(bounds) == 4

    def test_get_bounds_no_data(self, mapper):
        """Test getting bounds with no data."""
        with pytest.raises(ValueError, match="No GeoDataFrame provided"):
            mapper.get_bounds()

    def test_save_to_file_shapefile(self, mapper, sample_coordinates, output_dir):
        """Test saving GeoDataFrame to shapefile."""
        gdf = mapper.create_points_from_coordinates(sample_coordinates)
        output_path = output_dir / "test_output.shp"

        mapper.save_to_file(gdf, output_path)

        assert output_path.exists()
        # Verify the file can be read back
        loaded_gdf = gpd.read_file(output_path)
        assert len(loaded_gdf) == len(gdf)

    def test_save_to_file_geojson(self, mapper, sample_coordinates, output_dir):
        """Test saving GeoDataFrame to GeoJSON."""
        gdf = mapper.create_points_from_coordinates(sample_coordinates)
        output_path = output_dir / "test_output.geojson"

        mapper.save_to_file(gdf, output_path, driver="GeoJSON")

        assert output_path.exists()
        # Verify the file can be read back
        loaded_gdf = gpd.read_file(output_path)
        assert len(loaded_gdf) == len(gdf)

    def test_save_to_file_creates_directory(self, mapper, sample_coordinates, tmp_path):
        """Test that save_to_file creates directories if they don't exist."""
        gdf = mapper.create_points_from_coordinates(sample_coordinates)
        nested_path = tmp_path / "new_dir" / "nested_dir" / "test_output.shp"

        mapper.save_to_file(gdf, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_integration_workflow(
        self, mapper, test_shapefile_path, sample_coordinates, output_dir
    ):
        """Test a complete workflow integration."""
        # Read base map
        mapper.read_osm_shapefile(test_shapefile_path)
        assert mapper.base_map is not None

        # Create theft points
        attributes = {"incident_id": [1, 2, 3], "severity": ["high", "medium", "low"]}
        points_gdf = mapper.create_points_from_coordinates(
            sample_coordinates, attributes
        )
        assert mapper.theft_points is not None

        # Create visualizations
        fig, _ = mapper.plot_static_map(add_basemap=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        interactive_map = mapper.create_interactive_map(popup_columns=["incident_id"])
        assert isinstance(interactive_map, folium.Map)

        # Save results
        output_path = output_dir / "integration_test_output.shp"
        mapper.save_to_file(points_gdf, output_path)
        assert output_path.exists()

        # Get bounds
        bounds = mapper.get_bounds()
        assert len(bounds) == 4


class TestGISMapperEdgeCases:
    """Test edge cases and error conditions for GISMapper."""

    def test_coordinate_validation_edge_cases(self):
        """Test coordinate validation with edge case values."""
        mapper = GISMapper()

        # Test boundary values
        boundary_coords = [
            (-180, -90),  # Southwest corner
            (180, 90),  # Northeast corner
            (0, 0),  # Null island
        ]

        gdf = mapper.create_points_from_coordinates(boundary_coords)
        assert len(gdf) == 3

    def test_empty_shapefile_handling(self, tmp_path):
        """Test handling of empty shapefiles."""
        # Create an empty GeoDataFrame and save it
        empty_gdf = gpd.GeoDataFrame({"col1": []}, geometry=[], crs=WGS84_CRS)
        empty_shapefile = tmp_path / "empty.shp"
        empty_gdf.to_file(empty_shapefile)

        mapper = GISMapper()
        result_gdf = mapper.read_osm_shapefile(empty_shapefile)

        assert len(result_gdf) == 0
        assert isinstance(result_gdf, gpd.GeoDataFrame)

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        mapper = GISMapper()

        # Create a larger dataset
        import numpy as np

        rng = np.random.default_rng(42)  # Use new random generator
        n_points = 1000
        lons = rng.uniform(-180, 180, n_points)
        lats = rng.uniform(-90, 90, n_points)
        coordinates = list(zip(lons, lats))

        gdf = mapper.create_points_from_coordinates(coordinates)
        assert len(gdf) == n_points

    def test_crs_handling_with_none(self, tmp_path):
        """Test handling of shapefiles with no CRS."""
        # Create a GeoDataFrame without CRS
        geometry = [Point(-74, 40), Point(-87, 41)]
        gdf_no_crs = gpd.GeoDataFrame({"id": [1, 2]}, geometry=geometry)
        # Explicitly set crs to None
        gdf_no_crs.crs = None

        shapefile_path = tmp_path / "no_crs.shp"
        gdf_no_crs.to_file(shapefile_path)

        mapper = GISMapper()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_gdf = mapper.read_osm_shapefile(shapefile_path)

            assert len(w) == 1
            assert "No CRS found" in str(w[0].message)
            assert result_gdf.crs == WGS84_CRS


if __name__ == "__main__":
    pytest.main([__file__])
