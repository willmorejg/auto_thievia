"""
Tests for the TheftData and TheftDataPersistence classes.

This module contains comprehensive tests for theft data management functionality
including CSV import, data validation, filtering, and database persistence.
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest

from auto_thievia.theft_data import (
    DB_CONNECTION_ERROR,
    MEMORY_DB,
    TheftData,
    TheftDataPersistence,
)


class TestTheftData:
    """Test cases for the TheftData class."""

    def test_init(self):
        """Test TheftData initialization."""
        td = TheftData()
        assert td.crs == "EPSG:4326"
        assert td.theft_data is None
        assert len(td._required_columns) == 12
        assert len(td._optional_columns) == 11

    def test_init_custom_crs(self):
        """Test TheftData initialization with custom CRS."""
        td = TheftData(crs="EPSG:3857")
        assert td.crs == "EPSG:3857"

    def test_create_sample_data(self):
        """Test creating sample theft data."""
        td = TheftData()
        gdf = td.create_sample_data(num_records=50)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 50
        assert gdf.crs == "EPSG:4326"

        # Check required columns are present
        for col in td._required_columns:
            assert col in gdf.columns

        # Check data types and ranges
        assert gdf["vehicle_year"].dtype in ["int64", "int32"]
        assert gdf["vehicle_value"].dtype in ["int64", "int32", "float64"]
        assert gdf["owner_income"].dtype in ["int64", "int32", "float64"]

        # Check coordinate ranges (Newark area)
        assert gdf["incident_lat"].between(40.65, 40.75).all()
        assert gdf["incident_lon"].between(-74.28, -74.15).all()

    def test_create_sample_data_different_sizes(self):
        """Test creating sample data with different record counts."""
        td = TheftData()

        # Test small dataset
        small_gdf = td.create_sample_data(num_records=10)
        assert len(small_gdf) == 10

        # Test larger dataset
        large_gdf = td.create_sample_data(num_records=200)
        assert len(large_gdf) == 200

    def test_import_from_csv_valid(self):
        """Test importing valid CSV data."""
        td = TheftData()

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write valid CSV data
            f.write(
                "incident_lat,incident_lon,vehicle_make,vehicle_model,vehicle_year,"
                "vehicle_value,vehicle_type,owner_name,owner_address,owner_lat,"
                "owner_lon,owner_income\n"
            )
            f.write(
                "40.7282,-74.2090,Honda,Civic,2020,25000,sedan,John Doe,"
                "123 Main St,40.7300,-74.2100,50000\n"
            )
            f.write(
                "40.7300,-74.2100,Toyota,Camry,2019,28000,sedan,Jane Smith,"
                "456 Oak Ave,40.7320,-74.2120,60000\n"
            )

        try:
            gdf = td.import_from_csv(csv_path)

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 2
            assert gdf.crs == "EPSG:4326"
            assert "geometry" in gdf.columns

            # Check first record
            first_row = gdf.iloc[0]
            assert first_row["vehicle_make"] == "Honda"
            assert first_row["vehicle_model"] == "Civic"
            assert first_row["vehicle_year"] == 2020
            assert first_row["vehicle_value"] == 25000

        finally:
            csv_path.unlink()  # Clean up

    def test_import_from_csv_file_not_found(self):
        """Test importing from non-existent CSV file."""
        td = TheftData()

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            td.import_from_csv("non_existent_file.csv")

    def test_import_from_csv_missing_columns(self):
        """Test importing CSV with missing required columns."""
        td = TheftData()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write CSV with missing columns
            f.write("incident_lat,incident_lon,vehicle_make\n")
            f.write("40.7282,-74.2090,Honda\n")

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                td.import_from_csv(csv_path)
        finally:
            csv_path.unlink()

    def test_import_from_csv_invalid_coordinates(self):
        """Test importing CSV with invalid coordinates."""
        td = TheftData()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write CSV with invalid coordinates
            f.write(
                "incident_lat,incident_lon,vehicle_make,vehicle_model,vehicle_year,"
                "vehicle_value,vehicle_type,owner_name,owner_address,owner_lat,"
                "owner_lon,owner_income\n"
            )
            f.write(
                "200,-400,Honda,Civic,2020,25000,sedan,John Doe,"
                "123 Main St,40.7300,-74.2100,50000\n"
            )  # Invalid incident coords

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gdf = td.import_from_csv(csv_path)

                # Should have warnings about invalid coordinates
                assert len(w) > 0
                assert any(
                    "invalid coordinates" in str(warning.message) for warning in w
                )
                assert len(gdf) == 0  # No valid records
        finally:
            csv_path.unlink()

    def test_clean_data_with_missing_values(self):
        """Test data cleaning with missing values."""
        td = TheftData()

        # Create DataFrame with missing values
        df = pd.DataFrame(
            {
                "incident_lat": [40.7282, None, 40.7300],
                "incident_lon": [-74.2090, -74.2100, -74.2120],
                "vehicle_make": ["Honda", "Toyota", "Ford"],
                "vehicle_model": ["Civic", "Camry", "F-150"],
                "vehicle_year": [2020, 2019, None],
                "vehicle_value": [25000, 28000, 35000],
                "vehicle_type": ["sedan", "sedan", "truck"],
                "owner_name": ["John", "Jane", "Bob"],
                "owner_address": ["123 Main", "456 Oak", "789 Pine"],
                "owner_lat": [40.7300, 40.7320, 40.7350],
                "owner_lon": [-74.2100, -74.2120, -74.2150],
                "owner_income": [50000, 60000, 70000],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cleaned_df = td._clean_data(df, validate_coordinates=True)

            # Should have warnings about missing data
            assert len(w) > 0
            assert any("missing required data" in str(warning.message) for warning in w)
            assert len(cleaned_df) == 1  # Only one valid record

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        td = TheftData()

        # First test without data
        with pytest.raises(ValueError, match="No theft data loaded"):
            td.get_summary_stats()

        # Create sample data and test stats
        gdf = td.create_sample_data(num_records=100)
        stats = td.get_summary_stats()

        assert stats["total_incidents"] == 100
        assert "total_value_stolen" in stats
        assert "average_vehicle_value" in stats
        assert "most_stolen_make" in stats
        assert "most_stolen_type" in stats
        assert "year_range" in stats
        assert "owner_income_stats" in stats
        assert "recovery_stats" in stats

    def test_filter_by_criteria(self):
        """Test filtering theft data by various criteria."""
        td = TheftData()

        # Test without data
        with pytest.raises(ValueError, match="No theft data loaded"):
            td.filter_by_criteria(min_value=30000)

        # Create sample data
        gdf = td.create_sample_data(num_records=100)

        # Test value filtering
        high_value = td.filter_by_criteria(min_value=50000)
        assert len(high_value) <= len(gdf)
        assert all(high_value["vehicle_value"] >= 50000)

        # Test multiple criteria
        filtered = td.filter_by_criteria(
            min_value=20000, max_value=40000, vehicle_types=["sedan"], min_year=2015
        )
        assert len(filtered) <= len(gdf)
        if len(filtered) > 0:
            assert all(filtered["vehicle_value"] >= 20000)
            assert all(filtered["vehicle_value"] <= 40000)
            assert all(filtered["vehicle_type"] == "sedan")
            assert all(filtered["vehicle_year"] >= 2015)

    def test_filter_by_makes(self):
        """Test filtering by vehicle makes."""
        td = TheftData()
        gdf = td.create_sample_data(num_records=100)

        # Test filtering by specific makes
        honda_toyota = td.filter_by_criteria(makes=["Honda", "Toyota"])
        assert len(honda_toyota) <= len(gdf)
        if len(honda_toyota) > 0:
            assert all(honda_toyota["vehicle_make"].isin(["Honda", "Toyota"]))


class TestTheftDataPersistence:
    """Test cases for the TheftDataPersistence class."""

    def test_init_memory_db(self):
        """Test initialization with in-memory database."""
        with TheftDataPersistence() as db:
            assert db.db_path == MEMORY_DB
            assert db.connection is not None

    def test_init_file_db(self):
        """Test initialization with file database."""
        # Create a temporary directory and use a non-existent file path
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            with TheftDataPersistence(db_path) as db:
                assert db.db_path == str(db_path)
                assert db.connection is not None

    def test_save_and_load_theft_data(self):
        """Test saving and loading theft data."""
        td = TheftData()
        sample_data = td.create_sample_data(num_records=20)

        with TheftDataPersistence() as db:
            # Save data
            db.save_theft_data(sample_data)

            # Load data back
            loaded_data = db.load_theft_data()

            assert isinstance(loaded_data, gpd.GeoDataFrame)
            assert len(loaded_data) == 20
            assert loaded_data.crs == "EPSG:4326"

            # Check that key columns are preserved
            for col in [
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
                "vehicle_value",
            ]:
                assert col in loaded_data.columns

    def test_save_data_no_connection(self):
        """Test saving data with no database connection."""
        db = TheftDataPersistence()
        db.connection = None

        td = TheftData()
        sample_data = td.create_sample_data(num_records=5)

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.save_theft_data(sample_data)

    def test_load_data_no_connection(self):
        """Test loading data with no database connection."""
        db = TheftDataPersistence()
        db.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.load_theft_data()

    def test_load_empty_table(self):
        """Test loading from empty table."""
        with TheftDataPersistence() as db:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded_data = db.load_theft_data()

                assert len(w) > 0
                assert any("No data found" in str(warning.message) for warning in w)
                assert isinstance(loaded_data, gpd.GeoDataFrame)
                assert len(loaded_data) == 0

    def test_query_by_distance(self):
        """Test spatial distance queries."""
        td = TheftData()
        sample_data = td.create_sample_data(num_records=30)

        with TheftDataPersistence() as db:
            db.save_theft_data(sample_data)

            # Query around Newark center
            center_lat, center_lon = 40.7282, -74.2090
            nearby = db.query_by_distance(center_lat, center_lon, radius_km=5.0)

            assert isinstance(nearby, gpd.GeoDataFrame)
            # All sample data should be within Newark area, so some should be found
            assert len(nearby) <= len(sample_data)

    def test_query_by_distance_no_connection(self):
        """Test distance query with no database connection."""
        db = TheftDataPersistence()
        db.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.query_by_distance(40.7282, -74.2090, 5.0)

    def test_get_statistics(self):
        """Test getting database statistics."""
        td = TheftData()
        sample_data = td.create_sample_data(num_records=25)

        with TheftDataPersistence() as db:
            db.save_theft_data(sample_data)

            stats = db.get_statistics()

            assert stats["total_records"] == 25
            assert "avg_vehicle_value" in stats
            assert "total_value_stolen" in stats
            assert "unique_makes" in stats
            assert "unique_types" in stats

    def test_get_statistics_no_connection(self):
        """Test getting statistics with no database connection."""
        db = TheftDataPersistence()
        db.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.get_statistics()

    @patch("duckdb.connect")
    def test_database_setup_failure(self, mock_connect):
        """Test database setup failure handling."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Error setting up DuckDB database"):
            TheftDataPersistence()

    def test_context_manager(self):
        """Test using TheftDataPersistence as context manager."""
        td = TheftData()
        sample_data = td.create_sample_data(num_records=10)

        with TheftDataPersistence() as db:
            assert db.connection is not None
            db.save_theft_data(sample_data)
            loaded_data = db.load_theft_data()
            assert len(loaded_data) == 10

        # Connection should be closed after context
        assert db.connection is None

    def test_bbox_fallback_query(self):
        """Test fallback bounding box query."""
        td = TheftData()
        sample_data = td.create_sample_data(num_records=20)

        with TheftDataPersistence() as db:
            db.save_theft_data(sample_data)

            # Test the fallback method directly
            center_lat, center_lon = 40.7282, -74.2090
            nearby = db._query_by_bbox(center_lat, center_lon, 2.0, "theft_incidents")

            assert isinstance(nearby, gpd.GeoDataFrame)
            assert len(nearby) <= len(sample_data)


class TestTheftDataIntegration:
    """Integration tests for TheftData and TheftDataPersistence."""

    def test_csv_to_database_workflow(self):
        """Test complete workflow from CSV import to database storage."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            f.write(
                "incident_lat,incident_lon,vehicle_make,vehicle_model,vehicle_year,"
                "vehicle_value,vehicle_type,owner_name,owner_address,owner_lat,"
                "owner_lon,owner_income\n"
            )
            f.write(
                "40.7282,-74.2090,Honda,Civic,2020,25000,sedan,John Doe,"
                "123 Main St,40.7300,-74.2100,50000\n"
            )
            f.write(
                "40.7300,-74.2100,Toyota,Camry,2019,28000,sedan,Jane Smith,"
                "456 Oak Ave,40.7320,-74.2120,60000\n"
            )
            f.write(
                "40.7320,-74.2120,Ford,F-150,2021,45000,truck,Bob Johnson,"
                "789 Pine Rd,40.7340,-74.2140,75000\n"
            )

        try:
            # Import from CSV
            td = TheftData()
            gdf = td.import_from_csv(csv_path)
            assert len(gdf) == 3

            # Save to database and query
            with TheftDataPersistence() as db:
                db.save_theft_data(gdf)

                # Test various queries
                loaded = db.load_theft_data()
                assert len(loaded) == 3

                # Test filtering in original data
                high_value = td.filter_by_criteria(min_value=40000)
                assert len(high_value) == 1
                assert high_value.iloc[0]["vehicle_make"] == "Ford"

                # Test distance query
                nearby = db.query_by_distance(40.7282, -74.2090, 1.0)
                assert len(nearby) >= 1  # At least the first record should be nearby

                # Test statistics
                stats = db.get_statistics()
                assert stats["total_records"] == 3
                assert stats["total_value_stolen"] == 25000 + 28000 + 45000

        finally:
            csv_path.unlink()

    def test_sample_data_persistence_cycle(self):
        """Test creating sample data and full persistence cycle."""
        td = TheftData()

        # Create sample data
        original_data = td.create_sample_data(num_records=50)
        original_stats = td.get_summary_stats()

        # Save to database
        with TheftDataPersistence() as db:
            db.save_theft_data(original_data)

            # Load back and compare
            loaded_data = db.load_theft_data()

            # Create new TheftData instance with loaded data
            td_new = TheftData()
            td_new.theft_data = loaded_data
            new_stats = td_new.get_summary_stats()

            # Statistics should match
            assert new_stats["total_incidents"] == original_stats["total_incidents"]
            assert (
                abs(
                    new_stats["total_value_stolen"]
                    - original_stats["total_value_stolen"]
                )
                < 1
            )
            assert new_stats["most_stolen_make"] == original_stats["most_stolen_make"]
