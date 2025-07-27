"""
Tests for the RecoveryData and RecoveryDataPersistence classes.

This module contains comprehensive tests for recovery data management functionality
including CSV import, data validation, filtering, and database persistence.
"""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest

from auto_thievia.recovery_data import (
    RecoveryData,
    RecoveryDataPersistence,
)
from auto_thievia.duckdb_persistence import MEMORY_DB, DB_CONNECTION_ERROR


class TestRecoveryData:
    """Test cases for the RecoveryData class."""

    def test_init(self):
        """Test RecoveryData initialization."""
        rd = RecoveryData()
        assert rd.crs == "EPSG:4326"
        assert rd.recovery_data is None
        assert len(rd._required_columns) == 5
        assert len(rd._optional_columns) == 17

    def test_init_custom_crs(self):
        """Test RecoveryData initialization with custom CRS."""
        custom_crs = "EPSG:3857"
        rd = RecoveryData(crs=custom_crs)
        assert rd.crs == custom_crs

    def test_import_from_csv_success(self):
        """Test successful CSV import."""
        rd = RecoveryData()

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write sample CSV data
            f.write(
                "recovery_lat,recovery_lon,recovery_date,recovery_location_name,vehicle_condition\n"
            )
            f.write("40.7282,-74.2090,2023-01-15,Newark Recovery Site,Good\n")
            f.write("40.7300,-74.2100,2023-01-16,Essex Recovery Site,Fair\n")

        try:
            gdf = rd.import_from_csv(csv_path)

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 2
            assert gdf.crs == "EPSG:4326"
            assert "geometry" in gdf.columns

            # Check first record
            first_row = gdf.iloc[0]
            assert first_row["recovery_location_name"] == "Newark Recovery Site"
            assert first_row["vehicle_condition"] == "Good"

        finally:
            csv_path.unlink()  # Clean up

    def test_import_from_csv_file_not_found(self):
        """Test importing from non-existent CSV file."""
        rd = RecoveryData()

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            rd.import_from_csv("non_existent_file.csv")

    def test_import_from_csv_missing_columns(self):
        """Test importing CSV with missing required columns."""
        rd = RecoveryData()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write CSV with missing columns
            f.write("recovery_lat,recovery_lon,recovery_date\n")
            f.write("40.7282,-74.2090,2023-01-15\n")

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                rd.import_from_csv(csv_path)
        finally:
            csv_path.unlink()

    def test_import_from_csv_invalid_coordinates(self):
        """Test importing CSV with invalid coordinates."""
        rd = RecoveryData()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            # Write CSV with invalid coordinates
            f.write(
                "recovery_lat,recovery_lon,recovery_date,recovery_location_name,vehicle_condition\n"
            )
            f.write("200,-400,2023-01-15,Bad Location,Good\n")  # Invalid coords

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gdf = rd.import_from_csv(csv_path)

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
        rd = RecoveryData()

        # Create DataFrame with missing values
        df = pd.DataFrame(
            {
                "recovery_lat": [40.7282, None, 40.7300],
                "recovery_lon": [-74.2090, -74.2100, -74.2120],
                "recovery_date": ["2023-01-15", "2023-01-16", "2023-01-17"],
                "recovery_location_name": ["Site A", "Site B", "Site C"],
                "vehicle_condition": ["Good", "Fair", None],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cleaned_df = rd._clean_data(df, validate_coordinates=True)

            # Should have warnings about missing data
            assert len(w) > 0
            assert any("missing required data" in str(warning.message) for warning in w)
            assert len(cleaned_df) == 1  # Only one valid record

    def test_create_sample_data(self):
        """Test creating sample recovery data."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=30)

        assert isinstance(sample_data, gpd.GeoDataFrame)
        assert len(sample_data) == 30
        assert sample_data.crs == "EPSG:4326"
        assert "geometry" in sample_data.columns

        # Check required columns are present
        required_cols = [
            "recovery_id",
            "recovery_lat",
            "recovery_lon",
            "recovery_date",
            "recovery_location_name",
            "vehicle_condition",
        ]
        for col in required_cols:
            assert col in sample_data.columns

        # Check that some records have relationships
        assert "incident_id" in sample_data.columns
        assert "suspect_id" in sample_data.columns

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        rd = RecoveryData()

        # First test without data
        with pytest.raises(ValueError, match="No recovery data loaded"):
            rd.get_summary_stats()

        # Create sample data and test stats
        gdf = rd.create_sample_data(num_records=50)
        stats = rd.get_summary_stats()

        assert stats["total_recoveries"] == 50
        assert "recoveries_with_theft_link" in stats
        assert "recoveries_with_suspect_link" in stats
        assert "criminal_locations" in stats
        assert "recovery_agencies" in stats
        assert "condition_breakdown" in stats
        assert "recovery_methods" in stats
        assert "case_status_breakdown" in stats

    def test_filter_by_criteria(self):
        """Test filtering recovery data by various criteria."""
        rd = RecoveryData()

        # Test without data
        with pytest.raises(ValueError, match="No recovery data loaded"):
            rd.filter_by_criteria(vehicle_conditions=["Good"])

        # Create sample data
        gdf = rd.create_sample_data(num_records=50)

        # Test condition filtering
        good_condition = rd.filter_by_criteria(vehicle_conditions=["Good", "Excellent"])
        assert len(good_condition) <= len(gdf)
        if len(good_condition) > 0:
            assert all(good_condition["vehicle_condition"].isin(["Good", "Excellent"]))

        # Test criminal locations filter
        criminal_only = rd.filter_by_criteria(criminal_locations_only=True)
        assert len(criminal_only) <= len(gdf)
        if len(criminal_only) > 0:
            assert all(criminal_only["is_criminal_location"] == True)

        # Test relationship filters
        with_theft = rd.filter_by_criteria(with_theft_link=True)
        assert len(with_theft) <= len(gdf)
        if len(with_theft) > 0:
            assert all(with_theft["incident_id"].notna())

    def test_get_recoveries_by_location_type(self):
        """Test getting recoveries by criminal location type."""
        rd = RecoveryData()

        # Test without data
        with pytest.raises(ValueError, match="No recovery data loaded"):
            rd.get_recoveries_by_location_type()

        # Create sample data
        gdf = rd.create_sample_data(num_records=50)
        location_counts = rd.get_recoveries_by_location_type()

        assert isinstance(location_counts, dict)


class TestRecoveryDataPersistence:
    """Test cases for the RecoveryDataPersistence class."""

    def test_init_memory_db(self):
        """Test initialization with in-memory database."""
        with RecoveryDataPersistence() as db:
            assert db.db_persistence.db_path == MEMORY_DB
            assert db.db_persistence.connection is not None

    def test_init_file_db(self):
        """Test initialization with file database."""
        # Create a temporary directory and use a non-existent file path
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            with RecoveryDataPersistence(db_path) as db:
                assert db.db_persistence.db_path == str(db_path)
                assert db.db_persistence.connection is not None

    def test_save_and_load_recovery_data(self):
        """Test saving and loading recovery data."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=20)

        with RecoveryDataPersistence() as db:
            # Save data
            db.save_recovery_data(sample_data)

            # Load data back
            loaded_data = db.load_recovery_data()

            assert isinstance(loaded_data, gpd.GeoDataFrame)
            assert len(loaded_data) == 20
            assert loaded_data.crs == "EPSG:4326"

            # Check that key columns are preserved
            for col in [
                "recovery_id",
                "recovery_date",
                "recovery_location_name",
                "vehicle_condition",
            ]:
                assert col in loaded_data.columns

    def test_save_data_no_connection(self):
        """Test saving data with no database connection."""
        db = RecoveryDataPersistence()
        db.db_persistence.connection = None

        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=5)

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.save_recovery_data(sample_data)

    def test_load_data_no_connection(self):
        """Test loading data with no database connection."""
        db = RecoveryDataPersistence()
        db.db_persistence.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.load_recovery_data()

    def test_load_empty_table(self):
        """Test loading from empty table."""
        with RecoveryDataPersistence() as db:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded_data = db.load_recovery_data()

                assert len(w) > 0
                assert any("No data found" in str(warning.message) for warning in w)
                assert isinstance(loaded_data, gpd.GeoDataFrame)
                assert len(loaded_data) == 0

    def test_query_by_distance(self):
        """Test spatial distance queries."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=30)

        with RecoveryDataPersistence() as db:
            db.save_recovery_data(sample_data)

            # Query around Newark center
            center_lat, center_lon = 40.7282, -74.2090
            nearby = db.query_by_distance(center_lat, center_lon, radius_km=10.0)

            assert isinstance(nearby, gpd.GeoDataFrame)
            # All sample data should be within Newark area, so some should be found
            assert len(nearby) <= len(sample_data)

    def test_query_by_distance_no_connection(self):
        """Test distance query with no database connection."""
        db = RecoveryDataPersistence()
        db.db_persistence.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.query_by_distance(40.7282, -74.2090, 5.0)

    def test_get_recoveries_with_relationships(self):
        """Test getting recoveries with relationships to theft/suspect data."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=25)

        with RecoveryDataPersistence() as db:
            db.save_recovery_data(sample_data)

            # Get recoveries with relationships
            with_relationships = db.get_recoveries_with_relationships()

            assert isinstance(with_relationships, gpd.GeoDataFrame)
            # Should be fewer than total since not all have relationships
            assert len(with_relationships) <= len(sample_data)

    def test_get_criminal_location_recoveries(self):
        """Test getting recoveries from criminal locations."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=25)

        with RecoveryDataPersistence() as db:
            db.save_recovery_data(sample_data)

            # Get criminal location recoveries
            criminal_recoveries = db.get_criminal_location_recoveries()

            assert isinstance(criminal_recoveries, gpd.GeoDataFrame)
            # Should be fewer than total since not all are from criminal locations
            assert len(criminal_recoveries) <= len(sample_data)

    def test_get_statistics(self):
        """Test getting database statistics."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=25)

        with RecoveryDataPersistence() as db:
            db.save_recovery_data(sample_data)

            stats = db.get_statistics()

            assert stats["total_records"] == 25
            assert "recoveries_with_theft_link" in stats
            assert "recoveries_with_suspect_link" in stats
            assert "criminal_location_recoveries" in stats
            assert "unique_agencies" in stats
            assert "unique_conditions" in stats

    def test_get_statistics_no_connection(self):
        """Test getting statistics with no database connection."""
        db = RecoveryDataPersistence()
        db.db_persistence.connection = None

        with pytest.raises(ValueError, match=DB_CONNECTION_ERROR):
            db.get_statistics()

    @patch("auto_thievia.duckdb_persistence.duckdb.connect")
    def test_database_setup_failure(self, mock_connect):
        """Test database setup failure handling."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Error setting up DuckDB database"):
            RecoveryDataPersistence()

    def test_context_manager(self):
        """Test using RecoveryDataPersistence as context manager."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=10)

        with RecoveryDataPersistence() as db:
            assert db.db_persistence.connection is not None
            db.save_recovery_data(sample_data)
            loaded_data = db.load_recovery_data()
            assert len(loaded_data) == 10

        # Connection should be closed after context
        assert db.db_persistence.connection is None


class TestRecoveryDataIntegration:
    """Integration tests for RecoveryData and RecoveryDataPersistence."""

    def test_csv_to_database_workflow(self):
        """Test complete workflow from CSV import to database storage."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

            f.write(
                "recovery_lat,recovery_lon,recovery_date,recovery_location_name,"
                "vehicle_condition,incident_id,suspect_id,is_criminal_location,"
                "recovery_agency,case_status\n"
            )
            f.write(
                "40.7282,-74.2090,2023-01-15,Site A,Good,INC001,SUS001,true,"
                "Newark PD,Closed\n"
            )
            f.write(
                "40.7300,-74.2100,2023-01-16,Site B,Fair,INC002,,false,"
                "State Police,Open\n"
            )
            f.write(
                "40.7320,-74.2120,2023-01-17,Site C,Poor,,SUS003,true,"
                "Essex Sheriff,Closed\n"
            )

        try:
            # Import from CSV
            rd = RecoveryData()
            gdf = rd.import_from_csv(csv_path)
            assert len(gdf) == 3

            # Save to database and query
            with RecoveryDataPersistence() as db:
                db.save_recovery_data(gdf)

                # Test various queries
                loaded = db.load_recovery_data()
                assert len(loaded) == 3

                # Test filtering in original data
                criminal_locations = rd.filter_by_criteria(criminal_locations_only=True)
                assert len(criminal_locations) == 2

                # Test distance query
                nearby = db.query_by_distance(40.7282, -74.2090, 1.0)
                assert len(nearby) >= 1  # At least the first record should be nearby

                # Test relationship queries
                with_relationships = db.get_recoveries_with_relationships()
                assert (
                    len(with_relationships) == 3
                )  # All have at least one relationship

                # Test statistics
                stats = db.get_statistics()
                assert stats["total_records"] == 3
                assert stats["criminal_location_recoveries"] >= 2

        finally:
            csv_path.unlink()

    def test_sample_data_persistence_cycle(self):
        """Test creating sample data and full persistence cycle."""
        rd = RecoveryData()

        # Create sample data
        original_data = rd.create_sample_data(num_records=40)
        original_stats = rd.get_summary_stats()

        # Save to database
        with RecoveryDataPersistence() as db:
            db.save_recovery_data(original_data)

            # Load back and compare
            loaded_data = db.load_recovery_data()

            # Create new RecoveryData instance with loaded data
            rd_new = RecoveryData()
            rd_new.recovery_data = loaded_data
            new_stats = rd_new.get_summary_stats()

            # Statistics should match
            assert new_stats["total_recoveries"] == original_stats["total_recoveries"]
            assert new_stats["recovery_agencies"] == original_stats["recovery_agencies"]

    def test_filtering_and_queries_integration(self):
        """Test integration of filtering and database queries."""
        rd = RecoveryData()
        sample_data = rd.create_sample_data(num_records=30)

        with RecoveryDataPersistence() as db:
            db.save_recovery_data(sample_data)

            # Test various filters
            good_condition = rd.filter_by_criteria(
                vehicle_conditions=["Good", "Excellent"]
            )
            criminal_locations = rd.filter_by_criteria(criminal_locations_only=True)
            with_relationships = rd.filter_by_criteria(with_theft_link=True)

            # Test database queries
            stats = db.get_statistics()
            criminal_db = db.get_criminal_location_recoveries()
            relationship_db = db.get_recoveries_with_relationships()

            # Verify consistency
            assert len(criminal_db) <= stats["total_records"]
            assert len(relationship_db) <= stats["total_records"]
            assert stats["criminal_location_recoveries"] == len(criminal_db)
