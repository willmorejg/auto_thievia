"""
Test cases for SuspectData class.

This module contains comprehensive tests for the SuspectData class functionality
including CSV import/export, data validation, filtering, and analysis features.
"""

import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from auto_thievia.suspect_data import SuspectData, SuspectDataPersistence


class TestSuspectData:
    """Test cases for the SuspectData class."""

    def test_initialization(self):
        """Test SuspectData initialization."""
        suspect_data = SuspectData()
        assert suspect_data.crs == "EPSG:4326"
        assert suspect_data.suspect_data is None
        assert len(suspect_data._required_columns) == 8
        assert len(suspect_data._optional_columns) >= 10

    def test_initialization_custom_crs(self):
        """Test SuspectData initialization with custom CRS."""
        custom_crs = "EPSG:3857"
        suspect_data = SuspectData(crs=custom_crs)
        assert suspect_data.crs == custom_crs

    def test_create_sample_data_default(self):
        """Test creating sample data with default parameters."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data()

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 50  # default num_records
        assert gdf.crs == "EPSG:4326"
        assert "geometry" in gdf.columns
        assert all(col in gdf.columns for col in suspect_data._required_columns)

        # Check that all geometries are Points
        assert all(isinstance(geom, Point) for geom in gdf.geometry)

        # Check coordinate ranges (should be around Newark area)
        assert gdf["address_lat"].between(40.60, 40.80).all()
        assert gdf["address_lon"].between(-74.35, -74.10).all()

    def test_create_sample_data_custom_size(self):
        """Test creating sample data with custom size."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=25)

        assert len(gdf) == 25
        assert "suspect_id" in gdf.columns
        assert gdf["suspect_id"].nunique() == 25  # All unique IDs

    def test_import_from_csv_success(self):
        """Test successful CSV import."""
        suspect_data = SuspectData()

        # Create temporary CSV file with valid data
        data = {
            "suspect_name": ["John Doe", "Jane Smith"],
            "suspect_address": ["123 Main St", "456 Oak Ave"],
            "address_lat": [40.7357, 40.7580],
            "address_lon": [-74.1724, -74.1400],
            "last_arrest_date": ["2024-01-15", "2023-12-20"],
            "arrest_location": ["Newark PD", "Elizabeth PD"],
            "arrest_lat": [40.7357, 40.7580],
            "arrest_lon": [-74.1724, -74.1400],
        }

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            gdf = suspect_data.import_from_csv(csv_path)

            assert isinstance(gdf, gpd.GeoDataFrame)
            assert len(gdf) == 2
            assert gdf.crs == "EPSG:4326"
            assert all(isinstance(geom, Point) for geom in gdf.geometry)
            assert gdf["suspect_name"].tolist() == ["John Doe", "Jane Smith"]

            # Check that data is stored in the instance
            assert suspect_data.suspect_data is not None
            assert len(suspect_data.suspect_data) == 2

        finally:
            Path(csv_path).unlink()

    def test_import_from_csv_file_not_found(self):
        """Test CSV import with non-existent file."""
        suspect_data = SuspectData()

        with pytest.raises(FileNotFoundError):
            suspect_data.import_from_csv("non_existent_file.csv")

    def test_import_from_csv_missing_columns(self):
        """Test CSV import with missing required columns."""
        suspect_data = SuspectData()

        # Create CSV with missing required columns
        data = {
            "suspect_name": ["John Doe"],
            "address_lat": [40.7357],
            # Missing other required columns
        }

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            with pytest.raises(ValueError) as excinfo:
                suspect_data.import_from_csv(csv_path)
            assert "Missing required columns" in str(excinfo.value)
        finally:
            Path(csv_path).unlink()

    def test_import_from_csv_invalid_coordinates(self):
        """Test CSV import with invalid coordinates."""
        suspect_data = SuspectData()

        # Create CSV with invalid coordinates
        data = {
            "suspect_name": ["John Doe", "Jane Smith"],
            "suspect_address": ["123 Main St", "456 Oak Ave"],
            "address_lat": [91.0, 40.7580],  # Invalid latitude > 90
            "address_lon": [-74.1724, -200.0],  # Invalid longitude < -180
            "last_arrest_date": ["2024-01-15", "2023-12-20"],
            "arrest_location": ["Newark PD", "Elizabeth PD"],
            "arrest_lat": [40.7357, 40.7580],
            "arrest_lon": [-74.1724, -74.1400],
        }

        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gdf = suspect_data.import_from_csv(csv_path)
                assert len(w) > 0
                assert "invalid coordinates" in str(w[0].message).lower()
                assert len(gdf) == 0  # All rows should be filtered out
        finally:
            Path(csv_path).unlink()

    def test_get_summary_stats_no_data(self):
        """Test getting summary stats with no data loaded."""
        suspect_data = SuspectData()

        with pytest.raises(ValueError) as excinfo:
            suspect_data.get_summary_stats()
        assert "No suspect data loaded" in str(excinfo.value)

    def test_get_summary_stats_with_data(self):
        """Test getting summary stats with loaded data."""
        suspect_data = SuspectData()
        suspect_data.create_sample_data(num_records=20)

        stats = suspect_data.get_summary_stats()

        assert stats["total_suspects"] == 20
        assert "unique_suspects" in stats
        assert "average_age" in stats
        assert "age_range" in stats
        assert "gender_distribution" in stats
        assert "risk_level_distribution" in stats

        # Check arrest date statistics
        assert "arrest_date_range" in stats
        assert "recent_arrests_count" in stats

        # Check bail statistics
        assert "bail_stats" in stats
        assert "average" in stats["bail_stats"]
        assert "median" in stats["bail_stats"]

    def test_filter_by_criteria_no_data(self):
        """Test filtering with no data loaded."""
        suspect_data = SuspectData()

        with pytest.raises(ValueError) as excinfo:
            suspect_data.filter_by_criteria(risk_levels=["High"])
        assert "No suspect data loaded" in str(excinfo.value)

    def test_filter_by_criteria_risk_levels(self):
        """Test filtering by risk levels."""
        suspect_data = SuspectData()
        suspect_data.create_sample_data(num_records=30)

        # Filter for high and critical risk
        filtered = suspect_data.filter_by_criteria(risk_levels=["High", "Critical"])

        assert isinstance(filtered, gpd.GeoDataFrame)
        assert len(filtered) <= 30
        assert all(risk in ["High", "Critical"] for risk in filtered["risk_level"])

    def test_filter_by_criteria_age_range(self):
        """Test filtering by age range."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Filter for young adults (18-25)
        filtered = suspect_data.filter_by_criteria(min_age=18, max_age=25)

        assert isinstance(filtered, gpd.GeoDataFrame)
        assert all(age >= 18 and age <= 25 for age in filtered["suspect_age"])

    def test_filter_by_criteria_gender(self):
        """Test filtering by gender."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Filter for males only
        filtered = suspect_data.filter_by_criteria(gender="Male")

        assert isinstance(filtered, gpd.GeoDataFrame)
        assert all(gender == "Male" for gender in filtered["suspect_gender"])

    def test_filter_by_criteria_recent_arrests(self):
        """Test filtering by recent arrests."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Filter for recent arrests (last 30 days)
        filtered = suspect_data.filter_by_criteria(
            recent_arrests_only=True, days_threshold=30
        )

        assert isinstance(filtered, gpd.GeoDataFrame)
        # Should have fewer records than total
        assert len(filtered) <= 30

        # Check that all arrests are within the threshold
        threshold_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        assert all(date >= threshold_date for date in filtered["last_arrest_date"])

    def test_get_high_risk_suspects(self):
        """Test getting high risk suspects."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        high_risk = suspect_data.get_high_risk_suspects()

        assert isinstance(high_risk, gpd.GeoDataFrame)
        assert all(risk in ["High", "Critical"] for risk in high_risk["risk_level"])

    def test_get_repeat_offenders_no_data(self):
        """Test getting repeat offenders with no data."""
        suspect_data = SuspectData()

        with pytest.raises(ValueError) as excinfo:
            suspect_data.get_repeat_offenders()
        assert "No suspect data loaded" in str(excinfo.value)

    def test_get_repeat_offenders_with_data(self):
        """Test getting repeat offenders with data."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        repeat_offenders = suspect_data.get_repeat_offenders()

        assert isinstance(repeat_offenders, gpd.GeoDataFrame)
        # Should only contain repeat offenders or career criminals
        valid_history = ["Repeat Offender", "Career Criminal"]
        assert all(
            history in valid_history for history in repeat_offenders["criminal_history"]
        )

    def test_get_suspects_by_distance_no_data(self):
        """Test getting suspects by distance with no data."""
        suspect_data = SuspectData()

        with pytest.raises(ValueError) as excinfo:
            suspect_data.get_suspects_by_distance(40.7357, -74.1724, 1.0)
        assert "No suspect data loaded" in str(excinfo.value)

    def test_get_suspects_by_distance_address(self):
        """Test getting suspects by distance from their addresses."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Use Newark center coordinates
        center_lat, center_lon = 40.7357, -74.1724
        nearby = suspect_data.get_suspects_by_distance(
            center_lat, center_lon, radius_km=5.0, location_type="address"
        )

        assert isinstance(nearby, gpd.GeoDataFrame)
        assert len(nearby) <= 30

        # Check that returned suspects are within reasonable bounds
        degree_offset = 5.0 / 111.0  # Approximate conversion
        for _, suspect in nearby.iterrows():
            assert abs(suspect["address_lat"] - center_lat) <= degree_offset
            assert abs(suspect["address_lon"] - center_lon) <= degree_offset

    def test_get_suspects_by_distance_arrest(self):
        """Test getting suspects by distance from their arrest locations."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Use Newark center coordinates
        center_lat, center_lon = 40.7357, -74.1724
        nearby = suspect_data.get_suspects_by_distance(
            center_lat, center_lon, radius_km=5.0, location_type="arrest"
        )

        assert isinstance(nearby, gpd.GeoDataFrame)
        assert len(nearby) <= 30

    def test_get_suspects_by_distance_invalid_location_type(self):
        """Test getting suspects by distance with invalid location type."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=10)

        with pytest.raises(ValueError) as excinfo:
            suspect_data.get_suspects_by_distance(
                40.7357, -74.1724, 1.0, location_type="invalid"
            )
        assert "location_type must be 'address' or 'arrest'" in str(excinfo.value)

    def test_create_arrest_points_gdf_no_data(self):
        """Test creating arrest points GDF with no data."""
        suspect_data = SuspectData()

        with pytest.raises(ValueError) as excinfo:
            suspect_data.create_arrest_points_gdf()
        assert "No suspect data loaded" in str(excinfo.value)

    def test_create_arrest_points_gdf_with_data(self):
        """Test creating arrest points GDF with data."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=20)

        arrest_gdf = suspect_data.create_arrest_points_gdf()

        assert isinstance(arrest_gdf, gpd.GeoDataFrame)
        assert len(arrest_gdf) == 20
        assert arrest_gdf.crs == "EPSG:4326"

        # Check that geometry is based on arrest coordinates
        for i, row in arrest_gdf.iterrows():
            point = row.geometry
            assert isinstance(point, Point)
            assert point.x == row["arrest_lon"]
            assert point.y == row["arrest_lat"]

    def test_clean_data_missing_required(self):
        """Test data cleaning with missing required values."""
        suspect_data = SuspectData()

        # Create DataFrame with some missing required values
        data = {
            "suspect_name": ["John Doe", None, "Jane Smith"],
            "suspect_address": ["123 Main St", "456 Oak Ave", "789 Pine St"],
            "address_lat": [40.7357, 40.7580, None],
            "address_lon": [-74.1724, -74.1400, -74.1500],
            "last_arrest_date": ["2024-01-15", "2023-12-20", "2024-03-01"],
            "arrest_location": ["Newark PD", "Elizabeth PD", "Union PD"],
            "arrest_lat": [40.7357, 40.7580, 40.7600],
            "arrest_lon": [-74.1724, -74.1400, -74.1500],
        }

        df = pd.DataFrame(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cleaned_df = suspect_data._clean_data(df, validate_coordinates=True)

            # Should warn about missing data
            assert len(w) > 0
            assert "missing required data" in str(w[0].message).lower()

            # Should only have 1 valid row (first one)
            assert len(cleaned_df) == 1
            assert cleaned_df.iloc[0]["suspect_name"] == "John Doe"

    def test_clean_data_invalid_dates(self):
        """Test data cleaning with invalid dates."""
        suspect_data = SuspectData()

        # Create DataFrame with invalid dates
        data = {
            "suspect_name": ["John Doe", "Jane Smith"],
            "suspect_address": ["123 Main St", "456 Oak Ave"],
            "address_lat": [40.7357, 40.7580],
            "address_lon": [-74.1724, -74.1400],
            "last_arrest_date": ["invalid-date", "2023-12-20"],
            "arrest_location": ["Newark PD", "Elizabeth PD"],
            "arrest_lat": [40.7357, 40.7580],
            "arrest_lon": [-74.1724, -74.1400],
            "suspect_age": ["invalid", 25],
            "bail_amount": ["not-a-number", 5000],
        }

        df = pd.DataFrame(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cleaned_df = suspect_data._clean_data(df, validate_coordinates=True)

            # Should warn about invalid dates and ages
            warning_messages = [str(warning.message).lower() for warning in w]
            assert any("invalid arrest dates" in msg for msg in warning_messages)
            assert any("invalid ages" in msg for msg in warning_messages)

            # Both rows should remain (dates/ages are optional for validity)
            assert len(cleaned_df) == 2

    def test_validate_coordinates(self):
        """Test coordinate validation."""
        suspect_data = SuspectData()

        # Create DataFrame with invalid coordinates
        data = {
            "suspect_name": ["John Doe", "Jane Smith", "Bob Wilson"],
            "address_lat": [40.7357, 91.0, 40.7580],  # Second is invalid (>90)
            "address_lon": [-74.1724, -74.1400, -200.0],  # Third is invalid (<-180)
            "arrest_lat": [40.7357, 40.7580, 40.7600],
            "arrest_lon": [-74.1724, -74.1400, -74.1500],
        }

        df = pd.DataFrame(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validated_df = suspect_data._validate_coordinates(df)

            # Should warn about invalid coordinates
            assert len(w) > 0
            assert "invalid coordinates" in str(w[0].message).lower()

            # Should only have 1 valid row (first one)
            assert len(validated_df) == 1
            assert validated_df.iloc[0]["suspect_name"] == "John Doe"


class TestSuspectDataIntegration:
    """Integration tests for SuspectData class."""

    def test_full_workflow_with_csv(self):
        """Test complete workflow: create data, export to CSV, import from CSV."""
        suspect_data = SuspectData()

        # Create sample data
        original_gdf = suspect_data.create_sample_data(num_records=10)

        with TemporaryDirectory() as temp_dir:
            # Export to CSV
            csv_path = Path(temp_dir) / "test_suspects.csv"
            export_df = original_gdf.drop(columns="geometry")
            export_df.to_csv(csv_path, index=False)

            # Create new instance and import
            new_suspect_data = SuspectData()
            imported_gdf = new_suspect_data.import_from_csv(csv_path)

            # Verify data integrity
            assert len(imported_gdf) == len(original_gdf)
            assert list(imported_gdf.columns) == list(original_gdf.columns)

            # Check a few key fields
            assert (
                imported_gdf["suspect_name"].tolist()
                == original_gdf["suspect_name"].tolist()
            )
            assert (
                imported_gdf["risk_level"].tolist()
                == original_gdf["risk_level"].tolist()
            )

    def test_filtering_chain(self):
        """Test chaining multiple filters."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=50)

        # Chain multiple filters
        filtered = suspect_data.filter_by_criteria(
            risk_levels=["High", "Critical"], min_age=25, max_age=45, gender="Male"
        )

        assert isinstance(filtered, gpd.GeoDataFrame)
        # Verify all conditions are met
        assert all(risk in ["High", "Critical"] for risk in filtered["risk_level"])
        assert all(age >= 25 and age <= 45 for age in filtered["suspect_age"])
        assert all(gender == "Male" for gender in filtered["suspect_gender"])

    def test_analysis_workflow(self):
        """Test typical analysis workflow."""
        suspect_data = SuspectData()
        gdf = suspect_data.create_sample_data(num_records=30)

        # Get summary statistics
        stats = suspect_data.get_summary_stats()
        assert stats["total_suspects"] == 30

        # Get high-risk suspects
        high_risk = suspect_data.get_high_risk_suspects()
        assert len(high_risk) <= 30

        # Get repeat offenders
        repeat_offenders = suspect_data.get_repeat_offenders()
        assert len(repeat_offenders) <= 30

        # Get suspects near a location
        nearby = suspect_data.get_suspects_by_distance(40.7357, -74.1724, 2.0)
        assert len(nearby) <= 30

        # Create arrest points visualization data
        arrest_gdf = suspect_data.create_arrest_points_gdf()
        assert len(arrest_gdf) == 30
        assert arrest_gdf.crs == gdf.crs


class TestSuspectDataPersistence:
    """Test cases for the SuspectDataPersistence class."""

    def test_initialization(self):
        """Test SuspectDataPersistence initialization."""
        with SuspectDataPersistence() as db_manager:
            assert db_manager.db_persistence is not None
            assert db_manager.db_persistence.connection is not None

    def test_initialization_with_file_db(self):
        """Test SuspectDataPersistence initialization with file database."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_suspects.db"
            with SuspectDataPersistence(db_path) as db_manager:
                assert db_manager.db_persistence is not None
                assert str(db_path) in str(db_manager.db_persistence.db_path)

    def test_save_and_load_suspect_data(self):
        """Test saving and loading suspect data."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=10)

        with SuspectDataPersistence() as db_manager:
            # Save data
            db_manager.save_suspect_data(sample_data)

            # Load data
            loaded_data = db_manager.load_suspect_data()

            assert isinstance(loaded_data, gpd.GeoDataFrame)
            assert len(loaded_data) == 10
            assert loaded_data.crs == "EPSG:4326"
            assert all(isinstance(geom, Point) for geom in loaded_data.geometry if geom)

    def test_query_suspects_by_address_distance(self):
        """Test querying suspects by address distance."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=20)

        with SuspectDataPersistence() as db_manager:
            db_manager.save_suspect_data(sample_data)

            # Query by distance
            center_lat, center_lon = 40.7357, -74.1724
            nearby = db_manager.query_suspects_by_address_distance(
                center_lat, center_lon, radius_km=5.0
            )

            assert isinstance(nearby, gpd.GeoDataFrame)
            assert len(nearby) <= 20  # Should be subset

    def test_query_suspects_by_arrest_distance(self):
        """Test querying suspects by arrest distance."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=20)

        with SuspectDataPersistence() as db_manager:
            db_manager.save_suspect_data(sample_data)

            # Query by arrest distance
            center_lat, center_lon = 40.7357, -74.1724
            nearby = db_manager.query_suspects_by_arrest_distance(
                center_lat, center_lon, radius_km=5.0
            )

            assert isinstance(nearby, gpd.GeoDataFrame)
            assert len(nearby) <= 20  # Should be subset

    def test_query_suspects_by_criteria(self):
        """Test querying suspects by various criteria."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=30)

        with SuspectDataPersistence() as db_manager:
            db_manager.save_suspect_data(sample_data)

            # Query by risk level
            high_risk = db_manager.query_suspects_by_criteria(
                risk_levels=["High", "Critical"]
            )
            assert isinstance(high_risk, gpd.GeoDataFrame)
            if len(high_risk) > 0:
                assert all(
                    risk in ["High", "Critical"] for risk in high_risk["risk_level"]
                )

            # Query by age range
            young_adults = db_manager.query_suspects_by_criteria(min_age=18, max_age=25)
            assert isinstance(young_adults, gpd.GeoDataFrame)
            if len(young_adults) > 0:
                assert all(
                    age >= 18 and age <= 25 for age in young_adults["suspect_age"]
                )

            # Query by gender
            males = db_manager.query_suspects_by_criteria(gender="Male")
            assert isinstance(males, gpd.GeoDataFrame)
            if len(males) > 0:
                assert all(gender == "Male" for gender in males["suspect_gender"])

    def test_get_suspect_statistics(self):
        """Test getting suspect statistics."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=25)

        with SuspectDataPersistence() as db_manager:
            db_manager.save_suspect_data(sample_data)

            stats = db_manager.get_suspect_statistics()

            assert isinstance(stats, dict)
            assert stats["total_suspects"] == 25
            assert "unique_suspects" in stats
            assert "avg_age" in stats
            assert "min_age" in stats
            assert "max_age" in stats
            assert "risk_levels_count" in stats

    def test_context_manager(self):
        """Test context manager functionality."""
        suspect_data = SuspectData()
        sample_data = suspect_data.create_sample_data(num_records=5)

        # Test with context manager
        with SuspectDataPersistence() as db_manager:
            db_manager.save_suspect_data(sample_data)
            loaded_data = db_manager.load_suspect_data()
            assert len(loaded_data) == 5

        # Connection should be closed after context manager
        # This is implicitly tested by the fact that no exceptions are raised


if __name__ == "__main__":
    pytest.main([__file__])
