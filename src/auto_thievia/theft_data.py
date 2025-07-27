"""
Theft Data module for auto_thievia package.

This module provides functionality for importing auto theft data from CSV files,
managing theft records with vehicle and owner information, and persisting data
to DuckDB databases using geopandas dataframes.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .duckdb_persistence import DuckDbPersistence
from .gis_mapper import WGS84_CRS


class TheftData:
    """
    A class for managing auto theft data including vehicle and owner information.

    This class handles importing theft data from CSV files, validating coordinates,
    and creating geopandas dataframes with comprehensive theft incident information.
    """

    def __init__(self, crs: str = WGS84_CRS):
        """
        Initialize the TheftData manager.

        Args:
            crs (str): Coordinate Reference System to use. Defaults to WGS84.
        """
        self.crs = crs
        self.theft_data = None
        self._required_columns = {
            # Incident location
            "incident_lat",
            "incident_lon",
            # Vehicle information
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
            "vehicle_value",
            "vehicle_type",
            # Owner information
            "owner_name",
            "owner_address",
            "owner_lat",
            "owner_lon",
            "owner_income",
        }
        self._optional_columns = {
            "incident_id",
            "incident_date",
            "vehicle_color",
            "vehicle_vin",
            "owner_phone",
            "owner_email",
            "recovery_status",
            "insurance_claim",
            "police_report_id",
            "theft_method",
            "location_type",
        }

    def import_from_csv(
        self,
        csv_path: Union[str, Path],
        encoding: str = "utf-8",
        validate_coordinates: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Import auto theft data from a CSV file.

        Args:
            csv_path (Union[str, Path]): Path to the CSV file.
            encoding (str): File encoding. Defaults to 'utf-8'.
            validate_coordinates (bool): Whether to validate coordinate values.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing theft incident data.

        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
            ValueError: If required columns are missing or data is invalid.
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding=encoding)

            # Validate required columns
            missing_columns = self._required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Clean and validate data
            df = self._clean_data(df, validate_coordinates)

            # Create Point geometries for incident locations
            geometry = [
                Point(lon, lat)
                for lon, lat in zip(df["incident_lon"], df["incident_lat"])
            ]

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

            # Store the data
            self.theft_data = gdf

            return gdf

        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {str(e)}")

    def _clean_data(self, df: pd.DataFrame, validate_coordinates: bool) -> pd.DataFrame:
        """
        Clean and validate the theft data.

        Args:
            df (pd.DataFrame): Raw dataframe from CSV.
            validate_coordinates (bool): Whether to validate coordinates.

        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        # Make a copy to avoid modifying original
        df = df.copy()

        # Handle missing values in required columns
        required_cols = list(self._required_columns)
        missing_data = df[required_cols].isnull().any(axis=1)

        if missing_data.any():
            warnings.warn(
                f"Found {missing_data.sum()} rows with missing required data. These will be excluded."
            )
            df = df[~missing_data]

        if len(df) == 0:
            raise ValueError("No valid data rows after cleaning")

        # Validate and clean coordinate data
        if validate_coordinates:
            df = self._validate_coordinates(df)

        # Clean vehicle year (ensure it's integer)
        df["vehicle_year"] = pd.to_numeric(df["vehicle_year"], errors="coerce")
        invalid_years = df["vehicle_year"].isnull()
        if invalid_years.any():
            warnings.warn(
                f"Found {invalid_years.sum()} rows with invalid vehicle years"
            )
            df = df[~invalid_years]

        # Clean vehicle value (ensure it's numeric)
        df["vehicle_value"] = pd.to_numeric(df["vehicle_value"], errors="coerce")
        invalid_values = df["vehicle_value"].isnull()
        if invalid_values.any():
            warnings.warn(
                f"Found {invalid_values.sum()} rows with invalid vehicle values"
            )
            df = df[~invalid_values]

        # Clean owner income (ensure it's numeric)
        df["owner_income"] = pd.to_numeric(df["owner_income"], errors="coerce")
        invalid_income = df["owner_income"].isnull()
        if invalid_income.any():
            warnings.warn(
                f"Found {invalid_income.sum()} rows with invalid owner income"
            )
            df = df[~invalid_income]

        # Parse dates if incident_date column exists
        if "incident_date" in df.columns:
            df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate coordinate values for both incident and owner locations.

        Args:
            df (pd.DataFrame): Dataframe to validate.

        Returns:
            pd.DataFrame: Dataframe with valid coordinates only.
        """
        # Validate incident coordinates
        invalid_incident_coords = (
            (df["incident_lat"] < -90)
            | (df["incident_lat"] > 90)
            | (df["incident_lon"] < -180)
            | (df["incident_lon"] > 180)
        )

        # Validate owner coordinates
        invalid_owner_coords = (
            (df["owner_lat"] < -90)
            | (df["owner_lat"] > 90)
            | (df["owner_lon"] < -180)
            | (df["owner_lon"] > 180)
        )

        invalid_coords = invalid_incident_coords | invalid_owner_coords

        if invalid_coords.any():
            warnings.warn(
                f"Found {invalid_coords.sum()} rows with invalid coordinates. These will be excluded."
            )
            df = df[~invalid_coords]

        return df

    def create_sample_data(self, num_records: int = 100) -> gpd.GeoDataFrame:
        """
        Create sample theft data for testing and demonstration.

        Args:
            num_records (int): Number of sample records to create.

        Returns:
            gpd.GeoDataFrame: Sample theft data.
        """
        import numpy as np

        # Use reproducible random seed
        rng = np.random.default_rng(42)

        # Newark area bounding box
        lat_min, lat_max = 40.65, 40.75
        lon_min, lon_max = -74.28, -74.15

        # Sample vehicle makes and models
        vehicle_data = [
            ("Honda", "Civic", "sedan"),
            ("Toyota", "Camry", "sedan"),
            ("Ford", "F-150", "truck"),
            ("Chevrolet", "Silverado", "truck"),
            ("Nissan", "Altima", "sedan"),
            ("BMW", "X5", "suv"),
            ("Mercedes", "C-Class", "sedan"),
            ("Audi", "Q7", "suv"),
            ("Jeep", "Wrangler", "suv"),
            ("Hyundai", "Elantra", "sedan"),
        ]

        # Generate sample data
        data = []
        for i in range(num_records):
            # Random vehicle selection
            make, model, vtype = vehicle_data[rng.integers(0, len(vehicle_data))]

            # Generate coordinates within Newark area
            incident_lat = rng.uniform(lat_min, lat_max)
            incident_lon = rng.uniform(lon_min, lon_max)

            # Owner location (could be anywhere in broader area)
            owner_lat = rng.uniform(lat_min - 0.1, lat_max + 0.1)
            owner_lon = rng.uniform(lon_min - 0.1, lon_max + 0.1)

            # Generate other data
            year = rng.integers(2010, 2025)
            value = rng.integers(15000, 80000)
            income = rng.integers(30000, 150000)

            record = {
                "incident_id": f"NWK{i+1:04d}",
                "incident_lat": incident_lat,
                "incident_lon": incident_lon,
                "incident_date": pd.Timestamp.now()
                - pd.Timedelta(days=rng.integers(1, 365)),
                "vehicle_make": make,
                "vehicle_model": model,
                "vehicle_year": year,
                "vehicle_type": vtype,
                "vehicle_value": value,
                "vehicle_color": rng.choice(
                    ["Black", "White", "Silver", "Red", "Blue"]
                ),
                "owner_name": f"Owner_{i+1}",
                "owner_address": f"{rng.integers(100, 999)} Sample St, Newark, NJ",
                "owner_lat": owner_lat,
                "owner_lon": owner_lon,
                "owner_income": income,
                "recovery_status": rng.choice(
                    ["Recovered", "Not Recovered", "Partial"]
                ),
                "theft_method": rng.choice(
                    ["Carjacking", "Breaking", "Key Theft", "Unknown"]
                ),
            }
            data.append(record)

        # Create DataFrame and convert to GeoDataFrame
        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat) for lon, lat in zip(df["incident_lon"], df["incident_lat"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

        self.theft_data = gdf
        return gdf

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the theft data.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        if self.theft_data is None:
            raise ValueError("No theft data loaded. Import data first.")

        df = self.theft_data

        stats = {
            "total_incidents": len(df),
            "total_value_stolen": df["vehicle_value"].sum(),
            "average_vehicle_value": df["vehicle_value"].mean(),
            "most_stolen_make": (
                df["vehicle_make"].mode().iloc[0]
                if not df["vehicle_make"].empty
                else None
            ),
            "most_stolen_type": (
                df["vehicle_type"].mode().iloc[0]
                if not df["vehicle_type"].empty
                else None
            ),
            "year_range": {
                "oldest": df["vehicle_year"].min(),
                "newest": df["vehicle_year"].max(),
            },
            "owner_income_stats": {
                "mean": df["owner_income"].mean(),
                "median": df["owner_income"].median(),
                "min": df["owner_income"].min(),
                "max": df["owner_income"].max(),
            },
        }

        # Add recovery stats if available
        if "recovery_status" in df.columns:
            recovery_counts = df["recovery_status"].value_counts()
            stats["recovery_stats"] = recovery_counts.to_dict()

        return stats

    def filter_by_criteria(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        vehicle_types: Optional[List[str]] = None,
        makes: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter theft data by various criteria.

        Args:
            min_value (Optional[float]): Minimum vehicle value.
            max_value (Optional[float]): Maximum vehicle value.
            vehicle_types (Optional[List[str]]): Vehicle types to include.
            makes (Optional[List[str]]): Vehicle makes to include.
            min_year (Optional[int]): Minimum vehicle year.
            max_year (Optional[int]): Maximum vehicle year.

        Returns:
            gpd.GeoDataFrame: Filtered data.
        """
        if self.theft_data is None:
            raise ValueError("No theft data loaded. Import data first.")

        df = self.theft_data.copy()

        # Apply filters
        if min_value is not None:
            df = df[df["vehicle_value"] >= min_value]

        if max_value is not None:
            df = df[df["vehicle_value"] <= max_value]

        if vehicle_types is not None:
            df = df[df["vehicle_type"].isin(vehicle_types)]

        if makes is not None:
            df = df[df["vehicle_make"].isin(makes)]

        if min_year is not None:
            df = df[df["vehicle_year"] >= min_year]

        if max_year is not None:
            df = df[df["vehicle_year"] <= max_year]

        return df


class TheftDataPersistence:
    """
    A class for persisting theft data to DuckDB databases using the shared DuckDbPersistence layer.

    This class provides theft-specific database operations while leveraging
    the common database functionality from DuckDbPersistence.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """
        Initialize the persistence manager.

        Args:
            db_path (Union[str, Path]): Path to DuckDB database file.
                                       Use ":memory:" for in-memory database.
        """
        self.db_persistence = DuckDbPersistence(db_path)
        self.table_name = "theft_incidents"
        self._setup_table()

    def _setup_table(self):
        """Set up the theft incidents table."""
        # Define table schema for theft incidents
        schema = {
            "incident_id": "VARCHAR PRIMARY KEY",
            "incident_date": "TIMESTAMP",
            "incident_lat": "DOUBLE",
            "incident_lon": "DOUBLE",
            "vehicle_make": "VARCHAR",
            "vehicle_model": "VARCHAR",
            "vehicle_year": "INTEGER",
            "vehicle_type": "VARCHAR",
            "vehicle_value": "DOUBLE",
            "vehicle_color": "VARCHAR",
            "vehicle_vin": "VARCHAR",
            "owner_name": "VARCHAR",
            "owner_address": "VARCHAR",
            "owner_lat": "DOUBLE",
            "owner_lon": "DOUBLE",
            "owner_income": "DOUBLE",
            "owner_phone": "VARCHAR",
            "owner_email": "VARCHAR",
            "recovery_status": "VARCHAR",
            "insurance_claim": "DOUBLE",
            "police_report_id": "VARCHAR",
            "theft_method": "VARCHAR",
            "location_type": "VARCHAR",
            "geometry_wkt": "TEXT",
            "owner_lat_owner_lon_geometry": "TEXT",
        }

        self.db_persistence.create_table(self.table_name, schema)

    def save_theft_data(self, gdf: gpd.GeoDataFrame, table_name: Optional[str] = None):
        """
        Save theft data to the database.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing theft data.
            table_name (Optional[str]): Name of the table to save to.
        """
        table_name = table_name or self.table_name

        # Define required columns for theft data
        required_columns = [
            "incident_id",
            "incident_date",
            "incident_lat",
            "incident_lon",
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
            "vehicle_type",
            "vehicle_value",
            "vehicle_color",
            "vehicle_vin",
            "owner_name",
            "owner_address",
            "owner_lat",
            "owner_lon",
            "owner_income",
            "owner_phone",
            "owner_email",
            "recovery_status",
            "insurance_claim",
            "police_report_id",
            "theft_method",
            "location_type",
        ]

        # Use shared persistence layer
        self.db_persistence.save_data(
            gdf=gdf,
            table_name=table_name,
            geometry_column="geometry",
            primary_geometry_columns=["incident_lat", "incident_lon"],
            secondary_geometry_columns=["owner_lat", "owner_lon"],
            required_columns=required_columns,
            clear_existing=True,
        )

    def load_theft_data(self, table_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load theft data from the database.

        Args:
            table_name (Optional[str]): Name of the table to load from.

        Returns:
            gpd.GeoDataFrame: Loaded theft data.
        """
        table_name = table_name or self.table_name

        return self.db_persistence.load_data(
            table_name=table_name,
            geometry_column="geometry",
            geometry_wkt_column="geometry_wkt",
        )

    def query_by_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        table_name: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Query theft incidents within a specified distance from a point.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (Optional[str]): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Theft incidents within the specified distance.
        """
        table_name = table_name or self.table_name

        return self.db_persistence.query_by_distance(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            table_name=table_name,
            lat_column="incident_lat",
            lon_column="incident_lon",
            geometry_column="geometry",
            geometry_wkt_column="geometry_wkt",
        )

    def get_statistics(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database statistics for theft data.

        Args:
            table_name (Optional[str]): Name of the table to analyze.

        Returns:
            Dict[str, Any]: Database statistics.
        """
        table_name = table_name or self.table_name

        # Get basic statistics from shared layer
        basic_stats = self.db_persistence.get_statistics(
            table_name, columns=["vehicle_make", "vehicle_type", "recovery_status"]
        )

        # Add theft-specific statistics
        try:
            custom_stats_sql = f"""
            SELECT 
                AVG(vehicle_value) as avg_vehicle_value,
                SUM(vehicle_value) as total_value_stolen,
                MIN(incident_date) as earliest_incident,
                MAX(incident_date) as latest_incident,
                COUNT(DISTINCT vehicle_make) as unique_makes,
                COUNT(DISTINCT vehicle_type) as unique_types
            FROM {table_name}
            """

            result = self.db_persistence.execute_query(custom_stats_sql)
            if len(result) > 0:
                row = result.iloc[0]
                basic_stats.update(
                    {
                        "avg_vehicle_value": row["avg_vehicle_value"],
                        "total_value_stolen": row["total_value_stolen"],
                        "earliest_incident": row["earliest_incident"],
                        "latest_incident": row["latest_incident"],
                        "unique_makes": row["unique_makes"],
                        "unique_types": row["unique_types"],
                    }
                )

        except Exception as e:
            warnings.warn(f"Could not get extended statistics: {str(e)}")

        return basic_stats

    def close(self):
        """Close the database connection."""
        self.db_persistence.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
