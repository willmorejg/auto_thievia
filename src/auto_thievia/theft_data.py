"""
Theft Data module for auto_thievia package.

This module provides functionality for importing auto theft data from CSV files,
managing theft records with vehicle and owner information, and persisting data
to DuckDB databases using geopandas dataframes.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .gis_mapper import WGS84_CRS

# Constants
MEMORY_DB = ":memory:"
DB_CONNECTION_ERROR = "Database connection not established"


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
    A class for persisting theft data to DuckDB databases.

    This class handles creating DuckDB databases, storing geopandas dataframes,
    and querying theft data with spatial capabilities.
    """

    def __init__(self, db_path: Union[str, Path] = MEMORY_DB):
        """
        Initialize the persistence manager.

        Args:
            db_path (Union[str, Path]): Path to DuckDB database file.
                                       Use MEMORY_DB for in-memory database.
        """
        self.db_path = str(db_path) if db_path != MEMORY_DB else MEMORY_DB
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._setup_database()

    def _setup_database(self):
        """Set up the DuckDB database with spatial extensions."""
        try:
            self.connection = duckdb.connect(self.db_path)

            # Install and load spatial extension
            self.connection.execute("INSTALL spatial;")
            self.connection.execute("LOAD spatial;")

            # Create tables if they don't exist
            self._create_tables()

        except Exception as e:
            raise ValueError(f"Error setting up DuckDB database: {str(e)}")

    def _create_tables(self):
        """Create the theft data tables."""
        # Main theft incidents table
        theft_table_sql = """
        CREATE TABLE IF NOT EXISTS theft_incidents (
            incident_id VARCHAR PRIMARY KEY,
            incident_date TIMESTAMP,
            incident_lat DOUBLE,
            incident_lon DOUBLE,
            incident_geometry TEXT,
            vehicle_make VARCHAR,
            vehicle_model VARCHAR,
            vehicle_year INTEGER,
            vehicle_type VARCHAR,
            vehicle_value DOUBLE,
            vehicle_color VARCHAR,
            vehicle_vin VARCHAR,
            owner_name VARCHAR,
            owner_address VARCHAR,
            owner_lat DOUBLE,
            owner_lon DOUBLE,
            owner_geometry TEXT,
            owner_income DOUBLE,
            owner_phone VARCHAR,
            owner_email VARCHAR,
            recovery_status VARCHAR,
            insurance_claim DOUBLE,
            police_report_id VARCHAR,
            theft_method VARCHAR,
            location_type VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        self.connection.execute(theft_table_sql)

        # Note: Spatial indices not supported with TEXT geometry storage

    def save_theft_data(
        self, gdf: gpd.GeoDataFrame, table_name: str = "theft_incidents"
    ):
        """
        Save theft data to the database.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing theft data.
            table_name (str): Name of the table to save to.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            # Prepare data for insertion
            df = gdf.copy()

            # Convert geometry to WKT for storage
            df["incident_geometry"] = df.geometry.to_wkt()

            # Create owner geometry points if coordinates exist
            if "owner_lat" in df.columns and "owner_lon" in df.columns:
                owner_geometry = [
                    Point(lon, lat).wkt
                    for lon, lat in zip(df["owner_lon"], df["owner_lat"])
                ]
                df["owner_geometry"] = owner_geometry
            else:
                df["owner_geometry"] = None

            # Ensure all required columns exist with default values
            required_columns = [
                "incident_id",
                "incident_date",
                "incident_lat",
                "incident_lon",
                "incident_geometry",
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
                "owner_geometry",
                "owner_income",
                "owner_phone",
                "owner_email",
                "recovery_status",
                "insurance_claim",
                "police_report_id",
                "theft_method",
                "location_type",
            ]

            # Add missing columns with default values
            for col in required_columns:
                if col not in df.columns:
                    if col in ["incident_date"]:
                        df[col] = pd.Timestamp.now()
                    elif col in ["incident_id"]:
                        # Generate unique incident IDs
                        df[col] = [f"AUTO_{i+1:06d}" for i in range(len(df))]
                    elif col in [
                        "vehicle_year",
                        "vehicle_value",
                        "owner_income",
                        "insurance_claim",
                    ]:
                        df[col] = 0
                    elif col in [
                        "incident_lat",
                        "incident_lon",
                        "owner_lat",
                        "owner_lon",
                    ]:
                        df[col] = 0.0
                    else:
                        df[col] = None

            # Reorder columns to match table schema (excluding created_at which has DEFAULT)
            df = df[required_columns]

            # Drop the original geometry column if it still exists
            if "geometry" in df.columns:
                df = df.drop("geometry", axis=1)

            # Clear existing data and insert new data
            self.connection.execute(f"DELETE FROM {table_name}")
            self.connection.register("temp_df", df)

            # Insert all records with explicit column specification
            insert_sql = f"""
            INSERT INTO {table_name} ({', '.join(required_columns)})
            SELECT {', '.join(required_columns)} FROM temp_df
            """
            self.connection.execute(insert_sql)

            print(f"Successfully saved {len(df)} theft records to {table_name}")

        except Exception as e:
            raise ValueError(f"Error saving data to database: {str(e)}") from e

    def load_theft_data(self, table_name: str = "theft_incidents") -> gpd.GeoDataFrame:
        """
        Load theft data from the database.

        Args:
            table_name (str): Name of the table to load from.

        Returns:
            gpd.GeoDataFrame: Loaded theft data.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            # Query all data
            df = self.connection.execute(f"SELECT * FROM {table_name}").df()

            if len(df) == 0:
                warnings.warn(f"No data found in table {table_name}")
                return gpd.GeoDataFrame()

            # Convert WKT back to geometry
            from shapely import wkt

            # Create geometry from WKT strings
            geometries = []
            for wkt_str in df["incident_geometry"]:
                try:
                    if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                        geometries.append(wkt.loads(str(wkt_str)))
                    else:
                        geometries.append(None)
                except Exception:
                    geometries.append(None)

            df["geometry"] = geometries

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)

            # Drop the WKT columns
            columns_to_drop = ["incident_geometry", "owner_geometry"]
            gdf = gdf.drop(
                columns=[col for col in columns_to_drop if col in gdf.columns]
            )

            return gdf

        except Exception as e:
            raise ValueError(f"Error loading data from database: {str(e)}")

    def query_by_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        table_name: str = "theft_incidents",
    ) -> gpd.GeoDataFrame:
        """
        Query theft incidents within a specified distance from a point.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (str): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Theft incidents within the specified distance.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        # Use bounding box approach since we're storing WKT as TEXT
        return self._query_by_bbox(center_lat, center_lon, radius_km, table_name)

    def _query_by_bbox(
        self, center_lat: float, center_lon: float, radius_km: float, table_name: str
    ) -> gpd.GeoDataFrame:
        """
        Fallback method to query by bounding box.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (str): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Theft incidents within the bounding box.
        """
        # Rough conversion: 1 degree ≈ 111 km
        degree_offset = radius_km / 111.0

        min_lat = center_lat - degree_offset
        max_lat = center_lat + degree_offset
        min_lon = center_lon - degree_offset
        max_lon = center_lon + degree_offset

        query_sql = f"""
        SELECT * FROM {table_name}
        WHERE incident_lat BETWEEN {min_lat} AND {max_lat}
        AND incident_lon BETWEEN {min_lon} AND {max_lon}
        """

        df = self.connection.execute(query_sql).df()

        if len(df) == 0:
            return gpd.GeoDataFrame()

        from shapely import wkt

        # Create geometry from WKT strings
        geometries = []
        for wkt_str in df["incident_geometry"]:
            try:
                if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                    geometries.append(wkt.loads(str(wkt_str)))
                else:
                    geometries.append(None)
            except Exception:
                geometries.append(None)

        df["geometry"] = geometries
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)

        return gdf

    def get_statistics(self, table_name: str = "theft_incidents") -> Dict[str, Any]:
        """
        Get database statistics.

        Args:
            table_name (str): Name of the table to analyze.

        Returns:
            Dict[str, Any]: Database statistics.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(incident_date) as earliest_incident,
                MAX(incident_date) as latest_incident,
                AVG(vehicle_value) as avg_vehicle_value,
                SUM(vehicle_value) as total_value_stolen,
                COUNT(DISTINCT vehicle_make) as unique_makes,
                COUNT(DISTINCT vehicle_type) as unique_types
            FROM {table_name}
            """

            result = self.connection.execute(stats_sql).fetchone()

            stats = {
                "total_records": result[0],
                "earliest_incident": result[1],
                "latest_incident": result[2],
                "avg_vehicle_value": result[3],
                "total_value_stolen": result[4],
                "unique_makes": result[5],
                "unique_types": result[6],
            }

            return stats

        except Exception as e:
            raise ValueError(f"Error getting statistics: {str(e)}")

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
