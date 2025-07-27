"""
Recovery Data module for auto_thievia package.

This module provides functionality for importing auto recovery data from CSV files,
managing recovery records with relationships to theft and suspect data, and persisting
data to DuckDB databases using geopandas dataframes.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .duckdb_persistence import DuckDbPersistence
from .gis_mapper import WGS84_CRS


class RecoveryData:
    """
    A class for managing auto recovery data including relationships to theft and suspect data.

    This class handles importing recovery data from CSV files, validating coordinates,
    and creating geopandas dataframes with comprehensive recovery information.
    """

    def __init__(self, crs: str = WGS84_CRS):
        """
        Initialize the RecoveryData manager.

        Args:
            crs (str): Coordinate Reference System to use. Defaults to WGS84.
        """
        self.crs = crs
        self.recovery_data = None
        self._required_columns = {
            # Recovery location
            "recovery_lat",
            "recovery_lon",
            # Recovery information
            "recovery_date",
            "recovery_location_name",
            "vehicle_condition",
        }
        self._optional_columns = {
            "recovery_id",
            "incident_id",  # Link to theft data
            "suspect_id",  # Link to suspect data
            "recovery_agency",
            "recovery_officer",
            "is_criminal_location",  # Boolean: known criminal activity location
            "criminal_location_type",  # chop shop, abandoned lot, etc.
            "vehicle_damage_assessment",
            "parts_missing",
            "recovery_method",  # patrol, tip, investigation, etc.
            "recovery_time",
            "towed_to_location",
            "insurance_notified",
            "owner_notified",
            "case_status",
            "evidence_collected",
            "notes",
        }

    def import_from_csv(
        self,
        csv_path: Union[str, Path],
        encoding: str = "utf-8",
        validate_coordinates: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Import auto recovery data from a CSV file.

        Args:
            csv_path (Union[str, Path]): Path to the CSV file.
            encoding (str): File encoding. Defaults to 'utf-8'.
            validate_coordinates (bool): Whether to validate coordinate values.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing recovery data.

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

            # Create Point geometries for recovery locations
            geometry = [
                Point(lon, lat)
                for lon, lat in zip(df["recovery_lon"], df["recovery_lat"])
            ]

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

            # Store the data
            self.recovery_data = gdf

            return gdf

        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {str(e)}")

    def _clean_data(self, df: pd.DataFrame, validate_coordinates: bool) -> pd.DataFrame:
        """
        Clean and validate the recovery data.

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

        # Parse dates
        date_columns = ["recovery_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Parse recovery time if exists
        if "recovery_time" in df.columns:
            df["recovery_time"] = pd.to_datetime(df["recovery_time"], errors="coerce")

        # Clean boolean columns
        boolean_columns = [
            "is_criminal_location",
            "insurance_notified",
            "owner_notified",
        ]
        for col in boolean_columns:
            if col in df.columns:
                # Convert various representations to boolean
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .map(
                        {
                            "true": True,
                            "false": False,
                            "1": True,
                            "0": False,
                            "yes": True,
                            "no": False,
                            "y": True,
                            "n": False,
                        }
                    )
                )

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate coordinate values for recovery locations.

        Args:
            df (pd.DataFrame): Dataframe to validate.

        Returns:
            pd.DataFrame: Dataframe with valid coordinates only.
        """
        # Validate recovery coordinates
        invalid_coords = (
            (df["recovery_lat"] < -90)
            | (df["recovery_lat"] > 90)
            | (df["recovery_lon"] < -180)
            | (df["recovery_lon"] > 180)
        )

        if invalid_coords.any():
            warnings.warn(
                f"Found {invalid_coords.sum()} rows with invalid coordinates. These will be excluded."
            )
            df = df[~invalid_coords]

        return df

    def create_sample_data(self, num_records: int = 50) -> gpd.GeoDataFrame:
        """
        Create sample recovery data for testing and demonstration.

        Args:
            num_records (int): Number of sample records to create.

        Returns:
            gpd.GeoDataFrame: Sample recovery data.
        """
        import numpy as np

        # Use reproducible random seed
        rng = np.random.default_rng(42)

        # Newark area bounding box (slightly expanded for recovery locations)
        lat_min, lat_max = 40.60, 40.80
        lon_min, lon_max = -74.35, -74.10

        # Sample recovery locations and conditions
        recovery_agencies = [
            "Newark PD",
            "Essex County Sheriff",
            "State Police",
            "Port Authority",
        ]
        vehicle_conditions = ["Excellent", "Good", "Fair", "Poor", "Stripped", "Burned"]
        criminal_location_types = [
            "Chop Shop",
            "Abandoned Lot",
            "Warehouse",
            "Residential",
            "Street",
        ]
        recovery_methods = [
            "Patrol",
            "Tip",
            "Investigation",
            "Surveillance",
            "Traffic Stop",
        ]
        case_statuses = ["Open", "Closed", "Under Investigation", "Pending"]

        # Generate sample data
        data = []
        for i in range(num_records):
            # Generate coordinates within expanded Newark area
            recovery_lat = rng.uniform(lat_min, lat_max)
            recovery_lon = rng.uniform(lon_min, lon_max)

            # Random chance of being linked to theft/suspect data
            has_theft_link = rng.random() > 0.3  # 70% chance
            has_suspect_link = rng.random() > 0.6  # 40% chance

            # Random chance of being a criminal location
            is_criminal = rng.random() > 0.7  # 30% chance

            record = {
                "recovery_id": f"REC{i+1:04d}",
                "recovery_date": pd.Timestamp.now()
                - pd.Timedelta(days=rng.integers(1, 180)),
                "recovery_lat": recovery_lat,
                "recovery_lon": recovery_lon,
                "recovery_location_name": f"Recovery Site {i+1}",
                "vehicle_condition": rng.choice(vehicle_conditions),
                "recovery_agency": rng.choice(recovery_agencies),
                "recovery_officer": f"Officer_{rng.integers(1000, 9999)}",
                "is_criminal_location": is_criminal,
                "criminal_location_type": (
                    rng.choice(criminal_location_types) if is_criminal else None
                ),
                "vehicle_damage_assessment": rng.choice(
                    [
                        "Minor scratches",
                        "Broken windows",
                        "Engine damage",
                        "Body damage",
                        "Interior stripped",
                        "Total loss",
                    ]
                ),
                "parts_missing": rng.choice(
                    [
                        "None",
                        "Wheels",
                        "Battery",
                        "Catalytic converter",
                        "Engine parts",
                        "Interior components",
                        "Multiple parts",
                    ]
                ),
                "recovery_method": rng.choice(recovery_methods),
                "recovery_time": pd.Timestamp.now()
                - pd.Timedelta(
                    days=rng.integers(1, 180),
                    hours=rng.integers(0, 24),
                    minutes=rng.integers(0, 60),
                ),
                "insurance_notified": rng.choice([True, False]),
                "owner_notified": rng.choice([True, False]),
                "case_status": rng.choice(case_statuses),
                "evidence_collected": rng.choice([True, False]),
                "notes": f"Recovery notes for case {i+1}",
            }

            # Add optional relationship links
            if has_theft_link:
                record["incident_id"] = f"NWK{rng.integers(1, 1000):04d}"

            if has_suspect_link:
                record["suspect_id"] = f"SUS{rng.integers(1, 500):04d}"

            data.append(record)

        # Create DataFrame and convert to GeoDataFrame
        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat) for lon, lat in zip(df["recovery_lon"], df["recovery_lat"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

        self.recovery_data = gdf
        return gdf

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the recovery data.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        if self.recovery_data is None:
            raise ValueError("No recovery data loaded. Import data first.")

        df = self.recovery_data

        stats = {
            "total_recoveries": len(df),
            "recoveries_with_theft_link": (
                len(df[df["incident_id"].notna()]) if "incident_id" in df.columns else 0
            ),
            "recoveries_with_suspect_link": (
                len(df[df["suspect_id"].notna()]) if "suspect_id" in df.columns else 0
            ),
            "criminal_locations": (
                len(df[df["is_criminal_location"] == True])
                if "is_criminal_location" in df.columns
                else 0
            ),
            "recovery_agencies": (
                df["recovery_agency"].nunique()
                if "recovery_agency" in df.columns
                else 0
            ),
        }

        # Add condition statistics
        if "vehicle_condition" in df.columns:
            condition_counts = df["vehicle_condition"].value_counts()
            stats["condition_breakdown"] = condition_counts.to_dict()

        # Add recovery method statistics
        if "recovery_method" in df.columns:
            method_counts = df["recovery_method"].value_counts()
            stats["recovery_methods"] = method_counts.to_dict()

        # Add case status statistics
        if "case_status" in df.columns:
            status_counts = df["case_status"].value_counts()
            stats["case_status_breakdown"] = status_counts.to_dict()

        return stats

    def filter_by_criteria(
        self,
        vehicle_conditions: Optional[List[str]] = None,
        recovery_agencies: Optional[List[str]] = None,
        criminal_locations_only: bool = False,
        with_theft_link: Optional[bool] = None,
        with_suspect_link: Optional[bool] = None,
        case_status: Optional[List[str]] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter recovery data by various criteria.

        Args:
            vehicle_conditions (Optional[List[str]]): Vehicle conditions to include.
            recovery_agencies (Optional[List[str]]): Recovery agencies to include.
            criminal_locations_only (bool): Only include recoveries from criminal locations.
            with_theft_link (Optional[bool]): Filter by presence of theft data link.
            with_suspect_link (Optional[bool]): Filter by presence of suspect data link.
            case_status (Optional[List[str]]): Case statuses to include.
            min_date (Optional[str]): Minimum recovery date (YYYY-MM-DD format).
            max_date (Optional[str]): Maximum recovery date (YYYY-MM-DD format).

        Returns:
            gpd.GeoDataFrame: Filtered data.
        """
        if self.recovery_data is None:
            raise ValueError("No recovery data loaded. Import data first.")

        df = self.recovery_data.copy()

        # Apply filters
        if vehicle_conditions is not None and "vehicle_condition" in df.columns:
            df = df[df["vehicle_condition"].isin(vehicle_conditions)]

        if recovery_agencies is not None and "recovery_agency" in df.columns:
            df = df[df["recovery_agency"].isin(recovery_agencies)]

        if criminal_locations_only and "is_criminal_location" in df.columns:
            df = df[df["is_criminal_location"] == True]

        if with_theft_link is not None and "incident_id" in df.columns:
            if with_theft_link:
                df = df[df["incident_id"].notna()]
            else:
                df = df[df["incident_id"].isna()]

        if with_suspect_link is not None and "suspect_id" in df.columns:
            if with_suspect_link:
                df = df[df["suspect_id"].notna()]
            else:
                df = df[df["suspect_id"].isna()]

        if case_status is not None and "case_status" in df.columns:
            df = df[df["case_status"].isin(case_status)]

        # Date filtering
        if (
            min_date is not None or max_date is not None
        ) and "recovery_date" in df.columns:
            if min_date is not None:
                min_date_parsed = pd.to_datetime(min_date)
                df = df[df["recovery_date"] >= min_date_parsed]

            if max_date is not None:
                max_date_parsed = pd.to_datetime(max_date)
                df = df[df["recovery_date"] <= max_date_parsed]

        return df

    def get_recoveries_by_location_type(self) -> Dict[str, int]:
        """
        Get recovery counts by criminal location type.

        Returns:
            Dict[str, int]: Count of recoveries by location type.
        """
        if self.recovery_data is None:
            raise ValueError("No recovery data loaded. Import data first.")

        df = self.recovery_data

        if "criminal_location_type" not in df.columns:
            return {}

        # Count recoveries by criminal location type (excluding None/NaN)
        location_counts = df["criminal_location_type"].value_counts(dropna=True)
        return location_counts.to_dict()


class RecoveryDataPersistence:
    """
    A class for persisting recovery data to DuckDB databases using the shared DuckDbPersistence layer.

    This class provides recovery-specific database operations while leveraging
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
        self.table_name = "recovery_incidents"
        self._setup_table()

    def _setup_table(self):
        """Set up the recovery incidents table."""
        # Define table schema for recovery incidents
        schema = {
            "recovery_id": "VARCHAR PRIMARY KEY",
            "recovery_date": "TIMESTAMP",
            "recovery_lat": "DOUBLE",
            "recovery_lon": "DOUBLE",
            "recovery_location_name": "VARCHAR",
            "vehicle_condition": "VARCHAR",
            "incident_id": "VARCHAR",  # Foreign key to theft data
            "suspect_id": "VARCHAR",  # Foreign key to suspect data
            "recovery_agency": "VARCHAR",
            "recovery_officer": "VARCHAR",
            "is_criminal_location": "BOOLEAN",
            "criminal_location_type": "VARCHAR",
            "vehicle_damage_assessment": "VARCHAR",
            "parts_missing": "VARCHAR",
            "recovery_method": "VARCHAR",
            "recovery_time": "TIMESTAMP",
            "towed_to_location": "VARCHAR",
            "insurance_notified": "BOOLEAN",
            "owner_notified": "BOOLEAN",
            "case_status": "VARCHAR",
            "evidence_collected": "BOOLEAN",
            "notes": "TEXT",
            "geometry_wkt": "TEXT",
        }

        self.db_persistence.create_table(self.table_name, schema)

    def save_recovery_data(
        self, gdf: gpd.GeoDataFrame, table_name: Optional[str] = None
    ):
        """
        Save recovery data to the database.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing recovery data.
            table_name (Optional[str]): Name of the table to save to.
        """
        table_name = table_name or self.table_name

        # Define required columns for recovery data
        required_columns = [
            "recovery_id",
            "recovery_date",
            "recovery_lat",
            "recovery_lon",
            "recovery_location_name",
            "vehicle_condition",
            "incident_id",
            "suspect_id",
            "recovery_agency",
            "recovery_officer",
            "is_criminal_location",
            "criminal_location_type",
            "vehicle_damage_assessment",
            "parts_missing",
            "recovery_method",
            "recovery_time",
            "towed_to_location",
            "insurance_notified",
            "owner_notified",
            "case_status",
            "evidence_collected",
            "notes",
        ]

        # Use shared persistence layer
        self.db_persistence.save_data(
            gdf=gdf,
            table_name=table_name,
            geometry_column="geometry",
            primary_geometry_columns=["recovery_lat", "recovery_lon"],
            required_columns=required_columns,
            clear_existing=True,
        )

    def load_recovery_data(self, table_name: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load recovery data from the database.

        Args:
            table_name (Optional[str]): Name of the table to load from.

        Returns:
            gpd.GeoDataFrame: Loaded recovery data.
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
        Query recovery incidents within a specified distance from a point.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (Optional[str]): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Recovery incidents within the specified distance.
        """
        table_name = table_name or self.table_name

        return self.db_persistence.query_by_distance(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            table_name=table_name,
            lat_column="recovery_lat",
            lon_column="recovery_lon",
            geometry_column="geometry",
            geometry_wkt_column="geometry_wkt",
        )

    def get_recoveries_with_relationships(self) -> gpd.GeoDataFrame:
        """
        Get recovery data that has relationships to theft or suspect data.

        Returns:
            gpd.GeoDataFrame: Recovery data with theft or suspect relationships.
        """
        query_sql = f"""
        SELECT * FROM {self.table_name}
        WHERE incident_id IS NOT NULL OR suspect_id IS NOT NULL
        """

        df = self.db_persistence.execute_query(query_sql)

        if len(df) == 0:
            return gpd.GeoDataFrame()

        return self.db_persistence.load_data(
            table_name=self.table_name,
            geometry_column="geometry",
            geometry_wkt_column="geometry_wkt",
        ).loc[df.index]

    def get_criminal_location_recoveries(self) -> gpd.GeoDataFrame:
        """
        Get recovery data from known criminal locations.

        Returns:
            gpd.GeoDataFrame: Recovery data from criminal locations.
        """
        query_sql = f"""
        SELECT * FROM {self.table_name}
        WHERE is_criminal_location = true
        """

        df = self.db_persistence.execute_query(query_sql)

        if len(df) == 0:
            return gpd.GeoDataFrame()

        # Create geometry from stored WKT
        from shapely import wkt

        geometries = []
        for wkt_str in df["geometry_wkt"]:
            try:
                if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                    geometries.append(wkt.loads(str(wkt_str)))
                else:
                    geometries.append(None)
            except Exception:
                geometries.append(None)

        df["geometry"] = geometries
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)

        # Drop WKT column
        if "geometry_wkt" in gdf.columns:
            gdf = gdf.drop("geometry_wkt", axis=1)

        return gdf

    def get_statistics(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database statistics for recovery data.

        Args:
            table_name (Optional[str]): Name of the table to analyze.

        Returns:
            Dict[str, Any]: Database statistics.
        """
        table_name = table_name or self.table_name

        # Get basic statistics from shared layer
        basic_stats = self.db_persistence.get_statistics(
            table_name, columns=["recovery_agency", "vehicle_condition", "case_status"]
        )

        # Add recovery-specific statistics
        try:
            custom_stats_sql = f"""
            SELECT 
                COUNT(CASE WHEN incident_id IS NOT NULL THEN 1 END) as with_theft_link,
                COUNT(CASE WHEN suspect_id IS NOT NULL THEN 1 END) as with_suspect_link,
                COUNT(CASE WHEN is_criminal_location = true THEN 1 END) as criminal_locations,
                MIN(recovery_date) as earliest_recovery,
                MAX(recovery_date) as latest_recovery,
                COUNT(DISTINCT recovery_agency) as unique_agencies,
                COUNT(DISTINCT vehicle_condition) as unique_conditions
            FROM {table_name}
            """

            result = self.db_persistence.execute_query(custom_stats_sql)
            if len(result) > 0:
                row = result.iloc[0]
                basic_stats.update(
                    {
                        "recoveries_with_theft_link": row["with_theft_link"],
                        "recoveries_with_suspect_link": row["with_suspect_link"],
                        "criminal_location_recoveries": row["criminal_locations"],
                        "earliest_recovery": row["earliest_recovery"],
                        "latest_recovery": row["latest_recovery"],
                        "unique_agencies": row["unique_agencies"],
                        "unique_conditions": row["unique_conditions"],
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
