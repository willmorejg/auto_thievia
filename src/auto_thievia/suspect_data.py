"""
Suspect Data module for auto_thievia package.

This module provides functionality for importing suspect data from CSV files,
managing suspect records with personal and arrest information, and creating
geopandas dataframes for spatial analysis of suspect patterns.
"""

import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .gis_mapper import WGS84_CRS
from .duckdb_persistence import DuckDbPersistence, MEMORY_DB


# Constants
NO_DATA_ERROR_MSG = "No suspect data loaded. Import data first."


class SuspectData:
    """
    A class for managing suspect data related to auto theft incidents.

    This class handles importing suspect data from CSV files, validating coordinates,
    and creating geopandas dataframes with comprehensive suspect information including
    personal details and arrest history.
    """

    def __init__(self, crs: str = WGS84_CRS):
        """
        Initialize the SuspectData manager.

        Args:
            crs (str): Coordinate Reference System to use. Defaults to WGS84.
        """
        self.crs = crs
        self.suspect_data = None
        self._required_columns = {
            # Suspect personal information
            "suspect_name",
            "suspect_address",
            "address_lat",
            "address_lon",
            # Arrest information
            "last_arrest_date",
            "arrest_location",
            "arrest_lat",
            "arrest_lon",
        }
        self._optional_columns = {
            "suspect_id",
            "suspect_age",
            "suspect_gender",
            "suspect_phone",
            "suspect_email",
            "known_associates",
            "criminal_history",
            "arrest_charges",
            "arrest_officer",
            "case_number",
            "bail_amount",
            "court_date",
            "conviction_status",
            "parole_status",
            "risk_level",
            "gang_affiliation",
        }

    def import_from_csv(
        self,
        csv_path: Union[str, Path],
        encoding: str = "utf-8",
        validate_coordinates: bool = True,
    ) -> gpd.GeoDataFrame:
        """
        Import suspect data from a CSV file.

        Args:
            csv_path (Union[str, Path]): Path to the CSV file.
            encoding (str): File encoding. Defaults to 'utf-8'.
            validate_coordinates (bool): Whether to validate coordinate values.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing suspect data.

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

            # Create Point geometries for suspect addresses (primary geometry)
            geometry = [
                Point(lon, lat)
                for lon, lat in zip(df["address_lon"], df["address_lat"])
            ]

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

            # Store the data
            self.suspect_data = gdf

            return gdf

        except Exception as e:
            raise ValueError(f"Error reading CSV file {csv_path}: {str(e)}") from e

    def _clean_data(self, df: pd.DataFrame, validate_coordinates: bool) -> pd.DataFrame:
        """
        Clean and validate the suspect data.

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

        # Clean and parse date fields
        if "last_arrest_date" in df.columns:
            df["last_arrest_date"] = pd.to_datetime(
                df["last_arrest_date"], errors="coerce"
            )
            invalid_dates = df["last_arrest_date"].isnull()
            if invalid_dates.any():
                warnings.warn(
                    f"Found {invalid_dates.sum()} rows with invalid arrest dates"
                )

        # Clean age data if present
        if "suspect_age" in df.columns:
            df["suspect_age"] = pd.to_numeric(df["suspect_age"], errors="coerce")
            invalid_ages = df["suspect_age"].isnull()
            if invalid_ages.any():
                warnings.warn(f"Found {invalid_ages.sum()} rows with invalid ages")

        # Clean bail amount if present
        if "bail_amount" in df.columns:
            df["bail_amount"] = pd.to_numeric(df["bail_amount"], errors="coerce")

        # Parse court dates if present
        if "court_date" in df.columns:
            df["court_date"] = pd.to_datetime(df["court_date"], errors="coerce")

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate coordinate values for both address and arrest locations.

        Args:
            df (pd.DataFrame): Dataframe to validate.

        Returns:
            pd.DataFrame: Dataframe with valid coordinates only.
        """
        # Validate address coordinates
        invalid_address_coords = (
            (df["address_lat"] < -90)
            | (df["address_lat"] > 90)
            | (df["address_lon"] < -180)
            | (df["address_lon"] > 180)
        )

        # Validate arrest coordinates
        invalid_arrest_coords = (
            (df["arrest_lat"] < -90)
            | (df["arrest_lat"] > 90)
            | (df["arrest_lon"] < -180)
            | (df["arrest_lon"] > 180)
        )

        invalid_coords = invalid_address_coords | invalid_arrest_coords

        if invalid_coords.any():
            warnings.warn(
                f"Found {invalid_coords.sum()} rows with invalid coordinates. These will be excluded."
            )
            df = df[~invalid_coords]

        return df

    def create_sample_data(self, num_records: int = 50) -> gpd.GeoDataFrame:
        """
        Create sample suspect data for testing and demonstration.

        Args:
            num_records (int): Number of sample records to create.

        Returns:
            gpd.GeoDataFrame: Sample suspect data.
        """
        import numpy as np

        # Use reproducible random seed
        rng = np.random.default_rng(42)

        # Newark area bounding box (expanded for suspects)
        lat_min, lat_max = 40.60, 40.80
        lon_min, lon_max = -74.35, -74.10

        # Sample names and locations
        first_names = [
            "John",
            "Michael",
            "David",
            "James",
            "Robert",
            "William",
            "Richard",
            "Charles",
            "Thomas",
            "Daniel",
            "Matthew",
            "Anthony",
            "Christopher",
            "Maria",
            "Jennifer",
            "Lisa",
            "Sandra",
            "Michelle",
            "Patricia",
            "Nancy",
        ]

        last_names = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
            "Wilson",
            "Anderson",
            "Thomas",
            "Taylor",
            "Moore",
            "Jackson",
            "Martin",
        ]

        charges = [
            "Grand Theft Auto",
            "Auto Burglary",
            "Carjacking",
            "Receiving Stolen Property",
            "Auto Theft Conspiracy",
            "Vandalism",
            "Breaking and Entering",
            "Assault",
        ]

        risk_levels = ["Low", "Medium", "High", "Critical"]
        statuses = ["Active", "Paroled", "Incarcerated", "Released", "Pending Trial"]

        # Generate sample data
        data = []
        for i in range(num_records):
            # Generate suspect info
            first_name = rng.choice(first_names)
            last_name = rng.choice(last_names)
            full_name = f"{first_name} {last_name}"

            # Generate addresses within Newark area
            address_lat = rng.uniform(lat_min, lat_max)
            address_lon = rng.uniform(lon_min, lon_max)

            # Generate arrest location (could be anywhere in broader area)
            arrest_lat = rng.uniform(lat_min - 0.05, lat_max + 0.05)
            arrest_lon = rng.uniform(lon_min - 0.05, lon_max + 0.05)

            # Generate arrest date within last 3 years
            days_ago = rng.integers(1, 1095)  # 1-3 years
            arrest_date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)

            record = {
                "suspect_id": f"SUS{i+1:04d}",
                "suspect_name": full_name,
                "suspect_age": rng.integers(18, 65),
                "suspect_gender": rng.choice(["Male", "Female"]),
                "suspect_address": f"{rng.integers(100, 9999)} {rng.choice(['Main', 'Oak', 'Pine', 'Park', 'First', 'Second'])} {rng.choice(['St', 'Ave', 'Rd', 'Blvd'])}, Newark, NJ",
                "address_lat": address_lat,
                "address_lon": address_lon,
                "suspect_phone": f"({rng.integers(201, 999)}) {rng.integers(100, 999)}-{rng.integers(1000, 9999)}",
                "last_arrest_date": arrest_date,
                "arrest_location": f"{rng.choice(['Newark', 'Elizabeth', 'Union', 'Irvington'])} PD Station",
                "arrest_lat": arrest_lat,
                "arrest_lon": arrest_lon,
                "arrest_charges": rng.choice(charges),
                "arrest_officer": f"Officer {rng.choice(['Smith', 'Johnson', 'Brown', 'Davis'])}",
                "case_number": f"NWK-{rng.integers(2024, 2026)}-{rng.integers(10000, 99999)}",
                "bail_amount": rng.choice([0, 5000, 10000, 25000, 50000, 100000]),
                "conviction_status": rng.choice(statuses),
                "risk_level": rng.choice(risk_levels),
                "known_associates": rng.integers(0, 8),
                "criminal_history": rng.choice(
                    [
                        "First Offense",
                        "Repeat Offender",
                        "Career Criminal",
                        "Juvenile Record",
                    ]
                ),
                "gang_affiliation": rng.choice(
                    ["None", "Unknown", "Suspected", "Confirmed"]
                ),
                "parole_status": rng.choice(["N/A", "Active", "Completed", "Violated"]),
            }
            data.append(record)

        # Create DataFrame and convert to GeoDataFrame
        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat) for lon, lat in zip(df["address_lon"], df["address_lat"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)

        self.suspect_data = gdf
        return gdf

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the suspect data.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        if self.suspect_data is None:
            raise ValueError(NO_DATA_ERROR_MSG)

        df = self.suspect_data

        stats = {
            "total_suspects": len(df),
            "unique_suspects": (
                df["suspect_name"].nunique()
                if "suspect_name" in df.columns
                else len(df)
            ),
            "average_age": (
                df["suspect_age"].mean() if "suspect_age" in df.columns else None
            ),
            "age_range": {
                "min": df["suspect_age"].min() if "suspect_age" in df.columns else None,
                "max": df["suspect_age"].max() if "suspect_age" in df.columns else None,
            },
            "gender_distribution": (
                df["suspect_gender"].value_counts().to_dict()
                if "suspect_gender" in df.columns
                else {}
            ),
            "most_common_charge": (
                df["arrest_charges"].mode().iloc[0]
                if "arrest_charges" in df.columns and not df["arrest_charges"].empty
                else None
            ),
            "risk_level_distribution": (
                df["risk_level"].value_counts().to_dict()
                if "risk_level" in df.columns
                else {}
            ),
            "conviction_status_distribution": (
                df["conviction_status"].value_counts().to_dict()
                if "conviction_status" in df.columns
                else {}
            ),
        }

        # Add arrest date statistics if available
        if "last_arrest_date" in df.columns:
            stats["arrest_date_range"] = {
                "earliest": df["last_arrest_date"].min(),
                "latest": df["last_arrest_date"].max(),
            }

            # Recent arrests (within last 6 months)
            six_months_ago = pd.Timestamp.now() - pd.Timedelta(days=180)
            recent_arrests = df[df["last_arrest_date"] >= six_months_ago]
            stats["recent_arrests_count"] = len(recent_arrests)

        # Add bail statistics if available
        if "bail_amount" in df.columns:
            stats["bail_stats"] = {
                "average": df["bail_amount"].mean(),
                "median": df["bail_amount"].median(),
                "max": df["bail_amount"].max(),
                "held_without_bail": (df["bail_amount"] == 0).sum(),
            }

        return stats

    def filter_by_criteria(
        self,
        risk_levels: Optional[List[str]] = None,
        charges: Optional[List[str]] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        gender: Optional[str] = None,
        recent_arrests_only: bool = False,
        days_threshold: int = 180,
    ) -> gpd.GeoDataFrame:
        """
        Filter suspect data by various criteria.

        Args:
            risk_levels (Optional[List[str]]): Risk levels to include.
            charges (Optional[List[str]]): Charges to include.
            min_age (Optional[int]): Minimum age.
            max_age (Optional[int]): Maximum age.
            gender (Optional[str]): Gender filter.
            recent_arrests_only (bool): Only include recent arrests.
            days_threshold (int): Days threshold for recent arrests.

        Returns:
            gpd.GeoDataFrame: Filtered data.
        """
        if self.suspect_data is None:
            raise ValueError(NO_DATA_ERROR_MSG)

        df = self.suspect_data.copy()

        # Apply filters
        if risk_levels is not None and "risk_level" in df.columns:
            df = df[df["risk_level"].isin(risk_levels)]

        if charges is not None and "arrest_charges" in df.columns:
            df = df[df["arrest_charges"].isin(charges)]

        if min_age is not None and "suspect_age" in df.columns:
            df = df[df["suspect_age"] >= min_age]

        if max_age is not None and "suspect_age" in df.columns:
            df = df[df["suspect_age"] <= max_age]

        if gender is not None and "suspect_gender" in df.columns:
            df = df[df["suspect_gender"] == gender]

        if recent_arrests_only and "last_arrest_date" in df.columns:
            threshold_date = pd.Timestamp.now() - pd.Timedelta(days=days_threshold)
            df = df[df["last_arrest_date"] >= threshold_date]

        return df

    def get_high_risk_suspects(self) -> gpd.GeoDataFrame:
        """
        Get suspects classified as high risk or critical.

        Returns:
            gpd.GeoDataFrame: High risk suspects.
        """
        return self.filter_by_criteria(risk_levels=["High", "Critical"])

    def get_repeat_offenders(self) -> gpd.GeoDataFrame:
        """
        Get suspects with repeat offender or career criminal history.

        Returns:
            gpd.GeoDataFrame: Repeat offenders.
        """
        if self.suspect_data is None:
            raise ValueError(NO_DATA_ERROR_MSG)

        df = self.suspect_data
        if "criminal_history" in df.columns:
            return df[
                df["criminal_history"].isin(["Repeat Offender", "Career Criminal"])
            ]
        else:
            return gpd.GeoDataFrame()

    def get_suspects_by_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        location_type: str = "address",
    ) -> gpd.GeoDataFrame:
        """
        Get suspects within a specified distance from a point.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            location_type (str): 'address' or 'arrest' location.

        Returns:
            gpd.GeoDataFrame: Suspects within the specified distance.
        """
        if self.suspect_data is None:
            raise ValueError(NO_DATA_ERROR_MSG)

        # Simple bounding box approach
        degree_offset = radius_km / 111.0  # Rough conversion: 1 degree ≈ 111 km

        min_lat = center_lat - degree_offset
        max_lat = center_lat + degree_offset
        min_lon = center_lon - degree_offset
        max_lon = center_lon + degree_offset

        df = self.suspect_data

        if location_type == "address":
            filtered = df[
                (df["address_lat"].between(min_lat, max_lat))
                & (df["address_lon"].between(min_lon, max_lon))
            ]
        elif location_type == "arrest":
            filtered = df[
                (df["arrest_lat"].between(min_lat, max_lat))
                & (df["arrest_lon"].between(min_lon, max_lon))
            ]
        else:
            raise ValueError("location_type must be 'address' or 'arrest'")

        return filtered

    def create_arrest_points_gdf(self) -> gpd.GeoDataFrame:
        """
        Create a separate GeoDataFrame with arrest locations as geometry.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with arrest locations as points.
        """
        if self.suspect_data is None:
            raise ValueError(NO_DATA_ERROR_MSG)

        df = self.suspect_data.copy()

        # Create Point geometries for arrest locations
        arrest_geometry = [
            Point(lon, lat) for lon, lat in zip(df["arrest_lon"], df["arrest_lat"])
        ]

        # Create new GeoDataFrame with arrest locations
        arrest_gdf = gpd.GeoDataFrame(df, geometry=arrest_geometry, crs=self.crs)

        return arrest_gdf


class SuspectDataPersistence:
    """
    A class for persisting suspect data to DuckDB databases.

    This class handles creating DuckDB databases, storing geopandas dataframes,
    and querying suspect data with spatial capabilities.
    """

    def __init__(self, db_path: Union[str, Path] = MEMORY_DB):
        """
        Initialize the persistence manager.

        Args:
            db_path (Union[str, Path]): Path to DuckDB database file.
                                       Use MEMORY_DB for in-memory database.
        """
        self.db_persistence = DuckDbPersistence(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create the suspect data tables."""
        # Main suspect table schema
        suspect_table_schema = {
            "suspect_id": "VARCHAR PRIMARY KEY",
            "suspect_name": "VARCHAR",
            "suspect_age": "INTEGER",
            "suspect_gender": "VARCHAR",
            "suspect_address": "VARCHAR",
            "address_lat": "DOUBLE",
            "address_lon": "DOUBLE",
            "address_geometry_wkt": "TEXT",
            "suspect_phone": "VARCHAR",
            "suspect_email": "VARCHAR",
            "last_arrest_date": "TIMESTAMP",
            "arrest_location": "VARCHAR",
            "arrest_lat": "DOUBLE",
            "arrest_lon": "DOUBLE",
            "arrest_geometry_wkt": "TEXT",
            "arrest_charges": "VARCHAR",
            "arrest_officer": "VARCHAR",
            "case_number": "VARCHAR",
            "bail_amount": "DOUBLE",
            "court_date": "TIMESTAMP",
            "conviction_status": "VARCHAR",
            "parole_status": "VARCHAR",
            "risk_level": "VARCHAR",
            "known_associates": "INTEGER",
            "criminal_history": "VARCHAR",
            "gang_affiliation": "VARCHAR",
        }

        self.db_persistence.create_table("suspects", suspect_table_schema)

    def save_suspect_data(self, gdf: gpd.GeoDataFrame, table_name: str = "suspects"):
        """
        Save suspect data to the database.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing suspect data.
            table_name (str): Name of the table to save to.
        """
        # Define required columns for suspects
        required_columns = [
            "suspect_id",
            "suspect_name",
            "suspect_age",
            "suspect_gender",
            "suspect_address",
            "address_lat",
            "address_lon",
            "address_geometry_wkt",
            "suspect_phone",
            "suspect_email",
            "last_arrest_date",
            "arrest_location",
            "arrest_lat",
            "arrest_lon",
            "arrest_geometry_wkt",
            "arrest_charges",
            "arrest_officer",
            "case_number",
            "bail_amount",
            "court_date",
            "conviction_status",
            "parole_status",
            "risk_level",
            "known_associates",
            "criminal_history",
            "gang_affiliation",
        ]

        # Save data using the generic persistence layer
        self.db_persistence.save_data(
            gdf=gdf,
            table_name=table_name,
            geometry_column="geometry",
            primary_geometry_columns=["address_lat", "address_lon"],
            secondary_geometry_columns=["arrest_lat", "arrest_lon"],
            required_columns=required_columns,
            clear_existing=True,
        )

    def load_suspect_data(self, table_name: str = "suspects") -> gpd.GeoDataFrame:
        """
        Load suspect data from the database.

        Args:
            table_name (str): Name of the table to load from.

        Returns:
            gpd.GeoDataFrame: Loaded suspect data.
        """
        return self.db_persistence.load_data(
            table_name=table_name,
            geometry_column="geometry",
            geometry_wkt_column="address_geometry_wkt",
        )

    def query_suspects_by_address_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        table_name: str = "suspects",
    ) -> gpd.GeoDataFrame:
        """
        Query suspects within a specified distance from a point (by address).

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (str): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Suspects within the specified distance.
        """
        return self.db_persistence.query_by_distance(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            table_name=table_name,
            lat_column="address_lat",
            lon_column="address_lon",
            geometry_column="geometry",
            geometry_wkt_column="address_geometry_wkt",
        )

    def query_suspects_by_arrest_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        table_name: str = "suspects",
    ) -> gpd.GeoDataFrame:
        """
        Query suspects within a specified distance from a point (by arrest location).

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (str): Name of the table to query.

        Returns:
            gpd.GeoDataFrame: Suspects within the specified distance.
        """
        return self.db_persistence.query_by_distance(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            table_name=table_name,
            lat_column="arrest_lat",
            lon_column="arrest_lon",
            geometry_column="geometry",
            geometry_wkt_column="address_geometry_wkt",
        )

    def query_suspects_by_criteria(
        self,
        table_name: str = "suspects",
        risk_levels: Optional[List[str]] = None,
        charges: Optional[List[str]] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        gender: Optional[str] = None,
        recent_arrests_days: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Query suspects by various criteria using SQL.

        Args:
            table_name (str): Name of the table to query.
            risk_levels (Optional[List[str]]): Risk levels to include.
            charges (Optional[List[str]]): Charges to include.
            min_age (Optional[int]): Minimum age.
            max_age (Optional[int]): Maximum age.
            gender (Optional[str]): Gender filter.
            recent_arrests_days (Optional[int]): Days threshold for recent arrests.

        Returns:
            gpd.GeoDataFrame: Filtered suspect data.
        """
        # Build WHERE clause
        conditions = []

        if risk_levels:
            risk_list = "', '".join(risk_levels)
            conditions.append(f"risk_level IN ('{risk_list}')")

        if charges:
            charges_list = "', '".join(charges)
            conditions.append(f"arrest_charges IN ('{charges_list}')")

        if min_age is not None:
            conditions.append(f"suspect_age >= {min_age}")

        if max_age is not None:
            conditions.append(f"suspect_age <= {max_age}")

        if gender:
            conditions.append(f"suspect_gender = '{gender}'")

        if recent_arrests_days is not None:
            conditions.append(
                f"last_arrest_date >= (CURRENT_DATE - INTERVAL '{recent_arrests_days}' DAY)"
            )

        # Build full query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query_sql = f"SELECT * FROM {table_name} WHERE {where_clause}"

        # Execute query and convert to GeoDataFrame
        df = self.db_persistence.execute_query(query_sql)

        if len(df) == 0:
            return gpd.GeoDataFrame()

        # Convert to GeoDataFrame
        from shapely import wkt

        geometries = []
        for wkt_str in df["address_geometry_wkt"]:
            try:
                if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                    geometries.append(wkt.loads(str(wkt_str)))
                else:
                    geometries.append(None)
            except Exception:
                geometries.append(None)

        df["geometry"] = geometries
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84_CRS)

        # Drop WKT columns
        wkt_columns = [col for col in df.columns if "_wkt" in col.lower()]
        gdf = gdf.drop(columns=[col for col in wkt_columns if col in gdf.columns])

        return gdf

    def get_suspect_statistics(self, table_name: str = "suspects") -> Dict[str, Any]:
        """
        Get suspect database statistics.

        Args:
            table_name (str): Name of the table to analyze.

        Returns:
            Dict[str, Any]: Suspect statistics.
        """
        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_suspects,
                COUNT(DISTINCT suspect_name) as unique_suspects,
                AVG(suspect_age) as avg_age,
                MIN(suspect_age) as min_age,
                MAX(suspect_age) as max_age,
                COUNT(DISTINCT risk_level) as risk_levels_count,
                COUNT(DISTINCT arrest_charges) as unique_charges,
                COUNT(DISTINCT conviction_status) as conviction_statuses,
                MIN(last_arrest_date) as earliest_arrest,
                MAX(last_arrest_date) as latest_arrest,
                AVG(bail_amount) as avg_bail_amount,
                SUM(CASE WHEN bail_amount = 0 THEN 1 ELSE 0 END) as held_without_bail
            FROM {table_name}
            """

            result = self.db_persistence.execute_query(stats_sql)
            if len(result) > 0:
                row = result.iloc[0]
                stats = {
                    "total_suspects": int(row["total_suspects"]),
                    "unique_suspects": int(row["unique_suspects"]),
                    "avg_age": (
                        float(row["avg_age"]) if pd.notna(row["avg_age"]) else None
                    ),
                    "min_age": (
                        int(row["min_age"]) if pd.notna(row["min_age"]) else None
                    ),
                    "max_age": (
                        int(row["max_age"]) if pd.notna(row["max_age"]) else None
                    ),
                    "risk_levels_count": int(row["risk_levels_count"]),
                    "unique_charges": int(row["unique_charges"]),
                    "conviction_statuses": int(row["conviction_statuses"]),
                    "earliest_arrest": row["earliest_arrest"],
                    "latest_arrest": row["latest_arrest"],
                    "avg_bail_amount": (
                        float(row["avg_bail_amount"])
                        if pd.notna(row["avg_bail_amount"])
                        else None
                    ),
                    "held_without_bail": int(row["held_without_bail"]),
                }
                return stats
            else:
                return {"total_suspects": 0}

        except Exception as e:
            raise ValueError(f"Error getting suspect statistics: {str(e)}")

    def close(self):
        """Close the database connection."""
        self.db_persistence.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
