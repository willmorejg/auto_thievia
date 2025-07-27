"""
DuckDB Persistence module for auto_thievia package.

This module provides a generic DuckDB persistence layer that can be used by
different data classes (TheftData, SuspectData, etc.) for database operations.
"""

import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import duckdb
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt

from .gis_mapper import WGS84_CRS

# Constants
MEMORY_DB = ":memory:"
DB_CONNECTION_ERROR = "Database connection not established"


class DuckDbPersistence:
    """
    A generic class for persisting geospatial data to DuckDB databases.

    This class provides a common interface for different data types to store
    and retrieve data from DuckDB with spatial capabilities.
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

        except Exception as e:
            raise ValueError(f"Error setting up DuckDB database: {str(e)}")

    def create_table(self, table_name: str, schema: Dict[str, str]):
        """
        Create a table with the specified schema.

        Args:
            table_name (str): Name of the table to create.
            schema (Dict[str, str]): Column definitions {column_name: column_type}.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        # Build CREATE TABLE SQL
        columns = []
        for col_name, col_type in schema.items():
            columns.append(f"{col_name} {col_type}")

        # Add created_at timestamp
        columns.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        );
        """

        self.connection.execute(create_sql)

    def save_data(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        geometry_column: str = "geometry",
        primary_geometry_columns: Optional[List[str]] = None,
        secondary_geometry_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        clear_existing: bool = True,
    ):
        """
        Save geospatial data to the database.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing data to save.
            table_name (str): Name of the table to save to.
            geometry_column (str): Name of the primary geometry column.
            primary_geometry_columns (Optional[List[str]]): Lat/lon columns for primary geometry.
            secondary_geometry_columns (Optional[List[str]]): Lat/lon columns for secondary geometry.
            required_columns (Optional[List[str]]): List of required columns.
            clear_existing (bool): Whether to clear existing data before inserting.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            # Prepare data for insertion
            df = gdf.copy()

            # Convert primary geometry to WKT for storage
            primary_geometry_wkt_col = f"{geometry_column}_wkt"
            df[primary_geometry_wkt_col] = df[geometry_column].to_wkt()

            # Handle secondary geometry if specified
            if secondary_geometry_columns and len(secondary_geometry_columns) >= 2:
                lat_col, lon_col = (
                    secondary_geometry_columns[0],
                    secondary_geometry_columns[1],
                )
                if lat_col in df.columns and lon_col in df.columns:
                    secondary_geometry = [
                        Point(lon, lat).wkt if pd.notna(lon) and pd.notna(lat) else None
                        for lon, lat in zip(df[lon_col], df[lat_col])
                    ]
                    df[f"{lat_col}_{lon_col}_geometry"] = secondary_geometry

            # Handle required columns with default values
            if required_columns:
                for col in required_columns:
                    if col not in df.columns:
                        # Set appropriate default values based on column patterns
                        if "date" in col.lower() or "time" in col.lower():
                            df[col] = pd.Timestamp.now()
                        elif "id" in col.lower():
                            # Generate unique IDs
                            df[col] = [f"ID_{i+1:06d}" for i in range(len(df))]
                        elif any(
                            x in col.lower() for x in ["year", "age", "value", "amount"]
                        ):
                            df[col] = 0
                        elif any(
                            x in col.lower()
                            for x in ["lat", "lon", "latitude", "longitude"]
                        ):
                            df[col] = 0.0
                        else:
                            df[col] = None

                # Reorder columns to match required schema
                df = df[
                    required_columns
                    + [col for col in df.columns if col not in required_columns]
                ]

            # Drop the original geometry column
            if geometry_column in df.columns:
                df = df.drop(geometry_column, axis=1)

            # Clear existing data if requested
            if clear_existing:
                self.connection.execute(f"DELETE FROM {table_name}")

            # Register dataframe and insert
            self.connection.register("temp_df", df)

            # Get column names (excluding created_at which has DEFAULT)
            table_info = self.connection.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
            db_columns = [row[1] for row in table_info if row[1] != "created_at"]

            # Only use columns that exist in both the dataframe and the table
            insert_columns = [col for col in db_columns if col in df.columns]

            insert_sql = f"""
            INSERT INTO {table_name} ({', '.join(insert_columns)})
            SELECT {', '.join(insert_columns)} FROM temp_df
            """
            self.connection.execute(insert_sql)

            print(f"Successfully saved {len(df)} records to {table_name}")

        except Exception as e:
            raise ValueError(f"Error saving data to database: {str(e)}") from e

    def load_data(
        self,
        table_name: str,
        geometry_column: str = "geometry",
        geometry_wkt_column: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Load geospatial data from the database.

        Args:
            table_name (str): Name of the table to load from.
            geometry_column (str): Name to use for the geometry column.
            geometry_wkt_column (Optional[str]): Name of the WKT geometry column in DB.

        Returns:
            gpd.GeoDataFrame: Loaded geospatial data.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            # Query all data
            df = self.connection.execute(f"SELECT * FROM {table_name}").df()

            if len(df) == 0:
                warnings.warn(f"No data found in table {table_name}")
                return gpd.GeoDataFrame()

            # Auto-detect WKT geometry column if not specified
            if geometry_wkt_column is None:
                potential_geom_cols = [
                    col
                    for col in df.columns
                    if "geometry" in col.lower() or "_wkt" in col.lower()
                ]
                if potential_geom_cols:
                    geometry_wkt_column = potential_geom_cols[0]
                else:
                    raise ValueError(f"No geometry column found in table {table_name}")

            # Convert WKT back to geometry
            geometries = []
            for wkt_str in df[geometry_wkt_column]:
                try:
                    if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                        geometries.append(wkt.loads(str(wkt_str)))
                    else:
                        geometries.append(None)
                except Exception:
                    geometries.append(None)

            df[geometry_column] = geometries

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry_column, crs=WGS84_CRS)

            # Drop WKT columns
            wkt_columns = [
                col
                for col in df.columns
                if "_wkt" in col.lower() or "geometry" in col.lower()
            ]
            wkt_columns = [col for col in wkt_columns if col != geometry_column]
            gdf = gdf.drop(columns=[col for col in wkt_columns if col in gdf.columns])

            return gdf

        except Exception as e:
            raise ValueError(f"Error loading data from database: {str(e)}")

    def query_by_distance(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        table_name: str,
        lat_column: str = "lat",
        lon_column: str = "lon",
        geometry_column: str = "geometry",
        geometry_wkt_column: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Query data within a specified distance from a point.

        Args:
            center_lat (float): Center point latitude.
            center_lon (float): Center point longitude.
            radius_km (float): Search radius in kilometers.
            table_name (str): Name of the table to query.
            lat_column (str): Name of the latitude column.
            lon_column (str): Name of the longitude column.
            geometry_column (str): Name for the geometry column in result.
            geometry_wkt_column (Optional[str]): Name of the WKT geometry column in DB.

        Returns:
            gpd.GeoDataFrame: Data within the specified distance.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        # Use bounding box approach since we're storing WKT as TEXT
        degree_offset = radius_km / 111.0  # Rough conversion: 1 degree ≈ 111 km

        min_lat = center_lat - degree_offset
        max_lat = center_lat + degree_offset
        min_lon = center_lon - degree_offset
        max_lon = center_lon + degree_offset

        query_sql = f"""
        SELECT * FROM {table_name}
        WHERE {lat_column} BETWEEN {min_lat} AND {max_lat}
        AND {lon_column} BETWEEN {min_lon} AND {max_lon}
        """

        df = self.connection.execute(query_sql).df()

        if len(df) == 0:
            return gpd.GeoDataFrame()

        # Auto-detect WKT geometry column if not specified
        if geometry_wkt_column is None:
            potential_geom_cols = [
                col
                for col in df.columns
                if "geometry" in col.lower() or "_wkt" in col.lower()
            ]
            if potential_geom_cols:
                geometry_wkt_column = potential_geom_cols[0]
            else:
                raise ValueError(f"No geometry column found in query result")

        # Create geometry from WKT strings
        geometries = []
        for wkt_str in df[geometry_wkt_column]:
            try:
                if wkt_str and str(wkt_str) not in ["None", "nan", "null"]:
                    geometries.append(wkt.loads(str(wkt_str)))
                else:
                    geometries.append(None)
            except Exception:
                geometries.append(None)

        df[geometry_column] = geometries
        gdf = gpd.GeoDataFrame(df, geometry=geometry_column, crs=WGS84_CRS)

        return gdf

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a custom SQL query.

        Args:
            sql (str): SQL query to execute.

        Returns:
            pd.DataFrame: Query results.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        return self.connection.execute(sql).df()

    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """
        Get information about a table's structure.

        Args:
            table_name (str): Name of the table.

        Returns:
            pd.DataFrame: Table structure information.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        return self.connection.execute(f"PRAGMA table_info({table_name})").df()

    def get_statistics(
        self, table_name: str, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get basic statistics for a table.

        Args:
            table_name (str): Name of the table to analyze.
            columns (Optional[List[str]]): Specific columns to analyze.

        Returns:
            Dict[str, Any]: Table statistics.
        """
        if self.connection is None:
            raise ValueError(DB_CONNECTION_ERROR)

        try:
            # Get basic count
            count_result = self.connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            total_records = count_result[0] if count_result else 0

            stats = {"total_records": total_records, "table_name": table_name}

            if columns and total_records > 0:
                # Get statistics for specified columns
                for col in columns:
                    try:
                        col_stats = self.connection.execute(
                            f"""
                            SELECT 
                                COUNT({col}) as non_null_count,
                                COUNT(DISTINCT {col}) as unique_count
                            FROM {table_name}
                        """
                        ).fetchone()

                        stats[f"{col}_stats"] = {
                            "non_null_count": col_stats[0],
                            "unique_count": col_stats[1],
                        }
                    except Exception:
                        # Skip columns that can't be analyzed
                        continue

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
