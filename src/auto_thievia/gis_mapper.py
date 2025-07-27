"""
GIS Mapper module for auto_thievia package.

This module provides functionality for reading OSM shapefiles and plotting
points using geopandas dataframes for auto theft pattern analysis.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import contextily as ctx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from shapely.geometry import Point

# Constants
WEB_MERCATOR_CRS = "EPSG:3857"
WGS84_CRS = "EPSG:4326"


class GISMapper:
    """
    A class for handling GIS operations related to auto theft analysis.

    This class provides methods to read OSM shapefiles, create and manipulate
    geopandas dataframes, and visualize spatial data for identifying auto theft
    patterns and potential chop shop locations.
    """

    def __init__(self, crs: str = WGS84_CRS):
        """
        Initialize the GISMapper.

        Args:
            crs (str): Coordinate Reference System to use. Defaults to "EPSG:4326" (WGS84).
        """
        self.crs = crs
        self.base_map = None
        self.theft_points = None
        self._osm_data = {}

    def read_osm_shapefile(self, shapefile_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Read an OSM shapefile and return as a GeoDataFrame.

        Args:
            shapefile_path (Union[str, Path]): Path to the shapefile.

        Returns:
            gpd.GeoDataFrame: The loaded shapefile data.

        Raises:
            FileNotFoundError: If the shapefile doesn't exist.
            ValueError: If the file cannot be read as a valid shapefile.
        """
        shapefile_path = Path(shapefile_path)

        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

        try:
            gdf = gpd.read_file(shapefile_path)

            # Ensure CRS is set
            if gdf.crs is None:
                warnings.warn("No CRS found in shapefile, assuming EPSG:4326")
                gdf.crs = WGS84_CRS

            # Convert to desired CRS if different
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)

            self.base_map = gdf
            return gdf

        except Exception as e:
            raise ValueError(f"Error reading shapefile {shapefile_path}: {str(e)}")

    def create_points_from_coordinates(
        self,
        coordinates: List[Tuple[float, float]],
        attributes: Optional[Dict[str, List[Any]]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame of points from coordinate pairs.

        Args:
            coordinates (List[Tuple[float, float]]): List of (longitude, latitude) pairs.
            attributes (Optional[Dict[str, List[Any]]]): Optional attributes for each point.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the points.

        Raises:
            ValueError: If coordinates are invalid or attributes don't match coordinate count.
        """
        if not coordinates:
            raise ValueError("No coordinates provided")

        # Validate coordinates
        for i, (lon, lat) in enumerate(coordinates):
            if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                raise ValueError(f"Invalid coordinates at index {i}: ({lon}, {lat})")

        # Create Point geometries
        geometry = [Point(lon, lat) for lon, lat in coordinates]

        # Prepare data dictionary
        data = {"geometry": geometry}

        # Add attributes if provided
        if attributes:
            for key, values in attributes.items():
                if len(values) != len(coordinates):
                    raise ValueError(
                        f"Attribute '{key}' has {len(values)} values but "
                        f"there are {len(coordinates)} coordinates"
                    )
                data[key] = values

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        self.theft_points = gdf

        return gdf

    def create_points_from_dataframe(
        self, df: pd.DataFrame, lon_col: str = "longitude", lat_col: str = "latitude"
    ) -> gpd.GeoDataFrame:
        """
        Convert a pandas DataFrame with coordinate columns to a GeoDataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate data.
            lon_col (str): Name of the longitude column.
            lat_col (str): Name of the latitude column.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with Point geometries.

        Raises:
            KeyError: If specified coordinate columns don't exist.
            ValueError: If coordinate data is invalid.
        """
        if lon_col not in df.columns or lat_col not in df.columns:
            raise KeyError(
                f"Columns '{lon_col}' and/or '{lat_col}' not found in DataFrame"
            )

        # Check for missing values
        if df[lon_col].isna().any() or df[lat_col].isna().any():
            warnings.warn("Found missing coordinate values, they will be excluded")
            df = df.dropna(subset=[lon_col, lat_col])

        # Create Point geometries
        geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]

        # Create GeoDataFrame with all original columns except coordinate columns
        gdf_data = df.drop(columns=[lon_col, lat_col]).copy()
        gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, crs=self.crs)

        self.theft_points = gdf
        return gdf

    def plot_static_map(
        self,
        points_gdf: Optional[gpd.GeoDataFrame] = None,
        base_gdf: Optional[gpd.GeoDataFrame] = None,
        figsize: Tuple[int, int] = (12, 8),
        point_color: str = "red",
        point_size: int = 50,
        add_basemap: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, Axes]:
        """
        Create a static matplotlib plot of the GIS data.

        Args:
            points_gdf (Optional[gpd.GeoDataFrame]): Points to plot.
            base_gdf (Optional[gpd.GeoDataFrame]): Base map to plot.
            figsize (Tuple[int, int]): Figure size.
            point_color (str): Color for points.
            point_size (int): Size of points.
            add_basemap (bool): Whether to add a contextual basemap.
            title (Optional[str]): Plot title.

        Returns:
            Tuple[plt.Figure, Axes]: The figure and axes objects.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Use stored data if not provided
        if base_gdf is None:
            base_gdf = self.base_map
        if points_gdf is None:
            points_gdf = self.theft_points

        # Plot base map if available
        if base_gdf is not None:
            base_gdf.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.7)

        # Plot points if available
        if points_gdf is not None:
            # Convert to Web Mercator for basemap if needed
            if add_basemap and points_gdf.crs != WEB_MERCATOR_CRS:
                points_plot = points_gdf.to_crs(WEB_MERCATOR_CRS)
                ax_crs = WEB_MERCATOR_CRS
            else:
                points_plot = points_gdf
                ax_crs = points_gdf.crs

            points_plot.plot(ax=ax, color=point_color, markersize=point_size, alpha=0.7)

            # Add basemap
            if add_basemap:
                try:
                    ctx.add_basemap(
                        ax, crs=ax_crs, source=ctx.providers.CartoDB.Positron
                    )
                except Exception as e:
                    warnings.warn(f"Could not add basemap: {e}")

        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        elif points_gdf is not None:
            ax.set_title(
                f"Auto Theft Locations ({len(points_gdf)} incidents)",
                fontsize=14,
                fontweight="bold",
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()

        return fig, ax

    def create_interactive_map(
        self,
        points_gdf: Optional[gpd.GeoDataFrame] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        zoom_start: int = 10,
        popup_columns: Optional[List[str]] = None,
    ) -> folium.Map:
        """
        Create an interactive Folium map with the data.

        Args:
            points_gdf (Optional[gpd.GeoDataFrame]): Points to plot.
            center_lat (Optional[float]): Center latitude for map.
            center_lon (Optional[float]): Center longitude for map.
            zoom_start (int): Initial zoom level.
            popup_columns (Optional[List[str]]): Columns to include in popups.

        Returns:
            folium.Map: Interactive map object.
        """
        # Use stored data if not provided
        if points_gdf is None:
            points_gdf = self.theft_points

        # Calculate center if not provided
        if points_gdf is not None and (center_lat is None or center_lon is None):
            bounds = points_gdf.total_bounds  # [minx, miny, maxx, maxy]
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
        elif center_lat is None or center_lon is None:
            # Default to a generic location
            center_lat, center_lon = 39.8283, -98.5795  # Geographic center of US

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles="OpenStreetMap",
        )

        # Add points if available
        if points_gdf is not None:
            for idx, row in points_gdf.iterrows():
                # Create popup text
                if popup_columns:
                    popup_text = "<br>".join(
                        [
                            f"<b>{col}:</b> {row[col]}"
                            for col in popup_columns
                            if col in row.index
                        ]
                    )
                else:
                    popup_text = f"Auto Theft Incident<br>Index: {idx}"

                # Add marker
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    popup=folium.Popup(popup_text, max_width=200),
                    color="red",
                    fill=True,
                    fillColor="red",
                    fillOpacity=0.7,
                ).add_to(m)

        return m

    def get_bounds(
        self, gdf: Optional[gpd.GeoDataFrame] = None
    ) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of a GeoDataFrame.

        Args:
            gdf (Optional[gpd.GeoDataFrame]): GeoDataFrame to get bounds for.

        Returns:
            Tuple[float, float, float, float]: Bounds as (minx, miny, maxx, maxy).
        """
        if gdf is None:
            gdf = self.theft_points

        if gdf is None:
            raise ValueError("No GeoDataFrame provided and no theft points stored")

        return tuple(gdf.total_bounds)

    def save_to_file(
        self,
        gdf: gpd.GeoDataFrame,
        filepath: Union[str, Path],
        driver: str = "ESRI Shapefile",
    ) -> None:
        """
        Save a GeoDataFrame to a file.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame to save.
            filepath (Union[str, Path]): Output file path.
            driver (str): Output format driver.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(filepath, driver=driver)
