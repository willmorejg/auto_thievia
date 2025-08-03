"""
FastAPI REST API for Auto Thievia mapping services.

This module provides REST endpoints to generate and serve interactive maps
for auto theft investigation analysis.
"""

import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from .gis_mapper import GISMapper
from .theft_data import TheftData
from .suspect_data import SuspectData
from .recovery_data import RecoveryData
from .learning_models import LearningClass


# Pydantic models for API requests and responses
class MapRequest(BaseModel):
    """Request model for map generation."""

    coordinates: List[List[float]] = Field(
        ..., description="List of [lon, lat] coordinates"
    )
    attributes: Optional[Dict[str, List]] = Field(
        None, description="Additional attributes for points"
    )
    center_lat: Optional[float] = Field(None, description="Map center latitude")
    center_lon: Optional[float] = Field(None, description="Map center longitude")
    zoom_start: int = Field(10, description="Initial zoom level")
    popup_columns: Optional[List[str]] = Field(
        None, description="Columns to show in popups"
    )
    title: Optional[str] = Field(None, description="Map title")


class MapResponse(BaseModel):
    """Response model for map generation."""

    map_id: str = Field(..., description="Unique map identifier")
    map_url: str = Field(..., description="URL to access the map")
    bounds: List[float] = Field(..., description="Map bounds [minx, miny, maxx, maxy]")
    point_count: int = Field(..., description="Number of points on the map")
    created_at: str = Field(..., description="Map creation timestamp")


class TheftAnalysisRequest(BaseModel):
    """Request model for theft analysis."""

    file_path: Optional[str] = Field(None, description="Path to theft data CSV file")
    start_date: Optional[str] = Field(
        None, description="Analysis start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(None, description="Analysis end date (YYYY-MM-DD)")
    center_lat: float = Field(40.7357, description="Analysis center latitude")
    center_lon: float = Field(-74.1723, description="Analysis center longitude")
    radius_km: float = Field(10.0, description="Analysis radius in kilometers")


class SuspectAnalysisRequest(BaseModel):
    """Request model for suspect analysis."""

    risk_levels: Optional[List[str]] = Field(
        ["High", "Critical"], description="Risk levels to include"
    )
    include_arrests: bool = Field(True, description="Include arrest location data")
    days_threshold: int = Field(30, description="Recent activity threshold in days")


# Initialize FastAPI app
app = FastAPI(
    title="Auto Thievia API",
    description="REST API for auto theft investigation mapping and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
_gis_mapper = None
_theft_data = None
_suspect_data = None
_recovery_data = None
_learning_models = None
_output_dir = Path("data/output")


def get_gis_mapper() -> GISMapper:
    """Get or create GIS mapper instance."""
    global _gis_mapper
    if _gis_mapper is None:
        _gis_mapper = GISMapper()
    return _gis_mapper


def get_theft_data() -> TheftData:
    """Get or create theft data instance."""
    global _theft_data
    if _theft_data is None:
        _theft_data = TheftData()
    return _theft_data


def get_suspect_data() -> SuspectData:
    """Get or create suspect data instance."""
    global _suspect_data
    if _suspect_data is None:
        _suspect_data = SuspectData()
    return _suspect_data


def get_recovery_data() -> RecoveryData:
    """Get or create recovery data instance."""
    global _recovery_data
    if _recovery_data is None:
        _recovery_data = RecoveryData()
    return _recovery_data


def get_learning_models() -> LearningClass:
    """Get or create learning models instance."""
    global _learning_models
    if _learning_models is None:
        _learning_models = LearningClass()
    return _learning_models


def ensure_output_dir():
    """Ensure output directory exists."""
    _output_dir.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>Auto Thievia API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #007acc; }
            </style>
        </head>
        <body>
            <h1 class="header">🚗 Auto Thievia API</h1>
            <p>REST API for auto theft investigation mapping and analysis</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <span class="method">GET</span> /docs - Interactive API documentation
            </div>
            <div class="endpoint">
                <span class="method">POST</span> /maps/create - Create custom interactive map
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /maps/theft - Generate theft analysis map
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /maps/suspects - Generate suspect analysis map
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /maps/recovery - Generate recovery analysis map
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /maps/list - List available maps
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /analysis/theft - Perform theft pattern analysis
            </div>
            <div class="endpoint">
                <span class="method">GET</span> /analysis/suspects - Perform suspect analysis
            </div>
            
            <p><a href="/docs">📖 View API Documentation</a></p>
        </body>
    </html>
    """


@app.post("/maps/create", response_model=MapResponse)
async def create_custom_map(request: MapRequest):
    """Create a custom interactive map from coordinates and attributes."""
    try:
        mapper = get_gis_mapper()
        ensure_output_dir()

        # Convert coordinates to proper format
        coordinates = [(coord[0], coord[1]) for coord in request.coordinates]

        # Create points from coordinates
        gdf = mapper.create_points_from_coordinates(coordinates, request.attributes)

        # Create interactive map
        interactive_map = mapper.create_interactive_map(
            points_gdf=gdf,
            center_lat=request.center_lat,
            center_lon=request.center_lon,
            zoom_start=request.zoom_start,
            popup_columns=request.popup_columns,
        )

        # Generate unique map ID
        map_id = f"custom_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        map_path = _output_dir / f"{map_id}.html"

        # Save map
        interactive_map.save(str(map_path))

        # Get bounds
        bounds = list(mapper.get_bounds(gdf))

        return MapResponse(
            map_id=map_id,
            map_url=f"/maps/view/{map_id}",
            bounds=bounds,
            point_count=len(gdf),
            created_at=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Map creation failed: {str(e)}")


@app.get("/maps/theft")
async def generate_theft_map(
    file_path: Optional[str] = Query(None, description="Path to theft data CSV"),
    use_sample: bool = Query(True, description="Use sample data if no file provided"),
    center_lat: float = Query(40.7357, description="Map center latitude"),
    center_lon: float = Query(-74.1723, description="Map center longitude"),
    zoom_start: int = Query(12, description="Initial zoom level"),
):
    """Generate theft analysis interactive map."""
    try:
        theft_data = get_theft_data()
        mapper = get_gis_mapper()
        ensure_output_dir()

        # Load or generate theft data
        if file_path and Path(file_path).exists():
            data = theft_data.import_from_csv(file_path)
        else:
            # Generate sample data for demonstration
            data = theft_data.generate_sample_data(center_lat, center_lon, count=50)

        # Create interactive map
        popup_columns = [
            "vehicle_make",
            "vehicle_model",
            "vehicle_year",
            "vehicle_value",
        ]
        if "vehicle_type" in data.columns:
            popup_columns.append("vehicle_type")

        interactive_map = mapper.create_interactive_map(
            points_gdf=data,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom_start=zoom_start,
            popup_columns=popup_columns,
        )

        # Save map
        map_id = f"theft_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        map_path = _output_dir / f"{map_id}.html"
        interactive_map.save(str(map_path))

        return JSONResponse(
            {
                "map_id": map_id,
                "map_url": f"/maps/view/{map_id}",
                "point_count": len(data),
                "bounds": list(mapper.get_bounds(data)),
                "created_at": datetime.now().isoformat(),
                "data_source": "file" if file_path else "sample",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Theft map generation failed: {str(e)}"
        )


@app.get("/maps/suspects")
async def generate_suspect_map(
    risk_levels: str = Query(
        "High,Critical", description="Comma-separated risk levels"
    ),
    include_arrests: bool = Query(True, description="Include arrest locations"),
    zoom_start: int = Query(12, description="Initial zoom level"),
):
    """Generate suspect analysis interactive map."""
    try:
        suspect_data = get_suspect_data()
        mapper = get_gis_mapper()
        ensure_output_dir()

        # Parse risk levels
        risk_level_list = [level.strip() for level in risk_levels.split(",")]

        # Generate sample suspect data
        data = suspect_data.generate_sample_data(count=30)

        # Filter by risk levels
        filtered_data = data[data["risk_level"].isin(risk_level_list)]

        # Add color mapping for risk levels
        risk_colors = {
            "Low": "green",
            "Medium": "yellow",
            "High": "orange",
            "Critical": "red",
        }
        filtered_data = filtered_data.copy()
        filtered_data["marker_color"] = filtered_data["risk_level"].map(risk_colors)

        # Create map
        popup_columns = ["suspect_name", "risk_level", "last_known_address"]
        if include_arrests and "arrest_charges" in filtered_data.columns:
            popup_columns.append("arrest_charges")

        interactive_map = mapper.create_interactive_map(
            points_gdf=filtered_data, zoom_start=zoom_start, popup_columns=popup_columns
        )

        # Save map
        map_id = f"suspect_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        map_path = _output_dir / f"{map_id}.html"
        interactive_map.save(str(map_path))

        return JSONResponse(
            {
                "map_id": map_id,
                "map_url": f"/maps/view/{map_id}",
                "point_count": len(filtered_data),
                "risk_levels_included": risk_level_list,
                "bounds": list(mapper.get_bounds(filtered_data)),
                "created_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Suspect map generation failed: {str(e)}"
        )


@app.get("/maps/recovery")
async def generate_recovery_map(
    zoom_start: int = Query(12, description="Initial zoom level"),
    include_criminal_locations: bool = Query(
        True, description="Include criminal recovery locations"
    ),
):
    """Generate recovery analysis interactive map."""
    try:
        recovery_data = get_recovery_data()
        mapper = get_gis_mapper()
        ensure_output_dir()

        # Generate sample recovery data
        data = recovery_data.generate_sample_data(count=40)

        # Create map
        popup_columns = ["vehicle_id", "recovery_location", "condition_score"]
        if "days_stolen" in data.columns:
            popup_columns.append("days_stolen")

        interactive_map = mapper.create_interactive_map(
            points_gdf=data, zoom_start=zoom_start, popup_columns=popup_columns
        )

        # Save map
        map_id = f"recovery_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        map_path = _output_dir / f"{map_id}.html"
        interactive_map.save(str(map_path))

        return JSONResponse(
            {
                "map_id": map_id,
                "map_url": f"/maps/view/{map_id}",
                "point_count": len(data),
                "bounds": list(mapper.get_bounds(data)),
                "created_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recovery map generation failed: {str(e)}"
        )


@app.get("/maps/view/{map_id}", response_class=HTMLResponse)
async def view_map(map_id: str):
    """View a generated map by ID."""
    map_path = _output_dir / f"{map_id}.html"

    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map not found")

    return FileResponse(map_path, media_type="text/html")


@app.get("/maps/list")
async def list_maps():
    """List all available maps."""
    ensure_output_dir()

    maps = []
    for map_file in _output_dir.glob("*.html"):
        stat = map_file.stat()
        maps.append(
            {
                "map_id": map_file.stem,
                "filename": map_file.name,
                "size_kb": round(stat.st_size / 1024, 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "view_url": f"/maps/view/{map_file.stem}",
            }
        )

    # Sort by creation time (newest first)
    maps.sort(key=lambda x: x["created_at"], reverse=True)

    return {"maps": maps, "total_count": len(maps)}


@app.get("/maps/static/{map_id}")
async def generate_static_map(
    map_id: str,
    width: int = Query(800, description="Image width in pixels"),
    height: int = Query(600, description="Image height in pixels"),
    dpi: int = Query(150, description="Image DPI"),
):
    """Generate a static PNG version of a map."""
    try:
        # This is a placeholder for static map generation
        # You would need to recreate the map data and generate a static version
        mapper = get_gis_mapper()

        # For demonstration, create a simple static map
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.text(
            0.5,
            0.5,
            f"Static Map: {map_id}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )
        ax.set_title(f"Auto Thievia - {map_id}")

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="PNG", dpi=dpi, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close()

        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        return JSONResponse(
            {
                "map_id": map_id,
                "image_base64": img_base64,
                "format": "PNG",
                "width": width,
                "height": height,
                "dpi": dpi,
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Static map generation failed: {str(e)}"
        )


@app.get("/analysis/theft")
async def analyze_theft_patterns(
    file_path: Optional[str] = Query(None, description="Path to theft data CSV file"),
    start_date: Optional[str] = Query(
        None, description="Analysis start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = Query(None, description="Analysis end date (YYYY-MM-DD)"),
    center_lat: float = Query(40.7357, description="Analysis center latitude"),
    center_lon: float = Query(-74.1723, description="Analysis center longitude"),
    radius_km: float = Query(10.0, description="Analysis radius in kilometers"),
):
    """Perform theft pattern analysis."""
    try:
        theft_data = get_theft_data()
        learning_models = get_learning_models()

        # Generate or load data
        if file_path and Path(file_path).exists():
            data = theft_data.import_from_csv(file_path)
        else:
            data = theft_data.generate_sample_data(center_lat, center_lon, count=100)

        # Perform analysis
        stats = theft_data.get_data_statistics()

        # Basic clustering analysis (simplified)
        try:
            clusters = learning_models.theft_hotspot_detection(data)
            cluster_info = {
                "cluster_count": int(clusters.max() + 1),
                "clustered_points": int((clusters >= 0).sum()),
                "noise_points": int((clusters == -1).sum()),
            }
        except Exception:
            cluster_info = {"error": "Clustering analysis failed"}

        return JSONResponse(
            {
                "analysis_type": "theft_patterns",
                "data_summary": {
                    "total_incidents": len(data),
                    "date_range": f"{start_date or 'N/A'} to {end_date or 'N/A'}",
                    "analysis_center": [center_lat, center_lon],
                    "radius_km": radius_km,
                },
                "statistics": stats,
                "clustering": cluster_info,
                "generated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Theft analysis failed: {str(e)}")


@app.get("/analysis/suspects")
async def analyze_suspects(
    risk_levels: List[str] = Query(
        ["High", "Critical"], description="Risk levels to include"
    ),
    include_arrests: bool = Query(True, description="Include arrest location data"),
    days_threshold: int = Query(30, description="Recent activity threshold in days"),
):
    """Perform suspect analysis."""
    try:
        suspect_data = get_suspect_data()

        # Generate sample data
        data = suspect_data.generate_sample_data(count=50)

        # Filter by risk levels
        filtered_data = data[data["risk_level"].isin(risk_levels)]

        # Get statistics
        stats = suspect_data.get_data_statistics()

        # Risk level distribution
        risk_distribution = filtered_data["risk_level"].value_counts().to_dict()

        return JSONResponse(
            {
                "analysis_type": "suspect_analysis",
                "parameters": {
                    "risk_levels": risk_levels,
                    "include_arrests": include_arrests,
                    "days_threshold": days_threshold,
                },
                "summary": {
                    "total_suspects": len(data),
                    "filtered_suspects": len(filtered_data),
                    "risk_distribution": risk_distribution,
                },
                "statistics": stats,
                "generated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Suspect analysis failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "gis_mapper": "available",
            "theft_data": "available",
            "suspect_data": "available",
            "recovery_data": "available",
            "learning_models": "available",
        },
    }


# Mount static files
app.mount("/static", StaticFiles(directory=str(_output_dir)), name="static")


def main():
    """Run the API server."""
    print("🚗 Starting Auto Thievia API Server...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🗺️  Interactive API: http://localhost:8000/redoc")
    print("🌐 Main Interface: http://localhost:8000/")
    print("-" * 50)

    # Ensure output directory exists
    ensure_output_dir()

    # Run server
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
