#!/usr/bin/env python3
"""
Demonstration script for the GISMapper class.

This script shows how to use the GISMapper class for auto theft analysis.
"""

import sys
from pathlib import Path

# Add src to path so we can import auto_thievia
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    import pandas as pd

    from auto_thievia import GISMapper

    def demo_basic_functionality():
        """Demonstrate basic GISMapper functionality."""
        print("🗺️  GISMapper Demo - Basic Functionality")
        print("=" * 45)

        # Initialize mapper
        mapper = GISMapper()
        print("✅ GISMapper initialized")

        # Sample auto theft coordinates (Newark area, NJ)
        theft_coordinates = [
            (-74.1723, 40.7357),  # Newark, NJ
            (-74.2107, 40.6640),  # Elizabeth, NJ
            (-74.2632, 40.6976),  # Union, NJ
            (-74.1547, 40.7282),  # East Newark, NJ
            (-74.2090, 40.6581),  # Hillside, NJ
        ]

        # Sample theft attributes
        theft_attributes = {
            "incident_id": ["NWK001", "ELZ001", "UNI001", "ENK001", "HIL001"],
            "date": [
                "2024-01-15",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
            ],
            "vehicle_type": ["sedan", "suv", "truck", "sedan", "coupe"],
            "value_stolen": [25000, 45000, 35000, 20000, 30000],
        }

        # Create points from coordinates
        print(f"📍 Creating {len(theft_coordinates)} theft incident points...")
        theft_gdf = mapper.create_points_from_coordinates(
            theft_coordinates, theft_attributes
        )
        print(f"✅ Created GeoDataFrame with {len(theft_gdf)} points")

        # Display basic info
        print(f"📊 CRS: {theft_gdf.crs}")
        print(f"📊 Columns: {list(theft_gdf.columns)}")

        # Get bounds
        bounds = mapper.get_bounds()
        print(f"🗺️  Bounding box: {bounds}")

        # Create interactive map
        print("🌐 Creating interactive map...")
        interactive_map = mapper.create_interactive_map(
            popup_columns=["incident_id", "vehicle_type", "value_stolen"]
        )

        # Save map to file
        output_dir = Path("tests/data/output")
        output_dir.mkdir(exist_ok=True)
        map_file = output_dir / "demo_theft_map.html"
        interactive_map.save(str(map_file))
        print(f"💾 Interactive map saved to: {map_file}")

        # Save GeoDataFrame
        shapefile_path = output_dir / "demo_theft_points.shp"
        mapper.save_to_file(theft_gdf, shapefile_path)
        print(f"💾 Shapefile saved to: {shapefile_path}")

        return theft_gdf, interactive_map

    def demo_dataframe_workflow():
        """Demonstrate DataFrame-based workflow."""
        print("\n📊 GISMapper Demo - DataFrame Workflow")
        print("=" * 45)

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "longitude": [-74.1723, -74.2107, -74.2632, -74.1547],
                "latitude": [40.7357, 40.6640, 40.6976, 40.7282],
                "city": ["Newark", "Elizabeth", "Union", "East Newark"],
                "theft_count": [145, 89, 67, 34],
                "recovery_rate": [0.23, 0.31, 0.28, 0.35],
            }
        )

        print(f"📋 Sample DataFrame with {len(df)} rows")
        print(df.to_string(index=False))

        # Initialize mapper and create points
        mapper = GISMapper()
        gdf = mapper.create_points_from_dataframe(df)

        print(f"\n✅ Converted to GeoDataFrame with {len(gdf)} points")
        print(f"📊 Columns: {list(gdf.columns)}")

        return gdf

    def main():
        """Run all demonstrations."""
        try:
            print("🚗 Auto Thievia GIS Mapper Demonstration")
            print("🎯 Analyzing auto theft patterns in Newark, NJ area")
            print("=" * 55)

            # Run basic demo
            _, _ = demo_basic_functionality()

            # Run DataFrame demo
            demo_dataframe_workflow()

            print("\n🎉 Demo completed successfully!")
            print("📁 Check the tests/data/output/ directory for generated files:")
            print("   - demo_theft_map.html (interactive map)")
            print("   - demo_theft_points.shp (shapefile)")

            return 0

        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure all dependencies are installed:")
            print(
                "   pip install geopandas pandas matplotlib folium shapely contextily"
            )
            return 1
        except Exception as e:
            print(f"❌ Error during demo: {e}")
            return 1

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("💡 Make sure you're in the project directory and dependencies are installed")
    sys.exit(1)
