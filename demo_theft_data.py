#!/usr/bin/env python3
"""
Demonstration script for TheftData and TheftDataPersistence classes.

This script shows how to:
1. Create sample theft data
2. Import data from CSV
3. Persist data to DuckDB
4. Query and analyze data
5. Integrate with GISMapper for visualization
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_thievia import TheftData, TheftDataPersistence, GISMapper


def main():
    """Main demonstration function."""
    print("=== Auto Thievia - Theft Data Demo ===\n")

    # Initialize theft data manager
    print("1. Creating TheftData instance...")
    theft_manager = TheftData()

    # Create sample data
    print("2. Generating sample theft data...")
    sample_data = theft_manager.create_sample_data(num_records=50)
    print(f"   Created {len(sample_data)} sample theft records")
    print(f"   Data columns: {list(sample_data.columns)}")

    # Display summary statistics
    print("\n3. Summary statistics of sample data:")
    stats = theft_manager.get_summary_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Save sample data to CSV for demonstration
    print("\n4. Saving sample data to CSV...")
    csv_path = Path("data/sample_theft_data.csv")
    csv_path.parent.mkdir(exist_ok=True)

    # Convert to regular DataFrame for CSV export (drop geometry)
    df_for_csv = sample_data.drop(columns=["geometry"])
    df_for_csv.to_csv(csv_path, index=False)
    print(f"   Saved to: {csv_path}")

    # Demonstrate CSV import
    print("\n5. Importing data from CSV...")
    theft_manager_new = TheftData()
    imported_data = theft_manager_new.import_from_csv(csv_path)
    print(f"   Imported {len(imported_data)} records")

    # Demonstrate filtering
    print("\n6. Filtering high-value thefts...")
    high_value_thefts = theft_manager_new.filter_by_criteria(
        min_value=40000, vehicle_types=["sedan", "suv"]
    )
    print(f"   Found {len(high_value_thefts)} high-value thefts")

    # Demonstrate database persistence
    print("\n7. Testing database persistence...")
    with TheftDataPersistence(":memory:") as db:
        # Save data to database
        print("   Saving data to DuckDB...")
        db.save_theft_data(imported_data)

        # Load data back
        print("   Loading data from DuckDB...")
        loaded_data = db.load_theft_data()
        print(f"   Loaded {len(loaded_data)} records")

        # Get database statistics
        print("   Database statistics:")
        db_stats = db.get_statistics()
        for key, value in db_stats.items():
            print(f"     {key}: {value}")

        # Demonstrate spatial query
        print("   Performing spatial query...")
        # Use Newark center coordinates
        newark_lat, newark_lon = 40.7282, -74.2090
        nearby_thefts = db.query_by_distance(newark_lat, newark_lon, radius_km=2.0)
        print(f"   Found {len(nearby_thefts)} thefts within 2km of Newark center")

    # Integrate with GIS mapping
    print("\n8. Creating visualizations with GISMapper...")
    gis_mapper = GISMapper()

    # Create static map
    print("   Creating static map...")
    static_map_path = Path("data/output/theft_data_static_map.png")
    static_map_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = gis_mapper.plot_static_map(
        points_gdf=imported_data,
        title="Auto Theft Incidents - Newark Area",
        point_color="red",
        point_size=20,
    )
    fig.savefig(str(static_map_path), dpi=150, bbox_inches="tight")
    print(f"   Static map saved to: {static_map_path}")

    # Create interactive map
    print("   Creating interactive map...")
    interactive_map_path = Path("data/output/theft_data_interactive_map.html")

    # Add popup columns for interactive display
    popup_columns = [
        "vehicle_make",
        "vehicle_model",
        "vehicle_year",
        "vehicle_value",
        "vehicle_type",
    ]

    interactive_map = gis_mapper.create_interactive_map(
        points_gdf=imported_data,
        center_lat=newark_lat,
        center_lon=newark_lon,
        zoom_start=12,
        popup_columns=popup_columns,
    )
    interactive_map.save(str(interactive_map_path))
    print(f"   Interactive map saved to: {interactive_map_path}")

    print("\n=== Demo Complete ===")
    print("Check the 'data/output/' directory for generated maps and data files.")
    print(f"Total theft incidents processed: {len(imported_data)}")
    print(
        f"Total value of stolen vehicles: ${imported_data['vehicle_value'].sum():,.0f}"
    )


if __name__ == "__main__":
    main()
