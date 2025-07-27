#!/usr/bin/env python3
"""
Demonstration script for RecoveryData and RecoveryDataPersistence classes.

This script shows how to:
1. Create sample recovery data
2. Import data from CSV
3. Persist data to DuckDB
4. Query and analyze recovery data
5. Show relationships to theft and suspect data
6. Integrate with GISMapper for visualization
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_thievia import RecoveryData, RecoveryDataPersistence, GISMapper


def main():
    """Main demonstration function."""
    print("=== Auto Thievia - Recovery Data Demo ===\n")

    # Initialize recovery data manager
    print("1. Creating RecoveryData instance...")
    recovery_manager = RecoveryData()

    # Create sample data
    print("2. Generating sample recovery data...")
    sample_data = recovery_manager.create_sample_data(num_records=75)
    print(f"   Created {len(sample_data)} sample recovery records")
    print(f"   Data columns: {list(sample_data.columns)}")

    # Display summary statistics
    print("\n3. Summary statistics of sample data:")
    stats = recovery_manager.get_summary_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Show criminal location breakdown
    print("\n4. Criminal location breakdown:")
    location_breakdown = recovery_manager.get_recoveries_by_location_type()
    for location_type, count in location_breakdown.items():
        print(f"   {location_type}: {count} recoveries")

    # Save sample data to CSV for demonstration
    print("\n5. Saving sample data to CSV...")
    csv_path = Path("data/output/sample_recovery_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to regular DataFrame for CSV export (drop geometry)
    df_for_csv = sample_data.drop(columns=["geometry"])
    df_for_csv.to_csv(csv_path, index=False)
    print(f"   Saved to: {csv_path}")

    # Re-import from CSV to demonstrate CSV functionality
    print("\n6. Re-importing data from CSV...")
    imported_data = recovery_manager.import_from_csv(csv_path)
    print(f"   Imported {len(imported_data)} records")

    # Test filtering capabilities
    print("\n7. Testing data filtering...")

    # Filter by criminal locations
    criminal_recoveries = recovery_manager.filter_by_criteria(
        criminal_locations_only=True
    )
    print(f"   Recoveries from criminal locations: {len(criminal_recoveries)}")

    # Filter by vehicle condition
    poor_condition = recovery_manager.filter_by_criteria(
        vehicle_conditions=["Poor", "Stripped", "Burned"]
    )
    print(f"   Recoveries with poor vehicle condition: {len(poor_condition)}")

    # Filter by relationships
    with_theft_links = recovery_manager.filter_by_criteria(with_theft_link=True)
    with_suspect_links = recovery_manager.filter_by_criteria(with_suspect_link=True)
    print(f"   Recoveries linked to theft data: {len(with_theft_links)}")
    print(f"   Recoveries linked to suspect data: {len(with_suspect_links)}")

    # Test database persistence
    print("\n8. Testing database persistence...")
    db_path = Path("data/output/recovery_database.db")

    with RecoveryDataPersistence(db_path) as db:
        print("   Saving data to DuckDB...")
        db.save_recovery_data(imported_data)

        print("   Loading data from DuckDB...")
        loaded_data = db.load_recovery_data()
        print(f"   Loaded {len(loaded_data)} records")

        # Get database statistics
        print("   Database statistics:")
        db_stats = db.get_statistics()
        for key, value in db_stats.items():
            print(f"     {key}: {value}")

        # Demonstrate spatial query
        print("   Performing spatial query...")
        newark_lat, newark_lon = 40.7282, -74.2090
        nearby_recoveries = db.query_by_distance(newark_lat, newark_lon, radius_km=5.0)
        print(
            f"   Found {len(nearby_recoveries)} recoveries within 5km of Newark center"
        )

        # Query recoveries with relationships
        print("   Querying recoveries with relationships...")
        with_relationships = db.get_recoveries_with_relationships()
        print(f"   Found {len(with_relationships)} recoveries with theft/suspect links")

        # Query criminal location recoveries
        print("   Querying criminal location recoveries...")
        criminal_db_recoveries = db.get_criminal_location_recoveries()
        print(
            f"   Found {len(criminal_db_recoveries)} recoveries from criminal locations"
        )

    # Integrate with GIS mapping
    print("\n9. Creating visualizations with GISMapper...")
    gis_mapper = GISMapper()

    # Create static map for all recoveries
    print("   Creating static map for all recoveries...")
    static_map_path = Path("data/output/recovery_data_static_map.png")
    static_map_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = gis_mapper.plot_static_map(
        points_gdf=imported_data,
        title="Auto Recovery Locations",
        figsize=(12, 10),
        point_color="red",
        point_size=15,
    )
    fig.savefig(static_map_path, dpi=300, bbox_inches="tight")
    print(f"   Static map saved to: {static_map_path}")

    # Create interactive map for criminal location recoveries
    print("   Creating interactive map for criminal location recoveries...")
    interactive_map_path = Path("data/output/criminal_recovery_locations_map.html")

    if len(criminal_recoveries) > 0:
        interactive_map = gis_mapper.create_interactive_map(
            points_gdf=criminal_recoveries,
            popup_columns=[
                "recovery_id",
                "recovery_location_name",
                "vehicle_condition",
                "criminal_location_type",
                "recovery_agency",
            ],
        )
        interactive_map.save(str(interactive_map_path))
        print(f"   Interactive map saved to: {interactive_map_path}")
    else:
        print("   No criminal location recoveries to map")

    # Create map showing recovery relationships
    print("   Creating map showing recoveries with data relationships...")
    relationship_map_path = Path("data/output/recovery_relationships_map.html")

    if len(with_relationships) > 0:
        # Add a column to show relationship type
        relationship_data = with_relationships.copy()
        relationship_data["relationship_type"] = relationship_data.apply(
            lambda row: (
                "Both"
                if pd.notna(row.get("incident_id")) and pd.notna(row.get("suspect_id"))
                else (
                    "Theft Only"
                    if pd.notna(row.get("incident_id"))
                    else "Suspect Only" if pd.notna(row.get("suspect_id")) else "None"
                )
            ),
            axis=1,
        )

        relationship_map = gis_mapper.create_interactive_map(
            points_gdf=relationship_data,
            popup_columns=[
                "recovery_id",
                "recovery_location_name",
                "relationship_type",
                "incident_id",
                "suspect_id",
                "recovery_agency",
            ],
        )
        relationship_map.save(str(relationship_map_path))
        print(f"   Relationship map saved to: {relationship_map_path}")
    else:
        print("   No recoveries with relationships to map")

    print("\n=== Demo Complete ===")
    print("Check the 'data/output/' directory for generated maps and data files.")
    print(f"Total recovery incidents processed: {len(imported_data)}")
    print(f"Criminal location recoveries: {len(criminal_recoveries)}")
    print(f"Recoveries with data relationships: {len(with_relationships)}")


if __name__ == "__main__":
    # Import pandas here since we use it in the demo
    import pandas as pd

    main()
