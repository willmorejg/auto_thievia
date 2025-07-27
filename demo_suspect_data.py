"""
Demo script for SuspectData functionality.

This script demonstrates how to use the SuspectData class to:
1. Create sample suspect data
2. Export data to CSV
3. Import data from CSV
4. Filter and analyze suspect data
5. Generate visualizations with GIS mapping
"""

from pathlib import Path
import sys

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from auto_thievia.suspect_data import SuspectData, SuspectDataPersistence
from auto_thievia.gis_mapper import GISMapper
import matplotlib.pyplot as plt


def main():
    """Run the SuspectData demo."""
    print("=== SuspectData Demo ===\n")

    # Initialize the SuspectData class
    suspect_manager = SuspectData()

    # Create sample data
    print("1. Creating sample suspect data...")
    sample_data = suspect_manager.create_sample_data(num_records=75)
    print(f"Created {len(sample_data)} suspect records")
    print(f"Columns: {list(sample_data.columns)}")

    # Display sample records
    print("\n2. Sample suspect records:")
    print(
        sample_data[
            ["suspect_name", "suspect_age", "arrest_charges", "risk_level"]
        ].head()
    )

    # Export to CSV
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sample_suspect_data.csv"

    print(f"\n3. Exporting data to CSV: {csv_path}")
    # Convert to regular DataFrame for CSV export (remove geometry)
    export_df = sample_data.drop(columns="geometry")
    export_df.to_csv(csv_path, index=False)
    print("Export completed")

    # Import from CSV
    print("4. Importing data from CSV...")
    imported_data = suspect_manager.import_from_csv(csv_path)
    print(f"Imported {len(imported_data)} suspect records")

    # Get summary statistics
    print("\n5. Summary Statistics:")
    stats = suspect_manager.get_summary_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    # Filter data examples
    print("\n6. Data Filtering Examples:")

    # High risk suspects
    high_risk = suspect_manager.get_high_risk_suspects()
    print(f"  High risk suspects: {len(high_risk)}")

    # Repeat offenders
    repeat_offenders = suspect_manager.get_repeat_offenders()
    print(f"  Repeat offenders: {len(repeat_offenders)}")

    # Recent arrests (last 6 months)
    recent_arrests = suspect_manager.filter_by_criteria(
        recent_arrests_only=True, days_threshold=180
    )
    print(f"  Recent arrests (6 months): {len(recent_arrests)}")

    # Young adults (18-25)
    young_adults = suspect_manager.filter_by_criteria(min_age=18, max_age=25)
    print(f"  Young adult suspects (18-25): {len(young_adults)}")

    # Suspects near a specific location (Newark center)
    newark_center_lat, newark_center_lon = 40.7357, -74.1724
    nearby_suspects = suspect_manager.get_suspects_by_distance(
        newark_center_lat, newark_center_lon, radius_km=2.0, location_type="address"
    )
    print(f"  Suspects within 2km of Newark center: {len(nearby_suspects)}")

    # Generate GIS visualization
    print("\n7. Creating GIS visualization...")
    try:
        gis_mapper = GISMapper()

        # Create map with suspect locations (addresses)
        map_path = output_dir / "suspect_analysis_map.html"

        # Prepare data for mapping - use different colors for risk levels
        risk_colors = {
            "Low": "green",
            "Medium": "yellow",
            "High": "orange",
            "Critical": "red",
        }

        # Add color column based on risk level
        suspect_data_for_map = imported_data.copy()
        suspect_data_for_map["marker_color"] = suspect_data_for_map["risk_level"].map(
            risk_colors
        )
        suspect_data_for_map["marker_color"].fillna("blue", inplace=True)

        # Create the map
        folium_map = gis_mapper.create_interactive_map(
            points_gdf=suspect_data_for_map,
            popup_columns=["suspect_name", "risk_level", "arrest_charges"],
        )
        folium_map.save(str(map_path))
        print(f"Interactive map saved to: {map_path}")

        # Create a second map with arrest locations
        arrest_gdf = suspect_manager.create_arrest_points_gdf()
        arrest_gdf["marker_color"] = arrest_gdf["risk_level"].map(risk_colors)
        arrest_gdf["marker_color"].fillna("blue", inplace=True)

        arrest_map_path = output_dir / "suspect_arrest_locations_map.html"
        arrest_map = gis_mapper.create_interactive_map(
            points_gdf=arrest_gdf,
            popup_columns=["suspect_name", "arrest_location", "arrest_charges"],
        )
        arrest_map.save(str(arrest_map_path))
        print(f"Arrest locations map saved to: {arrest_map_path}")

        # Create static map for high-risk suspects only
        static_map_path = output_dir / "high_risk_suspects_map.png"
        fig, _ = gis_mapper.plot_static_map(
            points_gdf=high_risk,
            title="High Risk Suspects",
            point_color="red",
            point_size=100,
        )
        fig.savefig(static_map_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Static map of high-risk suspects saved to: {static_map_path}")

    except Exception as e:
        print(f"Error creating GIS visualization: {e}")
        print(
            "Note: This might be due to missing contextily or other mapping dependencies"
        )

    # Test DuckDB persistence functionality
    print("\n8. Testing DuckDB Persistence...")
    try:
        # Create database file path
        db_path = output_dir / "suspect_database.db"

        # Initialize persistence manager
        with SuspectDataPersistence(db_path) as db_manager:
            # Save suspect data to database
            print(f"Saving {len(imported_data)} suspects to database...")
            db_manager.save_suspect_data(imported_data)

            # Load data back from database
            print("Loading data from database...")
            loaded_data = db_manager.load_suspect_data()
            print(f"Loaded {len(loaded_data)} suspects from database")

            # Test distance queries (by address)
            newark_lat, newark_lon = 40.7357, -74.1724
            nearby_suspects_db = db_manager.query_suspects_by_address_distance(
                newark_lat, newark_lon, radius_km=2.0
            )
            print(
                f"Found {len(nearby_suspects_db)} suspects within 2km of Newark center (by address)"
            )

            # Test distance queries (by arrest location)
            arrest_nearby_db = db_manager.query_suspects_by_arrest_distance(
                newark_lat, newark_lon, radius_km=3.0
            )
            print(
                f"Found {len(arrest_nearby_db)} suspects within 3km of Newark center (by arrest location)"
            )

            # Test criteria-based queries
            high_risk_db = db_manager.query_suspects_by_criteria(
                risk_levels=["High", "Critical"]
            )
            print(f"Found {len(high_risk_db)} high-risk suspects in database")

            young_males_db = db_manager.query_suspects_by_criteria(
                min_age=18, max_age=30, gender="Male"
            )
            print(
                f"Found {len(young_males_db)} young male suspects (18-30) in database"
            )

            # Get database statistics
            db_stats = db_manager.get_suspect_statistics()
            print("Database statistics:")
            for key, value in db_stats.items():
                print(f"  {key}: {value}")

        print(f"Database saved to: {db_path}")

    except Exception as e:
        print(f"Error with DuckDB persistence: {e}")
        import traceback

        traceback.print_exc()

    # Analysis insights
    print("\n9. Analysis Insights:")

    # Age analysis
    if "suspect_age" in imported_data.columns:
        age_stats = imported_data["suspect_age"].describe()
        print("  Age statistics:")
        print(f"    Average age: {age_stats['mean']:.1f} years")
        print(f"    Age range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")

    # Risk level analysis
    if "risk_level" in imported_data.columns:
        risk_counts = imported_data["risk_level"].value_counts()
        total = len(imported_data)
        print("  Risk level distribution:")
        for risk, count in risk_counts.items():
            percentage = (count / total) * 100
            print(f"    {risk}: {count} ({percentage:.1f}%)")

    # Charge analysis
    if "arrest_charges" in imported_data.columns:
        charge_counts = imported_data["arrest_charges"].value_counts()
        print("  Most common charges:")
        for charge, count in charge_counts.head(5).items():
            print(f"    {charge}: {count}")

    # Geographic concentration
    print("  Geographic spread:")
    bounds = imported_data.bounds
    lat_range = bounds.maxy.max() - bounds.miny.min()
    lon_range = bounds.maxx.max() - bounds.minx.min()
    print(f"    Latitude range: {lat_range:.4f}° ({lat_range * 111:.1f} km)")
    print(f"    Longitude range: {lon_range:.4f}° ({lon_range * 111:.1f} km)")

    print("\n=== Demo completed successfully! ===")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
