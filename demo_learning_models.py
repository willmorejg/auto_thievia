"""
Demo script for LearningClass - Machine Learning Auto Theft Analysis.

This script demonstrates the machine learning capabilities of the auto_thievia package,
including theft hotspot detection, suspect activity clustering, and criminal activity prediction.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auto_thievia import GISMapper, LearningClass, RecoveryData, SuspectData, TheftData


def main():
    """Demonstrate LearningClass functionality."""
    print("=" * 60)
    print("Auto Theft Investigation - Machine Learning Demo")
    print("=" * 60)

    # Initialize data classes
    theft_data_class = TheftData()
    suspect_data_class = SuspectData()
    recovery_data_class = RecoveryData()

    # Create sample data
    print("\n1. Creating sample data...")
    theft_gdf = theft_data_class.create_sample_data(50)
    suspect_gdf = suspect_data_class.create_sample_data(30)
    recovery_gdf = recovery_data_class.create_sample_data(40)

    print(f"   - Created {len(theft_gdf)} theft records")
    print(f"   - Created {len(suspect_gdf)} suspect records")
    print(f"   - Created {len(recovery_gdf)} recovery records")

    # Initialize learning class
    print("\n2. Initializing machine learning models...")
    learning_class = LearningClass(random_state=42)

    # Load data into learning class
    learning_class.load_data(
        theft_data=theft_gdf, suspect_data=suspect_gdf, recovery_data=recovery_gdf
    )

    # Train theft location clustering
    print("\n3. Training theft location clustering models...")
    theft_results = learning_class.train_theft_location_clusters(
        eps=0.008, min_samples=3
    )

    print("   Theft clustering results:")
    print(f"   - Hotspots detected: {theft_results['n_hotspots']}")
    print(f"   - Noise ratio: {theft_results['noise_ratio']:.3f}")
    print(f"   - Silhouette score: {theft_results['silhouette_score']:.3f}")

    # Train suspect location clustering
    print("\n4. Training suspect activity clustering...")
    suspect_results = learning_class.train_suspect_location_clusters(
        eps=0.008, min_samples=2
    )

    print("   Suspect clustering results:")
    print(f"   - Activity clusters: {suspect_results['n_clusters']}")
    print(f"   - Noise ratio: {suspect_results['noise_ratio']:.3f}")
    print(f"   - Silhouette score: {suspect_results['silhouette_score']:.3f}")

    # Train criminal activity prediction
    print("\n5. Training criminal activity prediction model...")
    criminal_results = learning_class.train_criminal_activity_predictor()

    if criminal_results.get("status") != "insufficient_data":
        print("   Criminal activity prediction results:")
        print(f"   - MSE: {criminal_results['mse']:.4f}")
        print(
            f"   - Criminal locations used: {criminal_results['n_criminal_locations']}"
        )
        print(f"   - Training samples: {criminal_results['n_training_samples']}")
    else:
        print("   Insufficient criminal location data for training")

    # Train anomaly detection
    print("\n6. Training anomaly detection models...")
    anomaly_results = learning_class.train_anomaly_detectors()

    print("   Anomaly detection results:")
    print(f"   - Theft anomalies detected: {anomaly_results.get('theft_anomalies', 0)}")
    print(
        f"   - Suspect anomalies detected: {anomaly_results.get('suspect_anomalies', 0)}"
    )

    # Generate predictions
    print("\n7. Generating theft risk area predictions...")

    # Define area of interest (Newark area)
    newark_bounds = (-74.25, 40.70, -74.15, 40.80)

    theft_risk_areas = learning_class.predict_theft_risk_areas(
        grid_size=0.005, bounds=newark_bounds  # Smaller grid for more detail
    )

    print(f"   - Generated risk predictions for {len(theft_risk_areas)} grid cells")
    print(
        f"   - Risk score range: {theft_risk_areas['risk_score'].min():.4f} to {theft_risk_areas['risk_score'].max():.4f}"
    )

    # Generate suspect activity predictions
    print("\n8. Generating suspect activity area predictions...")

    suspect_activity_areas = learning_class.predict_suspect_activity_areas(
        grid_size=0.005, bounds=newark_bounds
    )

    print(
        f"   - Generated activity predictions for {len(suspect_activity_areas)} grid cells"
    )
    print(
        f"   - Activity score range: {suspect_activity_areas['activity_score'].min():.4f} to {suspect_activity_areas['activity_score'].max():.4f}"
    )

    # Criminal activity risk prediction
    high_risk_count = 0
    theft_gdf_with_risk = None

    if criminal_results.get("status") != "insufficient_data":
        print("\n9. Predicting criminal activity risk for theft locations...")

        theft_locations = theft_gdf[["geometry"]].copy()
        risk_scores = learning_class.predict_criminal_activity_risk(theft_locations)

        print(f"   - Predicted risk for {len(risk_scores)} theft locations")
        print(
            f"   - Risk score range: {min(risk_scores):.4f} to {max(risk_scores):.4f}"
        )

        # Add risk scores to theft data
        theft_gdf_with_risk = theft_gdf.copy()
        theft_gdf_with_risk["criminal_risk_score"] = risk_scores

        # Show high-risk locations
        high_risk_threshold = sorted(risk_scores, reverse=True)[
            min(5, len(risk_scores) - 1)
        ]
        high_risk_count = sum(
            1 for score in risk_scores if score >= high_risk_threshold
        )
        print(
            f"   - High-risk locations (score >= {high_risk_threshold:.4f}): {high_risk_count}"
        )

    # Model summary
    print("\n10. Model summary...")
    summary = learning_class.get_model_summary()

    print("   Trained models:")
    for model_name in summary["trained_models"]:
        print(f"   - {model_name}")

    print(f"\n   Available scalers: {len(summary['available_scalers'])}")
    print(f"   Available encoders: {len(summary['available_encoders'])}")

    if summary["clustering_results"]:
        print("   Clustering results:")
        if "theft" in summary["clustering_results"]:
            theft_cluster_info = summary["clustering_results"]["theft"]
            print(
                f"   - Theft: {theft_cluster_info['n_hotspots']} hotspots, {theft_cluster_info['n_kmeans_clusters']} k-means clusters"
            )

        if "suspect" in summary["clustering_results"]:
            suspect_cluster_info = summary["clustering_results"]["suspect"]
            print(
                f"   - Suspect: {suspect_cluster_info['n_clusters']} activity clusters"
            )

    # Save models
    print("\n11. Saving models...")
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "ml_models"
    learning_class.save_models(models_dir)
    print(f"   - Models saved to: {models_dir}")

    # Create visualizations
    print("\n12. Creating visualizations...")
    gis_mapper = GISMapper()

    # Visualize theft risk areas
    high_risk_areas = theft_risk_areas[
        theft_risk_areas["risk_score"] > theft_risk_areas["risk_score"].quantile(0.8)
    ]

    if len(high_risk_areas) > 0:
        theft_risk_map = gis_mapper.create_interactive_map(
            high_risk_areas,
            popup_columns=["risk_score"],
        )
        output_file = output_dir / "theft_risk_areas_map.html"
        theft_risk_map.save(str(output_file))
        print(f"   - Theft risk areas map: {output_file}")

    # Visualize suspect activity areas
    high_activity_areas = suspect_activity_areas[
        suspect_activity_areas["activity_score"]
        > suspect_activity_areas["activity_score"].quantile(0.8)
    ]

    if len(high_activity_areas) > 0:
        suspect_activity_map = gis_mapper.create_interactive_map(
            high_activity_areas,
            popup_columns=["activity_score"],
        )
        output_file = output_dir / "suspect_activity_areas_map.html"
        suspect_activity_map.save(str(output_file))
        print(f"   - Suspect activity areas map: {output_file}")

    # Visualize combined theft and risk data
    if theft_gdf_with_risk is not None:
        theft_risk_map = gis_mapper.create_interactive_map(
            theft_gdf_with_risk,
            popup_columns=[
                "incident_id",
                "vehicle_make",
                "vehicle_type",
                "criminal_risk_score",
            ],
        )
        output_file = output_dir / "theft_with_criminal_risk_map.html"
        theft_risk_map.save(str(output_file))
        print(f"   - Theft with criminal risk map: {output_file}")

    # Test model loading
    print("\n13. Testing model persistence...")
    new_learning_class = LearningClass(random_state=42)
    new_learning_class.load_models(models_dir)

    new_summary = new_learning_class.get_model_summary()
    print(f"   - Loaded {len(new_summary['trained_models'])} models successfully")

    print("\n" + "=" * 60)
    print("Machine Learning Demo Complete!")
    print("=" * 60)

    print("\nKey Insights:")
    print(
        f"- Identified {theft_results['n_hotspots']} theft hotspots from {len(theft_gdf)} incidents"
    )
    print(
        f"- Detected {suspect_results['n_clusters']} suspect activity clusters from {len(suspect_gdf)} suspects"
    )
    print(f"- Found {anomaly_results.get('theft_anomalies', 0)} unusual theft patterns")
    print(
        f"- Found {anomaly_results.get('suspect_anomalies', 0)} unusual suspect behaviors"
    )

    if criminal_results.get("status") != "insufficient_data":
        print(
            f"- Trained criminal activity predictor with {criminal_results['n_criminal_locations']} criminal locations"
        )
        print(f"- Identified {high_risk_count} high-risk theft locations")

    print(f"\nOutput files created in: {output_dir}")
    print("- ML models saved for future use")
    print("- Interactive maps generated for visualization")

    print("\nSelf-supervised learning techniques used:")
    print("- DBSCAN clustering for hotspot detection")
    print("- K-means clustering for pattern discovery")
    print("- Isolation Forest for anomaly detection")
    print("- Random Forest with proximity-based labels for risk prediction")
    print("- Nearest neighbors for criminal activity proximity scoring")


if __name__ == "__main__":
    main()
