"""
Tests for the LearningClass module.

This module tests machine learning functionality for predicting theft patterns,
suspect locations, and criminal activity areas.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.auto_thievia.gis_mapper import WGS84_CRS
from src.auto_thievia.learning_models import LearningClass


class TestLearningClass:
    """Test class for LearningClass functionality."""

    @pytest.fixture
    def sample_theft_data(self):
        """Create sample theft data for testing."""
        rng = np.random.default_rng(42)

        data = {
            "incident_id": [f"T{i:03d}" for i in range(1, 21)],
            "incident_date": pd.date_range("2024-01-01", periods=20),
            "vehicle_make": ["Honda", "Toyota", "Ford", "Chevrolet"] * 5,
            "vehicle_type": ["Sedan", "SUV", "Truck", "Coupe"] * 5,
            "vehicle_year": rng.integers(2010, 2024, 20),
            "vehicle_value": rng.integers(10000, 50000, 20),
            "theft_method": ["Break-in", "Key theft", "Carjacking", "Joy ride"] * 5,
            "owner_income": rng.integers(30000, 100000, 20),
            "latitude": rng.uniform(40.7, 40.8, 20),
            "longitude": rng.uniform(-74.3, -74.1, 20),
        }

        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84_CRS)

        return gdf

    @pytest.fixture
    def sample_suspect_data(self):
        """Create sample suspect data for testing."""
        rng = np.random.default_rng(42)

        data = {
            "suspect_id": [f"S{i:03d}" for i in range(1, 16)],
            "name": [f"Suspect {i}" for i in range(1, 16)],
            "age": rng.integers(18, 65, 15),
            "arrest_date": pd.date_range("2024-01-01", periods=15),
            "previous_arrests": rng.integers(0, 10, 15),
            "occupation": ["Unemployed", "Mechanic", "Driver", "Student", "Other"] * 3,
            "criminal_associations": [
                "None",
                "Gang A",
                "Gang B",
                "Organized Crime",
                "Solo",
            ]
            * 3,
            "address_latitude": rng.uniform(40.7, 40.8, 15),
            "address_longitude": rng.uniform(-74.3, -74.1, 15),
        }

        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(df["address_longitude"], df["address_latitude"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84_CRS)

        return gdf

    @pytest.fixture
    def sample_recovery_data(self):
        """Create sample recovery data for testing."""
        rng = np.random.default_rng(42)

        data = {
            "recovery_id": [f"R{i:03d}" for i in range(1, 21)],
            "incident_id": [f"T{i:03d}" for i in range(1, 21)],  # Links to theft data
            "suspect_id": [
                f"S{i%15+1:03d}" for i in range(20)
            ],  # Links to suspect data
            "recovery_date": pd.date_range("2024-01-02", periods=20),
            "recovery_latitude": rng.uniform(40.7, 40.8, 20),
            "recovery_longitude": rng.uniform(-74.3, -74.1, 20),
            "vehicle_condition": ["Good", "Fair", "Poor", "Totaled"] * 5,
            "is_criminal_location": [True if i % 3 == 0 else False for i in range(20)],
            "criminal_location_type": [
                "Chop Shop" if i % 3 == 0 else None for i in range(20)
            ],
        }

        df = pd.DataFrame(data)
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(df["recovery_longitude"], df["recovery_latitude"])
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84_CRS)

        return gdf

    @pytest.fixture
    def learning_class(self):
        """Create LearningClass instance for testing."""
        return LearningClass(random_state=42)

    def test_init(self, learning_class):
        """Test LearningClass initialization."""
        assert learning_class.random_state == 42
        assert learning_class.models == {}
        assert learning_class.scalers == {}
        assert learning_class.encoders == {}
        assert learning_class.theft_data is None
        assert learning_class.suspect_data is None
        assert learning_class.recovery_data is None

    def test_load_data(
        self,
        learning_class,
        sample_theft_data,
        sample_suspect_data,
        sample_recovery_data,
    ):
        """Test data loading functionality."""
        learning_class.load_data(
            theft_data=sample_theft_data,
            suspect_data=sample_suspect_data,
            recovery_data=sample_recovery_data,
        )

        assert learning_class.theft_data is not None
        assert learning_class.suspect_data is not None
        assert learning_class.recovery_data is not None
        assert len(learning_class.theft_data) == 20
        assert len(learning_class.suspect_data) == 15
        assert len(learning_class.recovery_data) == 20

    def test_prepare_spatial_features(self, learning_class, sample_theft_data):
        """Test spatial feature preparation."""
        features = learning_class._prepare_spatial_features(sample_theft_data)

        # Should have coordinates plus time features
        assert features.shape[0] == len(sample_theft_data)
        assert features.shape[1] >= 2  # At least coordinates

        # Check that coordinates are within expected range
        assert np.all(features[:, 0] >= -74.3)  # longitude
        assert np.all(features[:, 0] <= -74.1)
        assert np.all(features[:, 1] >= 40.7)  # latitude
        assert np.all(features[:, 1] <= 40.8)

    def test_prepare_theft_features(self, learning_class, sample_theft_data):
        """Test theft feature preparation."""
        features = learning_class._prepare_theft_features(sample_theft_data)

        assert features.shape[0] == len(sample_theft_data)
        assert features.shape[1] > 2  # More than just coordinates

        # Should include vehicle features, categorical features, etc.
        assert not np.any(np.isnan(features))

    def test_prepare_suspect_features(self, learning_class, sample_suspect_data):
        """Test suspect feature preparation."""
        features = learning_class._prepare_suspect_features(sample_suspect_data)

        assert features.shape[0] == len(sample_suspect_data)
        assert features.shape[1] > 2  # More than just coordinates

        # Should include demographic and criminal history features
        assert not np.any(np.isnan(features))

    def test_train_theft_location_clusters(self, learning_class, sample_theft_data):
        """Test theft location clustering training."""
        learning_class.load_data(theft_data=sample_theft_data)

        results = learning_class.train_theft_location_clusters()

        assert "n_hotspots" in results
        assert "noise_ratio" in results
        assert "silhouette_score" in results
        assert "n_records" in results

        assert results["n_records"] == 20
        assert 0 <= results["noise_ratio"] <= 1

        # Check that models were stored
        assert "theft_dbscan" in learning_class.models
        assert "theft_kmeans" in learning_class.models
        assert "theft" in learning_class.scalers
        assert learning_class.theft_clusters is not None

    def test_train_suspect_location_clusters(self, learning_class, sample_suspect_data):
        """Test suspect location clustering training."""
        learning_class.load_data(suspect_data=sample_suspect_data)

        results = learning_class.train_suspect_location_clusters()

        assert "n_clusters" in results
        assert "noise_ratio" in results
        assert "silhouette_score" in results
        assert "n_records" in results

        assert results["n_records"] == 15
        assert 0 <= results["noise_ratio"] <= 1

        # Check that models were stored
        assert "suspect_dbscan" in learning_class.models
        assert "suspect" in learning_class.scalers
        assert learning_class.suspect_clusters is not None

    def test_train_criminal_activity_predictor(
        self, learning_class, sample_recovery_data
    ):
        """Test criminal activity prediction training."""
        learning_class.load_data(recovery_data=sample_recovery_data)

        results = learning_class.train_criminal_activity_predictor()

        if results.get("status") != "insufficient_data":
            assert "mse" in results
            assert "n_criminal_locations" in results
            assert "n_training_samples" in results
            assert "n_test_samples" in results

            # Check that models were stored
            assert learning_class.criminal_activity_predictor is not None
            assert "criminal_activity" in learning_class.scalers
            assert "criminal_activity_nn" in learning_class.models

    def test_train_anomaly_detectors(
        self, learning_class, sample_theft_data, sample_suspect_data
    ):
        """Test anomaly detection training."""
        learning_class.load_data(
            theft_data=sample_theft_data, suspect_data=sample_suspect_data
        )

        results = learning_class.train_anomaly_detectors()

        assert "theft_anomalies" in results
        assert "suspect_anomalies" in results

        # Check that models were stored
        assert learning_class.theft_anomaly_detector is not None
        assert learning_class.suspect_anomaly_detector is not None
        assert "theft_anomaly" in learning_class.scalers
        assert "suspect_anomaly" in learning_class.scalers

    def test_predict_theft_risk_areas(self, learning_class, sample_theft_data):
        """Test theft risk area prediction."""
        learning_class.load_data(theft_data=sample_theft_data)
        learning_class.train_theft_location_clusters()

        # Test with bounds
        bounds = (-74.25, 40.72, -74.20, 40.77)
        risk_gdf = learning_class.predict_theft_risk_areas(
            grid_size=0.01, bounds=bounds
        )

        assert isinstance(risk_gdf, gpd.GeoDataFrame)
        assert "risk_score" in risk_gdf.columns
        assert "geometry" in risk_gdf.columns
        assert len(risk_gdf) > 0
        assert risk_gdf.crs == WGS84_CRS

    def test_predict_suspect_activity_areas(self, learning_class, sample_suspect_data):
        """Test suspect activity area prediction."""
        learning_class.load_data(suspect_data=sample_suspect_data)
        learning_class.train_suspect_location_clusters()

        bounds = (-74.25, 40.72, -74.20, 40.77)
        activity_gdf = learning_class.predict_suspect_activity_areas(
            grid_size=0.01, bounds=bounds
        )

        assert isinstance(activity_gdf, gpd.GeoDataFrame)
        assert "activity_score" in activity_gdf.columns
        assert "geometry" in activity_gdf.columns
        assert len(activity_gdf) > 0
        assert activity_gdf.crs == WGS84_CRS

    def test_predict_criminal_activity_risk(self, learning_class, sample_recovery_data):
        """Test criminal activity risk prediction."""
        learning_class.load_data(recovery_data=sample_recovery_data)

        # Train the model first
        results = learning_class.train_criminal_activity_predictor()

        if results.get("status") != "insufficient_data":
            # Create test locations
            test_locations = gpd.GeoDataFrame(
                {"geometry": [Point(-74.2, 40.75), Point(-74.15, 40.77)]}, crs=WGS84_CRS
            )

            risk_scores = learning_class.predict_criminal_activity_risk(test_locations)

            assert len(risk_scores) == 2
            assert all(isinstance(score, (int, float)) for score in risk_scores)

    def test_get_model_summary(
        self, learning_class, sample_theft_data, sample_suspect_data
    ):
        """Test model summary generation."""
        learning_class.load_data(
            theft_data=sample_theft_data, suspect_data=sample_suspect_data
        )

        # Train some models
        learning_class.train_theft_location_clusters()
        learning_class.train_suspect_location_clusters()

        summary = learning_class.get_model_summary()

        assert "trained_models" in summary
        assert "available_scalers" in summary
        assert "available_encoders" in summary
        assert "data_loaded" in summary
        assert "clustering_results" in summary

        assert summary["data_loaded"]["theft_data"] == True
        assert summary["data_loaded"]["suspect_data"] == True
        assert len(summary["trained_models"]) > 0

    def test_save_and_load_models(self, learning_class, sample_theft_data):
        """Test model saving and loading."""
        learning_class.load_data(theft_data=sample_theft_data)
        learning_class.train_theft_location_clusters()

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "models"

            # Save models
            learning_class.save_models(save_dir)

            # Verify files were created
            assert (save_dir / "theft_dbscan_model.pkl").exists()
            assert (save_dir / "theft_kmeans_model.pkl").exists()
            assert (save_dir / "theft_scaler.pkl").exists()

            # Create new instance and load models
            new_learning_class = LearningClass(random_state=42)
            new_learning_class.load_models(save_dir)

            # Verify models were loaded
            assert "theft_dbscan" in new_learning_class.models
            assert "theft_kmeans" in new_learning_class.models
            assert "theft" in new_learning_class.scalers

    def test_error_handling(self, learning_class):
        """Test error handling for missing data."""
        # Test without loading data
        with pytest.raises(ValueError, match="No theft data loaded"):
            learning_class.train_theft_location_clusters()

        with pytest.raises(ValueError, match="No suspect data loaded"):
            learning_class.train_suspect_location_clusters()

        with pytest.raises(ValueError, match="No recovery data loaded"):
            learning_class.train_criminal_activity_predictor()

        # Test prediction without training
        with pytest.raises(ValueError, match="No theft clustering model trained"):
            learning_class.predict_theft_risk_areas()

        with pytest.raises(ValueError, match="No suspect clustering model trained"):
            learning_class.predict_suspect_activity_areas()

        with pytest.raises(ValueError, match="No criminal activity model trained"):
            test_locations = gpd.GeoDataFrame(
                {"geometry": [Point(-74.2, 40.75)]}, crs=WGS84_CRS
            )
            learning_class.predict_criminal_activity_risk(test_locations)


class TestLearningClassIntegration:
    """Integration tests for LearningClass."""

    @pytest.fixture
    def learning_class(self):
        """Create LearningClass instance for testing."""
        return LearningClass(random_state=42)

    @pytest.fixture
    def full_learning_setup(self, learning_class):
        """Set up learning class with all data types."""
        rng = np.random.default_rng(42)

        # Create comprehensive test data
        theft_data = gpd.GeoDataFrame(
            {
                "incident_id": [f"T{i:03d}" for i in range(1, 31)],
                "incident_date": pd.date_range("2024-01-01", periods=30),
                "vehicle_make": ["Honda", "Toyota", "Ford"] * 10,
                "vehicle_type": ["Sedan", "SUV"] * 15,
                "vehicle_year": rng.integers(2015, 2024, 30),
                "vehicle_value": rng.integers(15000, 45000, 30),
                "theft_method": ["Break-in", "Key theft"] * 15,
                "owner_income": rng.integers(40000, 90000, 30),
                "geometry": [
                    Point(rng.uniform(-74.25, -74.15), rng.uniform(40.7, 40.8))
                    for _ in range(30)
                ],
            },
            crs=WGS84_CRS,
        )

        suspect_data = gpd.GeoDataFrame(
            {
                "suspect_id": [f"S{i:03d}" for i in range(1, 21)],
                "name": [f"Suspect {i}" for i in range(1, 21)],
                "age": rng.integers(20, 60, 20),
                "arrest_date": pd.date_range("2024-01-01", periods=20),
                "previous_arrests": rng.integers(0, 8, 20),
                "occupation": (
                    ["Unemployed", "Mechanic", "Driver", "Student", "Other"] * 4
                )[:20],
                "criminal_associations": (
                    ["None", "Gang A", "Gang B", "Organized Crime", "Solo"] * 4
                )[:20],
                "geometry": [
                    Point(rng.uniform(-74.25, -74.15), rng.uniform(40.7, 40.8))
                    for _ in range(20)
                ],
            },
            crs=WGS84_CRS,
        )

        recovery_data = gpd.GeoDataFrame(
            {
                "recovery_id": [f"R{i:03d}" for i in range(1, 26)],
                "incident_id": [f"T{i:03d}" for i in range(1, 26)],
                "suspect_id": [f"S{i%20+1:03d}" for i in range(25)],
                "recovery_date": pd.date_range("2024-01-02", periods=25),
                "vehicle_condition": (
                    ["Good", "Fair", "Poor", "Totaled", "Excellent"] * 5
                )[:25],
                "is_criminal_location": [
                    True if i % 4 == 0 else False for i in range(25)
                ],
                "criminal_location_type": [
                    "Chop Shop" if i % 4 == 0 else None for i in range(25)
                ],
                "geometry": [
                    Point(rng.uniform(-74.25, -74.15), rng.uniform(40.7, 40.8))
                    for _ in range(25)
                ],
            },
            crs=WGS84_CRS,
        )

        learning_class.load_data(
            theft_data=theft_data,
            suspect_data=suspect_data,
            recovery_data=recovery_data,
        )

        return learning_class

    def test_full_training_pipeline(self, full_learning_setup):
        """Test complete training pipeline."""
        learning_class = full_learning_setup

        # Train all models
        theft_results = learning_class.train_theft_location_clusters()
        suspect_results = learning_class.train_suspect_location_clusters()
        learning_class.train_criminal_activity_predictor()
        anomaly_results = learning_class.train_anomaly_detectors()

        # Verify all training completed successfully
        assert theft_results["n_records"] == 30
        assert suspect_results["n_records"] == 20
        assert anomaly_results["theft_anomalies"] >= 0
        assert anomaly_results["suspect_anomalies"] >= 0

        # Verify models exist
        summary = learning_class.get_model_summary()
        assert len(summary["trained_models"]) >= 4

    def test_full_prediction_pipeline(self, full_learning_setup):
        """Test complete prediction pipeline."""
        learning_class = full_learning_setup

        # Train models
        learning_class.train_theft_location_clusters()
        learning_class.train_suspect_location_clusters()

        # Test predictions
        bounds = (-74.24, 40.71, -74.16, 74.79)

        theft_risk = learning_class.predict_theft_risk_areas(
            grid_size=0.02, bounds=bounds
        )

        suspect_activity = learning_class.predict_suspect_activity_areas(
            grid_size=0.02, bounds=bounds
        )

        assert len(theft_risk) > 0
        assert len(suspect_activity) > 0
        assert all(score >= 0 for score in theft_risk["risk_score"])
        assert all(score >= 0 for score in suspect_activity["activity_score"])

    def test_model_persistence_integration(self, full_learning_setup):
        """Test model saving and loading with full setup."""
        learning_class = full_learning_setup

        # Train models
        learning_class.train_theft_location_clusters()
        learning_class.train_suspect_location_clusters()
        learning_class.train_anomaly_detectors()

        original_summary = learning_class.get_model_summary()

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "full_models"

            # Save models
            learning_class.save_models(save_dir)

            # Create new instance and load
            new_learning_class = LearningClass(random_state=42)
            new_learning_class.load_models(save_dir)

            # Load the same data
            new_learning_class.load_data(
                theft_data=learning_class.theft_data,
                suspect_data=learning_class.suspect_data,
                recovery_data=learning_class.recovery_data,
            )

            new_summary = new_learning_class.get_model_summary()

            # Verify key components match
            assert set(original_summary["trained_models"]) == set(
                new_summary["trained_models"]
            )
            assert set(original_summary["available_scalers"]) == set(
                new_summary["available_scalers"]
            )

    def test_cross_data_predictions(self, full_learning_setup):
        """Test predictions using data from multiple sources."""
        learning_class = full_learning_setup

        # Train with criminal activity data
        criminal_results = learning_class.train_criminal_activity_predictor()

        if criminal_results.get("status") != "insufficient_data":
            # Test prediction on theft locations
            theft_locations = learning_class.theft_data[["geometry"]].copy()

            risk_scores = learning_class.predict_criminal_activity_risk(theft_locations)

            assert len(risk_scores) == len(theft_locations)
            assert all(isinstance(score, (int, float)) for score in risk_scores)

            # Higher risk scores should be for locations closer to known criminal activity
            assert np.var(risk_scores) > 0  # Should have some variation
