"""
Learning Models module for auto_thievia package.

This module provides machine learning capabilities for predicting auto theft patterns,
suspect locations, and criminal activity areas using self-supervised learning approaches.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .gis_mapper import WGS84_CRS

# File constants
THEFT_CLUSTERS_FILE = "theft_clusters.pkl"
SUSPECT_CLUSTERS_FILE = "suspect_clusters.pkl"


class LearningClass:
    """
    A class for machine learning-based prediction of auto theft patterns, suspect locations,
    and criminal activity areas using self-supervised learning techniques.

    This class focuses on unsupervised and self-supervised approaches to discover patterns
    and make predictions without requiring labeled training data.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the LearningClass.

        Args:
            random_state (int): Random state for reproducible results.
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}

        # Data storage
        self.theft_data = None
        self.suspect_data = None
        self.recovery_data = None

        # Trained models and clusters
        self.theft_clusters = None
        self.suspect_clusters = None
        self.criminal_activity_clusters = None

        # Anomaly detection models
        self.theft_anomaly_detector = None
        self.suspect_anomaly_detector = None

        # Prediction models
        self.theft_location_predictor = None
        self.suspect_location_predictor = None
        self.criminal_activity_predictor = None

    def load_data(
        self,
        theft_data: Optional[gpd.GeoDataFrame] = None,
        suspect_data: Optional[gpd.GeoDataFrame] = None,
        recovery_data: Optional[gpd.GeoDataFrame] = None,
    ):
        """
        Load data for training and prediction.

        Args:
            theft_data (Optional[gpd.GeoDataFrame]): Auto theft incident data.
            suspect_data (Optional[gpd.GeoDataFrame]): Suspect data.
            recovery_data (Optional[gpd.GeoDataFrame]): Recovery data.
        """
        self.theft_data = theft_data
        self.suspect_data = suspect_data
        self.recovery_data = recovery_data

        print("Loaded data:")
        if theft_data is not None:
            print(f"  - Theft data: {len(theft_data)} records")
        if suspect_data is not None:
            print(f"  - Suspect data: {len(suspect_data)} records")
        if recovery_data is not None:
            print(f"  - Recovery data: {len(recovery_data)} records")

    def _prepare_spatial_features(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Prepare spatial features from GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): Input geodataframe.

        Returns:
            np.ndarray: Spatial features array.
        """
        # Extract coordinates
        coords = np.array(
            [
                [point.x, point.y]
                for point in gdf.geometry.apply(
                    lambda geom: geom.centroid if hasattr(geom, "centroid") else geom
                )
            ]
        )

        # Add time-based features if available
        features = coords.copy()

        if (
            "incident_date" in gdf.columns
            or "arrest_date" in gdf.columns
            or "recovery_date" in gdf.columns
        ):
            date_col = None
            if "incident_date" in gdf.columns:
                date_col = "incident_date"
            elif "arrest_date" in gdf.columns:
                date_col = "arrest_date"
            elif "recovery_date" in gdf.columns:
                date_col = "recovery_date"

            if date_col:
                dates = pd.to_datetime(gdf[date_col])
                hour_features = np.array(
                    [
                        dates.dt.hour,  # Hour of day
                        dates.dt.dayofweek,  # Day of week
                        dates.dt.month,  # Month
                        dates.dt.quarter,  # Quarter
                    ]
                ).T
                features = np.hstack([features, hour_features])

        return features

    def _prepare_theft_features(self, theft_df: gpd.GeoDataFrame) -> np.ndarray:
        """
        Prepare comprehensive features for theft prediction.

        Args:
            theft_df (gpd.GeoDataFrame): Theft data.

        Returns:
            np.ndarray: Feature matrix.
        """
        features = []

        # Spatial features
        spatial_features = self._prepare_spatial_features(theft_df)
        features.append(spatial_features)

        # Vehicle features
        if "vehicle_value" in theft_df.columns:
            vehicle_values = theft_df["vehicle_value"].fillna(
                theft_df["vehicle_value"].median()
            )
            features.append(np.array(vehicle_values).reshape(-1, 1))

        if "vehicle_year" in theft_df.columns:
            vehicle_years = theft_df["vehicle_year"].fillna(
                theft_df["vehicle_year"].median()
            )
            features.append(np.array(vehicle_years).reshape(-1, 1))

        # Categorical features (encoded)
        categorical_cols = ["vehicle_make", "vehicle_type", "theft_method"]
        for col in categorical_cols:
            if col in theft_df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded = self.encoders[col].fit_transform(
                        theft_df[col].fillna("Unknown")
                    )
                else:
                    encoded = self.encoders[col].transform(
                        theft_df[col].fillna("Unknown")
                    )
                features.append(encoded.reshape(-1, 1))

        # Socioeconomic features
        if "owner_income" in theft_df.columns:
            income = theft_df["owner_income"].fillna(theft_df["owner_income"].median())
            features.append(np.array(income).reshape(-1, 1))

        return np.hstack(features)

    def _prepare_suspect_features(self, suspect_df: gpd.GeoDataFrame) -> np.ndarray:
        """
        Prepare comprehensive features for suspect prediction.

        Args:
            suspect_df (gpd.GeoDataFrame): Suspect data.

        Returns:
            np.ndarray: Feature matrix.
        """
        features = []

        # Spatial features (address location)
        spatial_features = self._prepare_spatial_features(suspect_df)
        features.append(spatial_features)

        # Demographic features
        if "age" in suspect_df.columns:
            ages = suspect_df["age"].fillna(suspect_df["age"].median())
            features.append(np.array(ages).reshape(-1, 1))

        # Criminal history features
        if "previous_arrests" in suspect_df.columns:
            arrests = suspect_df["previous_arrests"].fillna(0)
            features.append(np.array(arrests).reshape(-1, 1))

        # Categorical features
        categorical_cols = ["occupation", "criminal_associations"]
        for col in categorical_cols:
            if col in suspect_df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    encoded = self.encoders[col].fit_transform(
                        suspect_df[col].fillna("Unknown")
                    )
                else:
                    encoded = self.encoders[col].transform(
                        suspect_df[col].fillna("Unknown")
                    )
                features.append(encoded.reshape(-1, 1))

        return np.hstack(features)

    def train_theft_location_clusters(
        self, eps: float = 0.01, min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Train clustering model to identify theft hotspots using self-supervised learning.

        Args:
            eps (float): DBSCAN epsilon parameter.
            min_samples (int): Minimum samples for DBSCAN cluster.

        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        if self.theft_data is None:
            raise ValueError("No theft data loaded. Call load_data() first.")

        print("Training theft location clustering model...")

        # Prepare features
        features = self._prepare_theft_features(self.theft_data)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers["theft"] = scaler

        # DBSCAN clustering for hotspot detection
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        theft_clusters = dbscan.fit_predict(features_scaled)

        # K-means for general pattern discovery
        n_clusters = max(3, len(np.unique(theft_clusters[theft_clusters != -1])))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        theft_kmeans_clusters = kmeans.fit_predict(features_scaled)

        self.theft_clusters = {
            "dbscan": theft_clusters,
            "kmeans": theft_kmeans_clusters,
            "features": features_scaled,
        }

        self.models["theft_dbscan"] = dbscan
        self.models["theft_kmeans"] = kmeans

        # Calculate metrics
        n_hotspots = len(np.unique(theft_clusters[theft_clusters != -1]))
        noise_ratio = np.sum(theft_clusters == -1) / len(theft_clusters)

        if n_hotspots > 1:
            silhouette_avg = silhouette_score(features_scaled, theft_clusters)
        else:
            silhouette_avg = 0

        results = {
            "n_hotspots": n_hotspots,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette_avg,
            "n_records": len(self.theft_data),
        }

        print(f"  - Identified {n_hotspots} theft hotspots")
        print(f"  - Noise ratio: {noise_ratio:.3f}")
        print(f"  - Silhouette score: {silhouette_avg:.3f}")

        return results

    def train_suspect_location_clusters(
        self, eps: float = 0.01, min_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Train clustering model to identify suspect activity areas.

        Args:
            eps (float): DBSCAN epsilon parameter.
            min_samples (int): Minimum samples for DBSCAN cluster.

        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        if self.suspect_data is None:
            raise ValueError("No suspect data loaded. Call load_data() first.")

        print("Training suspect location clustering model...")

        # Prepare features
        features = self._prepare_suspect_features(self.suspect_data)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers["suspect"] = scaler

        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        suspect_clusters = dbscan.fit_predict(features_scaled)

        self.suspect_clusters = {
            "dbscan": suspect_clusters,
            "features": features_scaled,
        }

        self.models["suspect_dbscan"] = dbscan

        # Calculate metrics
        n_clusters = len(np.unique(suspect_clusters[suspect_clusters != -1]))
        noise_ratio = np.sum(suspect_clusters == -1) / len(suspect_clusters)

        if n_clusters > 1:
            silhouette_avg = silhouette_score(features_scaled, suspect_clusters)
        else:
            silhouette_avg = 0

        results = {
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette_avg,
            "n_records": len(self.suspect_data),
        }

        print(f"  - Identified {n_clusters} suspect activity clusters")
        print(f"  - Noise ratio: {noise_ratio:.3f}")
        print(f"  - Silhouette score: {silhouette_avg:.3f}")

        return results

    def train_criminal_activity_predictor(self) -> Dict[str, Any]:
        """
        Train model to predict criminal activity areas using recovery data.

        Returns:
            Dict[str, Any]: Training results and metrics.
        """
        if self.recovery_data is None:
            raise ValueError("No recovery data loaded. Call load_data() first.")

        print("Training criminal activity prediction model...")

        # Focus on criminal locations from recovery data
        criminal_recoveries = self.recovery_data[
            self.recovery_data.get("is_criminal_location", False) == True
        ]

        if len(criminal_recoveries) < 5:
            warnings.warn("Insufficient criminal location data for training")
            return {"status": "insufficient_data"}

        # Prepare features for all recoveries
        recovery_coords = np.array(
            [
                [point.x, point.y]
                for point in self.recovery_data.geometry.apply(
                    lambda geom: geom.centroid if hasattr(geom, "centroid") else geom
                )
            ]
        )

        # Create self-supervised labels based on criminal location proximity
        criminal_coords = np.array(
            [
                [point.x, point.y]
                for point in criminal_recoveries.geometry.apply(
                    lambda geom: geom.centroid if hasattr(geom, "centroid") else geom
                )
            ]
        )

        # Use nearest neighbors to create risk scores
        nn_model = NearestNeighbors(n_neighbors=min(5, len(criminal_coords)))
        nn_model.fit(criminal_coords)

        distances, _ = nn_model.kneighbors(recovery_coords)
        risk_scores = 1.0 / (
            1.0 + distances.mean(axis=1)
        )  # Higher score = closer to criminal activity

        # Prepare additional features
        features = [recovery_coords]

        # Add temporal features if available
        if "recovery_date" in self.recovery_data.columns:
            dates = pd.to_datetime(self.recovery_data["recovery_date"])
            temporal_features = np.array(
                [dates.dt.hour, dates.dt.dayofweek, dates.dt.month]
            ).T
            features.append(temporal_features)

        # Add vehicle condition as feature
        if "vehicle_condition" in self.recovery_data.columns:
            condition_encoder = LabelEncoder()
            condition_encoded = condition_encoder.fit_transform(
                self.recovery_data["vehicle_condition"].fillna("Unknown")
            )
            features.append(np.array(condition_encoded).reshape(-1, 1))
            self.encoders["vehicle_condition"] = condition_encoder

        X = np.hstack(features)
        y = risk_scores

        # Train Random Forest regressor
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(X_train)
        x_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, max_depth=10
        )
        rf_model.fit(x_train_scaled, y_train)

        # Evaluate
        y_pred = rf_model.predict(x_test_scaled)
        mse = mean_squared_error(y_test, y_pred)

        self.criminal_activity_predictor = rf_model
        self.scalers["criminal_activity"] = scaler
        self.models["criminal_activity_nn"] = nn_model

        # Store feature importance
        self.feature_importance["criminal_activity"] = rf_model.feature_importances_

        results = {
            "mse": mse,
            "n_criminal_locations": len(criminal_recoveries),
            "n_training_samples": len(X_train),
            "n_test_samples": len(X_test),
        }

        print(f"  - MSE: {mse:.4f}")
        print(f"  - Criminal locations used: {len(criminal_recoveries)}")
        print(f"  - Training samples: {len(X_train)}")

        return results

    def train_anomaly_detectors(self) -> Dict[str, Any]:
        """
        Train anomaly detection models for unusual theft and suspect patterns.

        Returns:
            Dict[str, Any]: Training results.
        """
        print("Training anomaly detection models...")
        results = {}

        # Theft anomaly detection
        if self.theft_data is not None:
            theft_features = self._prepare_theft_features(self.theft_data)
            theft_scaler = StandardScaler()
            theft_features_scaled = theft_scaler.fit_transform(theft_features)

            theft_anomaly_model = IsolationForest(
                contamination=0.1, random_state=self.random_state
            )
            theft_anomalies = theft_anomaly_model.fit_predict(theft_features_scaled)

            self.theft_anomaly_detector = theft_anomaly_model
            self.scalers["theft_anomaly"] = theft_scaler

            n_theft_anomalies = np.sum(theft_anomalies == -1)
            results["theft_anomalies"] = n_theft_anomalies
            print(f"  - Theft anomalies detected: {n_theft_anomalies}")

        # Suspect anomaly detection
        if self.suspect_data is not None:
            suspect_features = self._prepare_suspect_features(self.suspect_data)
            suspect_scaler = StandardScaler()
            suspect_features_scaled = suspect_scaler.fit_transform(suspect_features)

            suspect_anomaly_model = IsolationForest(
                contamination=0.1, random_state=self.random_state
            )
            suspect_anomalies = suspect_anomaly_model.fit_predict(
                suspect_features_scaled
            )

            self.suspect_anomaly_detector = suspect_anomaly_model
            self.scalers["suspect_anomaly"] = suspect_scaler

            n_suspect_anomalies = np.sum(suspect_anomalies == -1)
            results["suspect_anomalies"] = n_suspect_anomalies
            print(f"  - Suspect anomalies detected: {n_suspect_anomalies}")

        return results

    def predict_theft_risk_areas(
        self,
        grid_size: float = 0.01,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Predict theft risk areas across a geographic grid.

        Args:
            grid_size (float): Size of grid cells in degrees.
            bounds (Optional[Tuple]): (min_x, min_y, max_x, max_y) bounds.

        Returns:
            gpd.GeoDataFrame: Grid with theft risk predictions.
        """
        if self.theft_clusters is None:
            raise ValueError(
                "No theft clustering model trained. Call train_theft_location_clusters() first."
            )

        # Determine bounds
        if bounds is None and self.theft_data is not None:
            bounds_series = self.theft_data.bounds
            bounds = (
                bounds_series.minx.min() - grid_size,
                bounds_series.miny.min() - grid_size,
                bounds_series.maxx.max() + grid_size,
                bounds_series.maxy.max() + grid_size,
            )
        elif bounds is None:
            # Default Newark area
            bounds = (-74.3, 40.6, -74.1, 40.8)

        # Create grid
        x_coords = np.arange(bounds[0], bounds[2], grid_size)
        y_coords = np.arange(bounds[1], bounds[3], grid_size)

        grid_points = []
        risk_scores = []

        for x in x_coords:
            for y in y_coords:
                grid_points.append((x, y))

                # Calculate risk based on proximity to theft clusters
                # If we have trained models, use them for prediction
                if "theft_kmeans" in self.models:
                    # Extend features to match training data dimensions
                    n_features = self.theft_clusters["features"].shape[1]
                    extended_features = np.zeros((1, n_features))
                    extended_features[0, :2] = [x, y]  # Set coordinates

                    cluster_id = self.models["theft_kmeans"].predict(extended_features)[
                        0
                    ]

                    # Calculate risk based on cluster density
                    cluster_mask = self.theft_clusters["kmeans"] == cluster_id
                    cluster_density = np.sum(cluster_mask) / len(
                        self.theft_clusters["kmeans"]
                    )

                    risk_scores.append(cluster_density)
                else:
                    risk_scores.append(0.0)

        # Create GeoDataFrame
        from shapely.geometry import Point

        geometries = [Point(x, y) for x, y in grid_points]

        risk_gdf = gpd.GeoDataFrame(
            {"risk_score": risk_scores, "geometry": geometries}, crs=WGS84_CRS
        )

        return risk_gdf

    def predict_suspect_activity_areas(
        self,
        grid_size: float = 0.01,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Predict suspect activity areas across a geographic grid.

        Args:
            grid_size (float): Size of grid cells in degrees.
            bounds (Optional[Tuple]): (min_x, min_y, max_x, max_y) bounds.

        Returns:
            gpd.GeoDataFrame: Grid with suspect activity predictions.
        """
        if self.suspect_clusters is None:
            raise ValueError(
                "No suspect clustering model trained. Call train_suspect_location_clusters() first."
            )

        # Similar implementation to theft risk areas but for suspects
        if bounds is None and self.suspect_data is not None:
            bounds_series = self.suspect_data.bounds
            bounds = (
                bounds_series.minx.min() - grid_size,
                bounds_series.miny.min() - grid_size,
                bounds_series.maxx.max() + grid_size,
                bounds_series.maxy.max() + grid_size,
            )
        elif bounds is None:
            bounds = (-74.3, 40.6, -74.1, 40.8)

        x_coords = np.arange(bounds[0], bounds[2], grid_size)
        y_coords = np.arange(bounds[1], bounds[3], grid_size)

        grid_points = []
        activity_scores = []

        for x in x_coords:
            for y in y_coords:
                grid_points.append((x, y))

                # Calculate activity score based on proximity to suspect clusters
                if self.suspect_data is not None:
                    distances = np.sqrt(
                        (self.suspect_data.geometry.x - x) ** 2
                        + (self.suspect_data.geometry.y - y) ** 2
                    )
                    min_distance = distances.min()
                    activity_score = 1.0 / (
                        1.0 + min_distance * 100
                    )  # Scale for degree units
                    activity_scores.append(activity_score)
                else:
                    activity_scores.append(0.0)

        from shapely.geometry import Point

        geometries = [Point(x, y) for x, y in grid_points]

        activity_gdf = gpd.GeoDataFrame(
            {"activity_score": activity_scores, "geometry": geometries}, crs=WGS84_CRS
        )

        return activity_gdf

    def predict_criminal_activity_risk(self, locations: gpd.GeoDataFrame) -> np.ndarray:
        """
        Predict criminal activity risk for given locations.

        Args:
            locations (gpd.GeoDataFrame): Locations to assess.

        Returns:
            np.ndarray: Risk scores for each location.
        """
        if self.criminal_activity_predictor is None:
            raise ValueError(
                "No criminal activity model trained. Call train_criminal_activity_predictor() first."
            )

        # Prepare features for prediction
        coords = np.array(
            [
                [point.x, point.y]
                for point in locations.geometry.apply(
                    lambda geom: geom.centroid if hasattr(geom, "centroid") else geom
                )
            ]
        )

        # Add minimal features to match training data
        features = [coords]

        # Add dummy temporal features (current time)
        current_hour = pd.Timestamp.now().hour
        current_dow = pd.Timestamp.now().dayofweek
        current_month = pd.Timestamp.now().month

        temporal_features = np.full(
            (len(locations), 3), [current_hour, current_dow, current_month]
        )
        features.append(temporal_features)

        # Add dummy vehicle condition feature
        if "vehicle_condition" in self.encoders:
            condition_features = np.zeros(
                (len(locations), 1)
            )  # Default to first category
            features.append(condition_features)

        X = np.hstack(features)
        x_scaled = self.scalers["criminal_activity"].transform(X)

        risk_scores = self.criminal_activity_predictor.predict(x_scaled)

        return risk_scores

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models.

        Returns:
            Dict[str, Any]: Model summary information.
        """
        summary = {
            "trained_models": list(self.models.keys()),
            "available_scalers": list(self.scalers.keys()),
            "available_encoders": list(self.encoders.keys()),
            "data_loaded": {
                "theft_data": self.theft_data is not None,
                "suspect_data": self.suspect_data is not None,
                "recovery_data": self.recovery_data is not None,
            },
            "clustering_results": {},
        }

        if self.theft_clusters is not None:
            summary["clustering_results"]["theft"] = {
                "n_hotspots": len(
                    np.unique(
                        self.theft_clusters["dbscan"][
                            self.theft_clusters["dbscan"] != -1
                        ]
                    )
                ),
                "n_kmeans_clusters": len(np.unique(self.theft_clusters["kmeans"])),
            }

        if self.suspect_clusters is not None:
            summary["clustering_results"]["suspect"] = {
                "n_clusters": len(
                    np.unique(
                        self.suspect_clusters["dbscan"][
                            self.suspect_clusters["dbscan"] != -1
                        ]
                    )
                )
            }

        if self.feature_importance:
            summary["feature_importance"] = self.feature_importance

        return summary

    def save_models(self, directory: Union[str, Path]):
        """
        Save all trained models to disk.

        Args:
            directory (Union[str, Path]): Directory to save models.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, directory / f"{name}_model.pkl")

        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, directory / f"{name}_scaler.pkl")

        # Save encoders
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, directory / f"{name}_encoder.pkl")

        # Save other components
        if self.theft_clusters is not None:
            joblib.dump(self.theft_clusters, directory / THEFT_CLUSTERS_FILE)

        if self.suspect_clusters is not None:
            joblib.dump(self.suspect_clusters, directory / SUSPECT_CLUSTERS_FILE)

        print(f"Models saved to {directory}")

    def load_models(self, directory: Union[str, Path]):
        """
        Load models from disk.

        Args:
            directory (Union[str, Path]): Directory containing saved models.
        """
        directory = Path(directory)

        # Load models
        for model_file in directory.glob("*_model.pkl"):
            name = model_file.stem.replace("_model", "")
            self.models[name] = joblib.load(model_file)

        # Load scalers
        for scaler_file in directory.glob("*_scaler.pkl"):
            name = scaler_file.stem.replace("_scaler", "")
            self.scalers[name] = joblib.load(scaler_file)

        # Load encoders
        for encoder_file in directory.glob("*_encoder.pkl"):
            name = encoder_file.stem.replace("_encoder", "")
            self.encoders[name] = joblib.load(encoder_file)

        # Load clusters
        if (directory / THEFT_CLUSTERS_FILE).exists():
            self.theft_clusters = joblib.load(directory / THEFT_CLUSTERS_FILE)

        if (directory / SUSPECT_CLUSTERS_FILE).exists():
            self.suspect_clusters = joblib.load(directory / SUSPECT_CLUSTERS_FILE)

        print(f"Models loaded from {directory}")
