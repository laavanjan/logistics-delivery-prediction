# =============================================================================
# XGBoost Model Unified Inference Pipeline
# Team: Laavanjan
# Task: Unified Inference Pipeline + API Development
# =============================================================================

import pickle
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostInferencePipeline:
    """
    Unified inference pipeline for XGBoost delivery time prediction model.

    This class handles:
    - Model loading
    - Data preprocessing
    - Input validation
    - Prediction generation
    - Error handling
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the inference pipeline.

        Args:
            model_path (str): Path to the trained XGBoost model
        """
        self.model_path = (
            model_path
            or "F:/MLE-intern/correct/Task6-XGBoost/xgboost_results/best_xgboost_model.pkl"
        )
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.is_loaded = False

        # Expected feature columns (from XGBoost training notebook, 50 features, correct order)
        self.expected_features = [
            "data",
            "route_type",
            "source_center",
            "source_name",
            "destination_center",
            "destination_name",
            "start_scan_to_end_scan",
            "is_cutoff",
            "cutoff_factor",
            "actual_distance_to_destination",
            "osrm_time",
            "osrm_distance",
            "factor",
            "segment_actual_time",
            "segment_osrm_time",
            "segment_osrm_distance",
            "segment_factor",
            "trip_creation_hour",
            "trip_creation_day",
            "trip_creation_weekday",
            "od_start_hour",
            "od_end_hour",
            "cutoff_hour",
            "planned_duration",
            "creation_to_start_mins",
            "start_to_cutoff_mins",
            "actual_vs_osrm_time",
            "segment_actual_vs_osrm_time",
            "distance_per_min",
            "time_difference",
            "segment_time_diff",
            "center_pair_count",
            "is_heavy_delay",
            "delay_category",
            "trip_creation_time_hour",
            "trip_creation_time_day",
            "trip_creation_time_weekday",
            "trip_creation_time_month",
            "od_start_time_hour",
            "od_start_time_day",
            "od_start_time_weekday",
            "od_start_time_month",
            "od_end_time_hour",
            "od_end_time_day",
            "od_end_time_weekday",
            "od_end_time_month",
            "cutoff_timestamp_hour",
            "cutoff_timestamp_day",
            "cutoff_timestamp_weekday",
            "cutoff_timestamp_month",
        ]
        
        # Debug print to help spot feature shape mismatch
        print(f"[DEBUG] Feature count: {len(self.expected_features)}")
        print(f"[DEBUG] Feature names: {self.expected_features}")

    def load_model(self) -> bool:
        """
        Load the trained XGBoost model and preprocessing components.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading XGBoost model from: {self.model_path}")

            # Load the main model
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            # Try to load feature names if available
            try:
                feature_path = Path(self.model_path).parent / "feature_names.pkl"
                if feature_path.exists():
                    with open(feature_path, "rb") as f:
                        self.feature_names = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load feature names: {e}")
                self.feature_names = self.expected_features

            # Try to load scaler if available
            try:
                scaler_path = Path(self.model_path).parent / "scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")

            self.is_loaded = True
            logger.info("✅ Model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean input data.

        Args:
            data (Dict): Input features dictionary

        Returns:
            Dict: Validated and cleaned data

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        validated_data = {}

        # Check for required features (at least the most important ones)
        critical_features = [
            "osrm_distance",
            "osrm_time",
            "actual_distance_to_destination",
        ]   
        missing_critical = [f for f in critical_features if f not in data]

        if missing_critical:
            raise ValueError(f"Missing critical features: {missing_critical}")

        # Validate and convert data types
        for feature in self.expected_features:
            if feature in data:
                value = data[feature]

                # Convert to appropriate type based on feature name
                if (
                    "time" in feature.lower()
                    or "distance" in feature.lower()
                    or "factor" in feature.lower()
                ):
                    try:
                        validated_data[feature] = (
                            float(value) if value is not None else 0.0
                        )
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Invalid numeric value for {feature}: {value}"
                        )

                elif feature in ["is_cutoff", "is_heavy_delay"]:
                    # Boolean features
                    if isinstance(value, bool):
                        validated_data[feature] = int(value)
                    elif str(value).lower() in ["true", "1", "yes"]:
                        validated_data[feature] = 1
                    else:
                        validated_data[feature] = 0

                elif feature in ["destination_center", "destination_name"]:
                    # Categorical features
                    validated_data[feature] = (
                        str(value) if value is not None else "unknown"
                    )

                else:
                    # Numeric features
                    try:
                        validated_data[feature] = (
                            float(value) if value is not None else 0.0
                        )
                    except (ValueError, TypeError):
                        validated_data[feature] = 0.0
            else:
                # Set default values for missing features
                if feature in ["is_cutoff", "is_heavy_delay"]:
                    validated_data[feature] = 0
                elif feature in ["destination_center", "destination_name"]:
                    validated_data[feature] = "unknown"
                else:
                    validated_data[feature] = 0.0

        return validated_data

    def preprocess_data(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the validated data for model prediction.

        Args:
            data (Dict): Validated input data

        Returns:
            np.ndarray: Preprocessed feature array
        """
        # Create DataFrame with expected feature order
        df = pd.DataFrame([data])

        # Handle categorical encoding if needed
        # This is a simplified version - you might need more sophisticated encoding
        categorical_features = ["destination_center", "destination_name"]
        for feature in categorical_features:
            if feature in df.columns:
                # Simple label encoding (in production, use saved encoders)
                df[feature] = pd.Categorical(df[feature]).codes

        # Ensure all expected features are present
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0.0

        # Reorder columns to match training
        df = df[self.expected_features]

        # Apply scaling if scaler is available
        if self.scaler is not None:
            try:
                scaled_data = self.scaler.transform(df)
                return scaled_data
            except Exception as e:
                logger.warning(f"Could not apply scaling: {e}")

        return df.values

    def predict(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Make predictions using the loaded model.

        Args:
            data: Input data (single dict or list of dicts)

        Returns:
            Dict: Prediction results with metadata
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "Model not loaded",
                    "predictions": None,
                }

        try:
            # Handle single prediction vs batch prediction
            if isinstance(data, dict):
                data_list = [data]
                single_prediction = True
            else:
                data_list = data
                single_prediction = False

            predictions = []

            for item in data_list:
                # Validate input
                validated_data = self.validate_input(item)

                # Preprocess data
                processed_data = self.preprocess_data(validated_data)

                # Make prediction
                pred = self.model.predict(processed_data)[0]

                # Convert prediction to readable format
                pred_minutes = pred / 60  # Convert seconds to minutes

                predictions.append(
                    {
                        "predicted_time_seconds": float(pred),
                        "predicted_time_minutes": float(pred_minutes),
                        "predicted_time_formatted": f"{int(pred_minutes//60)}h {int(pred_minutes%60)}m",
                    }
                )

            result = {
                "success": True,
                "predictions": predictions[0] if single_prediction else predictions,
                "model_info": {
                    "model_type": "XGBoost",
                    "version": "1.0",
                    "features_used": len(self.expected_features),
                },
            }

            logger.info(f"✅ Prediction successful: {result['predictions']}")
            return result

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg, "predictions": None}


# Convenience function for easy usage
def predict_delivery_time(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for making predictions.

    Args:
        input_data (Dict): Input features

    Returns:
        Dict: Prediction results
    """
    pipeline = XGBoostInferencePipeline()
    return pipeline.predict(input_data)


# Example usage and testing
if __name__ == "__main__":
    # Test the pipeline
    test_data = {
        "osrm_distance": 60000.0,
        "osrm_time": 800.0,
        "actual_distance_to_destination": 4800.0,
        "cutoff_factor": 1.3,
        "factor": 0.8,
        "is_cutoff": False,
        "time_difference": 120.0,
    }

    print("🚀 Testing XGBoost Inference Pipeline...")
    result = predict_delivery_time(test_data)
    print(f"📊 Result: {result}")
