import xgboost as xgb
import pandas as pd
import pickle
import os
import shap

# Get the absolute path to the artifacts folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

class InfralyticsModel:
    def __init__(self):
        self.model_timeline = None
        self.model_cost = None
        self.feature_names = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads the saved models and features from the 'artifacts' folder."""
        print("⚙️ Loading models from disk...")
        
        # 1. Load XGBoost Models
        self.model_timeline = xgb.XGBRegressor()
        self.model_timeline.load_model(os.path.join(ARTIFACTS_DIR, "model_timeline.json"))
        
        self.model_cost = xgb.XGBRegressor()
        self.model_cost.load_model(os.path.join(ARTIFACTS_DIR, "model_cost.json"))
        
        # 2. Load Feature Names
        with open(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"), "rb") as f:
            self.feature_names = pickle.load(f)
            
        # 3. Initialize SHAP Explainer (Fast for Trees)
        self.explainer = shap.TreeExplainer(self.model_timeline)
        print("✅ Models loaded successfully.")

    def predict(self, input_data):
        """
        Takes a dictionary of inputs, processes them, and returns predictions.
        """
        # Convert dict to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-Hot Encode
        input_processed = pd.get_dummies(input_df)
        
        # Align with training features (add missing cols with 0)
        input_aligned = input_processed.reindex(columns=self.feature_names, fill_value=0)
        input_aligned = input_aligned.astype(float)
        
        # Predict
        pred_delay = float(self.model_timeline.predict(input_aligned)[0])
        pred_cost = float(self.model_cost.predict(input_aligned)[0])
        
        # Calculate Severity
        severity = self._calculate_severity(pred_delay, pred_cost)
        
        # Calculate SHAP values for visualization
        shap_values = self.explainer.shap_values(input_aligned)
        
        return {
            "predicted_delay": pred_delay,
            "predicted_cost_overrun": pred_cost,
            "severity_score": severity,
            "shap_values": shap_values,
            "feature_names": self.feature_names
        }

    def _calculate_severity(self, delay, cost):
        # Normalize: Assume 365 days and 500 Lakhs are the "max risk" baselines
        d_score = min(delay / 365, 1.0) * 100
        c_score = min(cost / 500, 1.0) * 100
        return (0.6 * d_score) + (0.4 * c_score)

# Create a Singleton instance
# This ensures we only load the models ONCE when the server starts
model_service = InfralyticsModel()

def get_prediction(input_data):
    return model_service.predict(input_data)