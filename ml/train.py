import pandas as pd
import xgboost as xgb
import pickle
import os

# Define paths
ARTIFACTS_DIR = "ml/artifacts"
DATA_PATH = "projects_data.csv"  # Ensure this path is correct relative to where you run the script

if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

def train_and_save():
    print("⏳ Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    
    # Define features and targets
    features = df_processed.drop(
        ['actual_timeline_delay_days', 'cost_overrun_lakhs', 'lat', 'lon'], 
        axis=1, errors='ignore'
    )
    features = features.astype(float) # XGBoost requires float
    
    y_timeline = df_processed['actual_timeline_delay_days']
    y_cost = df_processed['cost_overrun_lakhs']
    
    feature_names = features.columns.tolist()

    print("🧠 Training Models...")
    # 1. Timeline Model
    model_timeline = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_timeline.fit(features, y_timeline)
    
    # 2. Cost Model
    model_cost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_cost.fit(features, y_cost)
    
    print("💾 Saving Artifacts...")
    
    # Save Models (XGBoost has its own save format which is faster/safer)
    model_timeline.save_model(os.path.join(ARTIFACTS_DIR, "model_timeline.json"))
    model_cost.save_model(os.path.join(ARTIFACTS_DIR, "model_cost.json"))
    
    # Save Feature Names (using pickle because it's a list)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
        
    print(f"✅ Success! Models saved to {ARTIFACTS_DIR}/")

if __name__ == "__main__":
    train_and_save()