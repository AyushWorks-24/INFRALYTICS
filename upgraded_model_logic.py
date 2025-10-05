import pandas as pd
import xgboost as xgb
import shap
import numpy as np

def train_models_and_explainer(data_path='projects_data.csv'):
    df = pd.read_csv(data_path)
    df_processed = pd.get_dummies(df, columns=['project_type', 'terrain'])
    features = df_processed.drop(['actual_timeline_delay_days', 'cost_overrun_lakhs', 'lat', 'lon'], axis=1, errors='ignore')
    y_timeline = df_processed['actual_timeline_delay_days']
    y_cost = df_processed['cost_overrun_lakhs']
    feature_names = features.columns.tolist()
    model_timeline = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_timeline.fit(features, y_timeline)
    model_cost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_cost.fit(features, y_cost)
    explainer_timeline = shap.TreeExplainer(model_timeline)
    print("✅ Models and SHAP explainer trained successfully.")
    return model_timeline, model_cost, explainer_timeline, feature_names

def calculate_severity_score(predicted_delay, predicted_cost_overrun):
    delay_score = min(predicted_delay / 365, 1.0) * 100
    cost_score = min(predicted_cost_overrun / 500, 1.0) * 100
    severity_score = (0.6 * delay_score) + (0.4 * cost_score)
    return severity_score

def get_predictions_and_hotspots(input_data, model_timeline, model_cost, explainer, feature_names):
    input_df = pd.DataFrame([input_data])
    input_processed = pd.get_dummies(input_df)
    input_aligned = input_processed.reindex(columns=feature_names, fill_value=0)
    predicted_delay = model_timeline.predict(input_aligned)[0]
    predicted_cost_overrun = model_cost.predict(input_aligned)[0]
    severity_score = calculate_severity_score(predicted_delay, predicted_cost_overrun)
    shap_values = explainer.shap_values(input_aligned)
    return {
        "predicted_delay": predicted_delay,
        "predicted_cost_overrun": predicted_cost_overrun,
        "severity_score": severity_score,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "input_aligned": input_aligned
    }