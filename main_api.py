# main_api.py

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Import your existing logic
# Make sure your ML logic file is accessible
from upgraded_model_logic import train_models_and_explainer
import lime.lime_tabular

# --- Application Setup ---
# Create the FastAPI app object
app = FastAPI(
    title="POWERGRID Project Risk API",
    description="API for predicting project timeline and cost overruns.",
    version="1.0.0"
)


# --- Pydantic Models for Input and Output ---
# This defines the structure of the data you expect in the request
class ProjectInput(BaseModel):
    project_type: str = Field(..., example="Overhead Line")
    terrain: str = Field(..., example="Hilly")
    material_cost_crore: float = Field(..., example=200)
    vendor_performance_rating: int = Field(..., example=3)
    historical_delays_project_count: int = Field(..., example=8)


# This defines the structure of the LIME explanation part of the response
class LimeExplanation(BaseModel):
    feature: str
    weight: float


# This defines the structure of the final JSON response
class PredictionResponse(BaseModel):
    predicted_delay_days: float
    predicted_cost_overrun_lakhs: float
    severity_score: float
    lime_explanation: list[LimeExplanation]


# --- Load Models and Explainers at Startup ---
# This is a global dictionary to hold our models so they are loaded only once
models = {}
historical_df = None


@app.on_event("startup")
def load_assets():
    """
    This function runs only once when the API server starts.
    It loads the ML models, explainers, and data into memory.
    """
    global models, historical_df
    print("--> Loading ML models and assets...")

    # Load historical data needed for LIME
    historical_df = pd.read_csv('projects_data.csv')

    # Load models and feature names
    model_timeline, model_cost, _, feature_names = train_models_and_explainer()

    # Create and store the LIME explainer
    X_train = pd.get_dummies(
        historical_df.drop(['actual_timeline_delay_days', 'cost_overrun_lakhs', 'lat', 'lon'], axis=1, errors='ignore'))
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['timeline_delay'],
        mode='regression'
    )

    # Store everything in the global 'models' dictionary
    models['timeline'] = model_timeline
    models['cost'] = model_cost
    models['lime_explainer'] = lime_explainer
    models['feature_names'] = feature_names

    print("--> Models and assets loaded successfully.")


# --- API Endpoint Definition ---
@app.post("/predict", response_model=PredictionResponse)
def predict(project_input: ProjectInput):
    """
    This is the main prediction endpoint.
    It takes project details as input and returns predictions and explanations.
    """
    # 1. Convert input data to a DataFrame
    input_df = pd.DataFrame([project_input.dict()])
    input_processed = pd.get_dummies(input_df)
    input_aligned = input_processed.reindex(columns=models['feature_names'], fill_value=0)

    # 2. Get predictions from the loaded models
    predicted_delay = models['timeline'].predict(input_aligned)[0]
    predicted_cost_overrun = models['cost'].predict(input_aligned)[0]

    # 3. Calculate severity score (can import this function or redefine it here)
    delay_score = min(predicted_delay / 365, 1.0) * 100
    cost_score = min(predicted_cost_overrun / 500, 1.0) * 100
    severity_score = (0.6 * delay_score) + (0.4 * cost_score)

    # 4. Get LIME explanation
    explanation = models['lime_explainer'].explain_instance(
        input_aligned.iloc[0].values,
        models['timeline'].predict,
        num_features=10
    )
    lime_results = [{"feature": feature, "weight": weight} for feature, weight in explanation.as_list()]

    # 5. Return the results in the format defined by PredictionResponse
    # We explicitly cast numpy types to standard Python types (float) for JSON compatibility
    return {
        "predicted_delay_days": float(predicted_delay),
        "predicted_cost_overrun_lakhs": float(predicted_cost_overrun),
        "severity_score": float(severity_score),
        "lime_explanation": lime_results
    }


# This part allows you to run the API directly from the command line for testing
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)