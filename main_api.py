import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Import your clean engine
from ml.predict import model_service 

app = FastAPI(
    title="INFRALYTICS API",
    description="Enterprise API for Infrastructure Risk Prediction",
    version="2.0.0"
)

# --- INPUT SCHEMA ---
class ProjectInput(BaseModel):
    project_type: str = Field(..., example="Overhead Line")
    terrain: str = Field(..., example="Hilly")
    material_cost_crore: float = Field(..., example=200.0)
    vendor_performance_rating: int = Field(..., ge=1, le=5, example=3)
    historical_delays_project_count: int = Field(..., ge=0, example=5)
    lat: Optional[float] = 28.6
    lon: Optional[float] = 77.2

# --- OUTPUT SCHEMA ---
class PredictionOutput(BaseModel):
    predicted_delay_days: int
    predicted_cost_overrun_lakhs: float
    severity_score: int
    risk_level: str
    top_risk_factors: List[str]

@app.get("/")
def home():
    return {"message": "Infralytics AI Engine is Running 🟢"}

@app.post("/predict", response_model=PredictionOutput)
def predict_risk(data: ProjectInput):
    try:
        # Convert Pydantic object to dict
        input_dict = data.dict()
        
        # Call the Inference Engine
        results = model_service.predict(input_dict)
        
        # Logic for Risk Level
        score = results['severity_score']
        risk = "High" if score > 60 else ("Medium" if score > 30 else "Low")
        
        # Get Top 3 Factors from SHAP values
        shap_vals = results['shap_values'][0]
        feature_names = results['feature_names']
        
        # Sort features by impact
        importance = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        top_factors = [f"{name} ({val:.2f})" for name, val in importance[:3]]

        return {
            "predicted_delay_days": int(results['predicted_delay']),
            "predicted_cost_overrun_lakhs": round(results['predicted_cost_overrun'], 2),
            "severity_score": int(score),
            "risk_level": risk,
            "top_risk_factors": top_factors
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)