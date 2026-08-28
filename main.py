import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from src.predict import predict_risk, load_prediction_assets
from src.explainability import explain_prediction

app = FastAPI(
    title="Oral Cancer Risk Prediction API",
    description="FastAPI backend exposing predictive model and explanation endpoints.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
REPORTS_DIR = os.path.join("D:\\arishaa jii", "reports")
MODELS_DIR = os.path.join("D:\\arishaa jii", "models")

class RiskAssessmentRequest(BaseModel):
    Age: int
    Gender: str
    Tobacco_Use: str
    Alcohol_Consumption: str
    Betel_Quid_Use: str
    HPV_Infection: str
    Chronic_Sun_Exposure: str
    Poor_Oral_Hygiene: str
    Diet_Fruits_Vegetables_Intake: str
    Family_History_of_Cancer: str
    Compromised_Immune_System: str
    Oral_Lesions: str
    Unexplained_Bleeding: str
    Difficulty_Swallowing: str
    White_or_Red_Patches_in_Mouth: str

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Age": 55,
                "Gender": "Male",
                "Tobacco Use": "Current",
                "Alcohol Consumption": "Current",
                "Betel Quid Use": "Current",
                "HPV Infection": "Yes",
                "Chronic Sun Exposure": "No",
                "Poor Oral Hygiene": "Poor",
                "Diet (Fruits & Vegetables Intake)": "Low",
                "Family History of Cancer": "Yes",
                "Compromised Immune System": "Compromised",
                "Oral Lesions": "Yes",
                "Unexplained Bleeding": "Yes",
                "Difficulty Swallowing": "Yes",
                "White or Red Patches in Mouth": "Yes"
            }
        }
    }

@app.get("/")
def read_root():
    return {
        "message": "Oral Cancer Risk Prediction API is running.",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metadata": "/metadata",
            "comparison": "/comparison"
        }
    }

@app.get("/health")
def health_check():
    try:
        # Verify prediction assets can load
        load_prediction_assets()
        return {"status": "healthy", "assets_loaded": True}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/predict")
def predict(request: Dict[str, Any]):
    """
    Predict risk percentage, category, and compute local counterfactual explanations.
    """
    try:
        # Standardize keys since JSON payload might contain spaces (e.g. "Tobacco Use" vs "Tobacco_Use")
        standard_input = {}
        for k, v in request.items():
            # Standardize key names
            norm_key = k.replace("_", " ")
            # Handle special naming mapping for frontend consistency
            if norm_key == "Diet Fruits Vegetables Intake":
                norm_key = "Diet (Fruits & Vegetables Intake)"
            elif norm_key == "White or Red Patches in Mouth":
                norm_key = "White or Red Patches in Mouth"
            elif norm_key == "HPV Infection Status":
                norm_key = "HPV Infection"
            elif norm_key == "Oral Hygiene Level":
                norm_key = "Poor Oral Hygiene"
            elif norm_key == "Immune System Status":
                norm_key = "Compromised Immune System"
            elif norm_key == "Oral Lesion Presence":
                norm_key = "Oral Lesions"
                
            standard_input[norm_key] = v

        # Run prediction
        res = predict_risk(standard_input)
        prob = res["probability"]
        prob_pct = res["probability_pct"]
        risk_cat = res["risk_category"]
        
        # Run explainability
        explanations = explain_prediction(standard_input, prob)
        
        return {
            "probability": prob,
            "probability_pct": prob_pct,
            "risk_category": risk_cat,
            "explanations": explanations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metadata")
def get_model_metadata():
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Model metadata not found. Please train models first.")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/comparison")
def get_model_comparison():
    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Model comparison data not found. Please train models first.")
    
    try:
        df = pd.read_csv(comparison_path)
        # Convert df to dictionary records
        records = df.to_dict(orient="records")
        return {"models": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading model comparison: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Default port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
