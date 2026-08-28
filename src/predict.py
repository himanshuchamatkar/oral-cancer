import os
import json
import pandas as pd
import numpy as np
import joblib

# Constants
MODELS_DIR = os.path.join("D:\\arishaa jii", "models")
CONFIG_DIR = os.path.join("D:\\arishaa jii", "config")

# Default thresholds
DEFAULT_LOW_THRESHOLD = 0.33
DEFAULT_HIGH_THRESHOLD = 0.66

_model = None
_preprocessor = None
_feature_config = None

def load_prediction_assets():
    """Load model, preprocessor, and configuration metadata if not already cached."""
    global _model, _preprocessor, _feature_config
    
    if _model is None:
        best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model file not found at {best_model_path}. Please train the models first.")
        _model = joblib.load(best_model_path)
        
    if _preprocessor is None:
        prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessing pipeline not found at {prep_path}. Please run preprocessing first.")
        _preprocessor = joblib.load(prep_path)
        
    if _feature_config is None:
        config_path = os.path.join(CONFIG_DIR, "feature_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Feature config not found at {config_path}.")
        with open(config_path, "r", encoding="utf-8") as f:
            _feature_config = json.load(f)
            
    return _model, _preprocessor, _feature_config

def predict_risk(input_data, low_threshold=DEFAULT_LOW_THRESHOLD, high_threshold=DEFAULT_HIGH_THRESHOLD):
    """
    Predict oral cancer risk category and probability for a single user's input.
    
    Parameters:
    - input_data: dict, e.g. {"Age": 45, "Gender": "Male", "Tobacco Use": "Current", ...}
    - low_threshold: float, threshold between Low and Moderate risk (default 0.33)
    - high_threshold: float, threshold between Moderate and High risk (default 0.66)
    
    Returns:
    - result_dict: dict containing predicted probability, category, and metadata.
    """
    model, preprocessor, feature_config = load_prediction_assets()
    
    # Map the UI inputs to the raw dataset columns and values
    mapped_record = {}
    for feat_name, feat_info in feature_config.items():
        dataset_col = feat_info["dataset_column"]
        user_val = input_data.get(feat_name, None)
        
        if user_val is None:
            # If feature is missing, pass NaN (imputer will handle it)
            mapped_record[dataset_col] = np.nan
        elif "mapping" in feat_info:
            # Apply mapping (e.g. "Current" -> "Yes", "Unknown" -> None)
            mapped_val = feat_info["mapping"].get(str(user_val), user_val)
            # If mapping maps to null, represent it as NaN for the scikit-learn imputer
            mapped_record[dataset_col] = mapped_val if mapped_val is not None else np.nan
        else:
            # No mapping defined, use directly
            mapped_record[dataset_col] = user_val
            
    # Convert single record to DataFrame
    df_row = pd.DataFrame([mapped_record])
    
    # Transform using preprocessor
    transformed_features = preprocessor.transform(df_row)
    
    # Compute prediction probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(transformed_features)[0, 1]
    else:
        # Fallback for models without predict_proba (should not happen for classification models trained here)
        prob = float(model.predict(transformed_features)[0])
        
    prob_pct = prob * 100
    
    # Classify risk category based on thresholds
    if prob < low_threshold:
        risk_category = "Low Risk"
    elif prob < high_threshold:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"
        
    return {
        "probability": float(prob),
        "probability_pct": float(prob_pct),
        "risk_category": risk_category,
        "mapped_record": mapped_record,
        "input_df": df_row
    }

if __name__ == "__main__":
    # Quick sanity check code
    try:
        model, prep, config = load_prediction_assets()
        print("Assets loaded successfully.")
        
        # Test input matching configuration keys
        sample_input = {
            "Age": 45,
            "Gender": "Male",
            "Tobacco Use": "Current",
            "Alcohol Consumption": "Current",
            "Betel Quid Use": "Current",
            "HPV Infection Status": "Yes",
            "Chronic Sun Exposure": "No",
            "Oral Hygiene Level": "Poor",
            "Dietary Habits (Fruit/Veggie Intake)": "Low",
            "Family History of Cancer": "Yes",
            "Immune System Status": "Compromised",
            "Oral Lesion Presence": "Yes",
            "Unexplained Oral Bleeding": "Yes",
            "Difficulty Swallowing": "Yes",
            "Persistent White/Red Patches": "Yes"
        }
        
        # We need to map sample_input keys from display names (labels) to config keys
        # For testing, let's construct it with exact config keys:
        test_input = {
            "Age": 45,
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
        res = predict_risk(test_input)
        print(f"Sample prediction result: Probability = {res['probability_pct']:.2f}%, Category = {res['risk_category']}")
    except Exception as e:
        print(f"Sanity check error (expected if models not trained yet): {e}")
