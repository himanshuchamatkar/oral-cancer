import os
import copy
from src.predict import predict_risk, load_prediction_assets

# Healthy baseline values for counterfactual calculations
HEALTHY_BASELINES = {
    "Age": 30, # baseline reference age
    "Gender": "Male", # reference
    "Tobacco Use": "Never",
    "Alcohol Consumption": "Never",
    "Betel Quid Use": "Never",
    "HPV Infection": "No",
    "Chronic Sun Exposure": "No",
    "Poor Oral Hygiene": "Good",
    "Diet (Fruits & Vegetables Intake)": "High",
    "Family History of Cancer": "No",
    "Compromised Immune System": "Normal",
    "Oral Lesions": "No",
    "Unexplained Bleeding": "No",
    "Difficulty Swallowing": "No",
    "White or Red Patches in Mouth": "No"
}

def explain_prediction(user_input, original_probability=None):
    """
    Explain the prediction for a single user input by evaluating the contribution
    of each risk factor using a counterfactual baseline approach.
    
    Parameters:
    - user_input: dict containing original user responses
    - original_probability: float, the original predicted probability (calculated if not provided)
    
    Returns:
    - explanations: list of dicts, sorted by contribution score, containing:
      - feature: feature name
      - label: UI display label
      - user_value: value entered by user
      - contribution: drop in probability when this factor is set to healthy baseline
      - explanation_text: clear text explaining the contribution
    """
    _, _, feature_config = load_prediction_assets()
    
    if original_probability is None:
        orig_res = predict_risk(user_input)
        original_probability = orig_res["probability"]
        
    contributions = []
    
    # Evaluate each feature's contribution compared to healthy baseline
    for feat_name, feat_info in feature_config.items():
        user_val = user_input.get(feat_name)
        baseline_val = HEALTHY_BASELINES.get(feat_name)
        
        # We only evaluate features where the user has a non-baseline risk value
        # For age, we evaluate if user's age is greater than baseline (e.g. > 30)
        is_risky = False
        if feat_name == "Age":
            try:
                is_risky = int(user_val) > baseline_val
            except (ValueError, TypeError):
                is_risky = False
        elif feat_name == "Diet (Fruits & Vegetables Intake)":
            is_risky = user_val in ["Low", "Moderate"]
        elif feat_name == "Poor Oral Hygiene":
            is_risky = user_val in ["Poor", "Moderate"]
        elif feat_name == "Compromised Immune System":
            is_risky = user_val == "Compromised"
        else:
            # For categorical yes/no or smoking options
            is_risky = user_val not in [baseline_val, "No", "Never", "Normal", "Good", "Unknown", "Unsure"]
            
        if is_risky:
            # Create counterfactual input: set this feature to its healthy baseline
            cf_input = copy.deepcopy(user_input)
            cf_input[feat_name] = baseline_val
            
            try:
                cf_res = predict_risk(cf_input)
                cf_probability = cf_res["probability"]
                
                # The drop in probability indicates how much this factor increased risk
                drop = original_probability - cf_probability
                
                if drop > 0.005: # Only include if it has a noticeable effect (> 0.5%)
                    contributions.append({
                        "feature": feat_name,
                        "label": feat_info["label"],
                        "user_value": user_val,
                        "contribution": drop,
                        "contribution_pct": drop * 100
                    })
            except Exception as e:
                # If prediction fails for this counterfactual, log it and proceed
                print(f"Explainability counterfactual failed for {feat_name}: {e}")
                
    # Sort contributions in descending order
    contributions = sorted(contributions, key=lambda x: x["contribution"], reverse=True)
    
    # Generate user-friendly explanation texts
    explanations = []
    for item in contributions:
        feat = item["feature"]
        val = item["user_value"]
        pct = item["contribution_pct"]
        lbl = item["label"]
        
        if feat == "Age":
            text = f"Age ({val} years) is associated with higher statistical risk compared to a baseline age of 30."
        elif feat == "Tobacco Use":
            text = f"Tobacco use status ('{val}') is a primary contributor, increasing estimated risk by {pct:.1f}%."
        elif feat == "Alcohol Consumption":
            text = f"Alcohol consumption ('{val}') increases risk, adding {pct:.1f}% to the model's estimate."
        elif feat == "Betel Quid Use":
            text = f"Chewing betel quid/areca nut ('{val}') is a significant oral carcinogen factor, contributing +{pct:.1f}%."
        elif feat == "Oral Lesions":
            text = f"The reported presence of oral lesions or sores contributes {pct:.1f}% to the estimated risk."
        elif feat == "White or Red Patches in Mouth":
            text = f"Having persistent white or red patches in the mouth adds {pct:.1f}% to the model's risk score."
        elif feat == "Unexplained Bleeding":
            text = f"Unexplained oral bleeding is a key clinical symptom, increasing predicted risk by {pct:.1f}%."
        elif feat == "Difficulty Swallowing":
            text = f"Difficulty swallowing is a clinical warning sign, contributing {pct:.1f}% to the risk score."
        elif feat == "Poor Oral Hygiene":
            text = f"Oral hygiene level ('{val}') increases risk by {pct:.1f}% compared to good hygiene."
        elif feat == "HPV Infection":
            text = f"HPV infection status ('{val}') is associated with increased statistical risk (+{pct:.1f}%)."
        elif feat == "Family History of Cancer":
            text = f"A family history of cancer is a contributing genetic risk factor (+{pct:.1f}%)."
        elif feat == "Compromised Immune System":
            text = f"Compromised immune status increases vulnerability to risk factors (+{pct:.1f}%)."
        elif feat == "Diet (Fruits & Vegetables Intake)":
            text = f"Dietary intake of fruits/vegetables ('{val}') increases risk compared to a high intake diet (+{pct:.1f}%)."
        else:
            text = f"The risk factor '{lbl}' (value: '{val}') contributes {pct:.1f}% to the model's prediction."
            
        explanations.append({
            "feature": feat,
            "label": lbl,
            "user_value": val,
            "contribution_pct": pct,
            "explanation_text": text
        })
        
    return explanations

if __name__ == "__main__":
    # Test the explanation code
    test_input = {
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
    try:
        exps = explain_prediction(test_input)
        print("Generated explanations successfully:")
        for e in exps[:3]:
            print(f"- {e['explanation_text']}")
    except Exception as e:
        print(f"Explainability sanity check error (expected if models not trained): {e}")
