import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from src.predict import predict_risk, DEFAULT_LOW_THRESHOLD, DEFAULT_HIGH_THRESHOLD
from src.explainability import explain_prediction

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# Page Configuration
st.set_page_config(
    page_title="Oral Cancer Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Healthcare Inspired, Premium Aesthetics)
st.markdown("""
<style>
    /* CSS for headers and text spacing */
    .title-text {
        color: #0d5c75;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #2c7a7b;
        font-size: 1.15rem;
        margin-bottom: 1.8rem;
    }
    
    /* Risk cards styling */
    .risk-card {
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: 'Inter', sans-serif;
    }
    .low-risk {
        background-color: #e6f4ea;
        border-left: 6px solid #137333;
        color: #137333;
    }
    .mod-risk {
        background-color: #fef7e0;
        border-left: 6px solid #b06000;
        color: #b06000;
    }
    .high-risk {
        background-color: #fce8e6;
        border-left: 6px solid #c5221f;
        color: #c5221f;
    }
    
    /* General containers styling */
    .info-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3182ce;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #edf2f7;
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        font-size: 0.92rem;
        color: #4a5568;
        line-height: 1.5;
    }
    .warning-box {
        background-color: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
        color: #9b2c2c;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration safely
def get_feature_config():
    config_path = os.path.join(CONFIG_DIR, "feature_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Helper to redirect page safely
def set_page(page_name):
    st.session_state.current_page = page_name
    st.rerun()

# Initialize session states
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar navigation
st.sidebar.markdown("<h2 style='color:#0d5c75; margin-bottom:0.2rem;'>🩺 Menu</h2>", unsafe_allow_html=True)
pages = ["Home", "Risk Assessment", "Prediction Result", "Model Performance", "About the Project", "Disclaimer"]
selected_page = st.sidebar.radio(
    "Go to:", 
    pages, 
    index=pages.index(st.session_state.current_page),
    key="nav_radio"
)

# Sync sidebar changes to session state
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

# Global Academic Disclaimer Footer
academic_disclaimer = """
**IMPORTANT NOTICE**: This application is an academic machine-learning prototype. It does not diagnose oral cancer, confirm the presence or absence of cancer, or replace professional medical examination, screening, biopsy, or clinical judgment. Predictions are experimental estimates based on historical research data and are not clinically validated. If you have persistent oral symptoms or health concerns, consult a qualified healthcare professional.
"""

# Render Pages
if st.session_state.current_page == "Home":
    st.markdown("<h1 class='title-text'>Oral Cancer Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Academic Machine-Learning Prototype for Risk Factor Assessment</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        Oral cancer is a serious global health concern. However, early awareness of critical risk factors can significantly impact outcomes. 
        This web application utilizes machine-learning classification models trained on patient risk datasets to estimate an individual's **oral cancer risk level** based on a combination of:
        - **Demographic Information** (Age, biological sex)
        - **Lifestyle Habits** (Tobacco use, alcohol consumption, betel quid/areca nut chewing, diet)
        - **Clinical Indicators** (Persistent lesions, sores, bleeding, difficulty swallowing, white/red patches)
        - **Genetic & Immunological Factors** (Family history of cancer, immune system status)

        ### How It Works
        1. **Complete Questionnaire**: Answer simple, controlled questions about your lifestyle and oral health symptoms.
        2. **Machine Learning Inference**: The selected best-performing ML model processes your inputs locally.
        3. **Risk Profile Generation**: Review your predicted risk category (Low, Moderate, High), estimated probability score, and primary contributing factors.
        """)
        
        st.markdown(f"<div class='disclaimer-box'>{academic_disclaimer}</div>", unsafe_allow_html=True)
        
        if st.button("Start Risk Assessment", type="primary"):
            set_page("Risk Assessment")
            
    with col2:
        st.markdown("<div style='background-color:#ebf8ff; border:1px solid #bee3f8; border-radius:12px; padding:1.5rem;'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Model Information")
        # Load metadata if available
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.metric("Selected Model Type", meta.get("model_name", "Random Forest Classifier"))
            st.metric("Model Test Accuracy", f"{meta['metrics']['accuracy']*100:.2f}%")
            st.metric("Model Test Recall", f"{meta['metrics']['recall']*100:.2f}%")
        else:
            st.warning("Note: Machine learning models are currently being trained. Run the train.py script to build models and inspect metrics.")
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == "Risk Assessment":
    st.markdown("<h1 class='title-text'>Risk Factor Questionnaire</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Please answer all questions accurately. Your responses are not stored and remain local to this session.</p>", unsafe_allow_html=True)
    
    feature_config = get_feature_config()
    
    if not feature_config:
        st.error("Feature configuration not found. Please ensure config/feature_config.json exists.")
    else:
        # Group features into form sections
        form_data = {}
        
        with st.form("risk_form"):
            # Section A: Demographics
            st.markdown("### 📋 Section A: Demographics")
            col1, col2 = st.columns(2)
            with col1:
                if "Age" in feature_config:
                    cfg = feature_config["Age"]
                    form_data["Age"] = st.number_input(
                        cfg["label"],
                        min_value=int(cfg["min"]),
                        max_value=int(cfg["max"]),
                        value=int(cfg["default"]),
                        help=cfg["description"]
                    )
            with col2:
                if "Gender" in feature_config:
                    cfg = feature_config["Gender"]
                    form_data["Gender"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            
            st.write("---")
            
            # Section B: Lifestyle Risk Factors
            st.markdown("### 🚬 Section B: Lifestyle Risk Factors")
            col3, col4, col5 = st.columns(3)
            with col3:
                if "Tobacco Use" in feature_config:
                    cfg = feature_config["Tobacco Use"]
                    form_data["Tobacco Use"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col4:
                if "Alcohol Consumption" in feature_config:
                    cfg = feature_config["Alcohol Consumption"]
                    form_data["Alcohol Consumption"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col5:
                if "Betel Quid Use" in feature_config:
                    cfg = feature_config["Betel Quid Use"]
                    form_data["Betel Quid Use"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
                    
            col6, col7 = st.columns(2)
            with col6:
                if "Poor Oral Hygiene" in feature_config:
                    cfg = feature_config["Poor Oral Hygiene"]
                    form_data["Poor Oral Hygiene"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col7:
                if "Diet (Fruits & Vegetables Intake)" in feature_config:
                    cfg = feature_config["Diet (Fruits & Vegetables Intake)"]
                    form_data["Diet (Fruits & Vegetables Intake)"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            
            st.write("---")
            
            # Section C: Medical & Genetic Factors
            st.markdown("### 🧬 Section C: Medical & Family Risk Factors")
            col8, col9, col10 = st.columns(3)
            with col8:
                if "Family History of Cancer" in feature_config:
                    cfg = feature_config["Family History of Cancer"]
                    form_data["Family History of Cancer"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col9:
                if "HPV Infection" in feature_config:
                    cfg = feature_config["HPV Infection"]
                    form_data["HPV Infection"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col10:
                if "Compromised Immune System" in feature_config:
                    cfg = feature_config["Compromised Immune System"]
                    form_data["Compromised Immune System"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
                    
            if "Chronic Sun Exposure" in feature_config:
                cfg = feature_config["Chronic Sun Exposure"]
                form_data["Chronic Sun Exposure"] = st.selectbox(
                    cfg["label"],
                    options=cfg["options"],
                    help=cfg["description"]
                )
            
            st.write("---")
            
            # Section D: Clinical Warning Symptoms
            st.markdown("### ⚠️ Section D: Clinical Symptoms & Warning Signs")
            st.info("The presence of these symptoms requires direct professional evaluation regardless of the model prediction.")
            col11, col12 = st.columns(2)
            with col11:
                if "Oral Lesions" in feature_config:
                    cfg = feature_config["Oral Lesions"]
                    form_data["Oral Lesions"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
                if "Unexplained Bleeding" in feature_config:
                    cfg = feature_config["Unexplained Bleeding"]
                    form_data["Unexplained Bleeding"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            with col12:
                if "White or Red Patches in Mouth" in feature_config:
                    cfg = feature_config["White or Red Patches in Mouth"]
                    form_data["White or Red Patches in Mouth"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
                if "Difficulty Swallowing" in feature_config:
                    cfg = feature_config["Difficulty Swallowing"]
                    form_data["Difficulty Swallowing"] = st.selectbox(
                        cfg["label"],
                        options=cfg["options"],
                        help=cfg["description"]
                    )
            
            submit_btn = st.form_submit_button("Assess Oral Cancer Risk", type="primary")
            
            if submit_btn:
                try:
                    # Run model prediction
                    res = predict_risk(form_data)
                    st.session_state.prediction_result = res
                    st.session_state.user_inputs = form_data
                    set_page("Prediction Result")
                except Exception as e:
                    st.error(f"Error during risk prediction: {e}. Ensure training pipeline was run and model files exist.")

elif st.session_state.current_page == "Prediction Result":
    st.markdown("<h1 class='title-text'>Prediction Result</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Model Prediction and Risk Explanations</p>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("No assessment completed yet.")
        if st.button("Go to Questionnaire"):
            set_page("Risk Assessment")
    else:
        res = st.session_state.prediction_result
        inputs = st.session_state.user_inputs
        prob_pct = res["probability_pct"]
        risk_cat = res["risk_category"]
        
        # Determine CSS class based on risk category
        card_class = "low-risk"
        if risk_cat == "Moderate Risk":
            card_class = "mod-risk"
        elif risk_cat == "High Risk":
            card_class = "high-risk"
            
        st.markdown(f"""
        <div class="risk-card {card_class}">
            <h2>ESTIMATED PROFILE: {risk_cat.upper()}</h2>
            <h3 style="margin-top:0.5rem; font-weight:400;">Model-Estimated Probability Score: <strong>{prob_pct:.1f}%</strong></h3>
            <p style="font-size:0.92rem; font-style:italic; margin-top:0.5rem;">
                Threshold default bands: Low Risk (&lt; 33%), Moderate Risk (33% - 66%), High Risk (&ge; 66%). 
                Derived from model outputs for educational demonstration.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Major Contributing Factors")
            
            # Generate feature explanations
            explanations = explain_prediction(inputs, res["probability"])
            
            if explanations:
                st.write("The following lifestyle or clinical attributes were identified as main risk drivers in the model's estimate:")
                for exp in explanations:
                    st.markdown(f"- **{exp['label']}** ({exp['user_value']}): {exp['explanation_text']}")
            else:
                st.write("No significant contributors identified. All reported inputs align with the healthy reference profile.")
                
            st.markdown("### 📋 Guidelines & Lifestyle Suggestions")
            if risk_cat == "Low Risk":
                st.info("💡 **Recommendation:** Maintain healthy oral hygiene habits (regular brushing, flossing), eat a balanced diet high in fruits and vegetables, avoid tobacco and excessive alcohol, and attend routine dental checkups.")
            elif risk_cat == "Moderate Risk":
                st.warning("⚠️ **Recommendation:** Consider discussing your lifestyle habits (such as tobacco or alcohol exposure) and any oral symptoms with a healthcare or dental professional. Lowering exposure to known risk factors can lower your overall risk profile.")
            else:
                st.error("🚨 **Recommendation:** Consider seeking professional oral-health evaluation, particularly if you have persistent oral lesions, white or red patches, unexplained bleeding, difficulty swallowing, or chronic pain. Avoid all tobacco, areca/betel quid, and alcohol exposure.")
                
        with col2:
            st.markdown("### 🏥 Clinical Warning Signs Report")
            
            # Identify if patient has symptoms
            reported_symptoms = []
            symptom_fields = ["Oral Lesions", "Unexplained Bleeding", "White or Red Patches in Mouth", "Difficulty Swallowing"]
            
            for s_field in symptom_fields:
                val = inputs.get(s_field, "No")
                if val in ["Yes", "Unsure"]:
                    reported_symptoms.append(f"{s_field} ({val})")
                    
            if reported_symptoms:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>🚨 Symptom Alert!</h4>
                    <p>You reported the presence or uncertainty of the following symptoms:</p>
                    <ul>
                        {"".join([f"<li>{s}</li>" for s in reported_symptoms])}
                    </ul>
                    <p><strong>Regardless of the model risk score</strong>, persistent oral lesions, sores, white or red patches, difficulty swallowing, or bleeding should be evaluated directly by an oral pathologist, dentist, or ENT physician immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ No key warning symptoms (lesions, bleeding, patches, swallowing difficulty) were reported in this questionnaire.")
                
            st.write("---")
            st.button("Perform Another Assessment", on_click=lambda: set_page("Risk Assessment"))
            
        st.markdown(f"<div class='disclaimer-box'>{academic_disclaimer}</div>", unsafe_allow_html=True)

elif st.session_state.current_page == "Model Performance":
    st.markdown("<h1 class='title-text'>Model Performance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Comparison of ML Models and Diagnostic Figures (Academic Demonstration)</p>", unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard details the model training, testing, and validation process. 
    In healthcare prediction tasks, we focus heavily on **Recall** (Sensitivity) to avoid False Negatives (missing high-risk individuals), while maintaining a reasonable **F1-Score** to control False Positives.
    """)
    
    # Check if comparison file exists
    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        df_comp = pd.read_csv(comparison_path)
        st.markdown("### 📊 Candidate Model Comparisons")
        st.dataframe(
            df_comp.style.format({
                "CV Accuracy": "{:.4%}", "CV Recall": "{:.4%}", "CV F1-Score": "{:.4%}",
                "Test Accuracy": "{:.4%}", "Test Precision": "{:.4%}", "Test Recall": "{:.4%}",
                "Test F1-Score": "{:.4%}", "Test ROC-AUC": "{:.4f}"
            })
        )
    else:
        st.warning("Model comparison data not found. Please train models using train.py.")
        
    st.write("---")
    st.markdown("### 📈 Best Model Evaluation Figures")
    
    col1, col2 = st.columns(2)
    with col1:
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Test Dataset", use_container_width=True)
        else:
            st.info("Confusion matrix figure not generated yet.")
            
        feat_path = os.path.join(FIGURES_DIR, "feature_importance.png")
        if os.path.exists(feat_path):
            st.image(feat_path, caption="Feature Importance / Attribute Weights", use_container_width=True)
        else:
            st.info("Feature importance figure not generated yet.")
            
    with col2:
        roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve (Receiver Operating Characteristic)", use_container_width=True)
        else:
            st.info("ROC curve figure not generated yet.")
            
        st.markdown("""
        #### Metric Meanings
        - **Accuracy**: Overall proportion of correct predictions. (Can be misleading in imbalanced datasets).
        - **Precision**: Of all patients predicted with high risk, how many actually had oral cancer.
        - **Recall (Sensitivity)**: Of all patients who actually had oral cancer, how many were correctly flagged by the model. **Critical in clinical screening!**
        - **F1-Score**: Harmonic mean of Precision and Recall. Used here as the primary metric to select the best model.
        - **ROC-AUC**: Represents the model's ability to distinguish between classes across all thresholds.
        """)

elif st.session_state.current_page == "About the Project":
    st.markdown("<h1 class='title-text'>About the Project</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Academic Prototype Architecture & Context</p>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Background
    This project was created as an academic prototype to demonstrate how machine learning algorithms (specifically classification models like Logistic Regression, Random Forest, and XGBoost) can be applied to patient-level lifestyle and clinical risk data to predict experimental risk levels.
    
    ### Pipeline Architecture
    Below is a layout of the end-to-end processing pipeline developed for this application:
    """)
    
    st.markdown("""
    ```
    +-----------------------------------------------------------+
    |                 Raw Dataset Acquisition                   |
    |  - Automated download from public research repositories    |
    |  - Saving patient demographic and lifestyle attributes    |
    +-----------------------------------------------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |               Data Cleaning & Isolation                   |
    |  - EXCLUSION of post-diagnosis fields (leakage mitigation)|
    |  - Split into 80% Train / 20% Test (Stratified)           |
    +-----------------------------------------------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |                 Preprocessing Pipelines                   |
    |  - Numerical Scaling (Age)                                |
    |  - Categorical One-Hot/Mode Imputations (Other factors)   |
    +-----------------------------------------------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |                 Model Training & Selection                |
    |  - Train Logistic Regression, Decision Tree, Random       |
    |    Forest, and XGBoost Classifiers                        |
    |  - Cross-Validation and selection based on Test F1-Score  |
    +-----------------------------------------------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |                  Streamlit User Interface                 |
    |  - Live inputs dynamically mapped from feature_config     |
    |  - Predict risk and analyze local feature attributions    |
    +-----------------------------------------------------------+
    ```
    """)
    
    st.markdown("""
    ### Target Audience
    - **Academic Evaluators**: Grading the ML implementation, training splits, leakage handling, and evaluations.
    - **General Public**: Accessing educational resources to understand how tobacco, alcohol, and clinical signs impact ML estimates.
    
    ### Future Scope
    - Integrating SHAP (SHapley Additive exPlanations) for local game-theory explanation.
    - Calibrating model probabilities to match clinical prevalence rates.
    - Testing on larger, Indian-specific or multi-center clinical validation datasets.
    """)

elif st.session_state.current_page == "Disclaimer":
    st.markdown("<h1 class='title-text'>Medical & Data Privacy Disclaimer</h1>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="risk-card high-risk" style="font-size:1.1rem; line-height:1.6;">
        <strong>⚠️ CRITICAL MEDICAL WARNING:</strong><br>
        This application is an academic machine-learning prototype and <strong>does NOT diagnose oral cancer</strong>, confirm the presence or absence of disease, or provide treatment options. 
        It is not an FDA-approved, CE-certified, or clinically validated medical device.
        <br><br>
        This assessment is intended solely for academic demonstration and general educational risk awareness. 
        <strong>It should never replace clinical screening, biopsy, diagnosis, or advice from a qualified dentist, oral pathologist, or medical professional.</strong>
        <br><br>
        If you have persistent oral symptoms (such as sores that do not heal in 2 weeks, red/white patches, difficulty swallowing, unexplained bleeding, or pain), you must seek immediate clinical evaluation from a healthcare provider.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔒 Data Privacy & PII Handling
    - **No PII Collection**: This application does not collect, request, or store any Personally Identifiable Information (PII) such as names, email addresses, phone numbers, location, or medical record identifiers.
    - **Session Storage Only**: All questionnaire answers and calculations are processed dynamically in the active memory of your web browser. No data is stored, saved to a database, or transmitted to any external server.
    - **Open-source & Local Execution**: You are running this prototype in a localized workspace, ensuring full control over the data.
    """)

# Render Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:0.8rem; color:#a0aec0;'>Oral Cancer Risk Assessment ML Application | Academic Assignment</p>", unsafe_allow_html=True)
