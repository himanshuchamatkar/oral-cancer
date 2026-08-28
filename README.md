# Oral Cancer Risk Prediction Using Machine Learning

An end-to-end Machine Learning web application that estimates an individual's experimental oral cancer risk level based on demographic, lifestyle, and early clinical factors. This is an academic assignment designed for demonstrations.

---

## 🏥 Critical Medical Disclaimer
**IMPORTANT**: This application is an academic machine-learning prototype intended for educational and risk-awareness purposes only. It **does not diagnose oral cancer**, confirm the presence or absence of cancer, or replace professional medical examination, screening, biopsy, or clinical judgment. Model predictions depend on the dataset and are not clinically validated. If you have persistent oral symptoms or concerns, consult a qualified healthcare professional.

---

## 📁 Project Structure

```
D:\arishaa jii\
│
├── app.py                     # Multi-page Streamlit application
├── requirements.txt           # Project python dependencies
├── README.md                  # Project overview & running instructions
├── PRD.md                     # Product Requirements Document
├── TRD.md                     # Technical Requirements Document
│
├── data/
│   ├── raw/                   # Raw oral cancer CSV dataset
│   ├── processed/             # Train/test splits and metadata
│   └── dataset_source.md      # Provenance and specs of raw data
│
├── models/
│   ├── best_model.pkl         # Trained classification model
│   ├── preprocessing.pkl      # Fitted ColumnTransformer pipeline
│   └── model_metadata.json    # Metrics and details of best model
│
├── notebooks/
│   └── eda_and_model_training.ipynb   # Academic demo notebook
│
├── src/
│   ├── data_preprocessing.py  # Data acquisition, cleaning, splits
│   ├── train.py               # Model training, comparisons, best selection
│   ├── evaluate.py            # Generates Confusion Matrix, ROC, importances
│   ├── predict.py             # Inference API wrapper
│   └── explainability.py      # Model-agnostic counterfactual attributions
│
├── config/
│   └── feature_config.json    # JSON mapping UI inputs to model features
│
└── reports/
    ├── model_comparison.csv   # Scores for all candidate models
    ├── metrics.json           # Performance metrics for best model
    └── figures/               # Evaluation charts (ROC, CM, Feature Importance)
```

---

## 🚀 Setup & Execution Instructions

Follow these steps to run the pipeline and start the application locally:

### 1. Initialize Virtual Environment & Install Dependencies
Ensure you are in the project folder `D:\arishaa jii`:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Acquisition
This downloads the 84,922 row public dataset from the target repository, mitigates data leakage by dropping post-diagnostic columns, splits the data, fits, and saves the preprocessing pipeline.
```powershell
python src/data_preprocessing.py
```

### 3. Run Model Training & Selection
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost (if available), evaluates them using stratified 5-fold cross validation, compares metrics on the test split, selects the best model based on F1-Score, and saves model files.
```powershell
python src/train.py
```

### 4. Generate Performance Evaluations
Generates the Confusion Matrix, ROC curve, and Feature Importance charts for the selected model.
```powershell
python src/evaluate.py
```

### 5. Launch the Streamlit Web Application
Starts the local development server and opens the multi-page web application in your browser.
```powershell
python -m streamlit run app.py
```

---

## 🧬 Machine Learning Performance Summary

The model pipeline is trained on **84,922 patient rows** across 15 features (excluding unique identifiers and 7 data-leakage columns). 

Models are evaluated on:
1. **Accuracy**: Overall prediction correctness.
2. **Precision**: Fraction of predicted high-risk profiles that are true positives.
3. **Recall (Sensitivity)**: Fraction of true positives correctly flagged. **Prioritized in clinical screening to avoid False Negatives.**
4. **F1-Score**: Joint balance of Precision and Recall.

All training splits, comparisons, and generated plots are outputted into `reports/` for academic grading and evaluation.
