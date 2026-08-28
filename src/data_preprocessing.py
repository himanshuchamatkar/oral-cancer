import os
import urllib.request
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Constants
DATA_DIR = os.path.join("D:\\arishaa jii", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join("D:\\arishaa jii", "models")
CONFIG_DIR = os.path.join("D:\\arishaa jii", "config")

DATASET_URL = "https://raw.githubusercontent.com/BhaveshBhakta/Oral-Cancer-Prediction-Using-ML/main/oral_cancer_prediction_dataset.csv"
RAW_FILE_PATH = os.path.join(RAW_DIR, "oral_cancer_prediction_dataset.csv")

def ensure_directories():
    """Ensure that all required directories exist."""
    for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, CONFIG_DIR]:
        os.makedirs(directory, exist_ok=True)

def download_dataset():
    """Download the oral cancer dataset programmatically."""
    ensure_directories()
    if not os.path.exists(RAW_FILE_PATH):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, RAW_FILE_PATH)
        print(f"Dataset downloaded and saved to {RAW_FILE_PATH}")
    else:
        print(f"Dataset already exists at {RAW_FILE_PATH}")

def write_dataset_source():
    """Write metadata documentation for the dataset source."""
    source_md_path = os.path.join(DATA_DIR, "dataset_source.md")
    content = """# Dataset Provenance

## Source
- **URL**: [GitHub Repository - Oral Cancer Prediction Using ML](https://github.com/BhaveshBhakta/Oral-Cancer-Prediction-Using-ML)
- **Direct Download Link**: [oral_cancer_prediction_dataset.csv](https://raw.githubusercontent.com/BhaveshBhakta/Oral-Cancer-Prediction-Using-ML/main/oral_cancer_prediction_dataset.csv)
- **Provenance**: Public dataset (originally sourced from Kaggle) containing patient-level oral cancer demographics, lifestyle factors, and clinical symptoms.

## Specifications
- **Number of Rows**: 84,922
- **Number of Features (Raw)**: 24 features + 1 target
- **Target Variable**: `Oral Cancer (Diagnosis)` ('Yes' / 'No')

## Data Leakage Mitigation
To prevent misleadingly high performance (near 100% accuracy), the following variables are strictly excluded from prediction models as they are determined post-diagnosis or represent the diagnosis directly:
- `Tumor Size (cm)`
- `Cancer Stage`
- `Treatment Type`
- `Survival Rate (5-Year, %)`
- `Cost of Treatment (USD)`
- `Economic Burden (Lost Workdays per Year)`
- `Early Diagnosis`
"""
    with open(source_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Dataset source documentation written to {source_md_path}")

def load_feature_config():
    """Load configuration for features used in prediction."""
    config_path = os.path.join(CONFIG_DIR, "feature_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_and_split():
    """Clean the raw dataset, build and fit preprocessing pipeline, and save splits."""
    ensure_directories()
    download_dataset()
    write_dataset_source()
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(RAW_FILE_PATH)
    
    # Load feature configuration
    feature_config = load_feature_config()
    
    # Extract prediction feature columns based on config keys/dataset_columns
    feature_mapping = {feat: info["dataset_column"] for feat, info in feature_config.items()}
    dataset_cols_to_keep = list(feature_mapping.values())
    
    target_col = "Oral Cancer (Diagnosis)"
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    
    X = df[dataset_cols_to_keep].copy()
    y = df[target_col].map({"Yes": 1, "No": 0})
    
    # Split into train and test sets (80% train, 20% test) with stratification
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify numerical and categorical features
    numerical_cols = []
    categorical_cols = []
    
    for feat, info in feature_config.items():
        col_name = info["dataset_column"]
        if info["type"] == "number":
            numerical_cols.append(col_name)
        else:
            categorical_cols.append(col_name)
            
    print(f"Numerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    # Define preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    # Fit the preprocessing pipeline on training data
    print("Fitting preprocessing pipeline on training data...")
    preprocessor.fit(X_train)
    
    # Save the preprocessing pipeline
    prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
    joblib.dump(preprocessor, prep_path)
    print(f"Saved preprocessing pipeline to {prep_path}")
    
    # Save raw split data sets (with columns matching config)
    train_df = X_train.copy()
    train_df["target"] = y_train
    
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    train_path = os.path.join(PROCESSED_DIR, "train_raw.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_raw.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved raw splits to {train_path} and {test_path}")
    
    # Also save transformed array shapes for verification
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"Transformed train shape: {X_train_trans.shape}")
    print(f"Transformed test shape: {X_test_trans.shape}")
    
    # Save the transformed categorical feature names for downstream diagnostics
    # This helps get column names after one-hot encoding
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        onehot_features = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(onehot_features)
        
        # Save feature names list
        feat_names_path = os.path.join(PROCESSED_DIR, "feature_names.json")
        with open(feat_names_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=4)
        print(f"Saved feature names list to {feat_names_path}")
    except Exception as e:
        print(f"Warning: Could not save feature names: {e}")

if __name__ == "__main__":
    preprocess_and_split()
