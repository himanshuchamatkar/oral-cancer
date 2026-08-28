import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Fallback import for XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost package is not installed or failed to load. falling back to scikit-learn models.")

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_evaluate():
    ensure_directories()
    
    # Load dataset splits
    train_path = os.path.join(PROCESSED_DIR, "train_raw.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_raw.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed datasets train_raw.csv or test_raw.csv not found. Please run preprocessing first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train_raw = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    
    X_test_raw = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # Load the preprocessing pipeline
    prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessing pipeline not found at {prep_path}. Please run preprocessing first.")
    preprocessor = joblib.load(prep_path)
    
    # Transform raw features
    print("Preprocessing training and test features...")
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Handle class imbalance for XGBoost
    neg_count = len(y_train) - sum(y_train)
    pos_count = sum(y_train)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
        
    # Dict to hold results
    model_results = []
    trained_models = {}
    
    print("\nTraining and evaluating models using 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        
        # Cross-validation for robust training metrics
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        
        print(f"CV F1-Score: {cv_f1:.4f} | CV Recall: {cv_recall:.4f} | CV Accuracy: {cv_accuracy:.4f}")
        
        # Fit model on entire training set
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        # Check if predict_proba is available (should be for all of these)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Calculate test metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"Test Accuracy:  {acc:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall:    {rec:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        print(f"Test ROC-AUC:   {auc:.4f}\n")
        
        model_results.append({
            "Model": model_name,
            "CV Accuracy": cv_accuracy,
            "CV Recall": cv_recall,
            "CV F1-Score": cv_f1,
            "Test Accuracy": acc,
            "Test Precision": prec,
            "Test Recall": rec,
            "Test F1-Score": f1,
            "Test ROC-AUC": auc,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })
        
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(model_results)
    comparison_csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved model comparisons to {comparison_csv_path}")
    
    # Model selection logic: Prioritize Test F1-Score to balance precision & recall,
    # as maximizing recall alone can lead to extreme false alarm rates, and accuracy doesn't reflect minority class detection.
    print("Selecting best model based on Test F1-Score...")
    results_df = results_df.sort_values(by="Test F1-Score", ascending=False)
    best_model_row = results_df.iloc[0]
    best_model_name = best_model_row["Model"]
    best_model = trained_models[best_model_name]
    
    print(f"Selected Best Model: {best_model_name} with Test F1: {best_model_row['Test F1-Score']:.4f} and Recall: {best_model_row['Test Recall']:.4f}")
    
    # Save the selected best model
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Saved best model to {best_model_path}")
    
    # Save metrics of best model to metrics.json
    best_metrics = {
        "selected_model": best_model_name,
        "accuracy": float(best_model_row["Test Accuracy"]),
        "precision": float(best_model_row["Test Precision"]),
        "recall": float(best_model_row["Test Recall"]),
        "f1_score": float(best_model_row["Test F1-Score"]),
        "roc_auc": float(best_model_row["Test ROC-AUC"]),
        "confusion_matrix": {
            "TP": int(best_model_row["TP"]),
            "FP": int(best_model_row["FP"]),
            "FN": int(best_model_row["FN"]),
            "TN": int(best_model_row["TN"])
        }
    }
    
    metrics_json_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=4)
    print(f"Saved best model metrics to {metrics_json_path}")
    
    # Save model metadata (metadata.json)
    # Check what features were used from feature config
    config_path = os.path.join(CONFIG_DIR, "feature_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
         features_list = list(json.load(f).keys())
         
    metadata = {
        "model_type": str(type(best_model).__name__),
        "model_name": best_model_name,
        "features_used": features_list,
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "accuracy": float(best_model_row["Test Accuracy"]),
            "recall": float(best_model_row["Test Recall"]),
            "f1_score": float(best_model_row["Test F1-Score"]),
            "roc_auc": float(best_model_row["Test ROC-AUC"])
        }
    }
    
    metadata_json_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_json_path}")

if __name__ == "__main__":
    train_and_evaluate()
