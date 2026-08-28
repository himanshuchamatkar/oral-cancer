import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

def ensure_directories():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_evaluation_plots():
    ensure_directories()
    
    # Load dataset splits
    test_path = os.path.join(PROCESSED_DIR, "test_raw.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test split not found at {test_path}. Please run preprocessing first.")
    test_df = pd.read_csv(test_path)
    
    X_test_raw = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # Load model and preprocessor
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
    
    if not os.path.exists(best_model_path) or not os.path.exists(prep_path):
        raise FileNotFoundError("Best model or preprocessing pipeline not found. Please train models first.")
        
    model = joblib.load(best_model_path)
    preprocessor = joblib.load(prep_path)
    
    # Preprocess test features
    X_test = preprocessor.transform(X_test_raw)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Setup aesthetic plotting
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14})
    
    # 1. Confusion Matrix
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # 2. ROC Curve
    print("Generating ROC Curve plot...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")
    
    # 3. Feature Importance
    print("Generating Feature Importance plot...")
    feat_names_path = os.path.join(PROCESSED_DIR, "feature_names.json")
    if not os.path.exists(feat_names_path):
        print("Feature names metadata not found; skipping feature importance plot.")
        return
        
    with open(feat_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
        
    # Get importances or coefficients
    importances = None
    title = "Feature Importance"
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importance ({type(model).__name__})"
    elif hasattr(model, "coef_"):
        # For Logistic Regression, use absolute coefficients
        importances = np.abs(model.coef_[0])
        title = f"Feature Coefficients Magnitude ({type(model).__name__})"
        
    if importances is not None and len(importances) == len(feature_names):
        feat_imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15) # Show top 15 features
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
        plt.title(title)
        plt.xlabel("Importance Score" if hasattr(model, "feature_importances_") else "Absolute Coefficient")
        plt.ylabel("Preprocessed Feature")
        plt.tight_layout()
        feat_imp_path = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(feat_imp_path, dpi=150)
        plt.close()
        print(f"Saved Feature Importance to {feat_imp_path}")
    else:
        print("Warning: Importances array size does not match feature names size. Skipping feature importance plot.")

if __name__ == "__main__":
    generate_evaluation_plots()
