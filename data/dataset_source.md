# Dataset Provenance

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
