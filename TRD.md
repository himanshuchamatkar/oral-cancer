# Technical Requirements Document (TRD)

## Technical Architecture Overview

The system is designed as an end-to-end Machine Learning web application structured for academic grading and risk awareness. It runs locally and maintains data containment strictly within `D:\arishaa jii`.

```
                    +-----------------------------+
                    |        Data Pipeline        |
                    | (src/data_preprocessing.py) |
                    +-----------------------------+
                                   |
                                   | Train/Test split
                                   v
+-----------------------+   Transformed   +--------------------+
| Preprocessing Pipeline | -------------> |   Model Pipeline   |
|   (models/prep.pkl)    |                | (src/train.py)     |
+-----------------------+                +--------------------+
           |                                       |
           | Load fitted                           | Load best fit
           v                                       v
+--------------------------------------------------------------+
|                        Streamlit Web UI                      |
|                           (app.py)                           |
+--------------------------------------------------------------+
```

---

## 1. Data Leakage Mitigation (Critical Design Rule)

A primary risk in clinical risk prediction models is **data leakage**, which occurs when features that are only determined *after* a definitive diagnosis are used as predictors. If included, they make the model artificially powerful (approaching 100% accuracy) but useless for pre-diagnostic risk screening.

The following columns from the raw dataset are strictly **excluded** from training features:
- `Tumor Size (cm)`: Measured post-diagnosis or during biopsy.
- `Cancer Stage`: Defined pathologically after diagnosis.
- `Treatment Type`: Prescribed after diagnosis (e.g. Surgery, Chemotherapy).
- `Survival Rate (5-Year, %)`: Outcome metric.
- `Cost of Treatment (USD)`: Outcome financial metric.
- `Economic Burden (Lost Workdays)`: Outcome metric.
- `Early Diagnosis`: Retrospective assessment of the diagnostic timeline.

The model only uses predictors that can be determined in a pre-clinical questionnaire: age, sex, lifestyle exposures (tobacco, alcohol, betel quid), family history, and visible early symptoms (lesions, bleeding, patches, swallowing difficulty).

---

## 2. Preprocessing & Feature Engineering Pipeline

The data preparation pipeline is built using scikit-learn's `ColumnTransformer` and `Pipeline` objects to prevent fit-leakage (fitting transformers only on the training set):
- **Numerical Features (Age)**: Imputed using the median (`SimpleImputer(strategy='median')`) and standardized to zero-mean and unit-variance (`StandardScaler`).
- **Categorical Features**: Imputed using the most frequent category (`SimpleImputer(strategy='most_frequent')`) and encoded using one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
- **Target Mapping**: Converted from textual outcomes (`Yes` / `No`) to binary integer flags (`1` / `0`).

---

## 3. Modeling & Evaluation Strategy

Four classification models are trained and compared:
1. **Logistic Regression**: Serves as a linear baseline, fitted with `class_weight='balanced'` to adjust for target imbalance.
2. **Decision Tree Classifier**: Fits non-linear boundaries with `max_depth=6` and `class_weight='balanced'` to maintain interpretability.
3. **Random Forest Classifier**: Fits an ensemble of trees with `n_estimators=100`, `max_depth=8`, and `class_weight='balanced'`.
4. **XGBoost Classifier**: A gradient booster utilizing `scale_pos_weight` to address imbalance. Wraps imports in try-except block to gracefully handle systems where xgboost might fail compilation.

### Validation Strategy
- **Stratified 5-Fold Cross-Validation** is run on the training split (80% of data) to check for stability.
- **Hold-out Test Set Evaluation** (20% of data) is used to calculate:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score (Selection Metric)
  - Area Under the ROC Curve (ROC-AUC)
  - Confusion Matrix

---

## 4. Model-Agnostic Explainability (Local Attributions)

Rather than relying on heavy and platform-dependent libraries (like SHAP), the system implements a model-agnostic **Counterfactual Local Attribution** algorithm in `src/explainability.py`:
1. For an entered profile, we compute the baseline prediction probability $P(\text{Risk})$.
2. For each active risk factor $F$ (e.g. user consumes tobacco, has poor hygiene, or has oral lesions), we construct a counterfactual input where $F$ is replaced with its healthy reference baseline (e.g. "Never" tobacco use, "Good" oral hygiene, "No" lesions).
3. We run the model on this counterfactual to obtain $P(\text{Risk} \setminus F)$.
4. The attribution score (contribution) is computed as:
   $$\text{Attribution}_F = P(\text{Risk}) - P(\text{Risk} \setminus F)$$
5. The factors causing the largest positive drop in risk are returned as the **Major Contributing Factors** and described to the user in natural language.
