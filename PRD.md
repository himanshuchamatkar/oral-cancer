# Product Requirements Document (PRD)

## Project Name
**Oral Cancer Risk Prediction Using Machine Learning**

## Project Folder
`D:\arishaa jii`

## 1. Project Overview
The project is a web-based machine-learning application that estimates an individual's **oral cancer risk level** based on selected demographic, lifestyle and oral-health-related risk factors.
The user will answer questions through dropdowns, radio buttons, numeric fields and other controlled inputs.
After submitting the form, the trained machine-learning model will analyze the provided risk factors and return:
- Predicted risk category
- Model probability/risk score
- Major contributing factors
- General awareness recommendation
- Medical disclaimer

The application MUST NOT claim that the user has cancer.
It MUST NOT replace a doctor, dentist, oral pathologist or clinical screening.

## 2. Problem Statement
Oral cancer has multiple associated risk factors including tobacco exposure, smoking, alcohol consumption, areca/betel-quid use, age, oral lesions and other demographic/clinical factors.
The purpose of this academic project is to demonstrate how machine-learning algorithms can learn patterns from tabular risk-factor data and generate an experimental risk estimate.
The system will provide an easy-to-use interface where a user can enter risk-factor information without needing technical knowledge.

## 3. Project Goal
Build an end-to-end machine-learning application that:
1. Obtains a suitable public tabular oral-cancer/risk-factor dataset automatically.
2. Stores the dataset inside the project folder.
3. Cleans and preprocesses the dataset.
4. Performs exploratory data analysis.
5. Trains multiple ML classification models.
6. Compares their performance.
7. Selects an appropriate model based on validation metrics.
8. Saves the trained model.
9. Provides a Streamlit web interface.
10. Accepts user responses through controlled form inputs.
11. Produces an experimental risk estimate.
12. Displays the result clearly.
13. Explains important contributing factors where technically possible.
14. Clearly communicates that the system is NOT a medical diagnostic tool.

## 4. Target Users
- **Primary User**: Student / academic evaluator demonstrating a machine-learning healthcare project.
- **Secondary User**: A general user who wants to understand how common risk factors affect an experimental ML-based risk estimate.

## 5. Core User Flow & Inputs
- **Demographics**: Age, Gender
- **Lifestyle**: Smoking, Tobacco Use, Alcohol, Betel Quid / Areca Nut, Poor Oral Hygiene, Diet
- **Medical/Family**: Family History of Cancer, HPV Infection, Compromised Immune System
- **Clinical Symptoms**: Oral Lesions, Persistent White/Red Patches, Unexplained Bleeding, Difficulty Swallowing

## 6. Dataset-Driven UI Rule
The UI MUST be generated based on the actual features used by the trained model.
Every production input must have a corresponding model feature.
The final application contains a feature configuration file describing features, displays, and options.

## 7. Model Selection and Evaluation
Train and compare:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost (with fallback)

Metrics required: Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix. Prioritize F1-Score/Recall to manage false negatives.

## 8. Disclaimer
Prominently display that the system is an academic prototype and does not replace medical screening. Do not collect Personally Identifiable Information (PII).
