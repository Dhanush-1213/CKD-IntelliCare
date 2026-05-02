
#  CKD-IntelliCare  
### Hybrid Clinical Decision Support System for Chronic Kidney Disease  

 A production-style healthcare AI system that combines **Machine Learning** and **Clinical Rule-Based Intelligence** to deliver accurate, interpretable, and realistic CKD predictions.

---

##  Project Overview

Chronic Kidney Disease (CKD) is often underdiagnosed in early stages. This project addresses that gap by building a **hybrid decision support system**:

-  **ML Model → Predicts CKD Risk**
-  **Rule-Based Engine → Determines CKD Stage using eGFR**

>  Designed with strict **data leakage prevention**, **realistic evaluation**, and **clinical alignment**

---

##  System Architecture

Patient Data  
↓  
Preprocessing + Leakage Guard  
↓  
Stratified K-Fold Training  
↓  
SMOTE (only on training folds)  
↓  
ML Models (XGBoost & LightGBM)  
↓  
Risk Prediction  
↓  
eGFR Calculation (Creatinine, Age, Sex)  
↓  
Rule-Based CKD Staging (Stage 1–5)  
↓  
Final Clinical Output  

---

##  Key Features

- Hybrid ML + Rule-Based System  
- CKD Risk Classification (Multi-class)  
- eGFR-based Clinical Staging  
- Bulk CSV Prediction Support  
- Streamlit Web Application  
- Leakage-Free Training Pipeline  
- Explainability Ready (SHAP / LIME compatible)  

---

##  Model Details

- Models: XGBoost, LightGBM  
- Evaluation: 5-Fold Stratified Cross Validation  
- Metric Focus: Macro F1 Score  
- Leakage Guard: Removes eGFR & derived features from ML  

---

##  CKD Staging Logic (Clinical Rule-Based)

- Stage 1: ≥ 90  
- Stage 2: 60–89  
- Stage 3: 30–59  
- Stage 4: 15–29  
- Stage 5: < 15  

---

##  Tech Stack

- Python  
- Scikit-learn  
- XGBoost, LightGBM  
- Pandas, NumPy  
- Streamlit  

---

##  How to Run

1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run app: `streamlit run webapp.py`  

---

##  Project Structure

CKD-IntelliCare/  
│  
├── train.py  
├── predict.py  
├── webapp.py  
├── models/  
├── data/  
├── requirements.txt  
└── README.md  

---

##  Author

Dhanush K  
PES University  
https://github.com/Dhanush-1213  

---

⭐ If you like this project, give it a star!
