
<div align="center">

<br/>


# CKD-IntelliCare

### Hybrid Clinical Decision Support System for Chronic Kidney Disease

*Combining Machine Learning precision with clinical rule-based intelligence — for interpretable, actionable, and trustworthy CKD prediction.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-00C853?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

[![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6F61?style=for-the-badge)](https://shap.readthedocs.io)
[![LIME](https://img.shields.io/badge/LIME-Local%20Interpretability-00ACC1?style=for-the-badge)](https://lime-ml.readthedocs.io)

<br/>

> **"Early detection saves kidneys. Interpretable AI saves trust."**

</div>

---

##  Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [System Architecture](#-system-architecture)
- [Core Features](#-core-features)
- [Modeling Approach](#-modeling-approach)
- [CKD Staging Logic](#-ckd-staging-logic)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)
- [Design Decisions](#-design-decisions)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

##  The Problem

Chronic Kidney Disease (CKD) affects over **850 million people worldwide** and is responsible for approximately **2.4 million deaths annually**. The tragedy? Most cases are entirely preventable — if caught early.

The core challenge:

- CKD is **asymptomatic** in Stages 1–3, meaning patients feel fine while their kidneys deteriorate
- Routine blood tests contain the signals needed for early detection — but interpreting them consistently is hard at scale
- Existing ML solutions often suffer from **data leakage**, **inflated metrics**, and **zero interpretability** — making them untrustworthy for real clinical use

CKD-IntelliCare was built to address all three.

---

##  The Solution

CKD-IntelliCare is a **hybrid clinical decision support system** that separates two fundamentally different tasks:

| Responsibility | Handled By |
|---|---|
| Estimating CKD risk from patient biomarkers | Machine Learning (XGBoost / LightGBM) |
| Assigning clinical CKD stage | Rule-based engine using eGFR (per KDIGO guidelines) |

This separation ensures that **ML does what it's good at** (pattern recognition under uncertainty) while **clinical logic does what it's good at** (deterministic, guideline-compliant staging). The result is a system that is both accurate and interpretable.

---

##  System Architecture

```
┌─────────────────────────────────────────────────┐
│                  PATIENT DATA                   │
│        (Biomarkers, Demographics, Labs)         │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│          PREPROCESSING + LEAKAGE GUARD          │
│  • Missing value imputation                     │
│  • Feature encoding & normalization             │
│  • Strict removal of eGFR-derived variables     │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                        ▼
┌──────────────────┐    ┌──────────────────────────┐
│  STRATIFIED      │    │  eGFR CALCULATION        │
│  K-FOLD CV       │    │  (Creatinine, Age, Sex)  │
│                  │    └────────────┬─────────────┘
│  SMOTE on train  │                 │
│  folds only      │                 ▼
│                  │    ┌──────────────────────────┐
│  Model Training  │    │  RULE-BASED CKD STAGING  │
│  XGBoost /       │    │  (KDIGO Guidelines)      │
│  LightGBM        │    └────────────┬─────────────┘
└──────┬───────────┘                 │
       │                             │
       ▼                             ▼
┌─────────────────────────────────────────────────┐
│              FINAL CLINICAL OUTPUT              │
│   Risk Score · CKD Stage · Confidence · SHAP   │
└─────────────────────────────────────────────────┘
```

---

##  Core Features

###  Hybrid Intelligence
A two-layer architecture that combines probabilistic ML predictions with deterministic clinical staging — mimicking how a nephrologist actually thinks.

###  Strict Leakage Prevention
eGFR and all eGFR-derived variables are explicitly excluded from ML features. This is non-negotiable — including them would make the model look brilliant on paper while being useless in deployment.

###  Clinically Honest Evaluation
SMOTE is applied **within training folds only** — never touching validation data. This prevents the silent optimism that plagues most medical ML papers.

###  Bulk CSV Inference
Upload a patient cohort file and receive predictions for the entire dataset in seconds — designed for real-world clinical workflows, not toy demos.

###  Explainability-Ready
Architecture is designed for SHAP and LIME integration, so clinicians can understand *why* a risk score was assigned, not just *what* it is.

###  Interactive Streamlit Interface
A clean, intuitive UI for single-patient assessment and bulk inference — no API knowledge required.

---

##  Modeling Approach

| Component | Choice | Rationale |
|---|---|---|
| **Primary Models** | XGBoost, LightGBM | State-of-the-art tabular performance; native handling of missing values |
| **Validation Strategy** | Stratified 5-Fold CV | Ensures class distribution is preserved across every fold |
| **Optimization Metric** | Macro F1 Score | Penalizes ignoring minority classes — critical in medical settings |
| **Imbalance Handling** | SMOTE (train folds only) | Synthetic oversampling without contaminating validation sets |
| **Leakage Control** | Explicit feature blocklist | eGFR, serum creatinine ratios, and all derived renal variables excluded |

> **Why Macro F1 and not accuracy?**  
> In CKD datasets, the majority class can represent 70–80% of samples. A model predicting "no CKD" for everyone would achieve high accuracy. Macro F1 forces the model to perform well on every stage — especially the critical early ones.

---

##  CKD Staging Logic

Staging follows **KDIGO 2012 Clinical Practice Guidelines**, using the CKD-EPI equation for eGFR estimation.

```
eGFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(−1.209) × 0.993^Age × [1.018 if Female]
```

| Stage | eGFR (mL/min/1.73m²) | Description | Clinical Action |
|---|---|---|---|
| **Stage 1** | ≥ 90 | Normal or high | Monitor; manage risk factors |
| **Stage 2** | 60 – 89 | Mildly decreased | Monitor; lifestyle intervention |
| **Stage 3** | 30 – 59 | Moderately decreased | Specialist referral recommended |
| **Stage 4** | 15 – 29 | Severely decreased | Prepare for renal replacement |
| **Stage 5** | < 15 | Kidney failure | Dialysis or transplant |

The rule engine handles edge cases, missing creatinine values, and sex-based coefficient correction automatically.

---

##  Project Structure

```
CKD-IntelliCare/
│
├──  train.py              # End-to-end model training pipeline
│   ├── Data loading & validation
│   ├── Leakage guard (feature blocklist)
│   ├── Stratified K-Fold + SMOTE
│   └── Model serialization
│
├──  predict.py            # Inference engine
│   ├── Single-patient prediction
│   ├── Bulk CSV inference
│   ├── eGFR calculation (CKD-EPI)
│   └── Rule-based staging
│
├──  webapp.py             # Streamlit application
│   ├── Patient input forms
│   ├── Bulk upload interface
│   └── Results dashboard
│
├──  models/               # Serialized artifacts
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── preprocessor.pkl
│
├──  data/                 # Dataset
│   └── ckd_dataset.csv
│
├──  requirements.txt
└──  README.md
```

---

##  Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Clone & Install

```bash
git clone https://github.com/Dhanush-1213/CKD-IntelliCare.git
cd CKD-IntelliCare
pip install -r requirements.txt
```

### Train the Models

```bash
python train.py
```

This will preprocess data, run stratified cross-validation, apply SMOTE, train both models, and serialize them to `models/`.

### Launch the App

```bash
streamlit run webapp.py
```

The interface will open at `http://localhost:8501`.

---

## 🚀 Usage Guide

### Single Patient Assessment

1. Open the Streamlit app
2. Enter patient biomarkers in the input form (serum creatinine, age, sex, hemoglobin, etc.)
3. Click **Predict** to receive:
   - CKD risk probability and class
   - eGFR value
   - Rule-based CKD stage
   - Feature importance visualization (if SHAP is enabled)

### Bulk Inference via CSV

Prepare a CSV with the required feature columns (see `data/sample_input.csv` for the template):

```bash
python predict.py --input patient_cohort.csv --output predictions.csv
```

Output includes one row per patient with risk score, predicted class, eGFR, and CKD stage.

---

##  Design Decisions

**Why separate ML from staging?**  
eGFR-based staging is a clinical standard — it should never be "learned" from data. Letting ML invent its own staging criteria would produce a black box with no clinical validity. Separation keeps each component auditable and independently updatable.

**Why not use a single end-to-end deep learning model?**  
Deep learning offers marginal gains on tabular medical data while sacrificing interpretability entirely. For a clinical decision support tool, the ability to explain a prediction to a doctor is not optional.

**Why SMOTE within folds only?**  
Applying SMOTE before splitting creates data leakage — synthetic samples derived from the full dataset appear in both training and validation sets, causing artificially inflated metrics. Fold-internal SMOTE is the only honest approach.

---

##  Roadmap

- [ ] SHAP global and local explanation views in webapp
- [ ] LIME per-prediction explanation panel
- [ ] GFR trajectory tracking (longitudinal patient support)
- [ ] FHIR-compatible patient data ingestion
- [ ] REST API wrapper for EHR system integration
- [ ] Uncertainty quantification (prediction confidence intervals)
- [ ] Albuminuria-based staging (full KDIGO G + A grid)

---

##  Author

<div align="center">

**Dhanush K**  
B.Tech Computer Science — Artificial Intelligence & Machine Learning  
PES University

[![GitHub](https://img.shields.io/badge/GitHub-Dhanush--1213-181717?style=for-the-badge&logo=github)](https://github.com/Dhanush-1213)

*Built with the belief that AI in healthcare must earn trust — one interpretable prediction at a time.*

</div>

---

<div align="center">

If CKD-IntelliCare was useful to you, consider leaving a ⭐ — it helps others find the project.

</div>
