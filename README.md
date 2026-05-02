# A Hybrid Machine Learning and Rule-Based Clinical Decision Support System for CKD

## Problem Statement

Chronic Kidney Disease (CKD) screening can be difficult because laboratory values, urine findings, and clinical history need to be interpreted together. This project is a research-oriented decision-support prototype that separates ML-based CKD status prediction from rule-based CKD stage interpretation.

This system is not a doctor replacement and does not provide a guaranteed diagnosis.

## Project Objective

The objective is to build an academic prototype that:

- predicts CKD status using machine learning,
- determines CKD stage using eGFR-based rules,
- reports probability-based confidence,
- provides stage-specific recommendations for qualified clinical review.

## Hybrid Approach

The project uses two separate components:

1. Machine-learning model: predicts the real UCI clinical label:
   - CKD
   - No_CKD
2. Rule-based stage engine: maps user-provided eGFR to CKD stage:
   - Stage 1: eGFR >= 90
   - Stage 2: 60 <= eGFR < 90
   - Stage 3a: 45 <= eGFR < 60
   - Stage 3b: 30 <= eGFR < 45
   - Stage 4: 15 <= eGFR < 30
   - Stage 5: eGFR < 15

eGFR is not used by the ML model. It is used only by the rule-based stage engine.

## Dataset Description

The dataset is the UCI Machine Learning Repository Chronic Kidney Disease dataset:

- Dataset: Chronic Kidney Disease
- Instances: 400
- Features: 24 clinical features plus class label
- Target labels: `ckd`, `notckd`
- DOI: `10.24432/C5G020`

The converted CSV is stored at:

```text
dataset/kidney_disease_dataset.csv
```

The original UCI variables include age, blood pressure, specific gravity, albumin, sugar, red blood cells, pus cells, pus cell clumps, bacteria, random blood glucose, blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count, hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, anemia, and class.

## Data Leakage Prevention

The pipeline removes columns that directly expose eGFR, CKD stage, prediction outputs, target-derived labels, risk scores, or stage-derived values. The model does not train on:

- eGFR
- eGFR-derived variables
- CKD stage
- stage-derived columns
- target-derived columns
- generated proxy labels

The earlier circular proxy target was removed. The model now trains only on the real UCI clinical class label.

## Model Details

The final training design keeps a simple ensemble comparison:

- XGBoost
- LightGBM
- Soft Voting Ensemble using XGBoost and LightGBM

CatBoost, StackingClassifier, Optuna, nested ensembling, artificial proxy labels, and artificial accuracy-window model selection were removed to reduce complexity and avoid result engineering.

## Soft Voting Explanation

The soft voting model trains XGBoost and LightGBM as base learners. During prediction, each model outputs class probabilities. The ensemble averages those probabilities and chooses the class with the highest averaged probability.

In the bundled validation run, the Soft Voting Ensemble has the same Accuracy, Macro F1, and Weighted F1 as XGBoost because the averaged probabilities produce the same class labels on the evaluated folds. Its ROC-AUC differs, which confirms that the ensemble is fitted separately and is not accidentally reusing the XGBoost model object.

## Training Pipeline

1. Load the UCI CKD CSV.
2. Clean column names.
3. Normalize categorical text.
4. Detect the real target column.
5. Drop leakage columns.
6. Encode the target with LabelEncoder.
7. Build a ColumnTransformer:
   - numerical features: median imputation plus StandardScaler
   - categorical features: most-frequent imputation plus OneHotEncoder
8. Apply SMOTE inside the training folds only.
9. Tune XGBoost and LightGBM with 5-fold GridSearchCV.
10. Build a soft voting ensemble from the tuned XGBoost and LightGBM models.
11. Evaluate XGBoost, LightGBM, and the ensemble using 5-fold StratifiedKFold.
12. Report mean and standard deviation for Accuracy, Precision Macro, Recall Macro, Macro F1, Weighted F1, and ROC-AUC.
13. Select the best model by mean Macro F1.
14. Save final model artifacts.

## Evaluation Metrics

The training script prints:

- Accuracy mean +/- standard deviation
- Precision Macro mean +/- standard deviation
- Recall Macro mean +/- standard deviation
- Macro F1 mean +/- standard deviation
- Weighted F1 mean +/- standard deviation
- ROC-AUC mean +/- standard deviation
- Out-of-fold classification report for the selected model
- Out-of-fold confusion matrix for the selected model
- Held-out test-set confusion matrix plot
- Held-out test-set ROC curve with AUC value

The model comparison table is saved to:

```text
models/model_metrics.csv
```

The selected model is chosen by Macro F1 because CKD screening is class-sensitive and accuracy alone can be misleading.

High accuracy is influenced by small dataset size; external validation is required for real-world deployment.

## Validation Artifacts

Running `python train.py` creates publication-oriented validation outputs under:

```text
models/validation_artifacts/
```

Generated artifacts include:

- `confusion_matrix.png`: held-out test confusion matrix with true and predicted labels.
- `roc_curve.png`: held-out test ROC curve with AUC shown in the legend.
- `ablation_study.csv`: comparison of the untuned LightGBM baseline and the tuned deployed model. Because the UCI CSV does not contain an eGFR column, this artifact measures hyperparameter tuning rather than eGFR removal.
- `soft_voting_diagnostic.txt`: note explaining why Soft Voting and XGBoost can share label-based metrics while still being separate fitted models.
- `shap_summary.png`: SHAP TreeExplainer summary plot.
- `shap_feature_importance.csv`: mean absolute SHAP feature ranking.
- `lime_sample_explanation.html`: LIME explanation for one sample prediction.
- `lime_sample_contributions.csv`: local feature contributions for the explained sample.
- `lime_sample_metadata.csv`: sample index, labels, probability, and selection rationale for the local LIME explanation.

## Explainability

The explainability layer uses:

- SHAP TreeExplainer for global feature contribution analysis.
- Mean absolute SHAP values for feature importance ranking.
- LIME tabular explanations for one local prediction.

The explanations are intended to support reviewer inspection and model transparency. They do not convert the prototype into a clinically validated diagnostic device.

## Hyperparameter-Tuning Ablation

The UCI CKD CSV used in this project does not contain an eGFR column. The leakage guard still searches for eGFR-like, stage-derived, prediction-derived, and target-derived columns, but there is no real eGFR-removal ablation to perform on this raw dataset.

The reported ablation therefore compares the untuned LightGBM baseline against the tuned deployed model:

| Model Version | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Untuned LightGBM baseline | computed at training time | computed at training time |
| Tuned deployed model | computed at training time | computed at training time |

The final deployed model remains leakage-safe. If a future dataset contains eGFR or eGFR-derived variables, the guard removes those columns from machine-learning training and reserves eGFR for the rule-based CKD stage engine.

## Limitations

- The UCI CKD dataset contains only 400 records, so reported performance can be optimistic.
- The project does not include external hospital validation.
- Missing values are handled statistically and may not reflect real clinical workflows.
- The model predicts CKD status only; CKD staging is handled separately through rule-based eGFR thresholds.
- This system is an academic decision-support prototype, not a medical device.

## How to Run

Create and activate a Python environment using Python 3.9 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run a CSV prediction:

```bash
python predict.py --input dataset/kidney_disease_dataset.csv --output predictions.csv
```

Start the Streamlit app:

```bash
streamlit run webapp.py
```

Then open:

```text
http://localhost:8501
```

## Folder Structure

```text
CKD-Clinical-Support/
├── dataset/
│   └── kidney_disease_dataset.csv
├── models/
│   ├── best_model.joblib
│   ├── preprocessors.joblib
│   ├── target_encoder.joblib
│   └── model_metrics.csv
├── train.py
├── predict.py
├── webapp.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

The model files are generated after running `python train.py`.

## Screenshots

Add screenshots after running the Streamlit app:

- Single prediction result
- Bulk CSV prediction result
- Model metrics table

## Research Paper Notes

In the methodology section, describe this as a hybrid ML and rule-based decision-support prototype. State that the synthetic-looking dataset and circular proxy target were removed. The final ML model uses the real UCI CKD labels, while the leakage guard programmatically removes eGFR-like, stage-derived, prediction-derived, and target-derived columns from ML training. In the current UCI CSV, no eGFR column is present; eGFR is used only when the Streamlit rule-based staging engine receives a user-provided value.

Mention that CatBoost, stacking, and Optuna were removed to reduce complexity. XGBoost and LightGBM were selected because gradient-boosted trees are strong baselines for small tabular clinical datasets. The soft voting ensemble averages class probabilities from XGBoost and LightGBM.

Report 5-fold StratifiedKFold results as mean +/- standard deviation. Select the final model by Macro F1 and explain that Macro F1 is preferred because accuracy can hide class imbalance. When discussing the ablation table, label it as the effect of hyperparameter tuning unless a future dataset actually contains eGFR and a genuine eGFR-removal experiment is run.

## Dataset Citation

Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020

## License

This project is released under the MIT License. See `LICENSE` for details.

## Disclaimer

This project is an academic prototype for decision support and research. It is not clinical advice, not a doctor replacement, and not a guaranteed diagnosis. Any medical interpretation should be reviewed by qualified healthcare professionals.
