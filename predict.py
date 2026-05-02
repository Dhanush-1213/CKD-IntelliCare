"""Prediction helpers for the hybrid CKD decision-support prototype."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Any

# Keep optional library caches inside the project/sandbox.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

import joblib
import numpy as np
import pandas as pd


MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_PREPROCESSOR_PATH = MODELS_DIR / "preprocessors.joblib"
DEFAULT_ENCODER_PATH = MODELS_DIR / "target_encoder.joblib"


STAGE_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "Stage 1": {
        "Key Goals": "Protect kidney function and manage underlying risk factors.",
        "Dietary Focus": "Balanced, heart-healthy meals with moderate sodium intake.",
        "Lifestyle": "Stay active, maintain healthy weight, avoid tobacco, and monitor blood pressure.",
        "What to Watch": "Albumin in urine, blood pressure trends, diabetes control, and repeat eGFR.",
    },
    "Stage 2": {
        "Key Goals": "Slow progression and control blood pressure and blood sugar.",
        "Dietary Focus": "Limit sodium and favor minimally processed foods.",
        "Lifestyle": "Regular exercise, medication adherence, and avoidance of nephrotoxic drugs unless prescribed.",
        "What to Watch": "Rising creatinine, urine albumin, hypertension, and worsening diabetes markers.",
    },
    "Stage 3": {
        "Key Goals": "Prevent complications and consider nephrology follow-up.",
        "Dietary Focus": "Renal-aware diet with attention to sodium, protein, phosphorus, and potassium as advised.",
        "Lifestyle": "Track home blood pressure, keep diabetes controlled, and review medicines with a clinician.",
        "What to Watch": "Anemia, bone-mineral abnormalities, edema, fatigue, and declining eGFR.",
    },
    "Stage 3a": {
        "Key Goals": "Prevent progression and monitor early CKD complications.",
        "Dietary Focus": "Renal-aware diet with attention to sodium and individualized protein guidance.",
        "Lifestyle": "Track blood pressure, review medicines, and maintain diabetes control when relevant.",
        "What to Watch": "Urine albumin, blood pressure, anemia, mineral balance, and eGFR trend.",
    },
    "Stage 3b": {
        "Key Goals": "Increase monitoring and manage complications more actively.",
        "Dietary Focus": "Individualized renal diet with closer attention to sodium, protein, potassium, and phosphorus.",
        "Lifestyle": "Consider nephrology follow-up, medication review, and symptom tracking.",
        "What to Watch": "Anemia, bone-mineral abnormalities, acidosis, edema, fatigue, and declining eGFR.",
    },
    "Stage 4": {
        "Key Goals": "Manage complications and prepare for advanced CKD care planning.",
        "Dietary Focus": "Individualized renal diet with stricter sodium, potassium, phosphorus, protein, and fluid guidance.",
        "Lifestyle": "Close follow-up, careful medication review, and planning for possible renal replacement therapy.",
        "What to Watch": "Fluid overload, electrolyte imbalance, acidosis, anemia, nausea, and worsening symptoms.",
    },
    "Stage 5": {
        "Key Goals": "Urgent specialist-led kidney failure management.",
        "Dietary Focus": "Dialysis- or kidney-failure-specific nutrition plan under clinical supervision.",
        "Lifestyle": "Follow nephrology instructions closely and seek timely care for severe symptoms.",
        "What to Watch": "Severe fatigue, breathlessness, swelling, confusion, chest pain, and dangerous electrolyte changes.",
    },
    "Stage unavailable": {
        "Key Goals": "Provide eGFR to determine CKD stage.",
        "Dietary Focus": "Use general kidney-healthy eating until stage is available.",
        "Lifestyle": "Discuss kidney risk factors with a qualified clinician.",
        "What to Watch": "Missing or invalid eGFR value.",
    },
}


def clean_column_name(column: object) -> str:
    text = str(column).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [clean_column_name(column) for column in cleaned.columns]
    return cleaned


def normalize_text_value(value: object) -> object:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = text.replace("\t", "").replace(",", "")
    if text in {"", "nan", "none", "null", "?", "-", "--"}:
        return np.nan

    text = re.sub(r"\s+", " ", text)
    replacements = {
        "notpresent": "not present",
        "not present": "not present",
        "notckd": "No_CKD",
        "no ckd": "No_CKD",
        "no_ckd": "No_CKD",
        "ckd": "CKD",
    }
    return replacements.get(text, text)


def normalize_categorical_text(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.select_dtypes(include=["object", "category"]).columns:
        normalized[column] = normalized[column].map(normalize_text_value)
    return normalized


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    for column in coerced.columns:
        if not pd.api.types.is_object_dtype(coerced[column]):
            continue

        non_missing = coerced[column].notna().sum()
        if non_missing == 0:
            continue

        numeric = pd.to_numeric(coerced[column], errors="coerce")
        parsed_ratio = numeric.notna().sum() / non_missing
        if parsed_ratio >= 0.9:
            coerced[column] = numeric

    return coerced


def is_leakage_column(column: str, target_column: str | None = None) -> bool:
    if target_column and column == target_column:
        return True

    leakage_patterns = (
        "egfr",
        "estimated_glomerular_filtration",
        "glomerular_filtration_rate",
        "ckd_stage",
        "disease_stage",
        "stage_",
        "_stage",
        "risk_score",
        "stage_score",
        "egfr_score",
        "target_encoded",
        "target_label",
        "predicted",
        "prediction",
    )

    if any(pattern in column for pattern in leakage_patterns):
        return True

    if re.search(r"(^|_)stage($|_)", column):
        return True

    if column in {"target", "label", "class", "classification"}:
        return True

    return False


def load_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    encoder_path: Path = DEFAULT_ENCODER_PATH,
) -> tuple[Any, dict[str, Any], Any]:
    model = joblib.load(model_path)
    preprocessors = joblib.load(preprocessor_path)
    target_encoder = joblib.load(encoder_path)
    return model, preprocessors, target_encoder


def get_row_value(row: pd.Series, aliases: tuple[str, ...]) -> Any | None:
    """Return a value from a row using cleaned-name aliases."""
    cleaned_aliases = {clean_column_name(alias) for alias in aliases}
    for column in row.index:
        cleaned = clean_column_name(column)
        if cleaned in cleaned_aliases:
            return row[column]
    return None


def calculate_egfr_ckd_epi_2021(
    age: float | int | str | None,
    serum_creatinine: float | int | str | None,
    sex: str | None = "male",
) -> float | None:
    """
    Calculate adult eGFR using the 2021 CKD-EPI creatinine equation.

    Required inputs:
    - age in years
    - serum creatinine in mg/dL
    - sex: male/female

    Note: CKD-EPI is an adult equation. For age < 18, this function returns None.
    """
    age_value = pd.to_numeric(age, errors="coerce")
    scr_value = pd.to_numeric(serum_creatinine, errors="coerce")

    if pd.isna(age_value) or pd.isna(scr_value):
        return None

    age_value = float(age_value)
    scr_value = float(scr_value)

    if age_value < 18 or scr_value <= 0:
        return None

    sex_text = str(sex or "male").strip().lower()
    is_female = sex_text in {"female", "f", "woman", "yes_female"}

    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    female_factor = 1.012 if is_female else 1.0

    scr_by_kappa = scr_value / kappa
    egfr = (
        142
        * min(scr_by_kappa, 1) ** alpha
        * max(scr_by_kappa, 1) ** -1.200
        * 0.9938 ** age_value
        * female_factor
    )
    return round(float(egfr), 2)


def get_egfr_value(row: pd.Series) -> float | None:
    # 1) If the user/dataset already provides eGFR, use it directly.
    for column in row.index:
        cleaned = clean_column_name(column)
        if cleaned == "egfr" or "egfr" in cleaned:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                return float(value)

    # 2) Otherwise calculate eGFR from age + serum creatinine + sex.
    age = get_row_value(
        row,
        (
            "age",
            "age_of_the_patient",
            "patient_age",
            "age_years",
        ),
    )
    serum_creatinine = get_row_value(
        row,
        (
            "serum_creatinine",
            "serum_creatinine_mg_dl",
            "serum creatinine (mg/dl)",
            "sc",
            "scr",
        ),
    )
    sex = get_row_value(
        row,
        (
            "sex",
            "gender",
            "patient_sex",
            "patient_gender",
        ),
    )

    return calculate_egfr_ckd_epi_2021(age, serum_creatinine, sex)


def get_ckd_stage(egfr: float | int | str | None) -> str:
    if egfr is None:
        return "Stage unavailable"

    value = pd.to_numeric(egfr, errors="coerce")
    if pd.isna(value):
        return "Stage unavailable"

    value = float(value)
    if value >= 90:
        return "Stage 1"
    if 60 <= value < 90:
        return "Stage 2"
    if 45 <= value < 60:
        return "Stage 3a"
    if 30 <= value < 45:
        return "Stage 3b"
    if 15 <= value < 30:
        return "Stage 4"
    return "Stage 5"


def prepare_features(df: pd.DataFrame, preprocessors: dict[str, Any]) -> Any:
    cleaned = clean_columns(df)
    cleaned = normalize_categorical_text(cleaned)
    cleaned = coerce_numeric_like_columns(cleaned)

    target_column = preprocessors.get("target_column")
    leakage_columns = set(preprocessors.get("leakage_columns", []))
    leakage_columns.update(
        column for column in cleaned.columns if is_leakage_column(column, target_column)
    )
    cleaned = cleaned.drop(columns=list(leakage_columns), errors="ignore")

    feature_columns = list(preprocessors["feature_columns"])
    for column in feature_columns:
        if column not in cleaned.columns:
            cleaned[column] = np.nan

    return cleaned[feature_columns]


def predict_dataframe(
    input_data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    model: Any | None = None,
    preprocessors: dict[str, Any] | None = None,
    target_encoder: Any | None = None,
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)

    if model is None or preprocessors is None or target_encoder is None:
        model, preprocessors, target_encoder = load_artifacts()

    egfr_values = [get_egfr_value(row) for _, row in df.iterrows()]
    stages = [get_ckd_stage(egfr) for egfr in egfr_values]
    features = prepare_features(df, preprocessors)

    if preprocessors.get("model_expects_raw_features", False):
        model_input = features
    else:
        model_input = preprocessors["preprocessor"].transform(features)

    encoded_predictions = model.predict(model_input)
    prediction_labels = target_encoder.inverse_transform(encoded_predictions.astype(int))

    probabilities = model.predict_proba(model_input)
    confidence_scores = probabilities.max(axis=1)

    output = pd.DataFrame(
        {
            "Prediction Result": prediction_labels,
            "Calculated eGFR": [
                f"{egfr:.2f}" if egfr is not None and pd.notna(egfr) else "Unavailable"
                for egfr in egfr_values
            ],
            "CKD Stage": stages,
            "Confidence Score": [f"{score * 100:.2f}%" for score in confidence_scores],
        }
    )

    for field in ["Key Goals", "Dietary Focus", "Lifestyle", "What to Watch"]:
        output[field] = [
            STAGE_RECOMMENDATIONS.get(stage, STAGE_RECOMMENDATIONS["Stage unavailable"])[
                field
            ]
            for stage in stages
        ]

    return output


def predict_single(input_data: dict[str, Any]) -> dict[str, Any]:
    return predict_dataframe(input_data).iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CKD risk and stage.")
    parser.add_argument("--input", required=True, help="CSV file with patient rows.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    predictions = predict_dataframe(input_df)

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
