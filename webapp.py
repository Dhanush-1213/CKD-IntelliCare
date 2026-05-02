"""Streamlit UI for the hybrid CKD decision-support prototype."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from predict import load_artifacts, predict_dataframe


st.set_page_config(page_title="CKD Clinical Support", layout="wide")


FIELD_LABELS = {
    "age_of_the_patient": "Age of the patient",
    "blood_pressure_mm_hg": "Blood pressure (mm/Hg)",
    "specific_gravity_of_urine": "Specific gravity of urine",
    "albumin_in_urine": "Albumin in urine",
    "sugar_in_urine": "Sugar in urine",
    "red_blood_cells_in_urine": "Red blood cells in urine",
    "pus_cells_in_urine": "Pus cells in urine",
    "pus_cell_clumps_in_urine": "Pus cell clumps in urine",
    "bacteria_in_urine": "Bacteria in urine",
    "random_blood_glucose_level_mg_dl": "Random blood glucose level (mg/dl)",
    "blood_urea_mg_dl": "Blood urea (mg/dl)",
    "serum_creatinine_mg_dl": "Serum creatinine (mg/dl)",
    "sodium_level_meq_l": "Sodium level (mEq/L)",
    "potassium_level_meq_l": "Potassium level (mEq/L)",
    "hemoglobin_level_gms": "Hemoglobin level (gms)",
    "packed_cell_volume": "Packed cell volume (%)",
    "white_blood_cell_count_cells_cumm": "White blood cell count (cells/cumm)",
    "red_blood_cell_count_millions_cumm": "Red blood cell count (millions/cumm)",
    "hypertension_yes_no": "Hypertension (yes/no)",
    "diabetes_mellitus_yes_no": "Diabetes mellitus (yes/no)",
    "coronary_artery_disease_yes_no": "Coronary artery disease (yes/no)",
    "appetite_good_poor": "Appetite (good/poor)",
    "pedal_edema_yes_no": "Pedal edema (yes/no)",
    "anemia_yes_no": "Anemia (yes/no)",
    "urine_protein_to_creatinine_ratio": "Urine protein-to-creatinine ratio",
    "urine_output_ml_day": "Urine output (ml/day)",
    "serum_albumin_level": "Serum albumin level",
    "cholesterol_level": "Cholesterol level",
    "parathyroid_hormone_pth_level": "Parathyroid hormone (PTH) level",
    "serum_calcium_level": "Serum calcium level",
    "serum_phosphate_level": "Serum phosphate level",
    "family_history_of_chronic_kidney_disease": "Family history of chronic kidney disease",
    "smoking_status": "Smoking status",
    "body_mass_index_bmi": "Body Mass Index (BMI)",
    "physical_activity_level": "Physical activity level",
    "duration_of_diabetes_mellitus_years": "Duration of diabetes mellitus (years)",
    "duration_of_hypertension_years": "Duration of hypertension (years)",
    "cystatin_c_level": "Cystatin C level",
    "urinary_sediment_microscopy_results": "Urinary sediment microscopy results",
    "c_reactive_protein_crp_level": "C-reactive protein (CRP) level",
    "interleukin_6_il_6_level": "Interleukin-6 (IL-6) level",
    "estimated_glomerular_filtration_rate_egfr": "Estimated Glomerular Filtration Rate (eGFR, optional)",
    "patient_sex": "Patient sex (for eGFR calculation)",
}

DEFAULT_VALUES = {
    "age_of_the_patient": 45.0,
    "blood_pressure_mm_hg": 80.0,
    "specific_gravity_of_urine": 1.015,
    "albumin_in_urine": 0.0,
    "sugar_in_urine": 0.0,
    "random_blood_glucose_level_mg_dl": 120.0,
    "blood_urea_mg_dl": 35.0,
    "serum_creatinine_mg_dl": 1.1,
    "sodium_level_meq_l": 138.0,
    "potassium_level_meq_l": 4.5,
    "hemoglobin_level_gms": 13.0,
    "packed_cell_volume": 40.0,
    "white_blood_cell_count_cells_cumm": 7500.0,
    "red_blood_cell_count_millions_cumm": 4.5,
    "urine_protein_to_creatinine_ratio": 0.2,
    "urine_output_ml_day": 1500.0,
    "serum_albumin_level": 4.0,
    "cholesterol_level": 180.0,
    "parathyroid_hormone_pth_level": 50.0,
    "serum_calcium_level": 9.0,
    "serum_phosphate_level": 3.5,
    "body_mass_index_bmi": 24.0,
    "duration_of_diabetes_mellitus_years": 0.0,
    "duration_of_hypertension_years": 0.0,
    "cystatin_c_level": 1.0,
    "c_reactive_protein_crp_level": 2.0,
    "interleukin_6_il_6_level": 2.0,
    "estimated_glomerular_filtration_rate_egfr": 0.0,
}

CATEGORICAL_OPTIONS = {
    "red_blood_cells_in_urine": ["normal", "abnormal"],
    "pus_cells_in_urine": ["normal", "abnormal"],
    "pus_cell_clumps_in_urine": ["not present", "present"],
    "bacteria_in_urine": ["not present", "present"],
    "hypertension_yes_no": ["no", "yes"],
    "diabetes_mellitus_yes_no": ["no", "yes"],
    "coronary_artery_disease_yes_no": ["no", "yes"],
    "appetite_good_poor": ["good", "poor"],
    "pedal_edema_yes_no": ["no", "yes"],
    "anemia_yes_no": ["no", "yes"],
    "family_history_of_chronic_kidney_disease": ["no", "yes"],
    "smoking_status": ["no", "yes"],
    "physical_activity_level": ["moderate", "low", "high"],
    "urinary_sediment_microscopy_results": ["normal", "abnormal"],
    "patient_sex": ["male", "female"],
}

FIELD_GROUPS = {
    "Patient Profile": [
        "age_of_the_patient",
        "patient_sex",
        "blood_pressure_mm_hg",
        "body_mass_index_bmi",
        "physical_activity_level",
        "smoking_status",
        "family_history_of_chronic_kidney_disease",
    ],
    "Laboratory Values": [
        "random_blood_glucose_level_mg_dl",
        "blood_urea_mg_dl",
        "serum_creatinine_mg_dl",
        "sodium_level_meq_l",
        "potassium_level_meq_l",
        "hemoglobin_level_gms",
        "packed_cell_volume",
        "white_blood_cell_count_cells_cumm",
        "red_blood_cell_count_millions_cumm",
        "serum_albumin_level",
        "cholesterol_level",
        "parathyroid_hormone_pth_level",
        "serum_calcium_level",
        "serum_phosphate_level",
        "cystatin_c_level",
        "c_reactive_protein_crp_level",
        "interleukin_6_il_6_level",
    ],
    "Urine Findings": [
        "specific_gravity_of_urine",
        "albumin_in_urine",
        "sugar_in_urine",
        "red_blood_cells_in_urine",
        "pus_cells_in_urine",
        "pus_cell_clumps_in_urine",
        "bacteria_in_urine",
        "urine_protein_to_creatinine_ratio",
        "urine_output_ml_day",
        "urinary_sediment_microscopy_results",
    ],
    "Clinical History": [
        "hypertension_yes_no",
        "diabetes_mellitus_yes_no",
        "coronary_artery_disease_yes_no",
        "appetite_good_poor",
        "pedal_edema_yes_no",
        "anemia_yes_no",
        "duration_of_diabetes_mellitus_years",
        "duration_of_hypertension_years",
    ],
}


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ckd-navy: #172554;
            --ckd-teal: #2563eb;
            --ckd-mint: #dbeafe;
            --ckd-sky: #eff6ff;
            --ckd-line: #c7d2fe;
            --ckd-text: #111827;
            --ckd-muted: #64748b;
            --ckd-button-hover: #1d4ed8;
            --ckd-soft-border: #bfdbfe;
            --ckd-divider: #dbeafe;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fbff 0%, #eff6ff 45%, #ffffff 100%);
            color: var(--ckd-text);
        }

        header[data-testid="stHeader"] {
            background: rgba(248, 251, 255, 0.96);
            border-bottom: 1px solid var(--ckd-divider);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            color: var(--ckd-navy);
            letter-spacing: 0;
        }

        h1 {
            font-size: 2.1rem;
            font-weight: 760;
            margin-bottom: 0.2rem;
        }

        .app-header {
            border-left: 5px solid var(--ckd-teal);
            padding: 0.75rem 0 0.75rem 1.1rem;
            margin-bottom: 1.2rem;
        }

        .app-subtitle {
            color: var(--ckd-muted);
            font-size: 1rem;
            margin-top: 0.25rem;
        }

        div[data-testid="stTabs"] button {
            color: var(--ckd-navy);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ckd-teal);
            border-bottom-color: var(--ckd-teal);
        }

        div[data-testid="stForm"],
        div[data-testid="stExpander"],
        div[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--ckd-line);
            border-radius: 8px;
            box-shadow: 0 10px 28px rgba(23, 37, 84, 0.08);
        }

        div[data-testid="stExpander"] summary {
            color: var(--ckd-navy);
            font-weight: 700;
            background: var(--ckd-mint);
            border-bottom: 1px solid var(--ckd-soft-border);
            border-radius: 8px 8px 0 0;
        }

        label, .stSelectbox label, .stNumberInput label, .stFileUploader label {
            color: var(--ckd-navy) !important;
            font-weight: 620 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid var(--ckd-soft-border) !important;
            border-radius: 8px !important;
            color: var(--ckd-text) !important;
            box-shadow: none !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span {
            color: var(--ckd-text) !important;
            font-weight: 600;
        }

        div[data-baseweb="input"] button {
            color: var(--ckd-teal) !important;
            background: var(--ckd-sky) !important;
        }

        div[data-testid="stNumberInput"] button {
            color: var(--ckd-teal) !important;
            background: var(--ckd-sky) !important;
            border-left: 1px solid var(--ckd-soft-border) !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: var(--ckd-mint) !important;
        }

        div[data-baseweb="select"] svg {
            color: var(--ckd-teal) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            background: var(--ckd-teal) !important;
            color: #ffffff !important;
            border: 1px solid var(--ckd-teal) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            min-height: 2.7rem !important;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        div[data-testid="stFormSubmitButton"] button * {
            color: #ffffff !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--ckd-button-hover) !important;
            border-color: var(--ckd-button-hover) !important;
            color: #ffffff !important;
        }

        .result-panel {
            background: #ffffff;
            border: 1px solid var(--ckd-line);
            border-radius: 8px;
            box-shadow: 0 12px 34px rgba(23, 37, 84, 0.10);
            padding: 1.15rem;
            margin-top: 1.15rem;
        }

        .result-title {
            color: var(--ckd-navy);
            font-size: 1.35rem;
            font-weight: 780;
            margin-bottom: 0.9rem;
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .metric-box {
            background: var(--ckd-sky);
            border: 1px solid var(--ckd-soft-border);
            border-radius: 8px;
            padding: 0.8rem;
        }

        .metric-label {
            color: var(--ckd-muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--ckd-navy);
            font-size: 1.05rem;
            font-weight: 760;
            overflow-wrap: anywhere;
        }

        .guidance-row {
            border-top: 1px solid var(--ckd-divider);
            padding: 0.75rem 0 0.15rem;
        }

        .guidance-label {
            color: var(--ckd-teal);
            font-size: 0.9rem;
            font-weight: 780;
            margin-bottom: 0.15rem;
        }

        .guidance-text {
            color: var(--ckd-text);
            line-height: 1.5;
        }

        .footer-note {
            color: var(--ckd-muted);
            font-size: 0.86rem;
            margin-top: 1.4rem;
        }

        @media (max-width: 780px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .result-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


def ordered_feature_columns(preprocessors: dict) -> list[str]:
    feature_columns = list(preprocessors["feature_columns"])
    # Not used by the ML model. This is collected only for the rule-based eGFR stage engine.
    if "patient_sex" not in feature_columns:
        feature_columns.append("patient_sex")
    return feature_columns


def render_input(feature: str):
    label = FIELD_LABELS.get(feature, feature.replace("_", " ").title())
    if feature in CATEGORICAL_OPTIONS:
        return st.selectbox(
            label,
            CATEGORICAL_OPTIONS[feature],
            key=f"single_{feature}",
        )

    return st.number_input(
        label,
        value=float(DEFAULT_VALUES.get(feature, 0.0)),
        format="%.3f",
        key=f"single_{feature}",
    )


def show_result(result: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-title">Kidney Health Result</div>
            <div class="result-grid">
                <div class="metric-box">
                    <div class="metric-label">Prediction Result</div>
                    <div class="metric-value">{escape(result["Prediction Result"])}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Calculated eGFR</div>
                    <div class="metric-value">{escape(result.get("Calculated eGFR", "Unavailable"))}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">CKD Stage</div>
                    <div class="metric-value">{escape(result["CKD Stage"])}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{escape(result["Confidence Score"])}</div>
                </div>
            </div>
            <div class="guidance-row">
                <div class="guidance-label">Key Goals</div>
                <div class="guidance-text">{escape(result["Key Goals"])}</div>
            </div>
            <div class="guidance-row">
                <div class="guidance-label">Dietary Focus</div>
                <div class="guidance-text">{escape(result["Dietary Focus"])}</div>
            </div>
            <div class="guidance-row">
                <div class="guidance-label">Lifestyle</div>
                <div class="guidance-text">{escape(result["Lifestyle"])}</div>
            </div>
            <div class="guidance-row">
                <div class="guidance-label">What to Watch</div>
                <div class="guidance-text">{escape(result["What to Watch"])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    model, preprocessors, target_encoder = cached_artifacts()
except FileNotFoundError:
    st.error("Model files were not found. Run `python train.py` first.")
    st.stop()


apply_styles()

st.markdown(
    """
    <div class="app-header">
        <h1>CKD Clinical Support</h1>
        <div class="app-subtitle">Hybrid risk prediction and eGFR-based stage guidance</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_bulk = st.tabs(["Single Prediction", "Bulk CSV Prediction"])

with tab_single:
    with st.form("single_prediction_form"):
        feature_columns = ordered_feature_columns(preprocessors)
        feature_set = set(feature_columns)
        patient_input = {}

        for group_name, group_fields in FIELD_GROUPS.items():
            visible_fields = [feature for feature in group_fields if feature in feature_set]
            if not visible_fields:
                continue

            with st.expander(group_name, expanded=group_name == "Patient Profile"):
                columns = st.columns(2)
                for index, feature in enumerate(visible_fields):
                    with columns[index % 2]:
                        patient_input[feature] = render_input(feature)

        grouped_fields = {
            feature for group_fields in FIELD_GROUPS.values() for feature in group_fields
        }
        remaining_fields = [
            feature for feature in feature_columns if feature not in grouped_fields
        ]
        if remaining_fields:
            with st.expander("Additional Values"):
                columns = st.columns(2)
                for index, feature in enumerate(remaining_fields):
                    with columns[index % 2]:
                        patient_input[feature] = render_input(feature)

        submitted = st.form_submit_button("Predict Kidney Health")

    if submitted:
        result = predict_dataframe(
            patient_input,
            model=model,
            preprocessors=preprocessors,
            target_encoder=target_encoder,
        ).iloc[0].to_dict()
        show_result(result)

with tab_bulk:
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        results = predict_dataframe(
            input_df,
            model=model,
            preprocessors=preprocessors,
            target_encoder=target_encoder,
        )
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ckd_predictions.csv",
            mime="text/csv",
        )

st.markdown(
    """
    <div class="footer-note">
        Academic decision-support prototype only. Use results with qualified clinical review.
    </div>
    """,
    unsafe_allow_html=True,
)
