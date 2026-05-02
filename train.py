"""Train the CKD decision-support model on the UCI CKD dataset.

This version uses the real UCI clinical target labels only. It does not create
proxy targets, does not use eGFR as an ML input, and does not select models to
hit a pre-decided accuracy range.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed.*",
)

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from xgboost import XGBClassifier


RANDOM_STATE = 42
DATASET_PATH = Path("dataset/kidney_disease_dataset.csv")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = MODELS_DIR / "validation_artifacts"
N_SPLITS = 5
TEST_SIZE = 0.20
METRIC_WARNING = (
    "High accuracy is influenced by small dataset size; external validation is "
    "required for real-world deployment."
)

TARGET_CANDIDATES = (
    "target",
    "class",
    "classification",
    "ckd",
    "diagnosis",
    "disease_status",
)


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
    text = re.sub(r"\s+", " ", text)

    if text in {"", "nan", "none", "null", "?", "-", "--"}:
        return np.nan

    replacements = {
        "notpresent": "not present",
        "not present": "not present",
        "present": "present",
        "ckd": "CKD",
        "notckd": "No_CKD",
        "not ckd": "No_CKD",
        "no_ckd": "No_CKD",
        "no ckd": "No_CKD",
    }
    return replacements.get(text, text)


def normalize_target_value(value: object) -> str:
    normalized = normalize_text_value(value)
    if pd.isna(normalized):
        raise ValueError("Target column contains missing values.")

    text = str(normalized).strip()
    lookup = {
        "ckd": "CKD",
        "CKD": "CKD",
        "notckd": "No_CKD",
        "No_CKD": "No_CKD",
        "not ckd": "No_CKD",
        "no_ckd": "No_CKD",
        "no ckd": "No_CKD",
    }
    return lookup.get(text, text)


def normalize_categorical_text(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    normalized = df.copy()
    excluded = set(exclude)
    for column in normalized.select_dtypes(include=["object", "category"]).columns:
        if column not in excluded:
            normalized[column] = normalized[column].map(normalize_text_value)
    return normalized


def coerce_numeric_like_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    coerced = df.copy()
    excluded = set(exclude)

    for column in coerced.columns:
        if column in excluded or not pd.api.types.is_object_dtype(coerced[column]):
            continue

        non_missing = coerced[column].notna().sum()
        if non_missing == 0:
            continue

        numeric = pd.to_numeric(coerced[column], errors="coerce")
        parsed_ratio = numeric.notna().sum() / non_missing
        if parsed_ratio >= 0.85:
            coerced[column] = numeric

    return coerced


def detect_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    for column in df.columns:
        values = {
            normalize_target_value(value)
            for value in df[column].dropna().astype(str).unique()
        }
        if {"CKD", "No_CKD"}.issubset(values):
            return column

    raise ValueError("Could not detect target column with CKD/No_CKD labels.")


def is_leakage_column(column: str, target_column: str) -> bool:
    if column == target_column:
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

    if column in {"target", "label", "class", "classification"} and column != target_column:
        return True

    return False


def drop_leakage_columns(
    X: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, list[str]]:
    leakage_columns = [
        column for column in X.columns if is_leakage_column(column, target_column)
    ]
    return X.drop(columns=leakage_columns, errors="ignore"), leakage_columns


def find_egfr_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if "egfr" in column
        or "estimated_glomerular_filtration" in column
        or "glomerular_filtration_rate" in column
    ]


def print_leakage_check(
    target_column: str, feature_columns: list[str], leakage_columns: list[str]
) -> None:
    egfr_columns = find_egfr_columns(feature_columns)
    target_used = target_column in feature_columns
    derived_target_columns = [
        column
        for column in feature_columns
        if column != target_column
        and any(
            pattern in column
            for pattern in (
                "target",
                "label",
                "predicted",
                "prediction",
                "risk_score",
                "stage_score",
                "target_encoded",
                "target_label",
            )
        )
    ]

    print("\nLeakage check")
    print("-------------")
    print(f"eGFR removed from training: {'YES' if not egfr_columns else 'NO'}")
    print(f"Target column used as input: {'YES' if target_used else 'NO'}")
    print(
        "Derived target/prediction features present: "
        f"{'YES' if derived_target_columns else 'NO'}"
    )
    if leakage_columns:
        print("Columns removed by leakage guard:")
        for column in leakage_columns:
            print(f"  - {column}")
    else:
        print("Columns removed by leakage guard: none")

    if not egfr_columns and not target_used and not derived_target_columns:
        print("No data leakage detected in training pipeline")
    else:
        print("WARNING: potential leakage indicators remain in the feature set")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numerical_pipeline, numerical_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_xgboost_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def make_lightgbm_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


def make_pipeline(
    model: Any, numerical_features: list[str], categorical_features: list[str]
) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(numerical_features, categorical_features)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def roc_auc_ovr_macro_score(estimator, X: pd.DataFrame, y: np.ndarray) -> float:
    probabilities = estimator.predict_proba(X)
    if probabilities.shape[1] == 2:
        return roc_auc_score(y, probabilities[:, 1])
    return roc_auc_score(y, probabilities, multi_class="ovr", average="macro")


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numerical_features = [
        column for column in X.columns if pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_features = [column for column in X.columns if column not in numerical_features]
    return numerical_features, categorical_features


def get_ckd_class_index(class_names: list[str]) -> int:
    for index, class_name in enumerate(class_names):
        if class_name.lower() == "ckd":
            return index
    return 1 if len(class_names) > 1 else 0


def score_accuracy_f1(
    estimator, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold
) -> dict[str, float]:
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1_macro"},
        n_jobs=1,
        error_score="raise",
    )
    return {
        "accuracy": float(np.mean(scores["test_accuracy"])),
        "f1": float(np.mean(scores["test_f1"])),
    }


def run_ablation_study(
    X_all_features: pd.DataFrame,
    X_without_egfr: pd.DataFrame,
    X_training: pd.DataFrame,
    y: np.ndarray,
    tuned_estimator,
    cv: StratifiedKFold,
    output_dir: Path,
) -> pd.DataFrame:
    all_num, all_cat = get_feature_groups(X_all_features)
    no_egfr_num, no_egfr_cat = get_feature_groups(X_without_egfr)
    egfr_columns = find_egfr_columns(X_all_features.columns)

    versions = [
        (
            "Untuned LightGBM baseline",
            make_pipeline(make_lightgbm_model(), all_num, all_cat),
            X_all_features,
            "Raw UCI feature set; no eGFR column is present."
            if not egfr_columns
            else "Raw feature set before eGFR exclusion.",
        ),
    ]

    if egfr_columns:
        versions.append(
            (
                "Untuned LightGBM without eGFR",
                make_pipeline(make_lightgbm_model(), no_egfr_num, no_egfr_cat),
                X_without_egfr,
                f"Removed eGFR-like columns: {', '.join(egfr_columns)}.",
            )
        )

    versions.append(
        (
            "Tuned deployed model",
            tuned_estimator,
            X_training,
            "Measures the effect of hyperparameter tuning under the leakage guard.",
        ),
    )

    records = []
    for version_name, estimator, features, interpretation in versions:
        scores = score_accuracy_f1(estimator, features, y, cv)
        records.append(
            {
                "Model Version": version_name,
                "Accuracy": scores["accuracy"],
                "Macro F1": scores["f1"],
                "Interpretation": interpretation,
            }
        )

    ablation_df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(output_dir / "ablation_study.csv", index=False)

    print("\nAblation study")
    print("--------------")
    if not egfr_columns:
        print(
            "Note: no eGFR column was present in the dataset CSV; this table "
            "therefore measures hyperparameter tuning, not eGFR exclusion."
        )
    print(ablation_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved ablation table to {output_dir / 'ablation_study.csv'}")
    return ablation_df


def soft_voting_diagnostic(
    xgb_estimator,
    voting_estimator,
    X: pd.DataFrame,
    y: np.ndarray,
) -> str:
    """Check whether matching XGBoost/Voting metrics come from matching labels."""
    xgb_model = clone(xgb_estimator)
    voting_model = clone(voting_estimator)
    xgb_model.fit(X, y)
    voting_model.fit(X, y)

    xgb_predictions = xgb_model.predict(X)
    voting_predictions = voting_model.predict(X)
    prediction_match_rate = float(np.mean(xgb_predictions == voting_predictions))

    xgb_probabilities = xgb_model.predict_proba(X)
    voting_probabilities = voting_model.predict_proba(X)
    max_probability_delta = float(
        np.max(np.abs(xgb_probabilities - voting_probabilities))
    )

    if prediction_match_rate == 1.0 and max_probability_delta > 0:
        return (
            "Soft Voting was fitted as a separate VotingClassifier. On the "
            "development set its predicted labels match XGBoost exactly, while "
            f"probabilities differ by up to {max_probability_delta:.6f}; matching "
            "accuracy/F1 reflects an identical decision boundary, not a reused "
            "model object."
        )

    return (
        "Soft Voting was fitted as a separate VotingClassifier. Development-set "
        f"label match with XGBoost: {prediction_match_rate:.4f}; max probability "
        f"difference: {max_probability_delta:.6f}."
    )


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Held-out Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve_plot(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> float:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    if len(class_names) == 2:
        positive_index = get_ckd_class_index(class_names)
        positive_mask = (y_true == positive_index).astype(int)
        fpr, tpr, _ = roc_curve(positive_mask, probabilities[:, positive_index])
        auc_value = roc_auc_score(positive_mask, probabilities[:, positive_index])
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_names[positive_index]} AUC = {auc_value:.3f}",
        )
    else:
        encoded = label_binarize(y_true, classes=np.arange(len(class_names)))
        auc_values = []
        for class_index, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(encoded[:, class_index], probabilities[:, class_index])
            class_auc = roc_auc_score(encoded[:, class_index], probabilities[:, class_index])
            auc_values.append(class_auc)
            ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} AUC = {class_auc:.3f}")
        auc_value = float(np.mean(auc_values))

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Held-out Test ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(auc_value)


def evaluate_held_out_test(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_encoder: LabelEncoder,
    output_dir: Path,
):
    holdout_model = clone(estimator)
    holdout_model.fit(X_train, y_train)

    predictions = holdout_model.predict(X_test)
    probabilities = holdout_model.predict_proba(X_test)
    class_names = list(target_encoder.classes_)
    roc_auc = save_roc_curve_plot(
        y_test, probabilities, class_names, output_dir / "roc_curve.png"
    )
    save_confusion_matrix_plot(
        y_test, predictions, class_names, output_dir / "confusion_matrix.png"
    )

    print("\nHeld-out test evaluation")
    print("------------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    print(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")

    return holdout_model, predictions, probabilities


def to_dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for transformer_name, _, columns in preprocessor.transformers_:
            if transformer_name == "remainder":
                continue
            names.extend(str(column) for column in columns)
        return names


def select_tree_model_for_shap(model_step: Any) -> tuple[Any, str]:
    if isinstance(model_step, VotingClassifier):
        named_estimators = getattr(model_step, "named_estimators_", {})
        for estimator_name in ("lightgbm", "xgboost"):
            if estimator_name in named_estimators:
                estimator = named_estimators[estimator_name]
                return estimator, f"Soft Voting Ensemble base {estimator_name}"
        first_name, first_estimator = next(iter(named_estimators.items()))
        return first_estimator, f"Soft Voting Ensemble base {first_name}"
    return model_step, model_step.__class__.__name__


def normalize_shap_values(
    shap_values: Any, class_index: int, class_count: int
) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[2] == class_count:
            return values[:, :, class_index]
        if values.shape[1] == class_count:
            return values[:, class_index, :]

    if class_count == 2 and class_index == 0:
        return -values
    return values


def generate_shap_artifacts(
    final_model,
    X: pd.DataFrame,
    target_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    try:
        import shap
    except ImportError:
        print("\nSHAP skipped: install `shap` to generate explainability artifacts.")
        return

    preprocessor = final_model.named_steps["preprocessor"]
    model_step = final_model.named_steps["model"]
    tree_model, model_label = select_tree_model_for_shap(model_step)
    class_names = list(target_encoder.classes_)
    ckd_index = get_ckd_class_index(class_names)

    transformed_X = to_dense_array(preprocessor.transform(X))
    feature_names = get_transformed_feature_names(preprocessor)
    sample_size = min(200, len(transformed_X))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(len(transformed_X), size=sample_size, replace=False)
    sample_X = transformed_X[sample_indices]

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(sample_X)
    shap_matrix = normalize_shap_values(shap_values, ckd_index, len(class_names))

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_matrix,
        sample_X,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP Summary for {class_names[ckd_index]} Prediction")
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.mean(np.abs(shap_matrix), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_path = output_dir / "shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nSHAP explainability")
    print("-------------------")
    print(f"TreeExplainer model: {model_label}")
    print(f"Saved SHAP summary plot to {summary_path}")
    print(f"Saved SHAP feature ranking to {importance_path}")
    print("Top SHAP features:")
    print(importance_df.head(10).to_string(index=False))


def humanize_lime_feature(description: str) -> str:
    text = description.replace("numerical__", "").replace("categorical__", "")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_raw_lime_matrix(
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[int], dict[int, list[str]], dict[str, list[str]]]:
    feature_columns = list(X.columns)
    numerical_features, categorical_features = get_feature_groups(X)
    lime_df = X.copy()

    for column in numerical_features:
        numeric_values = pd.to_numeric(lime_df[column], errors="coerce")
        lime_df[column] = numeric_values.fillna(numeric_values.median())

    categorical_names: dict[int, list[str]] = {}
    categorical_maps: dict[str, list[str]] = {}
    categorical_indices: list[int] = []
    for column in categorical_features:
        column_index = feature_columns.index(column)
        values = lime_df[column].where(lime_df[column].notna(), "missing").astype(str)
        categories = sorted(values.unique().tolist())
        category_lookup = {category: index for index, category in enumerate(categories)}
        lime_df[column] = values.map(category_lookup).astype(float)
        categorical_indices.append(column_index)
        categorical_names[column_index] = categories
        categorical_maps[column] = categories

    return (
        lime_df[feature_columns].astype(float).to_numpy(),
        feature_columns,
        categorical_indices,
        categorical_names,
        categorical_maps,
    )


def lime_array_to_raw_dataframe(
    values: np.ndarray,
    feature_columns: list[str],
    categorical_maps: dict[str, list[str]],
) -> pd.DataFrame:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    decoded: dict[str, Any] = {}
    for column_index, column in enumerate(feature_columns):
        if column in categorical_maps:
            categories = categorical_maps[column]
            codes = np.rint(array[:, column_index]).astype(int)
            codes = np.clip(codes, 0, len(categories) - 1)
            decoded[column] = [categories[code] for code in codes]
        else:
            decoded[column] = array[:, column_index]

    return pd.DataFrame(decoded, columns=feature_columns)


def generate_lime_explanation(
    final_model,
    X: pd.DataFrame,
    y: np.ndarray | None,
    target_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("\nLIME skipped: install `lime` to generate local explanations.")
        return

    (
        lime_matrix,
        feature_columns,
        categorical_indices,
        categorical_names,
        categorical_maps,
    ) = prepare_raw_lime_matrix(X)
    class_names = list(target_encoder.classes_)
    predictions = final_model.predict(X).astype(int)
    probabilities = final_model.predict_proba(X)
    ckd_index = get_ckd_class_index(class_names)
    ckd_probabilities = probabilities[:, ckd_index]

    if y is not None:
        true_positive_indices = np.flatnonzero((y == ckd_index) & (predictions == ckd_index))
    else:
        true_positive_indices = np.array([], dtype=int)

    if len(true_positive_indices):
        median_probability = float(np.median(ckd_probabilities[true_positive_indices]))
        sample_index = int(
            true_positive_indices[
                np.argmin(np.abs(ckd_probabilities[true_positive_indices] - median_probability))
            ]
        )
        selection_rationale = (
            "True-positive CKD case with CKD probability closest to the median "
            "among true-positive CKD predictions."
        )
    else:
        ckd_predictions = np.flatnonzero(predictions == ckd_index)
        sample_index = int(ckd_predictions[0]) if len(ckd_predictions) else 0
        selection_rationale = (
            "Fallback sample: first CKD prediction, because no true-positive CKD "
            "case was available for median-probability selection."
        )

    predicted_index = int(predictions[sample_index])

    def predict_from_lime(values: np.ndarray) -> np.ndarray:
        decoded_df = lime_array_to_raw_dataframe(
            values,
            feature_columns,
            categorical_maps,
        )
        return final_model.predict_proba(decoded_df)

    explainer = LimeTabularExplainer(
        lime_matrix,
        feature_names=[humanize_lime_feature(column) for column in feature_columns],
        class_names=class_names,
        categorical_features=categorical_indices,
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        lime_matrix[sample_index],
        predict_from_lime,
        labels=[predicted_index],
        num_features=10,
    )
    contributions = explanation.as_list(label=predicted_index)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "lime_sample_explanation.html"
    csv_path = output_dir / "lime_sample_contributions.csv"
    metadata_path = output_dir / "lime_sample_metadata.csv"
    explanation.save_to_file(str(html_path))
    pd.DataFrame(
        [
            {
                "feature_rule": humanize_lime_feature(feature_rule),
                "weight": weight,
                "prediction_class": class_names[predicted_index],
                "sample_index": sample_index,
            }
            for feature_rule, weight in contributions
        ]
    ).to_csv(csv_path, index=False)
    pd.DataFrame(
        [
            {
                "sample_index": sample_index,
                "true_label": class_names[int(y[sample_index])] if y is not None else "unknown",
                "predicted_label": class_names[predicted_index],
                "predicted_ckd_probability": float(ckd_probabilities[sample_index]),
                "selection_rationale": selection_rationale,
            }
        ]
    ).to_csv(metadata_path, index=False)

    print("\nLIME local explanation")
    print("----------------------")
    print(f"Explained sample row: {sample_index}")
    print(f"Selection rationale: {selection_rationale}")
    print(f"Sample prediction: {class_names[predicted_index]}")
    for feature_rule, weight in contributions:
        direction = "increased" if weight > 0 else "reduced"
        print(
            f"- {humanize_lime_feature(feature_rule)} {direction} evidence for "
            f"{class_names[predicted_index]} ({weight:+.4f})"
        )
    print(f"Saved LIME HTML explanation to {html_path}")
    print(f"Saved LIME contribution table to {csv_path}")
    print(f"Saved LIME sample metadata to {metadata_path}")


def cv_summary(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    best_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "roc_auc_ovr_macro": roc_auc_ovr_macro_score,
    }
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    record: dict[str, Any] = {
        "model": name,
        "best_params": json.dumps(best_params or {}),
    }
    print(f"\n{name} 5-fold cross-validation")
    print("-" * (len(name) + 24))

    for metric in scoring:
        values = scores[f"test_{metric}"]
        record[f"{metric}_mean"] = float(np.mean(values))
        record[f"{metric}_std"] = float(np.std(values))
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return record


def selected_oof_report(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
    target_encoder: LabelEncoder,
) -> dict[str, Any]:
    predictions = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1, method="predict")
    probabilities = cross_val_predict(
        estimator, X, y, cv=cv, n_jobs=1, method="predict_proba"
    )

    if probabilities.shape[1] == 2:
        roc_auc = roc_auc_score(y, probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(y, probabilities, multi_class="ovr", average="macro")

    class_names = list(target_encoder.classes_)
    report = classification_report(
        y,
        predictions,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y, predictions)

    print("\nSelected model out-of-fold evaluation")
    print("-------------------------------------")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(
        f"Precision Macro: {precision_score(y, predictions, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Recall Macro: {recall_score(y, predictions, average='macro', zero_division=0):.4f}"
    )
    print(f"Macro F1: {f1_score(y, predictions, average='macro', zero_division=0):.4f}")
    print(
        f"Weighted F1: {f1_score(y, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(f"ROC-AUC OVR Macro: {roc_auc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(pd.DataFrame(matrix, index=class_names, columns=class_names))

    return {
        "selected_accuracy_oof": accuracy_score(y, predictions),
        "selected_precision_macro_oof": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "selected_recall_macro_oof": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "selected_macro_f1_oof": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "selected_weighted_f1_oof": f1_score(
            y, predictions, average="weighted", zero_division=0
        ),
        "selected_roc_auc_ovr_macro_oof": roc_auc,
        "selected_classification_report_json": json.dumps(report),
        "selected_confusion_matrix_json": json.dumps(matrix.tolist()),
    }


def main() -> None:
    print("Loading UCI CKD dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = clean_columns(df)

    target_column = detect_target_column(df)
    print(f"Detected target column: {target_column}")

    df[target_column] = df[target_column].map(normalize_target_value)
    df = normalize_categorical_text(df, exclude=[target_column])
    df = coerce_numeric_like_columns(df, exclude=[target_column])
    df = df.dropna(subset=[target_column]).copy()

    X_all_features = df.drop(columns=[target_column])
    y = df[target_column]
    X_without_egfr = X_all_features.drop(
        columns=find_egfr_columns(X_all_features.columns),
        errors="ignore",
    )
    X, leakage_columns = drop_leakage_columns(X_all_features, target_column)

    print("Dropped leakage columns:")
    if leakage_columns:
        for column in leakage_columns:
            print(f"  - {column}")
    else:
        print("  - none")

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    print_leakage_check(target_column, list(X.columns), leakage_columns)

    numerical_features, categorical_features = get_feature_groups(X)

    print(f"Rows: {len(df)}")
    print("Target distribution:")
    print(y.value_counts())
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nValidation design: {N_SPLITS}-fold StratifiedKFold on "
        f"{len(X_dev)} development rows plus {len(X_test)} held-out test rows."
    )

    xgb_pipeline = make_pipeline(
        make_xgboost_model(), numerical_features, categorical_features
    )
    lgbm_pipeline = make_pipeline(
        make_lightgbm_model(), numerical_features, categorical_features
    )

    xgb_grid = {
        "model__n_estimators": [80, 120, 160],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.9],
        "model__colsample_bytree": [0.9],
    }
    lgbm_grid = {
        "model__n_estimators": [80, 120, 160],
        "model__num_leaves": [7, 15],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.9],
        "model__colsample_bytree": [0.9],
    }

    print("\nTuning XGBoost with 5-fold GridSearchCV...")
    xgb_search = GridSearchCV(
        xgb_pipeline,
        xgb_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    xgb_search.fit(X_dev, y_dev)
    print("Best XGBoost params:", xgb_search.best_params_)

    print("\nTuning LightGBM with 5-fold GridSearchCV...")
    lgbm_search = GridSearchCV(
        lgbm_pipeline,
        lgbm_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    lgbm_search.fit(X_dev, y_dev)
    print("Best LightGBM params:", lgbm_search.best_params_)

    tuned_xgb_model = clone(xgb_search.best_estimator_.named_steps["model"])
    tuned_lgbm_model = clone(lgbm_search.best_estimator_.named_steps["model"])
    voting_pipeline = make_pipeline(
        VotingClassifier(
            estimators=[
                ("xgboost", tuned_xgb_model),
                ("lightgbm", tuned_lgbm_model),
            ],
            voting="soft",
            n_jobs=1,
        ),
        numerical_features,
        categorical_features,
    )

    candidates = {
        "XGBoost": (xgb_search.best_estimator_, xgb_search.best_params_),
        "LightGBM": (lgbm_search.best_estimator_, lgbm_search.best_params_),
        "Soft Voting Ensemble": (
            voting_pipeline,
            {
                "xgboost": xgb_search.best_params_,
                "lightgbm": lgbm_search.best_params_,
            },
        ),
    }

    metrics_records = [
        cv_summary(name, estimator, X_dev, y_dev, cv, params)
        for name, (estimator, params) in candidates.items()
    ]
    metrics_df = pd.DataFrame(metrics_records)
    voting_note = soft_voting_diagnostic(
        xgb_search.best_estimator_,
        voting_pipeline,
        X_dev,
        y_dev,
    )
    metrics_df["diagnostic_note"] = ""
    metrics_df.loc[
        metrics_df["model"] == "Soft Voting Ensemble",
        "diagnostic_note",
    ] = voting_note

    print("\nSoft Voting diagnostic")
    print("----------------------")
    print(voting_note)

    best_index = metrics_df["macro_f1_mean"].astype(float).idxmax()
    best_model_name = str(metrics_df.loc[best_index, "model"])
    best_estimator = candidates[best_model_name][0]
    metrics_df["selected_model"] = metrics_df["model"] == best_model_name

    print(f"\nSelected best model by mean Macro F1: {best_model_name}")
    oof_metrics = selected_oof_report(best_estimator, X_dev, y_dev, cv, target_encoder)
    for key, value in oof_metrics.items():
        metrics_df.loc[best_index, key] = value

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate_held_out_test(
        best_estimator,
        X_dev,
        X_test,
        y_dev,
        y_test,
        target_encoder,
        ARTIFACTS_DIR,
    )
    run_ablation_study(
        X_all_features,
        X_without_egfr,
        X,
        y_encoded,
        best_estimator,
        cv,
        ARTIFACTS_DIR,
    )

    final_model = clone(best_estimator)
    final_model.fit(X, y_encoded)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(
        {
            "preprocessor": final_model.named_steps["preprocessor"],
            "feature_columns": list(X.columns),
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "leakage_columns": leakage_columns,
            "target_column": target_column,
            "target_strategy": "uci_real_clinical_labels",
            "cv_strategy": f"{N_SPLITS}-fold StratifiedKFold",
            "selection_rule": "highest_mean_macro_f1",
            "model_expects_raw_features": True,
        },
        MODELS_DIR / "preprocessors.joblib",
    )
    joblib.dump(target_encoder, MODELS_DIR / "target_encoder.joblib")
    metrics_df.to_csv(MODELS_DIR / "model_metrics.csv", index=False)

    generate_shap_artifacts(final_model, X, target_encoder, ARTIFACTS_DIR)
    generate_lime_explanation(final_model, X, y_encoded, target_encoder, ARTIFACTS_DIR)

    print("Saved model artifacts to models/.")
    print(f"\nMetric warning: {METRIC_WARNING}")


if __name__ == "__main__":
    main()
