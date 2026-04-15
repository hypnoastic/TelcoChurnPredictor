from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_loader import get_feature_lists

MODELS_DIR = Path("models")
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


@dataclass(frozen=True)
class ModelBundle:
    name: str
    pipeline: Any
    threshold: float
    metrics: dict[str, float]


def get_required_columns() -> list[str]:
    numeric_features, categorical_features = get_feature_lists()
    return categorical_features + numeric_features


@lru_cache(maxsize=1)
def load_model_bundles() -> dict[str, ModelBundle]:
    metrics = joblib.load(MODELS_DIR / "metrics.joblib")
    pipelines = {
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression_model.joblib"),
        "Decision Tree": joblib.load(MODELS_DIR / "decision_tree_model.joblib"),
    }
    bundles: dict[str, ModelBundle] = {}
    for name, pipeline in pipelines.items():
        model_metrics = metrics.get(name, {})
        bundles[name] = ModelBundle(
            name=name,
            pipeline=pipeline,
            threshold=float(model_metrics.get("Threshold", 0.5)),
            metrics=model_metrics,
        )
    return bundles


def get_available_models() -> list[str]:
    return list(load_model_bundles().keys())


def validate_customer_dataframe(df: pd.DataFrame) -> list[str]:
    required_columns = get_required_columns()
    return [column for column in required_columns if column not in df.columns]


def coerce_customer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features, _ = get_feature_lists()
    required_columns = get_required_columns()
    missing_columns = validate_customer_dataframe(df)
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    cleaned = df.copy()
    for column in numeric_features:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned[numeric_features].isnull().any().any():
        missing_numeric = cleaned[numeric_features].isnull().sum()
        bad_columns = [column for column, count in missing_numeric.items() if count > 0]
        raise ValueError(
            "Numeric columns contain invalid values after conversion: " + ", ".join(bad_columns)
        )

    return cleaned[required_columns]


def build_customer_frame(customer_payload: dict[str, Any]) -> pd.DataFrame:
    return coerce_customer_dataframe(pd.DataFrame([customer_payload]))


def _to_dense_row(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()[0]
    return np.asarray(matrix)[0]


def _friendly_feature_name(feature_name: str) -> str:
    renamed = (
        feature_name.replace("cat__", "")
        .replace("num__", "")
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
    )
    renamed = renamed.replace("SeniorCitizen", "Senior citizen")
    renamed = renamed.replace("MonthlyCharges", "Monthly charges")
    renamed = renamed.replace("TotalCharges", "Total charges")
    renamed = renamed.replace("tenure", "Tenure")
    return renamed


def _summarize_driver(feature_name: str, contribution: float, model_name: str) -> str:
    direction = "raises churn risk" if contribution >= 0 else "reduces churn risk"
    if model_name == "Decision Tree":
        direction = "was influential in the tree-based score"
    return f"{_friendly_feature_name(feature_name)} ({direction})"


def _extract_logistic_drivers(bundle: ModelBundle, row_df: pd.DataFrame, top_k: int = 3) -> list[str]:
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    poly = bundle.pipeline.named_steps["poly"]
    selector = bundle.pipeline.named_steps["selection"]
    classifier = bundle.pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(row_df)
    transformed = poly.transform(transformed)
    selected = selector.transform(transformed)

    base_features = preprocessor.get_feature_names_out()
    poly_features = poly.get_feature_names_out(base_features)
    selected_features = poly_features[selector.get_support()]
    contributions = _to_dense_row(selected) * classifier.coef_[0]

    ranked_indices = np.argsort(contributions)[::-1]
    positive_indices = [index for index in ranked_indices if contributions[index] > 0][:top_k]
    chosen_indices = positive_indices or ranked_indices[:top_k]
    return [
        _summarize_driver(selected_features[index], contributions[index], bundle.name)
        for index in chosen_indices
    ]


def _extract_tree_drivers(bundle: ModelBundle, row_df: pd.DataFrame, top_k: int = 3) -> list[str]:
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    classifier = bundle.pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(row_df)
    transformed_row = _to_dense_row(transformed)
    feature_names = preprocessor.get_feature_names_out()
    influences = np.abs(transformed_row) * classifier.feature_importances_

    ranked_indices = np.argsort(influences)[::-1]
    chosen_indices = [index for index in ranked_indices if influences[index] > 0][:top_k]
    return [
        _summarize_driver(feature_names[index], influences[index], bundle.name)
        for index in chosen_indices
    ]


def explain_prediction(row_df: pd.DataFrame, model_name: str) -> list[str]:
    bundle = load_model_bundles()[model_name]
    if model_name == "Logistic Regression":
        return _extract_logistic_drivers(bundle, row_df)
    return _extract_tree_drivers(bundle, row_df)


def predict_dataframe(df: pd.DataFrame, model_name: str = "Logistic Regression") -> pd.DataFrame:
    bundle = load_model_bundles()[model_name]
    customer_df = coerce_customer_dataframe(df)

    probabilities = bundle.pipeline.predict_proba(customer_df)[:, 1]
    predictions = np.where(probabilities >= bundle.threshold, "Yes (Churn)", "No (Retain)")

    records: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(customer_df.iterrows()):
        row_df = pd.DataFrame([row.to_dict()])
        drivers = explain_prediction(row_df, model_name)
        records.append(
            {
                **row.to_dict(),
                "selected_model": model_name,
                "churn_probability": float(probabilities[index]),
                "churn_probability_pct": f"{probabilities[index] * 100:.2f}%",
                "churn_prediction": predictions[index],
                "decision_threshold": bundle.threshold,
                "driver_1": drivers[0] if len(drivers) > 0 else "",
                "driver_2": drivers[1] if len(drivers) > 1 else "",
                "driver_3": drivers[2] if len(drivers) > 2 else "",
                "driver_summary": "; ".join(drivers),
            }
        )

    return pd.DataFrame(records)


def predict_single(customer_payload: dict[str, Any], model_name: str = "Logistic Regression") -> dict[str, Any]:
    result_df = predict_dataframe(pd.DataFrame([customer_payload]), model_name=model_name)
    result = result_df.iloc[0].to_dict()
    return {
        "model_name": result["selected_model"],
        "prediction": result["churn_prediction"],
        "probability": result["churn_probability"],
        "probability_pct": result["churn_probability_pct"],
        "threshold": result["decision_threshold"],
        "drivers": [value for value in [result["driver_1"], result["driver_2"], result["driver_3"]] if value],
        "driver_summary": result["driver_summary"],
        "customer_profile": build_customer_frame(customer_payload).iloc[0].to_dict(),
    }
