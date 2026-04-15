from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_clean_data
from src.eda import generate_eda_report
from src.evaluation import (
    plot_coefficients,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.inference import DATA_PATH, load_model_bundles

PLOTS_DIR = Path("plots")
EDA_DIR = PLOTS_DIR / "eda"
REPORTS_DIR = Path("reports")


def _required_plot_paths() -> list[Path]:
    return [
        PLOTS_DIR / "roc_comparison.png",
        PLOTS_DIR / "pr_curve.png",
        PLOTS_DIR / "logistic_regression_cm.png",
        PLOTS_DIR / "decision_tree_cm.png",
        PLOTS_DIR / "lr_coef.png",
        PLOTS_DIR / "dt_importance.png",
        EDA_DIR / "dist_tenure.png",
        EDA_DIR / "dist_MonthlyCharges.png",
        EDA_DIR / "dist_TotalCharges.png",
        EDA_DIR / "correlation_matrix.png",
    ]


def _generate_model_plots() -> None:
    bundles = load_model_bundles()
    data = load_and_clean_data(str(DATA_PATH))
    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    models_dict = {name: bundle.pipeline for name, bundle in bundles.items()}
    plot_roc_curve(models_dict, X_test, y_test, save_path=PLOTS_DIR / "roc_comparison.png")
    plot_precision_recall_curve(models_dict, X_test, y_test, save_path=PLOTS_DIR / "pr_curve.png")

    for name, bundle in bundles.items():
        probabilities = bundle.pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= bundle.threshold).astype(int)
        filename = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test,
            predictions,
            name,
            save_path=PLOTS_DIR / f"{filename}_cm.png",
        )

    lr_bundle = bundles["Logistic Regression"]
    lr_preprocessor = lr_bundle.pipeline.named_steps["preprocessor"]
    lr_poly = lr_bundle.pipeline.named_steps["poly"]
    lr_selector = lr_bundle.pipeline.named_steps["selection"]
    lr_classifier = lr_bundle.pipeline.named_steps["classifier"]
    preprocessor_feature_names = lr_preprocessor.get_feature_names_out()
    poly_feature_names = lr_poly.get_feature_names_out(preprocessor_feature_names)
    selected_feature_names = poly_feature_names[lr_selector.get_support()]
    plot_coefficients(
        lr_classifier,
        selected_feature_names,
        save_path=PLOTS_DIR / "lr_coef.png",
    )

    dt_bundle = bundles["Decision Tree"]
    dt_preprocessor = dt_bundle.pipeline.named_steps["preprocessor"]
    dt_classifier = dt_bundle.pipeline.named_steps["classifier"]
    plot_feature_importance(
        dt_classifier,
        dt_preprocessor.get_feature_names_out(),
        save_path=PLOTS_DIR / "dt_importance.png",
    )


@lru_cache(maxsize=1)
def ensure_runtime_assets() -> dict[str, str]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not all(path.exists() for path in _required_plot_paths()):
        generate_eda_report(str(DATA_PATH), ["tenure", "MonthlyCharges", "TotalCharges"], output_dir=str(REPORTS_DIR))
        _generate_model_plots()

    return {
        "plots_dir": str(PLOTS_DIR),
        "eda_dir": str(EDA_DIR),
        "reports_dir": str(REPORTS_DIR),
    }
