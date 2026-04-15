import tempfile
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

from src.inference import (
    get_available_models,
    get_required_columns,
    load_model_bundles,
    predict_dataframe,
    predict_single,
)
from src.runtime_assets import ensure_runtime_assets

APP_TITLE = "Customer Churn Prediction & Retention Strategist"
FORM_FIELD_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

BOOT_ERROR = ""
RUNTIME_ASSETS = {"plots_dir": "plots", "eda_dir": "plots/eda"}
MODEL_BUNDLES = {}

try:
    RUNTIME_ASSETS = ensure_runtime_assets()
    MODEL_BUNDLES = load_model_bundles()
except Exception as exc:
    BOOT_ERROR = str(exc)


def _build_customer_payload(*values):
    payload = dict(zip(FORM_FIELD_ORDER, values))
    payload["SeniorCitizen"] = int(bool(payload["SeniorCitizen"]))
    payload["tenure"] = int(payload["tenure"])
    payload["MonthlyCharges"] = float(payload["MonthlyCharges"])
    payload["TotalCharges"] = float(payload["TotalCharges"])
    return payload


def _format_drivers(drivers):
    if not drivers:
        return "No strong drivers were extracted from the selected model."
    return "\n".join(f"- {driver}" for driver in drivers)


def run_single_prediction(model_name, *values):
    try:
        result = predict_single(_build_customer_payload(*values), model_name=model_name)
    except Exception as exc:
        return f"Error: {exc}", "Error", "Prediction failed."

    details = (
        f"### Key churn drivers\n{_format_drivers(result['drivers'])}\n\n"
        f"Threshold used: `{result['threshold']:.3f}`"
    )
    return result["prediction"], result["probability_pct"], details


def score_csv_file(file_path, model_name):
    if not file_path:
        return "Upload a CSV file to score customers.", pd.DataFrame(), None

    try:
        input_df = pd.read_csv(file_path)
        result_df = predict_dataframe(input_df, model_name=model_name)
    except Exception as exc:
        return f"CSV scoring failed: {exc}", pd.DataFrame(), None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="churn_predictions_") as handle:
        result_df.to_csv(handle.name, index=False)
        download_path = handle.name

    churn_count = int((result_df["churn_prediction"] == "Yes (Churn)").sum())
    summary = (
        f"Scored `{len(result_df)}` customers with `{model_name}`. "
        f"Flagged `{churn_count}` high-risk customers."
    )
    return summary, result_df, download_path


def _metric_rows():
    rows = []
    for name, bundle in MODEL_BUNDLES.items():
        metrics = bundle.metrics
        rows.append(
            [
                name,
                f"{metrics.get('Accuracy', 0):.1%}",
                f"{metrics.get('Precision', 0):.1%}",
                f"{metrics.get('Recall', 0):.1%}",
                f"{metrics.get('F1 Score', 0):.1%}",
                f"{bundle.threshold:.4f}",
            ]
        )
    return rows


def _build_customer_form():
    components = {}
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Customer Profile")
            components["gender"] = gr.Dropdown(["Female", "Male"], label="Gender", value="Female")
            components["SeniorCitizen"] = gr.Checkbox(label="Senior Citizen")
            components["Partner"] = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            components["Dependents"] = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            components["tenure"] = gr.Slider(0, 72, label="Tenure (Months)", value=12)

            gr.Markdown("### Services")
            components["PhoneService"] = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            components["MultipleLines"] = gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines", value="No")
            components["InternetService"] = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
            components["OnlineSecurity"] = gr.Dropdown(["No", "Yes", "No internet service"], label="Online Security", value="No")
            components["OnlineBackup"] = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            components["DeviceProtection"] = gr.Dropdown(["No", "Yes", "No internet service"], label="Device Protection", value="No")
            components["TechSupport"] = gr.Dropdown(["No", "Yes", "No internet service"], label="Tech Support", value="No")

        with gr.Column(scale=1):
            gr.Markdown("### Engagement & Billing")
            components["StreamingTV"] = gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming TV", value="Yes")
            components["StreamingMovies"] = gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming Movies", value="Yes")
            components["Contract"] = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
            components["PaperlessBilling"] = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
            components["PaymentMethod"] = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                label="Payment Method",
                value="Electronic check",
            )
            components["MonthlyCharges"] = gr.Number(label="Monthly Charges ($)", value=70.0)
            components["TotalCharges"] = gr.Number(label="Total Charges ($)", value=1000.0)
    return components


theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
css = """
.gradio-container {max-width: 1280px !important;}
.app-note {padding: 12px 16px; border-radius: 10px; background: #eff6ff; border: 1px solid #bfdbfe;}
"""

with gr.Blocks(theme=theme, title=APP_TITLE, css=css) as demo:
    session_id = gr.State(str(uuid.uuid4()))
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(
        "ML-based churn prediction for Milestone 1, extended into an agentic retention workflow for Milestone 2."
    )

    if BOOT_ERROR:
        gr.Markdown(f"**Startup warning:** {BOOT_ERROR}")
    else:
        gr.Markdown(
            "<div class='app-note'>The app uses saved model artifacts at runtime. Missing plots are regenerated from the saved models and dataset without retraining.</div>"
        )

    with gr.Tab("Model Dashboard"):
        gr.Markdown("## Evaluation Summary")
        gr.Dataframe(
            headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Threshold"],
            value=_metric_rows(),
            interactive=False,
        )

        with gr.Row():
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "roc_comparison.png", label="ROC Comparison", show_label=True)
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "pr_curve.png", label="Precision-Recall Curve", show_label=True)

        with gr.Row():
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "logistic_regression_cm.png", label="Logistic Regression Confusion Matrix", show_label=True)
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "lr_coef.png", label="Top Logistic Coefficients", show_label=True)

        with gr.Row():
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "decision_tree_cm.png", label="Decision Tree Confusion Matrix", show_label=True)
            gr.Image(Path(RUNTIME_ASSETS["plots_dir"]) / "dt_importance.png", label="Decision Tree Feature Importance", show_label=True)

    with gr.Tab("EDA"):
        gr.Markdown("## Exploratory Data Analysis")
        with gr.Row():
            gr.Image(Path(RUNTIME_ASSETS["eda_dir"]) / "dist_tenure.png", label="Tenure Distribution", show_label=True)
            gr.Image(Path(RUNTIME_ASSETS["eda_dir"]) / "dist_MonthlyCharges.png", label="Monthly Charges Distribution", show_label=True)
        with gr.Row():
            gr.Image(Path(RUNTIME_ASSETS["eda_dir"]) / "dist_TotalCharges.png", label="Total Charges Distribution", show_label=True)
            gr.Image(Path(RUNTIME_ASSETS["eda_dir"]) / "correlation_matrix.png", label="Correlation Matrix", show_label=True)

    with gr.Tab("Single Customer Prediction"):
        model_selector = gr.Dropdown(get_available_models(), value="Logistic Regression", label="Scoring Model")
        form_components = _build_customer_form()
        predict_btn = gr.Button("Calculate Churn Risk", variant="primary")
        with gr.Row():
            output_label = gr.Textbox(label="Prediction")
            output_prob = gr.Textbox(label="Churn Probability")
        output_details = gr.Markdown()

        predict_btn.click(
            run_single_prediction,
            inputs=[model_selector] + [form_components[name] for name in FORM_FIELD_ORDER],
            outputs=[output_label, output_prob, output_details],
        )

    with gr.Tab("Batch CSV Scoring"):
        gr.Markdown(
            "## Upload customer data\n"
            f"Required columns: `{', '.join(get_required_columns())}`"
        )
        batch_model_selector = gr.Dropdown(get_available_models(), value="Logistic Regression", label="Scoring Model")
        csv_input = gr.File(label="Customer CSV", file_types=[".csv"], type="filepath")
        batch_btn = gr.Button("Score Uploaded CSV", variant="primary")
        batch_summary = gr.Markdown()
        batch_table = gr.Dataframe(label="Scored Customers", interactive=False)
        batch_download = gr.File(label="Download Scored CSV")

        batch_btn.click(
            score_csv_file,
            inputs=[csv_input, batch_model_selector],
            outputs=[batch_summary, batch_table, batch_download],
        )

    with gr.Tab("Agentic Retention Strategist"):
        gr.Markdown(
            "## Milestone 2 work in progress\n"
            "The LangGraph retention workflow will appear here once the agent, retrieval layer, and structured output pipeline are wired in."
        )
        gr.Textbox(value=session_id.value if hasattr(session_id, "value") else "", label="Session ID", interactive=False)

if __name__ == "__main__":
    demo.launch(share=False)
