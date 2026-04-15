import pandas as pd

from src.inference import predict_dataframe, predict_single, validate_customer_dataframe


SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 1000.0,
}


def test_predict_single_returns_probability_and_drivers():
    result = predict_single(SAMPLE_CUSTOMER, model_name="Logistic Regression")

    assert result["prediction"] in {"Yes (Churn)", "No (Retain)"}
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["drivers"]) >= 1


def test_predict_dataframe_appends_batch_output_columns():
    input_df = pd.DataFrame([SAMPLE_CUSTOMER, {**SAMPLE_CUSTOMER, "tenure": 48, "Contract": "Two year"}])
    result_df = predict_dataframe(input_df, model_name="Decision Tree")

    expected_columns = {
        "selected_model",
        "churn_probability",
        "churn_probability_pct",
        "churn_prediction",
        "decision_threshold",
        "driver_1",
        "driver_2",
        "driver_3",
        "driver_summary",
    }
    assert expected_columns.issubset(result_df.columns)
    assert len(result_df) == 2


def test_validate_customer_dataframe_reports_missing_columns():
    missing = validate_customer_dataframe(pd.DataFrame([{"gender": "Female"}]))
    assert "Contract" in missing
    assert "MonthlyCharges" in missing
