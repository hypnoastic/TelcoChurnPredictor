import gradio as gr
import pandas as pd
import joblib
import os
import numpy as np
import train

# ---------------------------------------------------------
# Trigger Training
# ---------------------------------------------------------
print("Starting Application...")
print("Triggering Advanced Model Training (This may take a moment)...")
try:
    train.main()
    print("Training Complete.")
except Exception as e:
    print(f"Training Failed: {e}")
    exit(1)

# Load Models and Metrics
MODELS_DIR = './models'
PLOTS_DIR = './plots'

try:
    lr_pipeline = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'))
    dt_pipeline = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_model.joblib'))
    metrics = joblib.load(os.path.join(MODELS_DIR, 'metrics.joblib'))
except FileNotFoundError as e:
    print(f"Error: Models not found. {e}")
    exit(1)

def predict_churn(model_name, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
                  MonthlyCharges, TotalCharges):
    
    # Create DataFrame
    data = {
        'gender': [gender],
        'SeniorCitizen': [int(SeniorCitizen)],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [int(tenure)],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [float(MonthlyCharges)],
        'TotalCharges': [float(TotalCharges)]
    }
    
    df = pd.DataFrame(data)
    
    # Select Model & Threshold
    if model_name == "Logistic Regression":
        model = lr_pipeline
        threshold = metrics['Logistic Regression'].get('Threshold', 0.5)
    elif model_name == "Decision Tree":
        model = dt_pipeline
        threshold = metrics['Decision Tree'].get('Threshold', 0.5)
        
    # Predict
    try:
        prob = model.predict_proba(df)[0][1]
        pred_class = 1 if prob >= threshold else 0
    except Exception as e:
        return f"Error: {str(e)}", "Error"

    label = "Yes (Churn)" if pred_class == 1 else "No (Retain)"
    prob_percent = f"{prob*100:.2f}%"
    
    return label, prob_percent

# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
# ---------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

css = """
.gradio-container {max_width: 1200px !important}

/* Light Mode Headings */
h1 {color: #0f172a !important; font-size: 40px !important;} /* Slate-900 */
h2 {color: #0369a1 !important; font-size: 28px !important;} /* Sky-700 */
h3 {color: #475569 !important; font-size: 20px !important;} /* Slate-600 */

/* Dark Mode Headings */
.dark h1 {color: #f8fafc !important;} 
.dark h2 {color: #ffffff !important;} /* Pure White */
.dark h3 {color: #94a3b8 !important;} 

/* Graph Container - Adaptive */
.gradio-image {
    height: 420px !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background-color: #f8fafc !important; /* Light bg */
}

.dark .gradio-image {
    border: 1px solid #334155 !important;
    background-color: #1e293b !important; /* Dark bg */
}

.gradio-image img {
    max-height: 400px !important;
    width: auto !important;
    object-fit: contain !important;
}
"""

with gr.Blocks(theme=theme, title="Telecom Churn Prediction", css=css) as demo:
    with gr.Row():
        gr.Markdown("# Telecom Customer Churn Prediction System")

    
    with gr.Tab("Model Performance Dashboard"):
        gr.Markdown("## Key Performance Indicators (Test Set)")
        
        # Metrics Table
        files_data = []
        for name, m in metrics.items():
            threshold_val = m.get('Threshold', 0.5)
            files_data.append([
                name, 
                f"{m['Accuracy']:.1%}", 
                f"{m['Precision']:.1%}", 
                f"{m['Recall']:.1%}", 
                f"{m['F1 Score']:.1%}",
                f"{threshold_val:.4f}"
            ])
        
        gr.Dataframe(
            headers=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Threshold"],
            value=files_data,
            interactive=False
        )
        
        gr.Markdown("---")
        gr.Markdown("## Visual Analysis")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. ROC Curve (Discrimination)")
                gr.Image(os.path.join(PLOTS_DIR, 'roc_comparison.png'), show_label=False)
            with gr.Column():
                gr.Markdown("### 2. Precision-Recall Curve (Trade-off)")
                gr.Image(os.path.join(PLOTS_DIR, 'pr_curve.png'), show_label=False)
        
        gr.Markdown("---")
        gr.Markdown("## Logistic Regression Analysis")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 3. Confusion Matrix")
                if os.path.exists(os.path.join(PLOTS_DIR, 'logistic_regression_cm.png')):
                    gr.Image(os.path.join(PLOTS_DIR, 'logistic_regression_cm.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")
            with gr.Column():
                gr.Markdown("### 4. Top Coefficients")
                if os.path.exists(os.path.join(PLOTS_DIR, 'lr_coef.png')):
                    gr.Image(os.path.join(PLOTS_DIR, 'lr_coef.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")

        gr.Markdown("---")
        gr.Markdown("## Decision Tree Analysis")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 5. Confusion Matrix")
                if os.path.exists(os.path.join(PLOTS_DIR, 'decision_tree_cm.png')):
                    gr.Image(os.path.join(PLOTS_DIR, 'decision_tree_cm.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")
            with gr.Column():
                gr.Markdown("### 6. Feature Importance")
                if os.path.exists(os.path.join(PLOTS_DIR, 'dt_importance.png')):
                    gr.Image(os.path.join(PLOTS_DIR, 'dt_importance.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")

    with gr.Tab("Exploratory Data Analysis"):
        gr.Markdown("## Data Distribution & Correlation")
        
        EDA_DIR = os.path.join(PLOTS_DIR, 'eda')
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Tenure Distribution")
                if os.path.exists(os.path.join(EDA_DIR, 'dist_tenure.png')):
                    gr.Image(os.path.join(EDA_DIR, 'dist_tenure.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")
            with gr.Column():
                gr.Markdown("### 2. Monthly Charges Distribution")
                if os.path.exists(os.path.join(EDA_DIR, 'dist_MonthlyCharges.png')):
                    gr.Image(os.path.join(EDA_DIR, 'dist_MonthlyCharges.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 3. Total Charges Distribution")
                if os.path.exists(os.path.join(EDA_DIR, 'dist_TotalCharges.png')):
                    gr.Image(os.path.join(EDA_DIR, 'dist_TotalCharges.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")
            with gr.Column():
                gr.Markdown("### 4. Correlation Matrix")
                if os.path.exists(os.path.join(EDA_DIR, 'correlation_matrix.png')):
                    gr.Image(os.path.join(EDA_DIR, 'correlation_matrix.png'), show_label=False)
                else:
                    gr.Markdown("Plot not found.")

    with gr.Tab("Prediction System"):
        gr.Markdown("## Customer Churn Probability Estimator")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Customer Profile")
                with gr.Group():
                    gender = gr.Dropdown(["Female", "Male"], label="Gender", value="Female")
                    SeniorCitizen = gr.Checkbox(label="Senior Citizen")
                    Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                    Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
                    tenure = gr.Slider(0, 72, label="Tenure (Months)", value=12)
                
                gr.Markdown("### Service Details")
                with gr.Group():
                    PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                    MultipleLines = gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines", value="No")
                    InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                    OnlineSecurity = gr.Dropdown(["No", "Yes", "No internet service"], label="Online Security", value="No")
                    OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                    DeviceProtection = gr.Dropdown(["No", "Yes", "No internet service"], label="Device Protection", value="No")
                    TechSupport = gr.Dropdown(["No", "Yes", "No internet service"], label="Tech Support", value="No")

            with gr.Column(scale=1):
                gr.Markdown("### Usage & Billing")
                with gr.Group():
                    StreamingTV = gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming TV", value="Yes")
                    StreamingMovies = gr.Dropdown(["No", "Yes", "No internet service"], label="Streaming Movies", value="Yes")
                    Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                    PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                    PaymentMethod = gr.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label="Payment Method", value="Electronic check")
                    MonthlyCharges = gr.Number(label="Monthly Charges ($)", value=70.0)
                    TotalCharges = gr.Number(label="Total Charges ($)", value=1000.0)
                
                gr.Markdown("### Prediction Control")
                with gr.Group():
                    model_selector = gr.Dropdown(["Logistic Regression", "Decision Tree"], label="Select Evaluation Model", value="Logistic Regression")
                    predict_btn = gr.Button("Calculate Churn Risk", variant="primary", size="lg")
                
                gr.Markdown("### Results")
                with gr.Group():
                    with gr.Row():
                        output_label = gr.Textbox(label="Prediction Result", scale=2)
                        output_prob = gr.Textbox(label="Probability Score", scale=1)

        predict_btn.click(
            predict_churn,
            inputs=[model_selector, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
                    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
                    MonthlyCharges, TotalCharges],
            outputs=[output_label, output_prob]
        )

if __name__ == "__main__":
    demo.launch(share=False)
