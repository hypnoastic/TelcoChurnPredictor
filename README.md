---
title: Telco Churn Predictor

colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Telecom Customer Churn Prediction System

## Overview

This project is a production-grade machine learning system designed to predict customer churn for a telecommunications provider. It utilizes a robust pipeline involving data preprocessing, feature engineering, and ensemble modeling to identify customers at risk of leaving. The system is deployed via an interactive web dashboard for real-time inference and business analytics.

## System Architecture

The system follows a modular architecture composed of three main layers:

### 1. Data Processing Layer

- **Data Ingestion**: Loads raw customer data from CSV files.
- **Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features using standard scalers.
- **Feature Engineering**: Generates interaction terms (Polynomial Features) and performs feature selection (Lasso Regression) to enhance model predictive power.
- **Balancing**: Implements SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the training dataset.

### 2. Model Training Layer

The system employs a multi-model approach to balance accuracy and interpretability:

- **Logistic Regression**: Optimized with Polynomial Features and Lasso Selection for high interpretability and strong baseline performance.
- **Decision Tree**: tailored for capturing non-linear patterns and providing clear decision rules.
- **Evaluation**: rigorous testing using Stratified K-Fold Cross-Validation. Metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 3. Application Layer (Dashboard)

A Gradio-based web interface serves as the end-user entry point:

- **Model Performance Dashboard**: Visualizes model metrics, ROC curves, Precision-Recall curves, Confusion Matrices, and Feature Importance methodology.
- **Exploratory Data Analysis (EDA)**: Displays data distributions and correlation matrices to understand underlying data patterns.
- **Prediction System**: Accepts real-time customer inputs and outputs a churn probability score along with a binary retention/churn decision.

## Project Structure

```
.
├── app.py                 # Main entry point for the Web Dashboard
├── train.py               # Training pipeline execution script
├── src/
│   ├── data_loader.py     # Data ingestion logic
│   ├── preprocessing.py   # Feature transformation pipelines
│   ├── models.py          # Model definitions
│   ├── evaluation.py      # Metric calculation and plotting libraries
│   └── eda.py             # Exploratory Data Analysis utilities
├── data/                  # Raw dataset directory
├── models/                # Serialized trained models (.joblib)
├── plots/                 # Generated visualization artifacts
├── reports/               # Text-based performance reports
└── requirements.txt       # Project dependencies
```

## Installation and Usage

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training

To retrain the models and regenerate analysis artifacts:

```bash
python train.py
```

### Running the Dashboard

To launch the interactive application:

```bash
python app.py
```

The application will be accessible at the local URL provided in the terminal (typically http://127.0.0.1:7860).

## Technologies Used

- **Language**: Python
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Interface**: Gradio
