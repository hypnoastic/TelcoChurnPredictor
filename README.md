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

Production ML system for predicting customer churn using Logistic Regression and Decision Trees with advanced feature engineering and SMOTE balancing.

## System Flow

```mermaid
flowchart TB
    A["<b>Raw Data</b><br/>7,043 customers<br/>21 features"] --> B["<b>Data Cleaning</b><br/>Handle missing values<br/>Encode target"]
    B --> C["<b>Train-Test Split</b><br/>75% train / 25% test<br/>Stratified sampling"]
    
    C --> D["<b>Numeric Processing</b><br/>StandardScaler<br/>4 features"]
    C --> E["<b>Categorical Processing</b><br/>OneHotEncoder<br/>15 features"]
    
    D --> F["<b>Feature Engineering</b><br/>PolynomialFeatures<br/>Lasso Selection"]
    E --> F
    
    F --> G["<b>SMOTE Balancing</b><br/>2.77:1 → 1:1<br/>Training set only"]
    
    G --> H["<b>Model Training</b><br/>Stratified 5-Fold CV"]
    
    H --> I["<b>Logistic Regression</b><br/>GridSearchCV<br/>C, penalty"]
    H --> J["<b>Decision Tree</b><br/>GridSearchCV<br/>max_depth, min_samples"]
    
    I --> K["<b>Threshold Optimization</b><br/>Maximize F1-Score<br/>Precision-Recall curve"]
    J --> K
    
    K --> L["<b>Model Evaluation</b><br/>Accuracy, Precision, Recall<br/>F1-Score, ROC-AUC"]
    
    L --> M["<b>Save Models</b><br/>.joblib files<br/>Scaler + Metrics"]
    
    M --> N["<b>Gradio Dashboard</b><br/>3 Tabs: Performance<br/>EDA, Predictions"]
    
    N --> O["<b>Churn Prediction</b><br/>Yes/No + Probability<br/>Real-time inference"]
    
    style A fill:#bbdefb,stroke:#0d47a1,stroke-width:2px,color:#000
    style B fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    style C fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000
    style D fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000
    style E fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000
    style F fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#000
    style G fill:#f8bbd0,stroke:#880e4f,stroke-width:2px,color:#000
    style H fill:#b2dfdb,stroke:#004d40,stroke-width:2px,color:#000
    style I fill:#b3e5fc,stroke:#01579b,stroke-width:2px,color:#000
    style J fill:#b3e5fc,stroke:#01579b,stroke-width:2px,color:#000
    style K fill:#dcedc8,stroke:#33691e,stroke-width:2px,color:#000
    style L fill:#f8bbd0,stroke:#880e4f,stroke-width:2px,color:#000
    style M fill:#d1c4e9,stroke:#311b92,stroke-width:2px,color:#000
    style N fill:#b2ebf2,stroke:#006064,stroke-width:2px,color:#000
    style O fill:#a5d6a7,stroke:#1b5e20,stroke-width:3px,color:#000
```

## Pipeline Overview

### 1. Data Processing
- Load CSV dataset (7,043 customers, 21 features)
- Handle missing values in TotalCharges (11 records)
- Encode target variable: Churn (Yes=1, No=0)
- Split: 75% train, 25% test (stratified)

### 2. Feature Engineering
- **Numeric Features** (4): tenure, MonthlyCharges, TotalCharges, SeniorCitizen
  - Apply StandardScaler (mean=0, std=1)
- **Categorical Features** (15): gender, Contract, InternetService, etc.
  - Apply OneHotEncoder (drop_first=True)
- **Advanced** (Logistic Regression only):
  - PolynomialFeatures (degree=2, interactions)
  - Lasso feature selection (30-50 features)
- **Output**: 30 engineered features

### 3. Imbalance Handling
- Original ratio: 2.77:1 (No Churn : Churn)
- Apply SMOTE oversampling → 1:1 balanced training set
- Maintains test set distribution for realistic evaluation

### 4. Model Training
- **Stratified 5-Fold Cross-Validation**
- **GridSearchCV** for hyperparameter tuning
- **Models**:
  - **Logistic Regression**: C, penalty, max_features
  - **Decision Tree**: max_depth, min_samples_leaf, class_weight
- **Scoring**: Accuracy (primary metric)

### 5. Threshold Optimization
- Generate probability predictions on test set
- Compute Precision-Recall curve
- Select threshold maximizing F1-Score
- Balances precision and recall for business needs

### 6. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: 
  - Confusion matrices
  - ROC curves (model comparison)
  - Precision-Recall curves
  - Feature importance (Decision Tree)
  - Coefficients (Logistic Regression)
  - Calibration curves

### 7. Deployment
- Save trained models (.joblib)
- Gradio web interface with 3 tabs:
  - **Performance Dashboard**: Metrics, ROC/PR curves, confusion matrices
  - **EDA**: Distribution plots, correlation matrix
  - **Prediction System**: Real-time churn prediction with probability scores

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train models
python train.py

# Launch dashboard
python app.py  # http://127.0.0.1:7860
```

## Project Structure

```
├── app.py              # Gradio interface
├── train.py            # Training pipeline
├── src/
│   ├── data_loader.py     # Data loading & cleaning
│   ├── preprocessing.py   # Feature transformers
│   ├── models.py          # Model definitions
│   └── evaluation.py      # Metrics & plotting
├── data/               # CSV dataset
├── models/             # Trained models (.joblib)
└── plots/              # Visualizations
```

## Key Technical Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| **Imbalance** | SMOTE | Synthetic oversampling for 2.77:1 ratio |
| **Scaling** | StandardScaler | Required for Logistic Regression |
| **Encoding** | OneHotEncoder | Avoids ordinal assumptions |
| **Features** | PolynomialFeatures | Captures interactions (tenure × contract) |
| **Selection** | Lasso (L1) | Reduces overfitting |
| **CV** | Stratified 5-Fold | Maintains class distribution |
| **Threshold** | F1-Optimized | Balances precision/recall |

## Technologies

Scikit-Learn • Imbalanced-Learn • Pandas • Gradio
