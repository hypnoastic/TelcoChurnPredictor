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
    A["Raw Data<br/>7,043 customers<br/>21 features"] --> B["Data Cleaning<br/>Handle missing values<br/>Encode target"]
    B --> C["Train-Test Split<br/>75% train / 25% test<br/>Stratified sampling"]
    
    C --> D["Numeric Processing<br/>StandardScaler<br/>4 features"]
    C --> E["Categorical Processing<br/>OneHotEncoder<br/>15 features"]
    
    D --> F["Feature Engineering<br/>PolynomialFeatures<br/>Lasso Selection"]
    E --> F
    
    F --> G["SMOTE Balancing<br/>2.77:1 → 1:1<br/>Training set only"]
    
    G --> H["Model Training<br/>Stratified 5-Fold CV"]
    
    H --> I["Logistic Regression<br/>GridSearchCV<br/>C, penalty"]
    H --> J["Decision Tree<br/>GridSearchCV<br/>max_depth, min_samples"]
    
    I --> K["Threshold Optimization<br/>Maximize F1-Score<br/>Precision-Recall curve"]
    J --> K
    
    K --> L["Model Evaluation<br/>Accuracy, Precision, Recall<br/>F1-Score, ROC-AUC"]
    
    L --> M["Save Models<br/>.joblib files<br/>Scaler + Metrics"]
    
    M --> N["Gradio Dashboard<br/>3 Tabs: Performance<br/>EDA, Predictions"]
    
    N --> O["Churn Prediction<br/>Yes/No + Probability<br/>Real-time inference"]
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style B fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style C fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style D fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style E fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style F fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style G fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style H fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    style I fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style J fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style K fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    style L fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style M fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style N fill:#e0f7fa,stroke:#00838f,stroke-width:2px
    style O fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
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
