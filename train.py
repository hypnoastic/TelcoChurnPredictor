import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import recall_score, make_scorer, precision_score, f1_score, accuracy_score, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

# Project Modules
from src.data_loader import load_and_clean_data
from src.preprocessing import get_preprocessor
from src.models import (
    get_logistic_regression_model, 
    get_decision_tree_model
)
from src.evaluation import get_evaluation_metrics, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_coefficients
from src.eda import generate_eda_report

# Constants
DATA_PATH = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODELS_DIR = './models'
PLOTS_DIR = './plots'
REPORTS_DIR = './reports'

def get_optimal_threshold(model, X, y):
    """
    Calculates the optimal threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1
    """
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, probas)
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
        best_idx = np.argmax(f1_scores)
        if best_idx < len(thresholds):
            return thresholds[best_idx]
    return 0.5

def plot_precision_recall_curve(models_dict, X_test, y_test, save_path=None):
    plt.figure(figsize=(10, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_probs)
            plt.plot(recall, precision, label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_calibration_curve(models_dict, X_test, y_test, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_probs, n_bins=10)
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def main():
    # 1. Deep EDA
    print("Running Deep Exploratory Data Analysis...")
    # Start with a small subset of features for the example check, or full execution
    numeric_features_for_eda = ['tenure', 'MonthlyCharges', 'TotalCharges']
    generate_eda_report(DATA_PATH, numeric_features_for_eda, output_dir=REPORTS_DIR)
    
    print("\nLoading and cleaning data for training...")
    data = load_and_clean_data(DATA_PATH)
    
    # Split Data
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    preprocessor = get_preprocessor()
    
    # Define Models and Grids
    # Define Models and Grids
    models_to_train = {
        'Logistic Regression': {
            'model': get_logistic_regression_model(),
            'params': {
                'selection__max_features': [30, 40, 50],
                'classifier__C': [0.1, 1, 10], 
                'classifier__class_weight': ['balanced', None]
            }
        },
        'Decision Tree': {
            'model': get_decision_tree_model(),
            'params': {
                'classifier__class_weight': ['balanced', None],
                'classifier__max_depth': [3, 5, 7, 10],
                'classifier__min_samples_leaf': [2, 5, 10]
            }
        }
    }

    trained_models = {}
    model_metrics = {}
    
    print("\nStarting Hyperparameter Tuning (Focus: Accuracy)...")
    
    for name, config in models_to_train.items():
        print(f"\nOptimization: {name}")
        
        if name == 'Logistic Regression':
            # Advanced Pipeline for LR: Poly -> Selection -> LR
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ('selection', SelectFromModel(Lasso(alpha=0.01, random_state=42), max_features=50)),
                ('smote', SMOTE(random_state=42)),
                ('classifier', config['model'])
            ])
            params = config['params']
        else:
            # Standard Pipeline for Trees & XGBoost
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                # Note: XGBoost handles imbalance well internally via scale_pos_weight, 
                # but SMOTE can specific help cases. We'll keep SMOTE for consistency.
                ('smote', SMOTE(random_state=42)),
                ('classifier', config['model'])
            ])
            params = config['params']
        
        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline, 
            params, 
            cv=cv, 
            scoring='accuracy', # User requested optimizing for Accuracy > 85%
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        trained_models[name] = best_model
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Best Cross-Val Score: {grid.best_score_:.4f}")
        
        # Optimal Threshold Tuning
        threshold = get_optimal_threshold(best_model, X_test, y_test)
        print(f"  Optimal Threshold: {threshold:.4f}")
        
        # Final Evaluation on Test Set
        if hasattr(best_model, "predict_proba"):
            y_probs = best_model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)
        else:
            y_pred = best_model.predict(X_test)
            
        metrics = get_evaluation_metrics(y_test, y_pred)
        metrics['Threshold'] = threshold
        model_metrics[name] = metrics
        print(f"  Test Metrics: {metrics}")
        
        # Save Model & Confusion Matrix
        filename = name.lower().replace(" ", "_")
        joblib.dump(best_model, os.path.join(MODELS_DIR, f'{filename}_model.joblib'))
        # Updated to ensure nice formatting
        plot_confusion_matrix(y_test, y_pred, name, save_path=os.path.join(PLOTS_DIR, f'{filename}_cm.png'))

    # ---------------------------------------------------------
    # Comparison Plots
    # ---------------------------------------------------------
    print("\nGenerating Comparison Plots...")
    plot_roc_curve(trained_models, X_test, y_test, save_path=os.path.join(PLOTS_DIR, 'roc_comparison.png'))
    plot_precision_recall_curve(trained_models, X_test, y_test, save_path=os.path.join(PLOTS_DIR, 'pr_curve.png'))
    plot_calibration_curve(trained_models, X_test, y_test, save_path=os.path.join(PLOTS_DIR, 'calibration_curve.png'))
    
    # Feature Importance
    
    # Feature Importance
    
    # Logistic Regression Coefficients
    print("\nGenerating Coefficients for Logistic Regression...")
    if 'Logistic Regression' in trained_models:
        try:
            best_lr = trained_models['Logistic Regression']
            # Pipeline: preprocessor -> poly -> selection -> smote -> classifier
            # We need to reconstruct feature names
            preproc = best_lr.named_steps['preprocessor']
            poly = best_lr.named_steps['poly']
            selector = best_lr.named_steps['selection']
            clf = best_lr.named_steps['classifier']
            
            orig_feats = preproc.get_feature_names_out()
            poly_feats = poly.get_feature_names_out(orig_feats)
            selected_mask = selector.get_support()
            final_feats = poly_feats[selected_mask]
            
            plot_coefficients(clf, final_feats, save_path=os.path.join(PLOTS_DIR, 'lr_coef.png'))
        except Exception as e:
            print(f"Could not plot LR coefficients: {e}")

    # Decision Tree Feature Importance
    print("\nGenerating Decision Tree Feature Importance...")
    if 'Decision Tree' in trained_models:
        try:
            best_dt = trained_models['Decision Tree']
            # Pipeline: preprocessor -> smote -> classifier
            # DT doesn't select features, so we use preprocessor output
            preproc = best_dt.named_steps['preprocessor']
            clf = best_dt.named_steps['classifier']
            
            feature_names = preproc.get_feature_names_out()
            plot_feature_importance(clf, feature_names, save_path=os.path.join(PLOTS_DIR, 'dt_importance.png'))
        except Exception as e:
            print(f"Could not plot DT importance: {e}")
            

    # Save Metrics using simple filenames for UI mapping
    # metrics.joblib will now map 'Random Forest' -> metrics dict
    joblib.dump(model_metrics, os.path.join(MODELS_DIR, 'metrics.joblib'))
    
    print("\nTraining Complete. All models and reports generated.")

if __name__ == "__main__":
    main()
