from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_evaluation_metrics(y_true, y_pred):
    """
    Calculates key performance metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Generates and optionally saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()

def plot_roc_curve(models_dict, X_test, y_test, save_path=None):
    """
    Generates a combined ROC curve for multiple models.
    models_dict: {'Model Name': model_object}
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plots feature importance for Decision Tree.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()

def plot_coefficients(model, feature_names, save_path=None):
    """
    Plots coefficients for Logistic Regression.
    """
    coefs = model.coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1][:10] # Top 10 by magnitude
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), coefs[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()
