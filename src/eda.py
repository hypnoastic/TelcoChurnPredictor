import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew

def check_missing_values(data):
    """Checks for missing values in the dataset."""
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    return missing

def check_class_imbalance(data, target_col='Churn'):
    """Checks and plots class imbalance."""
    counts = data[target_col].value_counts(normalize=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f'Class Distribution ({target_col})')
    plt.ylabel('Proportion')
    return counts

def check_correlations(data, numeric_features, save_path=None):
    """Plots correlation heatmap for numerical features."""
    corr = data[numeric_features].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        return plt.gcf()

def check_multicollinearity(data, numeric_features):
    """Calculates VIF for numerical features to detect multicollinearity."""
    # Drop rows with NaNs or infs just for VIF calculation
    X = data[numeric_features].dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(numeric_features))]
    return vif_data.sort_values(by="VIF", ascending=False)

def check_skewness(data, numeric_features):
    """Calculates skewness for numerical features."""
    skew_vals = data[numeric_features].apply(lambda x: skew(x.dropna()))
    return skew_vals.sort_values(ascending=False)

def detect_outliers(data, numeric_features):
    """Detects outliers using IQR method."""
    outlier_counts = {}
    for col in numeric_features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    return pd.Series(outlier_counts)

def generate_eda_report(data_path, numeric_features, target_col='Churn', output_dir='./reports'):
    """Generates a comprehensive EDA report."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data = pd.read_csv(data_path)
    # Basic cleaning for EDA (convert strings to numeric if needed)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
    if target_col in data.columns and data[target_col].dtype == 'object':
         data[target_col] = data[target_col].map({'Yes': 1, 'No': 0})

    report_path = os.path.join(output_dir, 'eda_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("Deep Exploratory Data Analysis Report\n")
        f.write("=====================================\n\n")
        
        # 1. Missing Values
        f.write("1. Missing Values:\n")
        missing = check_missing_values(data)
        if missing.empty:
            f.write("No missing values detected (after initial cleaning).\n")
        else:
            f.write(f"{missing}\n")
        f.write("\n")
        
        # 2. Class Imbalance
        f.write("2. Class Imbalance:\n")
        imbalance = check_class_imbalance(data, target_col)
        f.write(f"{imbalance}\n")
        if imbalance.min() < 0.2:
             f.write("Result: Moderate to High Imbalance detected. Resampling (SMOTE) is recommended.\n")
        f.write("\n")
        
        # 3. Skewness
        f.write("3. Skewness (Numeric Features):\n")
        skewness = check_skewness(data, numeric_features)
        f.write(f"{skewness}\n")
        f.write("Note: functional transformation (e.g. log) might be needed for highly skewed features (>1).\n")
        f.write("\n")

        # 4. Outliers
        f.write("4. Outliers (IQR Method):\n")
        outliers = detect_outliers(data, numeric_features)
        f.write(f"{outliers}\n")
        f.write("\n")
        
        # 5. Multicollinearity (VIF)
        f.write("5. Multicollinearity (Variance Inflation Factor):\n")
        vif = check_multicollinearity(data, numeric_features)
        f.write(f"{vif}\n")
        if vif['VIF'].max() > 10:
             f.write("Result: High Multicollinearity detected. Consider dropping or combining features.\n")
        else:
             f.write("Result: No severe multicollinearity detected.\n")
             
    # Generate Plots
    plots_dir = os.path.join(output_dir, '../plots/eda')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    check_correlations(data, numeric_features, save_path=os.path.join(plots_dir, 'correlation_matrix.png'))
    
    # Save Histogram of Distribution
    for col in numeric_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=col, hue=target_col, kde=True)
        plt.title(f'Distribution of {col} by Churn')
        plt.savefig(os.path.join(plots_dir, f'dist_{col}.png'))
        plt.close()

    print(f"EDA Report generated at {report_path}")
    print(f"EDA Plots saved to {plots_dir}")

if __name__ == "__main__":
    # Example usage for testing
    generate_eda_report('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv', ['tenure', 'MonthlyCharges', 'TotalCharges'])
