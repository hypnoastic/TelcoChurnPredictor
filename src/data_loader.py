import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    """
    Loads the customer churn dataset and performs initial cleaning.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Load Data
    data = pd.read_csv(filepath)
    
    # Data Cleaning: Convert TotalCharges to numeric, coerce errors to NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Fill missing values with mean
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
    
    # Drop customerID if it exists
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
    
    # Convert Target to Numeric
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        
    return data

def get_feature_lists():
    """
    Returns lists of numeric and categorical features.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    return numeric_features, categorical_features
