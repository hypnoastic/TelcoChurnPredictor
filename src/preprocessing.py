from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.data_loader import get_feature_lists

def get_preprocessor():
    """
    Creates and returns the ColumnTransformer for preprocessing.
    
    Returns:
        ColumnTransformer: Constructed preprocessor.
    """
    numeric_features, categorical_features = get_feature_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    return preprocessor
