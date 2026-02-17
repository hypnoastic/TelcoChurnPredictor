from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_logistic_regression_model():
    """Returns a Logistic Regression model with optimized default parameters."""
    return LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)

def get_decision_tree_model():
    """Returns a Decision Tree model with optimized default parameters."""
    return DecisionTreeClassifier(random_state=42, criterion='entropy')


