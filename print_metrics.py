import joblib
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
metrics = joblib.load('./models/metrics.joblib')
df = pd.DataFrame(metrics).T
print(df)
