
# import seaborn as sns
# import matplotlib.pyplot as plt
# import MultiColumnLabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import r2_score
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read CSV
data = pd.read_csv("cleaned_file.csv")



# -----------------------------
# 1. Create X and y
# -----------------------------
X = data.drop(['actual_productivity'], axis=1)   # Features
y = data['actual_productivity']                  # Target

# -----------------------------
# 2. Convert X into NumPy array
# -----------------------------
X = X.to_numpy()

# -----------------------------
# 3. Train–Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,       # Use 20% data for testing
    random_state=0       # Fix for reproducibility
)



print("Train/Test split completed.")
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


# print(pd.DataFrame(x_train).head())
# print("this is the TRUE VALUES: \n",pd.DataFrame(y_train).head())
# print(pd.DataFrame(x_test).head())
# print("this is the PREDICTED VALUES: \n",pd.DataFrame(y_test).head())

# FUNCTION TO RETURN SPLIT DATA FOR IMPORTING
def get_split_data():
    return x_train, x_test, y_train, y_test