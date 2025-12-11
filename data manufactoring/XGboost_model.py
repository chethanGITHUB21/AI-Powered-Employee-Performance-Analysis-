import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prepro_pipeline import get_split_data
import pickle

x_train, x_test, y_train, y_test = get_split_data()

# Train XGBoost Regressor Model
print("this is the xgboost data: \n")
model_xgb = xgb.XGBRegressor(n_estimators = 200, max_depth=5, learning_rate = 0.1)
model_xgb.fit(x_train, y_train.values.ravel()) #ravel() function is use for flatten the array from 3D array into 1D array

Pred_xgb = model_xgb.predict(x_test)
print("\n--- Model Evaluation (XGBoost) ---")
print("Mean Square Error: ", mean_squared_error(y_test, Pred_xgb))
print("Mean Absolute Error: ", mean_absolute_error(y_test, Pred_xgb))
print("R2 Score: {} ".format(r2_score(y_test, Pred_xgb)))
"""comparing the true values and predicted values in a dataframe of XGBoost"""
comparison_xgb = pd.DataFrame({
    "True": y_test.values,
    "Predicted": Pred_xgb
})
print(comparison_xgb.head())

with open("xgboost_productivity_model.pkl", "wb") as file:
    pickle.dump(model_xgb, file)
    
print("Model saved Successfully as 'xgboost_productivity_model.pkl'")