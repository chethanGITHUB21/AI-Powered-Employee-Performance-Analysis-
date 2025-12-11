from sklearn.linear_model import LinearRegression 
from split_data import get_split_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle


x_train, x_test, y_train, y_test = get_split_data()

# Train Linear Regression Model
print("this is the linear regression data: \n")
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
Pred_lr = model_lr.predict(x_test)
print("Mean Square Error: ", mean_squared_error(y_test, Pred_lr))
print("Mean Absolute Error: ", mean_absolute_error(y_test, Pred_lr))
print("R2 Score: {} ".format(r2_score(y_test, Pred_lr)))
"""comparing the true values and predicted values in a dataframe of linear regression"""
comparison = pd.DataFrame({
    "True": y_test.values,
    "Predicted": Pred_lr
})
print(comparison.head())

# Train Random Forest Regressor Model
print("this is the random forest data: \n")
model_rf = RandomForestRegressor(n_estimators=200, random_state=5)
model_rf.fit(x_train, y_train)

Pred_rf = model_rf.predict(x_test)
print("Mean Square Error: ", mean_squared_error(y_test, Pred_rf))
print("Mean Absolute Error: ", mean_absolute_error(y_test, Pred_rf))
print("R2 Score: {} ".format(r2_score(y_test, Pred_rf)))
"""comparing the true values and predicted values in a dataframe of random forest"""
comparison_rf = pd.DataFrame({
    "True": y_test.values,
    "Predicted": Pred_rf
})
print(comparison_rf.head())

# Train XGBoost Regressor Model
print("this is the xgboost data: \n")
model_xgb = xgb.XGBRegressor(n_estimators = 200, max_depth=5, learning_rate = 0.1)
model_xgb.fit(x_train, y_train)

Pred_xgb = model_xgb.predict(x_test)
print("Mean Square Error: ", mean_squared_error(y_test, Pred_xgb))
print("Mean Absolute Error: ", mean_absolute_error(y_test, Pred_xgb))
print("R2 Score: {} ".format(r2_score(y_test, Pred_xgb)))
"""comparing the true values and predicted values in a dataframe of XGBoost"""
comparison_xgb = pd.DataFrame({
    "True": y_test.values,
    "Predicted": Pred_xgb
})
print(comparison_xgb.head())