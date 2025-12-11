# AI-Powered-Employee-Performance-Analysis-

Using python I have build a simple employee performance analyser which gives you the performance on the basis of details you enter in webpage form.

# Employee Productivity Prediction – Machine Learning Web App

This project is a complete **end-to-end Machine Learning Application** that predicts **employee productivity** based on workplace factors such as SMV, overtime, idle time, incentives, number of workers, etc.

The project includes:

- Data preprocessing
- ML pipeline
- Train/Test split
- Model training (XGBoost)
- Model evaluation
- Saving the model using Pickle
- Flask backend
- HTML frontend form for predictions

---

## Tech Stack

### **Backend**

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Pickle

### **Frontend**

- HTML
- CSS
- (Optional) JavaScript

---

## Required Python Packages

- Install manually:
- pandas
- numpy
- scikit-learn
- xgboost
- flask
- pickle5

---

## 🧠 Machine Learning Pipeline

### **1. Load Dataset**

- Read CSV file
- Convert date column
- Clean missing values
- Fix inconsistent data types

### **2. Data Preprocessing**

- Drop unused columns
- Encode categorical features
- Normalize values if required
- Prepare X (features) and y (target)

### **3. Split Dataset**

Using `train_test_split()`:
