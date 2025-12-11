import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load file
df = pd.read_csv("garments_worker_productivity.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()
df['department'] = df['department'].str.strip().str.lower()
df['quarter'] = df['quarter'].str.strip().str.lower()

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_num'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday

df = df.drop(columns=['date'])

"""Dropping the day column because i already converted the day into integer by converting the day in number sequence like given
   monday = 0
   sunday = 6
   0-6 indexing of day_num"""
df = df.drop(columns=['day'])


# Fix missing values
df['wip'] = pd.to_numeric(df['wip'], errors="coerce")
df['wip'] = df['wip'].fillna(df['wip'].median())
df = df.fillna(0)

# Convert numeric columns properly
num_cols = ['team','targeted_productivity','smv','wip','over_time',
            'incentive','idle_time','idle_men','no_of_style_change',
            'no_of_workers']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Cap overtime outliers
df['over_time'] = df['over_time'].clip(0, 1500)

# Label encoding separately for each column
le_dept = LabelEncoder()
le_quarter = LabelEncoder()

df['department'] = le_dept.fit_transform(df['department'])
df['quarter'] = le_quarter.fit_transform(df['quarter'])
"""we can use MultiColumnLabelEncoder 
for converting all the catergorical columns into numerical values at onces
pip install MultiColumnLabelEncoder
import MultiColumnLabelEncoder
Mcle = MultiColumnLabelEncoder.MultiColumnLabelEncoder()
data = Mcle.fit_transform(df, columns = ['department','quarter'])"""

# Remove duplicates
df = df.drop_duplicates()


# Save cleaned data
df.to_csv("cleaned_file.csv", index=False)

print("Preprocessing Complete. File saved as cleaned_file.csv")


#=====================================Data Splitting=======================================

# -----------------------------
# 1. Create X and y
# -----------------------------
X = df.drop(['actual_productivity'], axis=1)   # Features
y = df['actual_productivity']                  # Target

# # -----------------------------
# # 2. Convert X into NumPy array
# # -----------------------------
# X = X.to_numpy()

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

#scaler dtandard deviation and mean
scaler = StandardScaler()
x_train[num_cols[:-1]] = scaler.fit_transform(x_train[num_cols[:-1]])

# Transform ONLY on TESTING data (using stats learned from the training set)
x_test[num_cols[:-1]] = scaler.transform(x_test[num_cols[:-1]])

# print(pd.DataFrame(x_train).head())
# print("this is the TRUE VALUES: \n",pd.DataFrame(y_train).head())
# print(pd.DataFrame(x_test).head())
# print("this is the PREDICTED VALUES: \n",pd.DataFrame(y_test).head())

# FUNCTION TO RETURN SPLIT DATA FOR IMPORTING
def get_split_data():
    return x_train, x_test, y_train, y_test
