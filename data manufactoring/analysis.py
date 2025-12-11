#use pip install package name(if not installed) to install the required packages before running the file 
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import MultiColumnLabelEncoder


#Load the raw data
df_raw = pd.read_csv("garments_worker_productivity.csv")
#Load the refined data for 
df_refine = pd.read_csv("cleaned_file.csv")
"""this two dataframes are used for comparing the data before and after cleaning process"""

#descriptive statistics
print(df.describe()) # type: ignore
#informastion about data types and missing values
print(df.shape()) # type: ignore
#using SeaBorn for heatMap visualization
sns.heatmap(df.corr(), annot=True, cmap="coolwarm") # type: ignore
plt.show()
#Checking for missing values
print(df.isnull().sum()) # type: ignore

#Merging the catergorical columns duplicate values
df['department'] = df['department'].apply(lambda x: 'finishing' if  x.replace("","") == 'finishing' else 'sweing') # type: ignore
print("This is the data before merging of departments columns: \n",df_raw['department'].value_counts())
print("This is the data after the merging of deparment column: \n",df_refine['department'].value_counts())

Mcle = MultiColumnLabelEncoder.MultiColumnLabelEncoder()
data = Mcle.fit_transform(df_raw, columns = ['department','day','quarter'])
print(data)

#comparing two values (Training and testing data)
import pandas as pd
from split_data import get_split_data

x-train, x_test, y_train, y_test = get_split_data()

