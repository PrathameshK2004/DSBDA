# ----------------------1.  Data Wrangling I ------------------------------------
# Perform the following operations using Python on any open source dataset (e.g., data.csv)
# 1. Import all the required Python Libraries.
# 2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear
#             description of the data and its source (i.e., URL of the web site).
# 3. Load the Dataset into pandas dataframe.
# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
# function to get some initial statistics. Provide variable descriptions. Types of variables etc.
# Check the dimensions of the data frame.
# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking
# the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the
# data set. If variables are not in the correct data type, apply proper type conversions.
# 6. Turn categorical variables into quantitative variables in Python.
#
# In addition to the codes and outputs, explain every operation that you do in the above steps and
# explain everything that you do to import/read/scrape the data set.

# 1. Import all the required Python Libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Titanic Dataset
df  = pd.read_csv("titanic.csv")
print(df)

# Step 3: Dataset Info
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df.shape)

#Step 4: Variable Descriptions
#From Titanic dataset (common columns):
# Survived: 0 = No, 1 = Yes (categorical/binary)
# Pclass: Passenger class (categorical - 1, 2, 3)
# Name, Sex, Ticket, Cabin, Embarked: categorical
# Age, Fare: numeric
# SibSp, Parch: discrete numeric (can be seen as categorical too)

print(df.dtypes)

#Step 5: Data Type Conversion (if needed)
df['pclass'] = df['pclass'].astype('category')
df['sex'] = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')
df['survived'] = df['survived'].astype('category')


# Step 6: Handle Missing Values
# View missing data
print(df.isnull().sum())

# Handling missing values
df['age'] = df['age'].fillna(df['age'].median())  # Fill missing ages with median
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])  # Fill missing embarked with mode
df['deck'] = df['deck'].fillna('Unknown')  # Fill deck with 'Unknown' or any placeholder

print(df.isnull().sum())

#Step 7: Normalize Numeric Columns (Optional)
# Normalize Age and Fare using Min-Max scaling
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['fare'] = (df['fare'] - df['fare'].min()) / (df['fare'].max() - df['fare'].min())


#Step 8: Convert Categorical to Numeric
# from sklearn.preprocessing import LabelEncoder
#
# le = LabelEncoder()
# df['Sex_encoded'] = le.fit_transform(df['sex'])           # male → 1, female → 0
# df['Embarked_encoded'] = le.fit_transform(df['embarked']) # S, C, Q → numbers

df['Sex_encoded'] = df['sex'].cat.rename_categories({'male': 1, 'female': 0})
print(df['Sex_encoded'])

df['Embarked_encoded'] = df['embarked'].cat.rename_categories({'S': 0, 'C': 1, 'Q': 2})
print(df['Embarked_encoded'])

#Final Step: Check Final DataFrame
df.head()
df.info()
