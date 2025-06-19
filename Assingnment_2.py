#2.  Data Wrangling II
# Create an “Academic performance” dataset of students and perform the following operations using
# Python.
# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
# inconsistencies, use any of the suitable techniques to deal with them.
# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable
# techniques to deal with them.
# 3. Apply data transformations on at least one of the variables. The purpose of this
# transformation should be one of the following reasons: to change the scale for better
# understanding of the variable, to convert a non-linear relation into a linear one, or to
# decrease the skewness and convert the distribution into a normal distribution.

# Reason and document your approach properly.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('StudentsPerformance.csv')
print(data)

# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
# inconsistencies, use any of the suitable techniques to deal with them.
print(data.isnull().sum())
data.ffill(inplace=True)     #Forward Fill
# data.bfill(inplace=True)     #Backward Fill

print(data)
print(data.head())
print(data.dtypes)


# print(data.select_dtypes(include=['object', 'category']).apply(pd.Series.unique))   there is no any string or object data type colume in dataset
# if there is such coloums then Convert object columns to category (optional but useful)  and then print(df[object_cols].apply(pd.Series.unique))



# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable
# techniques to deal with them.

# Find outliers using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1           # it gives middle part of data

# Boolean DataFrame where True = outlier
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

print("Number of outliers in each column:\n", outliers.sum())
#✅ Options to Handle Outliers:
# 1. Remove Outliers (Drop Rows)
# Remove rows that contain outliers.

# df = df[(df['column'] >= lower_limit) & (df['column'] <= upper_limit)]
# Use this when outliers are data entry errors or are not relevant for analysis.

# 2. Replace with Median or Mean (Capping)
# Cap outliers at a certain threshold to reduce their effect.

# df['column'] = np.where(df['column'] > upper_limit, upper_limit,
#                         np.where(df['column'] < lower_limit, lower_limit, df['column']))
# This keeps all rows but limits extreme values.

# 3. Transform the Data
# Apply transformations like log, sqrt, or Box-Cox to reduce the impact of outliers.

# df['transformed'] = np.log1p(df['column'])  # log1p handles 0 values safely
# Best used when you want to keep the data but reduce skewness.



# 3. Apply data transformations on at least one of the variables. The purpose of this
# transformation should be one of the following reasons: to change the scale for better
# understanding of the variable, to convert a non-linear relation into a linear one, or to
# decrease the skewness and convert the distribution into a normal distribution.

# Add small value to avoid log(0)
data['Placement_Score_log'] = np.log1p(data['Placement_Score'])

# Compare distributions if needed
print(data[['Placement_Score', 'Placement_Score_log']].describe())