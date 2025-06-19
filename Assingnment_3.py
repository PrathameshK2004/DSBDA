# Descriptive Statistics - Measures of Central Tendency and variability
# Perform the following operations on any open source dataset (e.g., data.csv)
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for
# a dataset (age, income etc.) with numeric variables grouped by one of the qualitative
# (categorical) variable. For example, if your categorical variable is age groups and
# quantitative variable is income, then provide summary statistics of income grouped by the
# age groups. Create a list that contains a numeric value for each response to the categorical
# variable.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
print(data.head())
print(data.isnull().sum())
print(data.dtypes)
grouped_data= data.groupby('Gender')[['Age','EstimatedSalary']].agg(['mean','median','min','max','std'])
print(grouped_data)

# Create a dictionary with numeric values for each 'Gender'
gender_numeric_values = data.groupby('Gender')[['Age', 'EstimatedSalary']].apply(lambda x: x.values.tolist()).to_dict()    # A lambda is a short way to write a function in one line.
print(gender_numeric_values)



# 2. Write a Python program to display some basic statistical details like percentile, mean,
# standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
# of iris.csv dataset.
#  Provide the codes with outputs and explain everything that you do in this step.

data1 = sns.load_dataset('Iris')
print(data1)

print(data1.isnull().sum())
print(data1.head(50))
print(data1.shape)
print(data1.describe())
print(data1.dtypes)

data1['species'] = data1['species'].astype('category')
print(data1.dtypes)

grouped_iris = data1.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].agg(['mean','median','std','min','max'])
print(grouped_iris)

# for percentile use describe
grouped_iris1 = data1.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].describe()
print(grouped_iris1)

grouped_iris2 = data1.groupby('species').describe(percentiles=[0.25,0.50,0.75])
print(grouped_iris2)