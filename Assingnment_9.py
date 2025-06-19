import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datas = sns.load_dataset("titanic")
print(datas.head())
print(datas.info)

# Drop rows with missing age values first
datas = datas.dropna(subset=['age'])

# Calculate Q1 and Q3
Q1 = datas['age'].quantile(0.25)
Q3 = datas['age'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers
filtered_data = datas[(datas['age'] >= lower_bound) & (datas['age'] <= upper_bound)]

# boxplot to show wheather people are survived or not
plt.figure(figsize=(10,6))
sns.boxplot(data=filtered_data, x='sex',y='age',hue='survived')

# Add title and labels
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()