import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

data = sns.load_dataset("titanic")
print(data)

print(data.info)


# histogram
plt.figure(figsize=(10,6))
sns.histplot(data=data,x='fare', kde=True, bins=30, color='skyblue')

plt.title('Distribution of Ticket Fares')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()


# Plot survival count based on gender
sns.countplot(data=data, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.show()

# Boxplot of fare by class
sns.boxplot(data=data, x='class', y='fare')
plt.title('Fare Distribution by Class')
plt.show()
