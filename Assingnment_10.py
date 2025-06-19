import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datas = sns.load_dataset("iris")
print(datas)
print(datas.shape)
print(datas.describe())
print(datas.isnull().sum())

# 1] list down the features and their types (e.g., numeric, nominal) available in the dataset.
print(datas.dtypes)
datas['species'] = datas['species'].astype('category')
print(datas.dtypes)

# 2] Create a histogram for each feature in the dataset to illustrate the feature distributions.
datas.hist(figsize=(10, 8), bins=15, edgecolor='black', color='skyblue')
plt.suptitle('Histograms of Iris Features')
plt.tight_layout()
plt.show()

# Boxplot
datas.plot(kind='box', subplots=True, layout=(2, 2), figsize=(10, 8), sharex=False, sharey=False)
plt.suptitle('Boxplots of Iris Features')
plt.tight_layout()
plt.show()


# 4] Compare Distributions & Identify Outliers
# 🟩 Observations from Histograms:
# Petal length and width clearly show three separate groups (good for classification).
# Sepal features have more overlapping distributions.
# Data is fairly well spread for all features, but petal length has more variation.

# 🟨 Observations from Boxplots:
# Petal width and petal length show clear separation between species.
# Sepal width has some outliers, especially in Setosa.
# No extreme outliers in petal features.
# Some mild outliers in sepal measurements.