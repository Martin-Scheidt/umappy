import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
# %matplotlib inline

ourData = pd.read_csv("C:/_dev/umappy/ourData.csv")
ourData.head()

ourData.dropna()
# str(ourData)
# ourData.Threshold_LP.value_counts()

# print(ourData.Threshold_LP.value_counts())

sns.pairplot(ourData.drop("Vh", axis=1), hue="DATE")

reducer = umap.UMAP()


ourData_data = ourData[
    [
        "Threshold_LP",
        "AP_peak",
        "rise"
    ]
].values
scaled_ourData_data =StandardScaler().fit_transform(ourData)
scaled_ourData_data.shape

embedding = reducer.fit_transform(scaled_ourData_data)
embedding.shape

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the ourData dataset', fontsize=24)