import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
# %matplotlib inline
sns.set_theme(style='white', context='notebook', rc={'figure.figsize':(14,10)})

test = pd.read_csv("./testData2.csv")
test.head()
# print(test)

test = test.dropna()
test.Position.value_counts()

sns.pairplot(test, hue='Name')

reducer = umap.UMAP()

test_data = test[
    [
        # "Name",
        # "Value1",
        "Value2",
        "Value3",
    ]
].values
scaled_test_data = StandardScaler().fit_transform(test_data)

# print(scaled_test_data)

embedding = reducer.fit_transform(scaled_test_data)
embedding.shape

# print(test_data)
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in test.Name.map({"Martin":0, "Hendrik":1, "Chiara":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP mit Chiaras Daten', fontsize=24);

plt.show()
