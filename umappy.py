import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
# %matplotlib inline
sns.set(style='white',
# context='notebook',
rc={'figure.figsize':(10,8)})

test = pd.read_csv("./Martin2.csv", sep=";", decimal=",", usecols=['maxdV', 'type'])[['type', 'maxdV']]
test.head()
print(test)
test = test.dropna()

standard_embedding = umap.UMAP(

).fit_transform(test)

plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='spectral')

# test = test.dropna()
# # test = test.drop('type')
# test.type.value_counts()

# sns.pairplot(test, hue='type')

# # sns.clustermap(test, hue='type')




# reducer = umap.UMAP()

# test_data = test[
#     [
#         "maxdV",
#         # "type",
#         # "rheobase",
#         # # "Name",
#         # "Vh",
#         # # "Rin",
#         # "Tau",
#         # "Cm",
#         # "Rin",
#         # "Thres",
#         # # "Threshold_LP",
#         # "Thres_Time",
#         # "AP_duration",
#         # # "Rise",
#         # "ADP",
#         # "ADP_area",
#         # # "FAHP",
#         # # "FAHP_time",

#     ]
# ].values
# scaled_test_data = umap.UMAP(
#     # n_neighbors=30,
#     # min_dist=0.0,
#     # n_components=2,
#     # random_state=42,
# ).fit_transform(test_data)

# print(scaled_test_data)

# embedding = reducer.fit_transform(scaled_test_data)
# embedding.shape

# print("###################################")
# print(embedding)

# print(test_data)
# plt.scatter(scaled_test_data[:, 0], scaled_test_data[:, 1],
# c=sns.color_palette(), s=0.1, cmap='Spectral')
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP mit Chiaras Daten', fontsize=24);

plt.show()









# %matplotlib inline

# ourData = pd.read_csv("C:/_dev/umappy/ourData.csv")
# ourData.head()

# ourData = ourData.dropna()
# # str(ourData)
# # ourData.Threshold_LP.value_counts()

# # print(ourData.Threshold_LP.value_counts())

# sns.pairplot(ourData.drop("Vh", axis=1), hue='Rin')

# reducer = umap.UMAP()


# ourData_data = ourData[
#     [
#         "Threshold_LP",
#         "AP_peak",
#         "rise"
#     ]
# ].values
# scaled_ourData_data =StandardScaler().fit_transform(ourData)
# scaled_ourData_data.shape

# embedding = reducer.fit_transform(scaled_ourData_data)
# embedding.shape

# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
# )
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the ourData dataset', fontsize=24)