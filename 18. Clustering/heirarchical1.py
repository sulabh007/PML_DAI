from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

### Calculating the linkages
mergings = linkage(milkscaled,method='complete')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
dendrogram(mergings,
           labels=list(milk.index),
           leaf_rotation=45,
           leaf_font_size=13)
plt.ylabel("Cluster Distance")
plt.show()

################ sklearn ############################
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
clustering = AgglomerativeClustering(n_clusters=2,
                                     linkage='average') 
pipe = Pipeline([('SCl', scaler), ('CLUST', clustering)])
milkscaled=scaler.fit_transform(milk)
pipe.fit(milk)
print(clustering.labels_)
print(silhouette_score(milkscaled,clustering.labels_))
