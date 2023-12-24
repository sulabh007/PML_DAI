import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

km = KMeans(n_clusters=2, random_state=23)
km.fit(milkscaled)

print(km.labels_)
print(silhouette_score(milkscaled,km.labels_))

###################################
clust = [2,3,4,5,6]
scores = []
for i in clust:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(milkscaled)
    sil = silhouette_score(milkscaled,km.labels_)
    scores.append(sil)
    
print("Best Clusters =", clust[np.argmax(scores)])
print("Best Score =", np.max(scores))

plt.scatter(clust, scores, c='red')
plt.plot(clust, scores, c="blue")
plt.xlabel("No. of Clusters")
plt.ylabel("Score")
plt.show()

############## WSS ######################

km = KMeans(n_clusters=4, random_state=23)
km.fit(milkscaled)
print(km.inertia_)

clust = [2,3,4,5,6,7,8]
scores = []
for i in clust:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(milkscaled)
    scores.append(km.inertia_)
    
plt.scatter(clust, scores, c='red')
plt.plot(clust, scores, c="blue")
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()


