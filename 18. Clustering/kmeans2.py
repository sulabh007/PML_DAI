import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("Protein.csv", index_col=0)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

clust = [2,3,4,5,6,7,8]
scores = []
for i in clust:
    km = KMeans(n_clusters=i, random_state=23)
    km.fit(df_scaled)
    scores.append(km.inertia_)
    
plt.scatter(clust, scores, c='red')
plt.plot(clust, scores, c="blue")
plt.title("Scree Plot")
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()


#######################################
km = KMeans(n_clusters=3, random_state=23)
km.fit(df_scaled)
print(km.labels_)
print(km.predict(df_scaled))

clust_df = df.copy()
clust_df['Cluster'] = km.labels_
clust_df.sort_values(by='Cluster')

####################### PCA ###########################
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt 

prcomp = PCA()
scores = prcomp.fit_transform(df_scaled)
print(prcomp.explained_variance_ratio_)
df_PCs = pd.DataFrame({'PC1':scores[:,0],
                       'PC2':scores[:,1],
                       'Cluster':km.labels_})
df_PCs['Cluster'] = df_PCs['Cluster'].astype(object)

sns.scatterplot(data=df_PCs, x='PC1',palette="bright",
                y='PC2',hue='Cluster')
for i in np.arange(0, df.shape[0] ):
    plt.text(scores[i,0], scores[i,1], list(df.index)[i])
plt.show()


print(clust_df.groupby(by='Cluster').mean() )

################# milk ###########################

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

km = KMeans(n_clusters=3, random_state=23)
km.fit(milkscaled)


prcomp = PCA()
scores = prcomp.fit_transform(milkscaled)
print(prcomp.explained_variance_ratio_)
df_PCs = pd.DataFrame({'PC1':scores[:,0],
                       'PC2':scores[:,1],
                       'Cluster':km.labels_})
df_PCs['Cluster'] = df_PCs['Cluster'].astype(object)

sns.scatterplot(data=df_PCs, x='PC1',palette="bright",
                y='PC2',hue='Cluster')
for i in np.arange(0, milk.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(milk.index)[i],fontsize=8)

plt.legend(loc='best')
plt.show()

clust_df = milk.copy()
clust_df['Cluster'] = km.labels_
clust_df.sort_values(by='Cluster')

print(clust_df.groupby(by='Cluster').mean() )

