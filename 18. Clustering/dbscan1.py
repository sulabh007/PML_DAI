from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler()
milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(milkscaled)
print(clust_DB.labels_)

milkscaled['Clust'] = clust_DB.labels_
milk_scl_inliers = milkscaled[milkscaled['Clust']!=-1]
print(silhouette_score(milk_scl_inliers.iloc[:,:-1],
                 milk_scl_inliers.iloc[:,-1]) )

milkscaled=scaler.fit_transform(milk)

milkscaled = pd.DataFrame(milkscaled,
                          columns=milk.columns,
                          index=milk.index)

eps_range = np.linspace(0.1, 1, 10)
mp_range = [2,3,4,5]
a = []
for e in eps_range:
    for m in mp_range:
        clust_DB = DBSCAN(eps=e, min_samples=m)
        clust_DB.fit(milkscaled.iloc[:,:5])
        distincts = len(np.unique(clust_DB.labels_))
        if distincts > 2:
            milkscaled['Clust'] = clust_DB.labels_
            milk_scl_inliers = milkscaled[milkscaled['Clust']!=-1]
            sil = silhouette_score(milk_scl_inliers.iloc[:,:-1],
                             milk_scl_inliers.iloc[:,-1]) 
            a.append([e,m,distincts,sil])

df_results = pd.DataFrame(a, columns=['eps','mp','n_clust','sil'])            
df_results.sort_values(by='sil',ascending=False,inplace=True)
print("Best:\n", df_results.iloc[0])
best_eps = df_results.iloc[0]['eps']
best_mp = int(df_results.iloc[0]['mp'])
########## Plotting PCs #################
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt 
prcomp = PCA()
scores = prcomp.fit_transform(milkscaled.iloc[:,:5])
clust_DB = DBSCAN(eps=best_eps, min_samples=best_mp)
clust_DB.fit(milkscaled.iloc[:,:5])
print(clust_DB.labels_)

df_PCs = pd.DataFrame({'PC1':scores[:,0],
                       'PC2':scores[:,1],
                       'Cluster':clust_DB.labels_})
df_PCs['Cluster'] = df_PCs['Cluster'].astype(object)

sns.scatterplot(data=df_PCs, x='PC1',palette="bright",
                y='PC2',hue='Cluster')
for i in np.arange(0, milk.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(milk.index)[i],fontsize=6)
plt.legend(loc='best')
plt.show()

################# USArrests ##########################
arrests = pd.read_csv("USArrests.csv", index_col=0)

scaler = StandardScaler()
ar_scaled=scaler.fit_transform(arrests)

ar_scaled = pd.DataFrame(ar_scaled,
                          columns=arrests.columns,
                          index=arrests.index)

eps_range = np.linspace(0.1, 1, 10)
mp_range = [2,3,4,5]
a = []
for e in eps_range:
    for m in mp_range:
        clust_DB = DBSCAN(eps=e, min_samples=m)
        clust_DB.fit(ar_scaled.iloc[:,:4])
        distincts = len(np.unique(clust_DB.labels_))
        if distincts > 2:
            ar_scaled['Clust'] = clust_DB.labels_
            arr_scl_inliers = ar_scaled[ar_scaled['Clust']!=-1]
            sil = silhouette_score(arr_scl_inliers.iloc[:,:-1],
                             arr_scl_inliers.iloc[:,-1]) 
            a.append([e,m,distincts,sil])

df_results = pd.DataFrame(a, columns=['eps','mp','n_clust','sil'])            
df_results.sort_values(by='sil',ascending=False,inplace=True)
print("Best:\n", df_results.iloc[0])
best_eps = df_results.iloc[0]['eps']
best_mp = int(df_results.iloc[0]['mp'])
########## Plotting PCs #################
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt 
prcomp = PCA()
scores = prcomp.fit_transform(ar_scaled.iloc[:,:4])
clust_DB = DBSCAN(eps=best_eps, min_samples=best_mp)
clust_DB.fit(ar_scaled.iloc[:,:4])
print(clust_DB.labels_)

df_PCs = pd.DataFrame({'PC1':scores[:,0],
                       'PC2':scores[:,1],
                       'Cluster':clust_DB.labels_})
df_PCs['Cluster'] = df_PCs['Cluster'].astype(object)

sns.scatterplot(data=df_PCs, x='PC1',palette="bright",
                y='PC2',hue='Cluster')
for i in np.arange(0, arrests.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(arrests.index)[i],fontsize=6)
plt.legend(loc='best')
plt.show()


################# USArrests ##########################
nut = pd.read_csv("nutrient.csv", index_col=0)

scaler = StandardScaler()
ar_scaled=scaler.fit_transform(nut)

ar_scaled = pd.DataFrame(ar_scaled,
                          columns=nut.columns,
                          index=nut.index)

eps_range = np.linspace(0.1, 1, 10)
mp_range = [2,3,4,5]
a = []
for e in eps_range:
    for m in mp_range:
        clust_DB = DBSCAN(eps=e, min_samples=m)
        clust_DB.fit(ar_scaled.iloc[:,:-1])
        distincts = len(np.unique(clust_DB.labels_))
        if distincts > 2:
            ar_scaled['Clust'] = clust_DB.labels_
            arr_scl_inliers = ar_scaled[ar_scaled['Clust']!=-1]
            sil = silhouette_score(arr_scl_inliers.iloc[:,:-1],
                             arr_scl_inliers.iloc[:,-1]) 
            a.append([e,m,distincts,sil])

df_results = pd.DataFrame(a, columns=['eps','mp','n_clust','sil'])            
df_results.sort_values(by='sil',ascending=False,inplace=True)
print("Best:\n", df_results.iloc[0])
best_eps = df_results.iloc[0]['eps']
best_mp = int(df_results.iloc[0]['mp'])
########## Plotting PCs #################
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt 
prcomp = PCA()
scores = prcomp.fit_transform(ar_scaled.iloc[:,:-1])
clust_DB = DBSCAN(eps=best_eps, min_samples=best_mp)
clust_DB.fit(ar_scaled.iloc[:,:-1])
print(clust_DB.labels_)

df_PCs = pd.DataFrame({'PC1':scores[:,0],
                       'PC2':scores[:,1],
                       'Cluster':clust_DB.labels_})
df_PCs['Cluster'] = df_PCs['Cluster'].astype(object)

sns.scatterplot(data=df_PCs, x='PC1',palette="bright",
                y='PC2',hue='Cluster')
for i in np.arange(0, nut.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(nut.index)[i],fontsize=6)
plt.legend(loc='best')
plt.show()