import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 

milk = pd.read_csv("milk.csv", index_col=0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(milk)
prcomp = PCA()

scores = prcomp.fit_transform(scaled_data)

v_covar = np.cov(scaled_data, rowvar=False)
values, vectors = np.linalg.eig(v_covar)
print("Eigen values =",values)
print("Variances of PC Scores = ", np.var(scores, axis=0, ddof=1) )
print(prcomp.explained_variance_)

tot_var = np.sum(prcomp.explained_variance_)
print("% age variations:")
print((prcomp.explained_variance_/tot_var)*100)
print("Cumulative %age variations explained:")
print(np.cumsum(prcomp.explained_variance_ratio_*100))

##################################################
from pca import pca
model = pca()
results = model.fit_transform(scaled_data,col_labels=milk.columns,
                              row_labels=list(milk.index))
model.biplot(label=True,legend=True)
for i in np.arange(0, milk.shape[0] ):
    plt.text(scores[i,0], scores[i,1], list(milk.index)[i])
plt.show()



######### Nutrient ############
from sklearn.pipeline import Pipeline
nut = pd.read_csv("nutrient.csv", index_col=0)

scaler = StandardScaler()
prcomp = PCA()
pipe_PCA = Pipeline([('SCL', scaler),('COMP', prcomp)])
scores = pipe_PCA.fit_transform(nut)

print("Variances of PC Scores = " )
print(prcomp.explained_variance_)

print("Cumulative %age variations explained:")
print(np.cumsum(prcomp.explained_variance_ratio_*100))

######################### Biplot ##################################
scaled_data = scaler.fit_transform(nut)
model = pca()
results = model.fit_transform(scaled_data,col_labels=nut.columns,
                              row_labels=list(nut.index))
scores = prcomp.fit_transform(scaled_data)
model.biplot(label=True,legend=True)
for i in np.arange(0, nut.shape[0] ):
    plt.text(scores[i,0], scores[i,1], list(nut.index)[i])
plt.show()


                   