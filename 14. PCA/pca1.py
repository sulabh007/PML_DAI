import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

prcomp = PCA()
scores = prcomp.fit_transform(X)

print(np.cumsum(prcomp.explained_variance_ratio_))

PCs = pd.DataFrame({'PC1':scores[:,0],
                    'PC2':scores[:,1],
                    'Class':y})
sns.scatterplot(data=PCs, x='PC1',
                y='PC2', hue='Class')
plt.show()



                   