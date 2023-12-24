import pandas as pd 
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

tsne = TSNE(random_state=23, 
            perplexity=40).set_output(transform='pandas')
X_tsne = tsne.fit_transform(X)

tsne_pd = pd.concat([X_tsne, y], axis=1)

sns.scatterplot(data=tsne_pd, x='tsne0',
                y='tsne1', hue='Class')
plt.title("t-SNE")
plt.show()


################ Grid Search CV ######################
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

scores = []
perplex = [5,15,30,35, 45, 50]
for i in perplex:
    tsne = TSNE(random_state=23, perplexity=i)
    X_tsne = tsne.fit_transform(X)
    svm = SVC(kernel='linear')
    results = cross_val_score(svm, X_tsne, y, cv=kfold, verbose=3)
    scores.append( results.mean() )   

print("Best score =", np.max(scores))
print("Best perplexity =", perplex[np.argmax(scores)])


############ HR #####################
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 

hr = pd.read_csv("HR_comma_sep.csv")
X = hr.drop('left', axis=1)
y = hr['left']
ohc = OneHotEncoder(sparse_output=False,drop='first')
scaler = StandardScaler()
ct_enc = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       (scaler,
        make_column_selector(dtype_include=['int64','float64'])),
       verbose_feature_names_out=False).set_output(transform="pandas")
X_enc = ct_enc.fit_transform(X)


scores = []
perplex = [20,30, 45]
for i in perplex:
    tsne = TSNE(random_state=23, perplexity=i)
    X_tsne = tsne.fit_transform(X_enc)
    svm = SVC(kernel='linear')
    results = cross_val_score(svm, X_tsne, y, cv=kfold, n_jobs=-1)
    scores.append( results.mean() )   
    print(results.mean())

print("Best score =", np.max(scores))
print("Best perplexity =", perplex[np.argmax(scores)])
