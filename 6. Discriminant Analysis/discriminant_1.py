import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 

brupt = pd.read_csv("data.csv")
X = brupt.drop('Bankrupt?', axis=1)
y = brupt['Bankrupt?']

#### LDA
lda = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'solver': ['svd','lsqr','eigen']}
gcv = GridSearchCV(lda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##### QDA
qda = QuadraticDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {}
gcv = GridSearchCV(qda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_score_)
