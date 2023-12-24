import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import numpy as np 

chem = pd.read_csv("ChemicalProcess.csv")
#print(chem.isnull().sum())

X = chem.drop('Yield', axis=1)
y = chem['Yield']

#### Simple Imputer
imp = SimpleImputer()
elastic = ElasticNet()
pipe = Pipeline([('IMPUTE', imp),('ELASTIC', elastic)])
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'ELASTIC__alpha':alphas,'ELASTIC__l1_ratio':l1_rat,
          'IMPUTE__strategy':['mean','median']}

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

### KNN Imputation
imp_knn = KNNImputer()
elastic = ElasticNet()
pipe = Pipeline([('IMPUTE', imp_knn),('ELASTIC', elastic)])
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'ELASTIC__alpha':alphas,'ELASTIC__l1_ratio':l1_rat,
          'IMPUTE__n_neighbors':[1,2,3,4,5]}

gcv = GridSearchCV(pipe, param_grid=params,cv=kfold,
                   n_jobs=-1)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
