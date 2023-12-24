from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import pandas as pd 
from sklearn.model_selection import KFold, RandomizedSearchCV
import numpy as np 
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 

concrete = pd.read_csv("Concrete_Data.csv")
y = concrete['Strength']
X = concrete.drop('Strength', axis=1)

elastic = ElasticNet()
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe_knn = Pipeline([('SCL',scaler),('NG',knn)])
dtr = DecisionTreeRegressor(random_state=23)

xgb = XGBRegressor(random_state=23)
lbgm = LGBMRegressor(random_state=23)
rf = RandomForestRegressor(random_state=23)
cgbm = CatBoostRegressor(random_state=23)

stack = StackingRegressor([('ELASTIC',elastic),('KNN', pipe_knn),
                           ('TREE',dtr)], passthrough=True)
print(stack.get_params())

params = {'ELASTIC__l1_ratio':np.linspace(0.001, 0.99, 4),
          'ELASTIC__alpha': [0.1, 1],
          'KNN__NG__n_neighbors':[1,5,10],
          'TREE__min_samples_leaf':[1,5,10],
          'TREE__min_samples_split':[2,5,10],
          'final_estimator':[xgb,lbgm,rf,cgbm]}

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv_stack = RandomizedSearchCV(stack, param_distributions=params,
                               verbose=3, cv=kfold,
                               random_state=23, n_iter=100)
gcv_stack.fit(X, y)
print(gcv_stack.best_params_)
print(gcv_stack.best_score_)


bm_stack = gcv_stack.best_estimator_

from joblib import dump 
dump(bm_stack , 'stack.joblib')




