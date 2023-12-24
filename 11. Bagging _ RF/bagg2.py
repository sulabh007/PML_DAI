import pandas as pd 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

concrete = pd.read_csv("Concrete_Data.csv")
y = concrete['Strength']
X = concrete.drop('Strength', axis=1)

dtr = DecisionTreeRegressor(random_state=23)
bagg = BaggingRegressor(dtr,random_state=23)
print(bagg.get_params())
params = {'estimator__max_depth':[None, 3, 5],
          'estimator__min_samples_split':[2, 5, 10],
          'estimator__min_samples_leaf':[1,5,10],
          'n_estimators':[10,30,50]}
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(bagg, param_grid=params,cv=kfold,verbose=3)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
