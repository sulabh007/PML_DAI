import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np 

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

#################### Grid Search CV ###################
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsRegressor()
params = {'n_neighbors': np.arange(1,11)}

# log loss
gcv = GridSearchCV(knn, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

####### with scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,11),
          'SCL':[StandardScaler(),MinMaxScaler()]}

gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############### Medical Costs ###########################
medical = pd.read_csv("insurance.csv")
dum_med = pd.get_dummies(medical, drop_first=True)
X = dum_med.drop('charges', axis=1)
y = dum_med['charges']

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
scaler = StandardScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,11),
          'SCL':[StandardScaler(),MinMaxScaler()]}

gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########### Elastic Net ##################
from sklearn.linear_model import ElasticNet

l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'alpha':alphas,'l1_ratio':l1_rat}
elastic = ElasticNet()
gcv = GridSearchCV(elastic, param_grid=params,cv=kfold,
                       scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

