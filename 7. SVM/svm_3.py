import pandas as pd 
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np 
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline 

concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete[['Strength']]

scaler_y = StandardScaler()
y_scl = scaler_y.fit_transform(y)
################### Linear SVM ##########################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVR(kernel='linear')
pipe = Pipeline([('SCL', scaler),('SVM', svm)])

params = { 'SCL':[StandardScaler(), MinMaxScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__epsilon': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold)
gcv.fit(X, y_scl[:,0])
print(gcv.best_params_)
print(gcv.best_score_)

################### Radial SVM ##########################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVR(kernel='rbf')
pipe = Pipeline([('SCL', scaler),('SVM', svm)])

params = { 'SCL':[StandardScaler(), MinMaxScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma':np.linspace(0.001, 5, 10),
          'SVM__epsilon': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold)
gcv.fit(X, y_scl[:,0])
print(gcv.best_params_)
print(gcv.best_score_)
bm_std = gcv.best_estimator_

### MM
scaler_y = MinMaxScaler()
y_scl = scaler_y.fit_transform(y)
################### Linear SVM ##########################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVR(kernel='linear')
pipe = Pipeline([('SCL', scaler),('SVM', svm)])

params = { 'SCL':[StandardScaler(), MinMaxScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__epsilon': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold)
gcv.fit(X, y_scl[:,0])
print(gcv.best_params_)
print(gcv.best_score_)

################### Radial SVM ##########################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVR(kernel='rbf')
pipe = Pipeline([('SCL', scaler),('SVM', svm)])

params = { 'SCL':[StandardScaler(), MinMaxScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma':np.linspace(0.001, 5, 10),
          'SVM__epsilon': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold)
gcv.fit(X, y_scl[:,0])
print(gcv.best_params_)
print(gcv.best_score_)
bm_mm = gcv.best_estimator_

############## Inferencing ##########
tst_conc = pd.read_csv("testConcrete.csv")
y_scl = bm_mm.predict(tst_conc)
min_y = y.min()
max_y = y.max()
y_scl*(max_y.values[0] - min_y.values[0]) + min_y.values[0]
# OR
scaler_y.inverse_transform(y_scl.reshape(-1,1))


########## Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV
scaler_y = MinMaxScaler()
y_scl = scaler_y.fit_transform(y)

scaler = StandardScaler()
svm = SVR(kernel='rbf')
pipe = Pipeline([('SCL', scaler),('SVM', svm)])

params = { 'SCL':[StandardScaler(), MinMaxScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma':np.linspace(0.001, 5, 10),
          'SVM__epsilon': np.linspace(0.001, 5, 10)}
rgcv = RandomizedSearchCV(pipe, param_distributions=params,verbose=3,
                   cv=kfold, n_iter=100, random_state=23)
rgcv.fit(X, y_scl[:,0])
print(rgcv.best_params_)
print(rgcv.best_score_)
