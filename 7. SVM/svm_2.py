import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline 

img = pd.read_csv("Image_Segmention.csv")
lbl = LabelEncoder()
img['Class'] = lbl.fit_transform(img['Class'])
print(lbl.classes_)

X = img.drop('Class', axis=1)
y = img['Class']


################### Linear SVM ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVC(kernel='linear',
          probability=True, random_state=23)
pipe = Pipeline([('SCL', scaler),('SVM', svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### MM
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = MinMaxScaler()
svm = SVC(kernel='linear',
          probability=True, random_state=23)
pipe = Pipeline([('SCL', scaler),('SVM', svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################### Radial SVM ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = StandardScaler()
svm = SVC(kernel='rbf',
          probability=True, random_state=23)
pipe = Pipeline([('SCL', scaler),('SVM', svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma':  list(np.linspace(0.001, 5, 10))+['scale','auto'],
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### MM
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
scaler = MinMaxScaler()
svm = SVC(kernel='rbf',
          probability=True, random_state=23)
pipe = Pipeline([('SCL', scaler),('SVM', svm)])
params = {'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma': list(np.linspace(0.001, 5, 10))+['scale','auto'],
          'SVM__decision_function_shape':['ovo','ovr']}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)

########### all in one
svm = SVC(kernel='rbf',
          probability=True, random_state=23)
pipe = Pipeline([('SCL', scaler),('SVM', svm)])
params = {'SCL':[MinMaxScaler(),StandardScaler()],
          'SVM__C': np.linspace(0.001, 5, 10),
          'SVM__gamma': list(np.linspace(0.001, 5, 10))+['scale','auto'],
          'SVM__decision_function_shape':['ovo','ovr'],
          'SVM__kernel':['linear','rbf']}
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


