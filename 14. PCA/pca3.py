import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC 

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Acc with Original =", accuracy_score(y_test, y_pred))

################## PCA ########################
prcomp = PCA()
scores = prcomp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scores[:,:3], y, 
                                                    test_size=0.3,
                                                    random_state=23)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Acc with PC =", accuracy_score(y_test, y_pred))

############### PCAs separately for train and test ###############
from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23) 
prcomp = PCA(n_components=3)
svm = SVC(kernel='linear')
pipe_svc = Pipeline([('PC', prcomp),('SVM', svm)])
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
print("Acc with PC =", accuracy_score(y_test, y_pred))              

################ Grid Search CV ######################
from sklearn.model_selection import GridSearchCV, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
prcomp = PCA(n_components=3)
svm = SVC(kernel='linear')
pipe_svc = Pipeline([('PC', prcomp),('SVM', svm)])
print(pipe_svc.get_params())
params = {'PC__n_components':[2,3,4,5]}
gcv = GridSearchCV(pipe_svc, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

####################### HR #################################
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler 

hr = pd.read_csv("HR_comma_sep.csv")
X = hr.drop('left', axis=1)
y = hr['left']
ohc = OneHotEncoder(sparse_output=False,drop='first')
ct_enc = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_include=['int64','float64'])),
       verbose_feature_names_out=False).set_output(transform="pandas")
scaler = StandardScaler()
prcomp = PCA()
svm = SVC(kernel='linear')
pipe_svc = Pipeline([('TRNF', ct_enc),('SCL', scaler),
                     ('PC', prcomp),('SVM', svm)])
params = {'PC__n_components':[2,3,4,5,6,7,8,9],
          'SVM__C':np.linspace(0.001, 3, 5)}
gcv = GridSearchCV(pipe_svc, param_grid=params,cv=kfold,
                   verbose=3)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

