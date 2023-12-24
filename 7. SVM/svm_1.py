import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
bcancer['Class'] = lbl.fit_transform(bcancer['Class'])
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
svm = SVC(kernel='linear',
          probability=True, random_state=23)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = svm.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))

################### Linear SVM ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'C': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


################### Polynomial SVM ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
svm = SVC(kernel='poly',
          probability=True, random_state=23)
params = {'C': np.linspace(0.001, 5, 10),
          'degree':[1,2,3], 'coef0':[0.0, 0.3]}
gcv = GridSearchCV(svm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


################### Radial SVM ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
svm = SVC(kernel='rbf',
          probability=True, random_state=23)
params = {'C': np.linspace(0.001, 5, 10),
          'gamma': np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(svm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
