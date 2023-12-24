import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np 

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
X = cancer.drop('Class', axis=1)
y = cancer['Class']

print(y.value_counts(normalize=True)*100)
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
print(y_train.value_counts(normalize=True)*100)
print(y_train.value_counts())

print(y_test.value_counts(normalize=True)*100)
print(y_test.value_counts())

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#################### Grid Search CV ###################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()

params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga']}
# Default scoring=acc score
gcv = GridSearchCV(lr, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
# log loss
gcv = GridSearchCV(lr, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

###
best_model = gcv.best_estimator_
print(best_model.coef_)

############ HR ###################
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()

params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga']}
# Default scoring=acc score
gcv = GridSearchCV(lr, param_grid=params,
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
# log loss
gcv = GridSearchCV(lr, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
