import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import numpy as np 

brupt = pd.read_csv("data.csv")
X = brupt.drop('Bankrupt?', axis=1)
y = brupt['Bankrupt?']

print(y.value_counts(normalize=True)*100)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
print(y_train.value_counts(normalize=True)*100)
print(y_test.value_counts(normalize=True)*100)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))

y_pred = lr.predict(X_test)
print(accuracy_score(y_test, y_pred))

####################### Kfold ##########################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()
# Default scoring=acc score
results = cross_val_score(lr, X, y, cv=kfold)
print(results.mean())
# log loss
results = cross_val_score(lr, X, y, cv=kfold,
                          scoring='neg_log_loss')
print(results.mean())

#################### Grid Search CV ###################
from sklearn.model_selection import GridSearchCV 
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
