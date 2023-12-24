import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
bcancer['Class'] = lbl.fit_transform(bcancer['Class'])
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)

lr = LogisticRegression()
svm = SVC(kernel='linear', random_state=23, probability=True)
dtc = DecisionTreeClassifier(random_state=23)
rf = RandomForestClassifier(random_state=23)

# w/o pass through
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)],
                           final_estimator=rf)
stack.fit(X_train, y_train) # Excecutes all ops on train set
y_pred = stack.predict(X_test) # Excecutes all ops on test set
print(accuracy_score(y_test, y_pred))

# with pass through
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)],
                           final_estimator=rf, passthrough=True)
stack.fit(X_train, y_train) # Excecutes all ops on train set
y_pred = stack.predict(X_test) # Excecutes all ops on test set
print(accuracy_score(y_test, y_pred))


####### other model
gbm = GradientBoostingClassifier(random_state=23)
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)],
                           final_estimator=gbm, passthrough=True)
stack.fit(X_train, y_train) # Excecutes all ops on train set
y_pred = stack.predict(X_test) # Excecutes all ops on test set
print(accuracy_score(y_test, y_pred))


############### Grid Search CV ##################
gbm = GradientBoostingClassifier(random_state=23)
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)],
                           final_estimator=gbm, passthrough=True)
print(stack.get_params())
params = {'TREE__max_depth': [2,3,4,5,6,None],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,3,5,7,10,15],
          'LR__penalty':['l1','l2',None],
          'SVM__C': np.linspace(0.001, 3, 5),
          'final_estimator__learning_rate':[0.1, 0.2],
          'final_estimator__max_depth':  [2,3,4,5,6,None]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_stack = RandomizedSearchCV(stack, param_distributions=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss', n_iter=100,
                   random_state=23)
gcv_stack.fit(X, y)
print(gcv_stack.best_params_)
print(gcv_stack.best_score_)
