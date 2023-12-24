import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier 

img = pd.read_csv("Image_Segmention.csv")
lbl = LabelEncoder()
img['Class'] = lbl.fit_transform(img['Class'])
print(lbl.classes_)

X = img.drop('Class', axis=1)
y = img['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
############# Logistic
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)
lr_ll = log_loss(y_test, y_pred_prob)
print("Alone Logistic =", lr_ll)

#########################
bagg = BaggingClassifier(lr, n_estimators=15,oob_score=True,
                         random_state=23)
bagg.fit(X_train, y_train)
y_pred_prob = bagg.predict_proba(X_test)
bagg_ll = log_loss(y_test, y_pred_prob)
print("Bagging =", bagg_ll)

y_pred = bagg.predict(X_test)
print(accuracy_score(y_test, y_pred))

print("OOB Score =", bagg.oob_score_)

############# SVM-Linear
from sklearn.svm import SVC
svm = SVC(kernel='linear', probability=True, random_state=23)
svm.fit(X_train, y_train)
y_pred_prob = svm.predict_proba(X_test)
svm_ll = log_loss(y_test, y_pred_prob)
print("Alone SVM =", svm_ll)

#########################
bagg = BaggingClassifier(svm, n_estimators=50,
                         random_state=23)
bagg.fit(X_train, y_train)
y_pred_prob = bagg.predict_proba(X_test)
bagg_ll = log_loss(y_test, y_pred_prob)
print("Bagging =", bagg_ll)

########## Grid Search CV - Models #################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()
svm = SVC(kernel='linear', probability=True, random_state=23)
dtc = DecisionTreeClassifier(random_state=23)
bagg = BaggingClassifier(random_state=23)
print(bagg.get_params())
params = {'estimator':[lr, svm, dtc],
          'n_estimators':[10, 15, 30, 40, 50]}
gcv = GridSearchCV(bagg, param_grid=params,
                   cv=kfold, verbose=3,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



