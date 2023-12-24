import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import matplotlib.pyplot as plt

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
cancer['Class'] = lbl.fit_transform(cancer['Class'])
X = cancer.drop('Class', axis=1)
y = cancer['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_prob = lr.predict_proba(X_test)[:,1]

fpr, tpr, thres = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.ylabel("Sensitivity")
plt.xlabel("1-Specificity")
plt.show()

### AUC
print(roc_auc_score(y_test, y_pred_prob))

#################### Grid Search CV ###################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()

params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga']}
gcv = GridSearchCV(lr, param_grid=params,scoring='roc_auc',
                   cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


################ Multi-class *******

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
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)

### AUC OVR
print(roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))

### AUC OVO
print(roc_auc_score(y_test, y_pred_prob, multi_class='ovo'))
