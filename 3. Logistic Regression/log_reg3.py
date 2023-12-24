import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import numpy as np 
from sklearn.preprocessing import LabelEncoder

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
lr = LogisticRegression(multi_class='multinomial')
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

y_pred = lr.predict(X_test)
print(accuracy_score(y_test, y_pred))

#################### Grid Search CV ###################
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
lr = LogisticRegression()
params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga'],
          'multi_class':['ovr','multinomial']}
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
bm = gcv.best_estimator_
print(bm.coef_)
print(bm.intercept_)

############### inferencing #################
tst_img = pd.read_csv("tst_img.csv")
# y_pred_prob = bm.predict_proba(tst_img)
# np.argmax(y_pred_prob, axis=1)
# OR
y_pred = bm.predict(tst_img)
print(lbl.inverse_transform(y_pred))
