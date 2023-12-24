import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 


mowers = pd.read_csv("RidingMowers.csv")
lbl = LabelEncoder()
mowers['Response'] = lbl.fit_transform(mowers['Response'])

X = mowers.drop('Response', axis=1)
y = mowers['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = knn.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))

#################### Grid Search CV ###################

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()

params = {'n_neighbors': np.arange(1,11)}

# log loss
gcv = GridSearchCV(knn, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################ Breast Cancer ######################

cancer = pd.read_csv("BreastCancer.csv", index_col=0)
X = cancer.drop('Class', axis=1)
y = cancer['Class']


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()

params = {'n_neighbors': np.arange(1,11)}

# log loss
gcv = GridSearchCV(knn, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#################### Image Segmentation #################

img = pd.read_csv("Image_Segmention.csv")
lbl = LabelEncoder()
img['Class'] = lbl.fit_transform(img['Class'])
print(lbl.classes_)

X = img.drop('Class', axis=1)
y = img['Class']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()

params = {'n_neighbors': np.arange(1,11)}

# log loss
gcv = GridSearchCV(knn, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
