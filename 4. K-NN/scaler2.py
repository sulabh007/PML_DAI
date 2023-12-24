import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler

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
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = knn.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))


############### with scaling #################
scaler = StandardScaler()
X_scl_trn = scaler.fit_transform(X_train)
X_scl_tst = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_scl_trn, y_train)
y_pred = knn.predict(X_scl_tst)
print(accuracy_score(y_test, y_pred))

y_pred_prob = knn.predict_proba(X_scl_tst)
print(log_loss(y_test, y_pred_prob))

######### with pipeline ###############
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=1)
pipe = Pipeline([('SCL',scaler),('KNN',knn)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))


#################### Grid Search CV ###################
from sklearn.preprocessing import MinMaxScaler
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,11),
          'SCL':[StandardScaler(),MinMaxScaler()]}
# log loss
gcv = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)


