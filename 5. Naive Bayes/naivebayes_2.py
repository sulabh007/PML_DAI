import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 

brupt = pd.read_csv("data.csv")
X = brupt.drop('Bankrupt?', axis=1)
y = brupt['Bankrupt?']

nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'var_smoothing': np.linspace(0, 1, 20)}
gcv = GridSearchCV(nb, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########### HR ##########################
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
############### Logistic 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga'],
          'l1_ratio':np.linspace(0,1,10)}

gcv_lr = GridSearchCV(lr, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_lr.fit(X, y)
print(gcv_lr.best_params_)
print(gcv_lr.best_score_)

############## kNN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,51),
          'SCL':[StandardScaler(),MinMaxScaler()]}
gcv_knn = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_knn.fit(X, y)
print(gcv_knn.best_params_)
print(gcv_knn.best_score_)

############# GaussianNB
nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'var_smoothing': np.linspace(0, 1, 20)}
gcv_nb = GridSearchCV(nb, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_nb.fit(X, y)
print(gcv_nb.best_params_)
print(gcv_nb.best_score_)

################# Image Segmentation ##################
from sklearn.preprocessing import LabelEncoder

img = pd.read_csv("Image_Segmention.csv")
lbl = LabelEncoder()
img['Class'] = lbl.fit_transform(img['Class'])
print(lbl.classes_)

X = img.drop('Class', axis=1)
y = img['Class']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
############### Logistic 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

params = {'penalty':['l1','l2','elasticnet',None],
          'solver':['lbfgs', 'liblinear', 
                    'newton-cg', 'newton-cholesky', 'sag', 'saga'],
          'l1_ratio':np.linspace(0,1,10),
          'multi_class':['ovr','multinomial']}

gcv_lr = GridSearchCV(lr, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_lr.fit(X, y)
print(gcv_lr.best_params_)
print(gcv_lr.best_score_)

############## kNN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
scaler = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,21),
          'SCL':[StandardScaler(),MinMaxScaler()]}
gcv_knn = GridSearchCV(pipe, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_knn.fit(X, y)
print(gcv_knn.best_params_)
print(gcv_knn.best_score_)

############# GaussianNB
nb = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'var_smoothing': np.linspace(0, 1, 20)}
gcv_nb = GridSearchCV(nb, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_nb.fit(X, y)
print(gcv_nb.best_params_)
print(gcv_nb.best_score_)
