import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline 

train = pd.read_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\train.csv")
X = train.drop('label', axis=1)
y = train['label']
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

############### Logistic 
lr = LogisticRegression()
params = {'penalty':['l1','l2',None],
          'multi_class':['multinomial']}
gcv_lr = GridSearchCV(lr, param_grid=params,verbose=2,
                   cv=kfold, scoring='neg_log_loss',
                   n_jobs=-1)
gcv_lr.fit(X, y)
print(gcv_lr.best_params_)
print(gcv_lr.best_score_)

###### Inferencing
bm_logistic = gcv_lr.best_estimator_
test = pd.read_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\test.csv")
y_pred = bm_logistic.predict(test)
submit_lg = pd.DataFrame({'ImageId':np.arange(1,28001),
                          'Label':y_pred })
submit_lg.to_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\sbtOct_logReg.csv",
                 index=False)

######### Job serialization
# from joblib import dump, load
# dump(bm_logistic, r"C:\Training\Kaggle\Competitions\Digit Recognizer\logistic.joblib") 

# load_logistic = load(r"C:\Training\Kaggle\Competitions\Digit Recognizer\logistic.joblib")


############ KNN
scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL',scaler),('KNN',knn)])

print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,6)}
# log loss
gcv = GridSearchCV(pipe, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

###### Inferencing
bm_knn = gcv.best_estimator_
test = pd.read_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\test.csv")
y_pred = bm_knn.predict(test)
submit_lg = pd.DataFrame({'ImageId':np.arange(1,28001),
                          'Label':y_pred })
submit_lg.to_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\sbtOct_KNN.csv",
                 index=False)

########### Gaussian NB
nb = GaussianNB()
params = {'var_smoothing': np.linspace(0, 1, 20)}
gcv_nb = GridSearchCV(nb, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_nb.fit(X, y)
print(gcv_nb.best_params_)
print(gcv_nb.best_score_)

###### Inferencing
bm_nb = gcv_nb.best_estimator_
test = pd.read_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\test.csv")
y_pred = bm_nb.predict(test)
submit_lg = pd.DataFrame({'ImageId':np.arange(1,28001),
                          'Label':y_pred })
submit_lg.to_csv(r"C:\Training\Kaggle\Competitions\Digit Recognizer\sbtOct_nb.csv",
                 index=False)