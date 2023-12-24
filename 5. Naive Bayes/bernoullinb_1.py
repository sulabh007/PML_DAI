import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 

telecom = pd.read_csv("Telecom.csv")
dum_telecom = pd.get_dummies(telecom, drop_first=True)
X = dum_telecom.drop('Response_Y', axis=1)
y = dum_telecom['Response_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=23,
                                                    stratify=y)
nb = BernoulliNB()
nb.fit(X_train, y_train) # Model Building: Apriori Probs Calculated

y_probs = nb.predict_proba(X_test) # Posterior Probs Calculated
y_pred = nb.predict(X_test) # Applying built on test data

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

################# Cancer ########################################
cancer = pd.read_csv("Cancer.csv", index_col=0)
dum_cancer = pd.get_dummies(cancer, drop_first=True)
X = dum_cancer.drop('Class_recurrence-events', axis=1)
y = dum_cancer['Class_recurrence-events']



nb = BernoulliNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'alpha': np.linspace(0, 5, 20)}
gcv = GridSearchCV(nb, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
