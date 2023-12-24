import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import numpy as np 

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
elastic = ElasticNet()
elastic.fit(X_train, y_train)

y_pred = elastic.predict(X_test)
print(r2_score(y_test, y_pred))

########## Grid Search CV #######################
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'alpha':alphas,'l1_ratio':l1_rat}
elastic = ElasticNet()
gcv = GridSearchCV(elastic, param_grid=params,cv=kfold,
                       scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

