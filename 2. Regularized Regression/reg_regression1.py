import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np 

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
print(mean_squared_error(y_test, y_pred))

############Ridge##################
params = [0.1,0.5,1,2,2.5,3]
errors = []
for a in params:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred))
 
print("Best Score:", np.min(errors))
i_min = np.argmin(errors)
print("Best Parameter:", params[i_min])


############Lasso##################

params = [0.1,0.5,1,2,2.5,3]
errors = []
for a in params:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    errors.append(mean_squared_error(y_test, y_pred))
 
print("Best Score:", np.min(errors))
i_min = np.argmin(errors)
print("Best Parameter:", params[i_min])

############ ElasticNet #############
from sklearn.metrics import mean_squared_error as mse
elastic = ElasticNet()
elastic.fit(X_train, y_train)

y_pred = elastic.predict(X_test)
print(mean_squared_error(y_test, y_pred))
###################################################
l1_rat = [0.1, 0.25, 0.5, 0.8, 0.9]
alpha = [0.1, 0.5, 1, 2, 2.5, 3]

errors = dict()
for a in alpha:
    for l in l1_rat:
        elastic = ElasticNet(alpha=a, l1_ratio=l)
        elastic.fit(X_train, y_train)
        y_pred = elastic.predict(X_test)
        errors['alpha:'+ str(a)+", l1_ratio:"+ str(l)] = mse(y_test, y_pred)
er_pd = pd.Series(errors)    
er_pd.sort_values()
