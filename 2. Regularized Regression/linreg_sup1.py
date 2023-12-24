import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np 

boston = pd.read_csv("Boston.csv")

lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=23)
X = boston.drop('medv', axis=1)
y = boston['medv']

train, test = train_test_split(boston, test_size=0.3,
                               random_state=23)
X_train = train.drop('medv', axis=1)
y_train = train['medv']
X_test = test.drop('medv', axis=1)
y_test = test['medv']
# OR
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
lr.fit(X_train, y_train)
#print(lr.coef_)
y_pred = lr.predict(X_test)
#print(np.mean((y_test-y_pred)**2))
print(mean_squared_error(y_test, y_pred))

########################################
dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_test)
#print(np.mean((y_test-y_pred)**2))
print(mean_squared_error(y_test, y_pred))

############ Housing #######################
housing = pd.read_csv("Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)

lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=23)

train, test = train_test_split(dum_hous, test_size=0.3,
                               random_state=23)

X_train = train.drop('price', axis=1)
y_train = train['price']
X_test = test.drop('price', axis=1)
y_test = test['price']

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

########################################
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
print(mean_squared_error(y_test, y_pred))
