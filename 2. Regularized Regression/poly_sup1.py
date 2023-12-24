import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

pizza = pd.read_csv("pizza.csv")

X = pizza[['Promote']]
y = pizza['Sales']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)
print(lr.intercept_, lr.coef_)

############ Boston #################
boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 
                                                    test_size=0.3,
                                                    random_state=23)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

########## California Housing #####################
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, 
                                      as_frame=True)
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 
                                                    test_size=0.3,
                                                    random_state=23)
dtr = DecisionTreeRegressor(random_state=23)
dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_test)
print(mean_squared_error(y_test, y_pred))

