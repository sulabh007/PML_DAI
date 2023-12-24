import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

pizza = pd.read_csv("pizza.csv")

X = pizza[['Promote']]
y = pizza['Sales']

X = sm.add_constant(X)

model = sm.OLS(y,X)
results = model.fit()
print(results.params)
print(results.summary())
#############################################

boston = pd.read_csv("Boston.csv")

X = boston.drop('medv', axis=1)
y = boston['medv']

X = sm.add_constant(X)

model = sm.OLS(y,X)
results = model.fit()
print(results.params)
print(results.summary())
