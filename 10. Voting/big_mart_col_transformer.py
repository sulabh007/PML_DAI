import pandas as pd 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt 
from sklearn.compose import make_column_transformer 
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline 

train = pd.read_csv(r"C:\Training\AV\Big Mart III\train_v9rqX0R.csv")

X_train = train.drop(['Item_Identifier','Outlet_Identifier', 
                      'Item_Outlet_Sales'],
                     axis=1)
y_train = train['Item_Outlet_Sales']

imp_mode = SimpleImputer(strategy='most_frequent')
imp_med = SimpleImputer(strategy='median')
ohc = OneHotEncoder(sparse_output=False)
ct_imp = make_column_transformer((imp_mode,
       make_column_selector(dtype_include=object)),
       (imp_med,
        make_column_selector(dtype_include=['int64','float64'])),
       verbose_feature_names_out=False).set_output(transform="pandas")
ct_enc = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_include=['int64','float64'])),
       verbose_feature_names_out=False).set_output(transform="pandas")
dtr = DecisionTreeRegressor(random_state=23)
pipe = Pipeline([('IMPUTE', ct_imp),('ENC',ct_enc),('TREE',dtr)])

pipe.fit(X_train, y_train)
############# Inferencing
test = pd.read_csv(r"C:\Training\AV\Big Mart III\test_AbJTz2l.csv")
X_test = test.drop(['Item_Identifier','Outlet_Identifier'],
                     axis=1)
predictions = pipe.predict(X_test)
