import pandas as pd 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np 

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']


kfold = KFold(n_splits=5, shuffle=True, random_state=23)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.3)
results = cross_val_score(elastic, X, y,
                          cv=kfold,
                          scoring='neg_mean_squared_error')
print(results.mean())

########## Tuning ###############
l1_rat = [0.1, 0.25, 0.5, 0.8, 0.9]
alpha = [0.1, 0.5, 1, 2, 2.5, 3]

errors = dict()
for a in alpha:
    for l in l1_rat:
        elastic = ElasticNet(alpha=a, l1_ratio=l)
        results = cross_val_score(elastic, X, y,
                                  cv=kfold,
                                  scoring='neg_mean_squared_error')
        errors['alpha:'+ str(a)+", l1_ratio:"+ str(l)] = results.mean()
er_pd = pd.Series(errors)    
er_pd.sort_values(ascending=False)

########## Grid Search CV #######################
from sklearn.model_selection import GridSearchCV
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'alpha':alphas,'l1_ratio':l1_rat}
elastic = ElasticNet()
gcv = GridSearchCV(elastic, param_grid=params,cv=kfold,
                       scoring='neg_mean_squared_error')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_)

################### Concrete ####################
concrete = pd.read_csv("Concrete_Data.csv")
y = concrete['Strength']
X = concrete.drop('Strength', axis=1)
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'alpha':alphas,'l1_ratio':l1_rat}
elastic = ElasticNet()
gcv = GridSearchCV(elastic, param_grid=params,cv=kfold,
                       scoring='neg_mean_squared_error')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

################ Inferencing ####################
# elastic = ElasticNet(alpha=3.7895263157894736,
#                      l1_ratio=0.001)
# elastic.fit(X,y)

best_model = gcv.best_estimator_
unlabel_data = pd.read_csv("testConcrete.csv")
y_pred = best_model.predict(unlabel_data)

######### Medical ########################
medical = pd.read_csv("insurance.csv")
dum_med = pd.get_dummies(medical, drop_first=True)

y = dum_med['charges']
X = dum_med.drop('charges', axis=1)
l1_rat =  np.linspace(0.001, 0.999, 20)
alphas =  np.linspace(0.001, 4, 20)
params = {'alpha':alphas,'l1_ratio':l1_rat}
elastic = ElasticNet()
gcv = GridSearchCV(elastic, param_grid=params,cv=kfold,
                       scoring='neg_mean_squared_error')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

################ Inferencing ####################
best_model = gcv.best_estimator_
unlabel_data = pd.read_csv("tst_insure.csv")
unlabel_data = pd.get_dummies(unlabel_data,
                              drop_first=True)
y_pred = best_model.predict(unlabel_data)
