import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
bcancer['Class'] = lbl.fit_transform(bcancer['Class'])
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=23)

################ X Gradient Boosting ####################

xgbm = XGBClassifier(random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'learning_rate':np.linspace(0.001, 0.999, 10),
          'n_estimators':[50, 100, 150]}

gcv_xgbm = GridSearchCV(xgbm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_xgbm.fit(X, y)
print(gcv_xgbm.best_params_)
print(gcv_xgbm.best_score_)

####
bm_xgbm = gcv_xgbm.best_estimator_

df_imp = pd.DataFrame({ 'variable':bm_xgbm.feature_names_in_,
                        'importance':bm_xgbm.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.title("XGBoost Feature Importance")
plt.ylabel("Importance")
plt.show()



################ Light GBM ####################
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'learning_rate':np.linspace(0.001, 0.999, 10),
          'n_estimators':[50, 100, 150]}

gcv_lgbm = GridSearchCV(lgbm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_lgbm.fit(X, y)
print(gcv_lgbm.best_params_)
print(gcv_lgbm.best_score_)

####
bm_lgbm = gcv_lgbm.best_estimator_

df_imp = pd.DataFrame({ 'variable':bm_lgbm.feature_name_,
                        'importance':bm_lgbm.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.title("Light GBM Feature Importance")
plt.ylabel("Importance")
plt.show()


################ Cat Boost ####################
from catboost import CatBoostClassifier
cgbm = CatBoostClassifier(random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'learning_rate':np.linspace(0.001, 0.999, 10),
          'n_estimators':[50, 100, 150]}

gcv_cgbm = GridSearchCV(cgbm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_cgbm.fit(X, y)
print(gcv_cgbm.best_params_)
print(gcv_cgbm.best_score_)

####
bm_cgbm = gcv_cgbm.best_estimator_

df_imp = pd.DataFrame({ 'variable':bm_cgbm.feature_names_,
                        'importance':bm_cgbm.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.title("Light GBM Feature Importance")
plt.ylabel("Importance")
plt.show()

