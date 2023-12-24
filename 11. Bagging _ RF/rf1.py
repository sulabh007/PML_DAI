import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
bcancer['Class'] = lbl.fit_transform(bcancer['Class'])
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

rf = RandomForestClassifier(random_state=23)
params = {'max_features': [3,4,5,6]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=23)
gcv_rf = GridSearchCV(rf, param_grid=params,scoring='neg_log_loss', 
                   cv=kfold, verbose=3)
gcv_rf.fit(X, y)
print(gcv_rf.best_params_)
print(gcv_rf.best_score_)

bm_rf = gcv_rf.best_estimator_

df_imp = pd.DataFrame({ 'variable':bm_rf.feature_names_in_,
                        'importance':bm_rf.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.title("RF Feature Importance")
plt.ylabel("Importance")
plt.show()

######## Decision Tree
params = {'max_depth': [2,3,4,5,6,None],
          'min_samples_split':[2,5,10],
          'min_samples_leaf':[1,3,5,7,10,15]}
dtc = DecisionTreeClassifier(random_state=23)
gcv_tree = GridSearchCV(dtc, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)

####
bm_tree = gcv_tree.best_estimator_

df_imp = pd.DataFrame({ 'variable':bm_tree.feature_names_in_,
                        'importance':bm_tree.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.title("Tree Feature Importance")
plt.ylabel("Importance")
plt.show()

from joblib import dump
dump(bm_rf, "RF.joblib")
