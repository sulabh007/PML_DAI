import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

y = dum_hr['left']
X = dum_hr.drop('left', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
dtc = DecisionTreeClassifier(random_state=23, 
                             max_depth=2)
dtc.fit(X_train, y_train)
#### Creating a graphical view of tree
plt.figure(figsize=(25,10))
plot_tree(dtc,feature_names=X_train.columns,
               class_names=['Stayed','Left'],
               filled=True,fontsize=12) 
plt.show()
############## 

y_pred = dtc.predict(X_test)
y_pred_prob = dtc.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))


######### Grid Search CV ###################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'min_samples_split':[2,5,10],
          'min_samples_leaf':[1,3,5,7,10,15]}
gcv = GridSearchCV(dtc, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

####
best_model = gcv.best_estimator_

df_imp = pd.DataFrame({ 'variable':best_model.feature_names_in_,
                        'importance':best_model.feature_importances_})

df_imp = df_imp[df_imp['importance']>0.0001]
df_imp = df_imp.sort_values(by='importance')
plt.barh(df_imp['variable'] ,df_imp['importance'])
plt.ylabel("Importance")
plt.show()
