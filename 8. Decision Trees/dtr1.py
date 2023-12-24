import pandas as pd 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt 

concrete = pd.read_csv("Concrete_Data.csv")
y = concrete['Strength']
X = concrete.drop('Strength', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
dtr = DecisionTreeRegressor(random_state=23, 
                             max_depth=2)
dtr.fit(X_train, y_train)
#### Creating a graphical view of tree
plt.figure(figsize=(25,10))
plot_tree(dtr,feature_names=X_train.columns,
               filled=True,fontsize=20) 
plt.show()
############## 

kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'min_samples_split':[2,3,4,5,6,7,8,9,10],
          'min_samples_leaf':np.arange(1,16)}
gcv = GridSearchCV(dtr, param_grid=params,verbose=3,
                   cv=kfold)
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