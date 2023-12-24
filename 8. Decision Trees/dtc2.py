import pandas as pd 
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 

wine = pd.read_csv(r"C:\Training\Kaggle\Competitions\Wine Quality\train.csv")

y = wine['quality']
X = wine.drop(['quality','Id'], axis=1)


######### Grid Search CV ###################
dtc = DecisionTreeClassifier(random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'min_samples_split':[2,3,4,5,6,7,8,9,10],
          'min_samples_leaf':np.arange(1,16)}
gcv = GridSearchCV(dtc, param_grid=params,verbose=3,
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

###### Inferencing
test = pd.read_csv(r"C:\Training\Kaggle\Competitions\Wine Quality\test.csv")
pred = best_model.predict(test.iloc[:,1:])
submit = pd.DataFrame({'Id':test.Id,
                       'quality':pred})

submit.to_csv(r"C:\Training\Kaggle\Competitions\Wine Quality\sbt_oct_dtc.csv",
              index=False)
