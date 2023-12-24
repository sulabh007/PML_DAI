import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

bcancer = pd.read_csv("BreastCancer.csv", index_col=0)
lbl = LabelEncoder()
bcancer['Class'] = lbl.fit_transform(bcancer['Class'])
X = bcancer.drop('Class', axis=1)
y = bcancer['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23,
                                                    stratify=y)
svm = SVC(probability=True, random_state=23)
lr = LogisticRegression()
nb = GaussianNB()
voting = VotingClassifier(estimators=[('SVM',svm),
                                       ('LR',lr), ('NB',nb)], 
                          voting='soft')
voting.fit(X_train, y_train)
y_pred_prob = voting.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))
y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

################ Grid Search CV ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
print(voting.get_params())
params = {'SVM__gamma':['scale','auto'],
          'SVM__C': np.linspace(0.001,5, 5),
          'LR__penalty':['l1','l2','elastic',None],
          'NB__var_smoothing':np.linspace(0.0001, 0.999, 5)}
gcv = GridSearchCV(voting, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########### Image Segmentation ###############

img = pd.read_csv("Image_Segmention.csv")
lbl = LabelEncoder()
img['Class'] = lbl.fit_transform(img['Class'])
print(lbl.classes_)

X = img.drop('Class', axis=1)
y = img['Class']

###### Unweighted
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
print(voting.get_params())
params = {'SVM__gamma':['scale','auto'],
          'SVM__C': np.linspace(0.001,5, 5),
          'LR__penalty':['l1','l2','elastic',None],
          'LR__multi_class':['ovr','multinomial']}
gcv = GridSearchCV(voting, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
unweighted_ll = gcv.best_score_

###### Only SVM
params = {'gamma':['scale','auto'],
          'C': np.linspace(0.001,5, 5)}
gcv_svm = GridSearchCV(svm, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_svm.fit(X, y)
print(gcv_svm.best_params_)
print(gcv_svm.best_score_)
svm_ll = gcv_svm.best_score_

###### Only LR
params = {'penalty':['l1','l2','elastic',None],
          'multi_class':['ovr','multinomial']}
gcv_lr = GridSearchCV(lr, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_lr.fit(X, y)
print(gcv_lr.best_params_)
print(gcv_lr.best_score_)
lr_ll = gcv_lr.best_score_

###### Only NB
params = {}
gcv_nb = GridSearchCV(nb, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv_nb.fit(X, y)
print(gcv_nb.best_params_)
print(gcv_nb.best_score_)
nb_ll = gcv_nb.best_score_

print("Unweighted =", unweighted_ll)
print("Only SVM =", svm_ll)
print("Only LR =", lr_ll)
print("Only NB =", nb_ll)

#### Weighted
voting = VotingClassifier(estimators=[('SVM',svm),
                                       ('LR',lr), ('NB',nb)], 
                          voting='soft', 
                          weights=[0.5,0.4,0.1])
params = {'SVM__gamma':['scale','auto'],
          'SVM__C': np.linspace(0.001,5, 5),
          'LR__penalty':['l1','l2','elastic',None],
          'LR__multi_class':['ovr','multinomial']}
gcv = GridSearchCV(voting, param_grid=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
weighted_ll = gcv.best_score_

print("Unweighted =", unweighted_ll)
print("Only SVM =", svm_ll)
print("Only LR =", lr_ll)
print("Only NB =", nb_ll)
print("Weighted =", weighted_ll)


######## Only Tree
dtc = DecisionTreeClassifier(random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth': [2,3,4,5,6,None],
          'min_samples_split':[2,5,10],
          'min_samples_leaf':[1,3,5,7,10,15]}
gcv_tree = GridSearchCV(dtc, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
tree_ll = gcv_tree.best_score_

print("Only SVM =", svm_ll)
print("Only LR =", lr_ll)
print("Only Tree =",tree_ll)


###### Unweighted

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
voting = VotingClassifier(estimators=[('SVM',svm),
                                       ('LR',lr), ('TREE',dtc)], 
                          voting='soft')

params = {'SVM__gamma':['scale','auto'],
          'SVM__C': np.linspace(0.001,5, 5),
          'LR__penalty':['l1','l2',None],
          'LR__multi_class':['ovr','multinomial'],
          'TREE__max_depth': [2,4,None],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,5,10,15]}
rgcv = RandomizedSearchCV(voting, param_distributions=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss', random_state=23,
                   n_iter=50)
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)
unweighted_ll = rgcv.best_score_

###### Weighted

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
voting = VotingClassifier(estimators=[('SVM',svm),
                                       ('LR',lr), ('TREE',dtc)], 
                          voting='soft',
                          weights=[0.45,0.35,0.2])

params = {'SVM__gamma':['scale','auto'],
          'SVM__C': np.linspace(0.001,5, 5),
          'LR__penalty':['l1','l2',None],
          'LR__multi_class':['ovr','multinomial'],
          'TREE__max_depth': [2,4,None],
          'TREE__min_samples_split':[2,5,10],
          'TREE__min_samples_leaf':[1,5,10,15]}
rgcv = RandomizedSearchCV(voting, param_distributions=params,verbose=3,
                   cv=kfold, scoring='neg_log_loss', random_state=23,
                   n_iter=50)
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)
weighted_ll = rgcv.best_score_

print("Unweighted =", unweighted_ll)
print("Only SVM =", svm_ll)
print("Only LR =", lr_ll)
print("Only Tree =", tree_ll)
print("Weighted =", weighted_ll)
