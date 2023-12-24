import pandas as pd 
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN

df = pd.read_csv("data.csv")
y = df['Bankrupt?']
X = df.drop('Bankrupt?', axis=1)

print(y.value_counts())
print(y.value_counts(normalize=True)*100)
########## Under Sampling ##################
u_sampler = RandomUnderSampler(random_state=23)
X_us, y_us = u_sampler.fit_resample(X, y)
print(y_us.value_counts())
print(y_us.value_counts(normalize=True)*100)
########## Over Sampling ##################
o_sampler = RandomOverSampler(random_state=23)
X_os, y_os = o_sampler.fit_resample(X, y)
print(y_os.value_counts())
print(y_os.value_counts(normalize=True)*100)
############ SMOTE ###########################
smote = SMOTE(random_state=23)
X_smote, y_smote = smote.fit_resample(X, y)
print(y_smote.value_counts())
print(y_smote.value_counts(normalize=True)*100)
########### ADASYN ###################
adasyn = ADASYN(random_state=23)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

print(y_adasyn.value_counts())
print(y_adasyn.value_counts(normalize=True)*100)

