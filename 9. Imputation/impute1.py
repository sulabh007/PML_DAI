import pandas as pd
from sklearn.impute import SimpleImputer

chem = pd.read_csv("ChemicalProcess.csv")
print(chem.isnull().sum())

imp = SimpleImputer()
imputed = imp.fit_transform(chem)
pd_imp = pd.DataFrame(imputed,columns=chem.columns)
print(pd_imp.isnull().sum().sum())

###### JobSalary2
job = pd.read_csv("JobSalary2.csv", index_col=0)

imp_mean = SimpleImputer()
imputed_mean = imp_mean.fit_transform(job)

imp_med = SimpleImputer(strategy='median')
imputed_med = imp_med.fit_transform(job)



