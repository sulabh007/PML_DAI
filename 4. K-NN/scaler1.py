import numpy as np
x_trn = np.array([[24000, 8],
                  [30000, 2],
                  [35000, 7],
                  [34500, 1]])

x_tst = np.array([[34000, 7],
                  [12000, 1]])

### Standard Scaling
col1 = x_trn[:,0]
trf_col1 = (col1 - col1.mean())/np.std(col1)
print(np.mean(trf_col1), np.std(trf_col1))
col2 = x_trn[:,1]
trf_col2 = (col2 - col2.mean())/np.std(col2)
print(np.mean(trf_col2), np.std(trf_col2))

### All columns at once
means = x_trn.mean(axis=0)
stds = x_trn.std(axis=0)
print((x_trn-means)/stds)
print((x_tst-means)/stds)
print(means)
print(stds)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(x_trn))
print(scaler.transform(x_tst))
print(scaler.mean_)
print(scaler.scale_)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit_transform(x_trn))
print(scaler.transform(x_tst))

