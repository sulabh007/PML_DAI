import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("Groceries.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
fp_df = pd.DataFrame(te_ary, columns=te.columns_)
one_freq = fp_df.sum().reset_index()
one_freq.columns=['Items', 'Freq']
print(one_freq.sort_values(by='Freq', ascending=False))
