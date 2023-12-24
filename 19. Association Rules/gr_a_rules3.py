import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import os 
import gradio as gr
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer")

cancer = pd.read_csv("Cancer.csv", index_col=0)
fp_df = pd.get_dummies(cancer,prefix_sep='=')


def gen_rules(min_sup, min_conf):
    itemsets = apriori(fp_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules = rules[rules['lift']>1]
    return rules.sort_values(by='lift', ascending=False)

demo = gr.Interface(gen_rules, 
                    inputs= [gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Support",
                                   minimum=0.0001, maximum=1),
                             gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Confidence",    
                                   minimum=0.0001, maximum=1)], 
                    outputs='dataframe')

if __name__ == "__main__":
    demo.launch()