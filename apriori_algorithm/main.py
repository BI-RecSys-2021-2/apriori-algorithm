import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

df = pd.read_csv("./data/input/groceries_categorized.csv")

#One hot encoding
oh = pd.get_dummies(df.drop(columns=['Item(s)'], axis=1), prefix='', prefix_sep='')
oh = oh.groupby(level=0, axis=1).sum()
print(oh)