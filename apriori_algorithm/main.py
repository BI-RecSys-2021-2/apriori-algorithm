import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

df = pd.read_csv("./data/input/groceries_categorized.csv")

#One hot encoding
oh = pd.get_dummies(df.drop(columns=['Item(s)'], axis=1), prefix='', prefix_sep='')
oh = oh.groupby(level=0, axis=1).sum()

fi = apriori(oh, min_support=0.02, use_colnames=True)

rules = association_rules(fi, metric="lift", min_threshold=1)
print(rules)

outputPath = r'/Users/yejoolee/Library/Mobile Documents/com~apple~CloudDocs/2021-2/비즈니스인텔리전스/팀플/backend/apriori_algorithm/data/output/minS_02_minL_1.csv'
rules.to_csv(outputPath, index=True)
