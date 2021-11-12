import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

df = pd.read_csv("./data/input/groceries_categorized.csv")

#One hot encoding
oh = pd.get_dummies(df.drop(columns=['Item(s)'], axis=1), prefix='', prefix_sep='')
oh = oh.groupby(level=0, axis=1).sum()

fi = apriori(oh, min_support=0.03, use_colnames=True)

rules = association_rules(fi, metric="lift", min_threshold=1)
#print(rules)

rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x))).astype("unicode")


outputPath = r'/Users/yejoolee/Library/Mobile Documents/com~apple~CloudDocs/2021-2/비즈니스인텔리전스/팀플/backend/apriori_algorithm/data/output/minS_03_minL_1.csv'
rules.to_csv(outputPath, index=True)

# max_i = 4
# for i, row in rules.iterrows():
#     print("Rule: " + list(row['antecedents'])[0] + " => " + list(row['consequents'])[0])
#     print("Support: " + str(round(row['support'], 2)))
#     print("Confidence: " + str(round(row['confidence'], 2)))
#     print("Lift: " + str(round(row['lift'], 2)))
#     print("========================================")
#     if i == max_i:
#         break

support = rules['support']
confidence = rules['confidence']

all_confidences = []
collective_strengths = []
cosine_similarities = []
for _, row in rules.iterrows():
    all_confidence_if = list(row['antecedents'])[0]
    all_confidence_then = list(row['consequents'])[0]
    if row['antecedent support'] <= row['consequent support']:
        all_confidence_if = list(row['consequents'])[0]
        all_confidence_then = list(row['antecedents'])[0]
    all_confidence = {all_confidence_if + ' => ' + all_confidence_then : \
                      row['support']/max(row['antecedent support'], row['consequent support'])}
    all_confidences.append(all_confidence)

    violation = row['antecedent support'] + row['consequent support'] - 2*row['support']
    ex_violation = 1-row['antecedent support']*row['consequent support'] - \
                   (1-row['antecedent support'])*(1-row['consequent support'])
    collective_strength = (1-violation)/(1-ex_violation)*(ex_violation/violation)
    collective_strengths.append(collective_strength)

    cosine_similarity = row['support']/np.sqrt(row['antecedent support']*row['consequent support'])
    cosine_similarities.append(cosine_similarity)
rules['all-confidence'] = all_confidences
rules['collective strength'] = collective_strengths
rules['cosine similarity'] = cosine_similarities

h = 347
s = 1
v = 1
colors = [
    mcl.hsv_to_rgb((h/360, 0.2, v)),
    mcl.hsv_to_rgb((h/360, 0.55, v)),
    mcl.hsv_to_rgb((h/360, 1, v))
]
cmap = LinearSegmentedColormap.from_list('my_cmap', colors, gamma=2)
measures = ['lift', 'leverage', 'conviction', 'all-confidence', 'collective strength', 'cosine similarity']

fig = plt.figure(figsize=(15, 10))
fig.set_facecolor('white')
for i, measure in enumerate(measures):
    ax = fig.add_subplot(320+i+1)
    if measure != 'all-confidence':
        scatter = ax.scatter(support, confidence, c=rules[measure], cmap=cmap)
    else:
        scatter = ax.scatter(support, confidence, c=rules['all-confidence'].map(lambda x: [v for k,v in x.items()][0]), cmap=cmap)
    ax.set_xlabel('support')
    ax.set_ylabel('confidence')
    ax.set_title(measure)

    fig.colorbar(scatter, ax=ax)
fig.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
