import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

df = pd.read_csv("../apriori_algorithm/data/output/minS_01_minL_1.csv")

def search_product(antecedent):
    condition = (df.antecedents == 'beef')
    result = df[condition]
    result = result.sort_values('lift', ascending=False)

    print(result)

search_product('beef')