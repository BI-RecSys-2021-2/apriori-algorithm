import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)

apriori = pd.read_csv("../apriori_algorithm/data/output/minS_01_minL_1.csv")
prod = pd.read_csv("./data/dataymx.csv", encoding='cp949')

def search_product(antecedent):
    condition = (apriori.antecedents == antecedent)
    result = apriori[condition]
    result = result.sort_values('lift', ascending=False)

    products = result['consequents'].values.tolist()
    print(result)
    print(products)

    return products

def priority(consequents):
    condition = prod['search'].isin(consequents)
    result = prod[condition]
    result = result.sort_values('freshness')

    print(result)

    return result

consequents = search_product('bottled water')
priority(consequents)
