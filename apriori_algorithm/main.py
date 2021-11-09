import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# for dirname, _, filenames in os.walk('./data/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

df = pd.read_csv("./data/input/groceries_categorized.csv")
df.head(10)
df.describe()
df.info()
