import pandas as pd

df = pd.read_csv("./housing-price/data/train.csv")

print(df.isna().sum())
