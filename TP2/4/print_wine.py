import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df["target"] = pd.Series(wine.target)
df.to_csv("wine.csv", sep=",", index=False)
