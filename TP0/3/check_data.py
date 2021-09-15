import pandas as pd
data = pd.read_csv('diagonal.test', sep = ',')
summary = data.describe()
summary = summary.transpose()
summary.head()