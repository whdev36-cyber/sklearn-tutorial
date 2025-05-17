import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/sklearn_test_data.csv.csv', encoding='utf-8')

X = df.drop(columns=['education'])

print(df)