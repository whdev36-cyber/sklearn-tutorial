import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/sklearn_test_data.csv', encoding='utf-8')

X = df.drop(columns=['education'])
y = df['income']

# print(X.head())
# print(X.shape)
# print(y.shape)
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train.head())
print(X_train.describe())
print(X_train.describe().round(3))