import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

headers = ['age','income','credit_score','education','employment_years','loan_amount','default']
df = pd.read_csv('data/sklearn_test_data.csv', encoding='utf-8', header=None, names=headers)

# print(df.head())
df = df.drop(columns=['education', 'employment_years'])
df = df.describe().round(3)

X1 = df.iloc[:, 0:13]
X2 = df.iloc[:, 0:13]

# ss = StandardScaler()
# X1 = ss.fit_transform(X1)
# X1 = pd.DataFrame(X1, columns=['income','credit_score','loan_amount','default'])
# df = X1.head()

print(tabulate(df, headers='keys', tablefmt=''))