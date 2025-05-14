from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)

x = [
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
    [0, 1, 1],
]

y = [
    't-shirt',
    'hoodie',
    'raincoat',
    'sweater',
    'raincoat',
    't-shirt',
    'jacket',
]

clf.fit(x, y)

print(clf.predict([[0, 0, 1]]))
print(clf.predict([[1, 1, 0]]))