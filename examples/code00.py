from sklearn import tree

# CREATE MODEL
clf = tree.DecisionTreeClassifier()

# TRAIN MODEL
x = [
    [1], # sunny
    [0]  # cloudy
]
y = ['sunny', 'cloudy']
clf = clf.fit(x, y)

# PREDICT
result = clf.predict([[0]])
print(result)
