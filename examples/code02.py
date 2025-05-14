from sklearn import tree
import matplotlib.pyplot as plt

clf = tree.DecisionTreeClassifier()

x = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1]
]
y = ['soup', 'sandwich', 'tea', 'none', 'pizza', 'cola', 'hotdog', 'water']
clf = clf.fit(x, y)

plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=['hungry', 'time', 'with_friends'],
    class_names=clf.classes_, filled=True)
plt.savefig('img/res01.png')