from sklearn import tree
import matplotlib.pyplot as plt

clf = tree.DecisionTreeClassifier()

# 😋 is hungry?,
# ⏳ is there enough time?,
# 👥 with friends?
x = [
    [1, 1, 0],  # hungry, time, alone -> soup
    [1, 0, 0],  # hungry, no time, alone -> sandwich
    [0, 1, 1],  # not hungry, time, with friends -> tea or cola
    [0, 0, 0],  # noy hungry, not time, alone -> none
]
y = ['soup', 'sandwich', 'tea', 'none']
clf = clf.fit(x, y)

plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=['Hungry', 'Time', 'With Fields'],
    class_names=clf.classes_, filled=True)
plt.savefig('img/res00.png')