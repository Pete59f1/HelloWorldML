from sklearn import tree
# Smooth are 1 and bumpy are 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# Apples are 0 oranges are 1
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[120, 1]]))
print(clf.predict([[150, 0]]))
