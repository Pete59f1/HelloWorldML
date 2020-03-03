from sklearn import tree
from sklearn.metrics import accuracy_score

# Training data
# Degrees in weather, 0 for clear sky, 1 for rain
xtrain = [[25, 0], [4, 1], [10, 1], [40, 1], [15, 0], [10, 0], [23, 1]]
# 0 for no jacket needed, 1 for jacket needed
ytrain = [0, 1, 1, 0, 0, 1, 0]

# Testing data
xtest = [[5, 1], [10, 0], [15, 1], [20, 0], [25, 1], [30, 0], [35, 1]]
ytest = [1, 1, 1, 0, 0, 0, 0]

# Training model
clf = tree.DecisionTreeClassifier()
clf.fit(xtrain, ytrain)

# Testing model
predictions = clf.predict(xtest)
print(clf.predict(xtest))
print(accuracy_score(ytest, predictions))
