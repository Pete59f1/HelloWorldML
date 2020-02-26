import numpy as np
from sklearn import tree
# Import dataset iris
from sklearn.datasets import load_iris
iris = load_iris()

# Printing some of the data
# print(iris.feature_names)
# print(iris.target_names)
print(iris.data[51])
# print(iris.target[0])

# The first setosa, versicolor and virginica
testdataindex = [0, 50, 100]

# Training data - used for training
train_target = np.delete(iris.target, testdataindex)
train_data = np.delete(iris.data, testdataindex, axis=0)

# Test data - used for testing our model
test_target = iris.target[testdataindex]
test_data = iris.data[testdataindex]

# Creating out tree classifier, and train it with our data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

# print(test_target)
# print(clf.predict(test_data))

# Trying to predict what kind of flower
print(clf.predict([[6.4, 3.2, 4.5, 1.5]]))
